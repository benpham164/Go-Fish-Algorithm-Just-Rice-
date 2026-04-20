"""
simulate_gofish.py
==================
A complete 4-player Go Fish engine that pits our competition bot against
three baseline opponents (random, greedy, tracker) and reports win rates,
average sets, and head-to-head matrices.

Tournaments included
--------------------
1. Fresh deals
   * 4 players, 5 cards each, 32-card stock.  Seating shuffled per game.
   * Default 200 games per matchup family.

2. Seeded from organizer history files
   * Take e.g. "mid-game.json".  Fast-forward an actual engine to a state
     that matches the file (using GameState as the public-info constraint
     and sampling hidden hands consistent with all knowns), then play it
     out against baselines.

Baseline bots intentionally cover the most common heuristic styles a
competition entry might use:

    RandomBot   - picks any legal ask uniformly.
    GreedyBot   - asks for the rank it has the most copies of, targeting
                  the player with the largest hand.
    TrackerBot  - lightweight memory of who asked for what; if anyone has
                  signaled a rank we hold, we ask THEM.  Otherwise greedy.

Run:
    python simulate_gofish.py                       # quick run
    python simulate_gofish.py --games 400 --seed 7
    python simulate_gofish.py --seeded-only
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict
from typing import Callable, Dict, List, Optional, Tuple

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)
import gofish_bot as gb                                            # noqa: E402

RANKS = gb.RANKS
SUITS = ["C", "D", "H", "S"]


# ---------------------------------------------------------------------------
# Bots
# ---------------------------------------------------------------------------

class BaseBot:
    """Common interface.  Each bot receives the public observation in the
    organizer JSON shape and returns {"Requestee":..., "Request":...}."""

    name = "base"

    def __init__(self, name: str) -> None:
        self.name = name

    def decide(self, observation: Dict, rng: random.Random) -> Dict[str, str]:
        raise NotImplementedError


class RandomBot(BaseBot):
    def decide(self, observation, rng):
        my_hand = [c[0] for c in observation["Hand"]]
        opps = [p["Name"] for p in observation["Players"]
                if p["Name"] != self.name and p["HandSize"] > 0]
        if not opps or not my_hand:
            return {"Requestee": "", "Request": ""}
        return {"Requestee": rng.choice(opps),
                "Request": rng.choice(my_hand)}


class GreedyBot(BaseBot):
    """Always ask for what we have most of, target the biggest hand."""

    def decide(self, observation, rng):
        my_hand = [c[0] for c in observation["Hand"]]
        if not my_hand:
            return {"Requestee": "", "Request": ""}
        cnt = Counter(my_hand)
        rank = max(cnt, key=lambda r: (cnt[r], rng.random()))
        opps = [(p["Name"], p["HandSize"])
                for p in observation["Players"]
                if p["Name"] != self.name and p["HandSize"] > 0]
        if not opps:
            return {"Requestee": "", "Request": rank}
        opps.sort(key=lambda x: -x[1])
        return {"Requestee": opps[0][0], "Request": rank}


class TrackerBot(BaseBot):
    """If any opponent has asked for a rank we hold, target them.  Else greedy."""

    def decide(self, observation, rng):
        my_hand = [c[0] for c in observation["Hand"]]
        if not my_hand:
            return {"Requestee": "", "Request": ""}
        my_set = set(my_hand)
        # Build "rank -> [opps that asked for it]" from history.
        signals: Dict[str, List[str]] = defaultdict(list)
        for t in observation.get("Turns", []):
            r = t.get("Rank")
            if r in my_set and t.get("Requester") != self.name:
                signals[r].append(t["Requester"])
        live = {p["Name"]: p["HandSize"] for p in observation["Players"]
                if p["Name"] != self.name and p["HandSize"] > 0}
        # Try to find a rank that someone showed interest in.
        for rank, askers in signals.items():
            recent = [a for a in askers if a in live]
            if recent:
                return {"Requestee": recent[-1], "Request": rank}
        # Fall back to greedy.
        return GreedyBot.decide(self, observation, rng)


class CompetitionBot(BaseBot):
    """Wraps gofish_bot.choose_move."""
    def __init__(self, name, mc_samples: int = 64, seed: Optional[int] = None):
        super().__init__(name)
        self.mc_samples = mc_samples
        self.seed = seed

    def decide(self, observation, rng):
        gs = gb.GameState.from_raw(observation, my_name=self.name)
        # Use rng to derive a per-decision seed for reproducibility.
        seed = rng.randint(0, 1 << 30) if self.seed is None else self.seed
        return gb.choose_move(gs, use_mc=True,
                              mc_samples=self.mc_samples, seed=seed)


# ---------------------------------------------------------------------------
# Game engine
# ---------------------------------------------------------------------------

class GoFishEngine:
    """Authoritative game state.  Bots only see the public observation."""

    def __init__(self, players: List[BaseBot], rng: random.Random,
                 deal_size: Optional[int] = None,
                 hands: Optional[Dict[str, List[str]]] = None,
                 deck: Optional[List[str]] = None,
                 sets: Optional[Dict[str, List[str]]] = None,
                 turns: Optional[List[Dict]] = None,
                 starting_player: int = 0) -> None:
        self.players = players
        self.names = [b.name for b in players]
        self.rng = rng

        if hands is None or deck is None:
            # Fresh deal.
            full = [r for r in RANKS for _ in SUITS]      # 52 cards
            rng.shuffle(full)
            n = len(players)
            d = deal_size if deal_size else (5 if n == 4 else 7)
            hands = {b.name: [full.pop() for _ in range(d)] for b in players}
            deck = full

        self.hands: Dict[str, List[str]] = {n: list(hands[n]) for n in self.names}
        self.deck: List[str] = list(deck)
        self.sets: Dict[str, List[str]] = (
            {n: list(sets.get(n, [])) for n in self.names}
            if sets is not None else {n: [] for n in self.names}
        )
        self.out_of_play: List[str] = sorted({r for v in self.sets.values() for r in v})
        self.turns: List[Dict] = list(turns or [])
        self.cur = starting_player
        self._check_initial_sets()

    # ---- helpers ----
    def _check_initial_sets(self) -> None:
        """A fresh deal can occasionally include 4-of-a-kind in one hand."""
        for n in self.names:
            self._collect_sets(n, last_rank=None)

    def _collect_sets(self, player: str, last_rank: Optional[str]) -> Optional[str]:
        """If `player` has 4 of any rank, lay it down and return that rank."""
        c = Counter(self.hands[player])
        for rank, cnt in list(c.items()):
            if cnt >= 4:
                self.hands[player] = [x for x in self.hands[player] if x != rank]
                self.sets[player].append(rank)
                if rank not in self.out_of_play:
                    self.out_of_play.append(rank)
                return rank
        return None

    def _observation_for(self, player: str) -> Dict:
        return {
            "Players": [
                {"Name": n, "Sets": len(self.sets[n]),
                 "HandSize": len(self.hands[n])}
                for n in self.names
            ],
            "Turns": list(self.turns),
            "OutOfPlay": list(self.out_of_play),
            "Hand": [[r, "?"] for r in self.hands[player]],
            "Deck": len(self.deck),
        }

    # ---- main loop ----
    def is_over(self) -> bool:
        # Standard Go Fish ends when all 13 sets are made or the game has
        # no legal moves (everyone with cards can only ask players with 0,
        # and the deck is empty).
        if sum(len(v) for v in self.sets.values()) >= 13:
            return True
        any_with_cards = any(len(self.hands[n]) > 0 for n in self.names)
        if not any_with_cards and not self.deck:
            return True
        return False

    def _living_opps(self, player: str) -> List[str]:
        return [n for n in self.names
                if n != player and len(self.hands[n]) > 0]

    def _safe_decision(self, bot: BaseBot, observation: Dict
                       ) -> Optional[Tuple[str, str]]:
        try:
            move = bot.decide(observation, self.rng)
        except Exception:
            return None
        if not isinstance(move, dict):
            return None
        rq, rk = move.get("Requestee"), move.get("Request")
        my_hand = self.hands[bot.name]
        opps = self._living_opps(bot.name)
        # Validate; if illegal, replace with random legal ask.
        if rq in opps and rk in my_hand and rk not in self.out_of_play:
            return rq, rk
        if not my_hand or not opps:
            return None
        return (self.rng.choice(opps), self.rng.choice(my_hand))

    def take_turn(self) -> None:
        player_name = self.names[self.cur]
        # Auto-draw if hand empty and deck nonempty.
        if not self.hands[player_name] and self.deck:
            self.hands[player_name].append(self.deck.pop())
        if not self.hands[player_name]:
            self.cur = (self.cur + 1) % len(self.names)
            return

        # Loop while the player keeps the turn.
        while True:
            obs = self._observation_for(player_name)
            decision = self._safe_decision(self.players[self.cur], obs)
            if decision is None:
                self.cur = (self.cur + 1) % len(self.names)
                return
            rq, rk = decision

            # Execute ask.
            other_hand = self.hands[rq]
            got = other_hand.count(rk)
            if got > 0:
                self.hands[rq] = [c for c in other_hand if c != rk]
                self.hands[player_name].extend([rk] * got)
                set_made = self._collect_sets(player_name, rk) or ""
                self.turns.append({
                    "Requester": player_name, "Requestee": rq,
                    "Rank": rk, "Received": got, "Set": set_made,
                })
                # Check if I can still ask (need cards in hand).
                if not self.hands[player_name] and self.deck:
                    self.hands[player_name].append(self.deck.pop())
                if not self.hands[player_name]:
                    break  # turn ends
                continue
            else:
                # Failed ask, draw from stock.
                self.turns.append({
                    "Requester": player_name, "Requestee": rq,
                    "Rank": rk, "Received": 0, "Set": "",
                })
                if not self.deck:
                    break
                drawn = self.deck.pop()
                self.hands[player_name].append(drawn)
                set_made = self._collect_sets(player_name, drawn)
                if drawn == rk:
                    continue   # lucky draw, keep turn
                break
        self.cur = (self.cur + 1) % len(self.names)

    def play(self, max_turns: int = 2000) -> Dict[str, int]:
        steps = 0
        while not self.is_over() and steps < max_turns:
            self.take_turn()
            steps += 1
        return {n: len(self.sets[n]) for n in self.names}


# ---------------------------------------------------------------------------
# Tournament drivers
# ---------------------------------------------------------------------------

def play_match(bot_factories: List[Callable[[str], BaseBot]],
               names: List[str],
               rng: random.Random,
               deal_size: int = 5) -> Dict[str, int]:
    bots = [factory(name) for factory, name in zip(bot_factories, names)]
    eng = GoFishEngine(bots, rng=rng, deal_size=deal_size)
    return eng.play()


def tournament(bot_factories: List[Callable[[str], BaseBot]],
               labels: List[str],
               games: int = 200, seed: int = 0,
               deal_size: int = 5) -> Dict[str, Dict[str, float]]:
    """Plays `games` games rotating seating to remove first-mover bias."""
    rng = random.Random(seed)
    n = len(bot_factories)
    wins = Counter()
    sets_total = Counter()
    ties = 0
    for g in range(games):
        rotation = (g % n)
        order = list(range(n))[rotation:] + list(range(n))[:rotation]
        match_factories = [bot_factories[i] for i in order]
        match_labels = [labels[i] for i in order]
        # Disambiguate names so two of the same kind can play together.
        seen: Counter = Counter()
        names = []
        for lbl in match_labels:
            seen[lbl] += 1
            names.append(f"{lbl}{seen[lbl]}" if seen[lbl] > 1 else lbl)
        result = play_match(match_factories, names, rng, deal_size=deal_size)
        max_sets = max(result.values())
        winners = [n for n, s in result.items() if s == max_sets]
        if len(winners) > 1:
            ties += 1
        for w in winners:
            wins[_label_of(w, match_labels, names)] += 1.0 / len(winners)
        for n_, s in result.items():
            sets_total[_label_of(n_, match_labels, names)] += s
    summary: Dict[str, Dict[str, float]] = {}
    for lbl in labels:
        summary[lbl] = {
            "win_rate": wins[lbl] / games,
            "avg_sets": sets_total[lbl] / games,
        }
    summary["_meta"] = {"games": games, "ties": ties}
    return summary


def _label_of(name: str, labels: List[str], names: List[str]) -> str:
    """Map seated name back to its bot family label (Strip trailing digit)."""
    idx = names.index(name)
    return labels[idx]


# ---------------------------------------------------------------------------
# Seeded tournament from organizer files
# ---------------------------------------------------------------------------

def seed_engine_from_file(path: str,
                          my_name: str,
                          baseline_factories: Dict[str, Callable[[str], BaseBot]],
                          rng: random.Random,
                          mc_samples: int = 64,
                          ) -> Tuple[GoFishEngine, Dict[str, str]]:
    """Build an engine whose state matches the organizer file.

    Steps:
        1. Parse the file via gofish_bot.GameState.
        2. Use Inference to compute which cards are pinned to opponents,
           which are in the deck pool, and remaining unseen copies.
        3. Sample a hidden world consistent with constraints (this lets us
           simulate from the *exact* public state in the file).
        4. Seat our CompetitionBot under my_name; assign each other live
           opponent a baseline bot (rotating through the provided dict).
    """
    with open(path) as f:
        raw = json.load(f)
    gs = gb.GameState.from_raw(raw, my_name=my_name)
    inf = gb.Inference(gs)
    mc = gb.MonteCarlo(gs, inf, n_samples=1, seed=rng.randint(0, 1 << 30))
    hands_unknown, deck = mc.sample_world()    # opp hands + remaining deck

    # Build the full hands dict including ours.
    hands = {gs.my_name: list(gs.my_hand)}
    for opp, cards in hands_unknown.items():
        hands[opp] = list(cards)

    # Seats: our CompetitionBot for my_name, baselines elsewhere.
    baseline_keys = list(baseline_factories.keys())
    seat_assignments: Dict[str, str] = {}
    bots: List[BaseBot] = []
    for i, name in enumerate(gs.players):
        if name == my_name:
            bots.append(CompetitionBot(name, mc_samples=mc_samples))
            seat_assignments[name] = "Competition"
        else:
            kind = baseline_keys[i % len(baseline_keys)]
            bots.append(baseline_factories[kind](name))
            seat_assignments[name] = kind

    # Sets per player as a list of completed ranks (we only know counts in the
    # JSON; create dummy placeholders for engine bookkeeping).
    sets_per_player = {n: ["?"] * gs.sets_by_player.get(n, 0)
                       for n in gs.players}
    # Make sure out_of_play accounts for those mock sets.
    for r in gs.out_of_play:
        # Attribute one of the unknown sets to whoever has the "?" slot,
        # or leave it floating — engine only uses the count for `is_over`.
        pass

    eng = GoFishEngine(
        bots,
        rng=rng,
        hands=hands,
        deck=deck,
        sets=sets_per_player,
        turns=gs.turns,
        starting_player=gs.players.index(my_name),
    )
    eng.out_of_play = list(gs.out_of_play)
    return eng, seat_assignments


def run_seeded_simulation(path: str, my_name: str,
                          games: int, seed: int,
                          mc_samples: int = 64) -> Dict[str, float]:
    rng = random.Random(seed)
    baseline_factories: Dict[str, Callable[[str], BaseBot]] = {
        "Greedy": GreedyBot,
        "Tracker": TrackerBot,
        "Random": RandomBot,
    }
    wins = Counter()
    sets_total = Counter()
    label_count: Counter = Counter()
    label_games = Counter()
    for g in range(games):
        eng, assignments = seed_engine_from_file(
            path, my_name, baseline_factories, rng, mc_samples=mc_samples,
        )
        result = eng.play()
        # Find winner
        max_sets = max(result.values())
        winners = [n for n, s in result.items() if s == max_sets]
        for n, s in result.items():
            label = assignments[n]
            sets_total[label] += s
            label_count[label] += 1
            label_games[label] += 1
        for w in winners:
            wins[assignments[w]] += 1.0 / len(winners)
    out: Dict[str, float] = {}
    for label, n in label_games.items():
        out[f"{label}.win_rate"] = wins[label] / games
        out[f"{label}.avg_sets"] = sets_total[label] / n
    out["games"] = games
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=200,
                    help="games per fresh-deal tournament")
    ap.add_argument("--seeded-games", type=int, default=80,
                    help="games per seeded-state tournament")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mc-samples", type=int, default=48,
                    help="Monte-Carlo samples for the competition bot "
                         "(lower = faster simulation)")
    ap.add_argument("--name", default="Cobra",
                    help="our bot's name in the simulation")
    ap.add_argument("--uploads", default="/sessions/brave-elegant-clarke/mnt/uploads")
    ap.add_argument("--fresh-only", action="store_true")
    ap.add_argument("--seeded-only", action="store_true")
    args = ap.parse_args()

    print("=" * 78)
    print(" GO FISH 4-PLAYER TOURNAMENT")
    print("=" * 78)

    if not args.seeded_only:
        # ---------- fresh deals ----------
        print(f"\n[Fresh deals] {args.games} games per matchup, deal size = 5\n")

        def comp(name): return CompetitionBot(name, mc_samples=args.mc_samples)

        matchups = [
            ("vs 3 Random",  [comp, RandomBot, RandomBot, RandomBot],
             ["Competition", "Random",  "Random",  "Random"]),
            ("vs 3 Greedy",  [comp, GreedyBot, GreedyBot, GreedyBot],
             ["Competition", "Greedy",  "Greedy",  "Greedy"]),
            ("vs 3 Tracker", [comp, TrackerBot, TrackerBot, TrackerBot],
             ["Competition", "Tracker", "Tracker", "Tracker"]),
            ("vs mixed",     [comp, GreedyBot, TrackerBot, RandomBot],
             ["Competition", "Greedy",  "Tracker", "Random"]),
        ]
        for title, factories, labels in matchups:
            res = tournament(factories, labels,
                             games=args.games, seed=args.seed)
            print(f"  {title}")
            print(f"    {'bot':<14}{'win rate':>12}{'avg sets':>14}")
            print(f"    {'-' * 38}")
            for lbl in dict.fromkeys(labels):
                if lbl == "_meta":
                    continue
                row = res[lbl]
                print(f"    {lbl:<14}{row['win_rate']*100:>10.1f}%"
                      f"{row['avg_sets']:>14.2f}")
            print(f"    games={res['_meta']['games']}, "
                  f"ties={res['_meta']['ties']}\n")

    if not args.fresh_only:
        # ---------- seeded from files ----------
        print(f"[Seeded from organizer files] "
              f"{args.seeded_games} games each\n")
        files = [
            "first-turn.json",
            "first-turn-fillable.json",
            "mid-game.json",
            "late-game.json",
        ]
        for fname in files:
            path = os.path.join(args.uploads, fname)
            if not os.path.isfile(path):
                continue
            # Pick our name appropriately for each file's player roster.
            with open(path) as f:
                roster = [p["Name"] for p in json.load(f).get("Players", [])]
            our_name = "steven" if "steven" in roster else (
                args.name if args.name in roster else (roster[0] if roster else args.name)
            )
            print(f"  -- {fname} (we play as {our_name!r}) --")
            res = run_seeded_simulation(path, our_name,
                                        games=args.seeded_games,
                                        seed=args.seed,
                                        mc_samples=args.mc_samples)
            for label in ("Competition", "Greedy", "Tracker", "Random"):
                k_w = f"{label}.win_rate"
                k_s = f"{label}.avg_sets"
                if k_w in res:
                    print(f"    {label:<14}"
                          f"win={res[k_w]*100:>5.1f}%   "
                          f"avg sets={res[k_s]:.2f}")
            print()

    print("Done.")


if __name__ == "__main__":
    main()
