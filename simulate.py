#!/usr/bin/env python3
"""
===============================================================================
  GO FISH COMPETITION SIMULATOR
===============================================================================

Simulates 4-player Go Fish games (7 cards each, 24-card stock) where our
"Cowork Shark" bot (from go_fish_bot.py) plays against three opponent
archetypes that mirror the styles observed in the provided history files:

    * StickyGreedyBot       -- "vaughn" style
        Always asks for the rank it holds most copies of, and persists on
        that rank until it gives up (got zero AND drew something else).

    * TargetLockedBot       -- "haris" / "alyser" style
        Picks one opponent and keeps asking THEM for whatever rank it has
        the most of. Switches target only when the current target is
        empty-handed or the bot just completed a set.

    * MemoryInformantBot    -- "steven" style
        Uses recent turn history: if any opponent has asked for rank R
        (or received R in a transfer), and I have R, I ask them first.
        Otherwise falls back to most-held-rank / random valid target.

All opponents are reasonable Go Fish players. They follow the same
legal-ask rule (must hold the rank they request).

Deck / deal:
    * Standard 52-card deck, shuffled with a seed for reproducibility.
    * 4 players, 7 cards each (matches the organizer's history files
      exactly; the task doc's "4 players -> 5 cards" variant is also
      supported via a flag but defaults to 7).

Outputs:
    * Per-bot win rate
    * Per-bot average sets completed
    * Head-to-head records vs each opponent type
    * Optional verbose game-by-game trace

Run:
    python3 simulate.py                         # 1000 games, seed 0
    python3 simulate.py --games 5000 --seed 42  # larger sample
    python3 simulate.py --games 20 --verbose    # inspect actual play
"""

from __future__ import annotations

import argparse
import os
import random
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from go_fish_bot import (                                  # noqa: E402
    CANONICAL_RANKS,
    GoFishBot,
    LegalMove,
    build_game_state,
    clean_state_dict,
)

SUITS = ["H", "D", "C", "S"]
HAND_SIZE_BY_PLAYER_COUNT = {2: 7, 3: 7, 4: 7}   # defaults per task spec


# ============================================================================
#  BASE PLAYER INTERFACE
# ============================================================================

@dataclass
class PlayerRuntime:
    """Live per-player state inside the simulator."""
    name: str
    hand: List[Tuple[str, str]] = field(default_factory=list)
    sets: List[str] = field(default_factory=list)

    def hand_size(self) -> int:
        return len(self.hand)

    def ranks_in_hand(self) -> List[str]:
        return sorted({r for r, _ in self.hand})

    def count(self, rank: str) -> int:
        return sum(1 for r, _ in self.hand if r == rank)

    def take_all_of_rank(self, rank: str) -> List[Tuple[str, str]]:
        taken = [c for c in self.hand if c[0] == rank]
        self.hand = [c for c in self.hand if c[0] != rank]
        return taken

    def lay_down_sets(self) -> List[str]:
        """
        Any rank with >=4 copies gets laid down; returns the ranks laid.
        (Drawing extra copies beyond 4 isn't possible in a legal game, but
        we compute safely anyway.)
        """
        counts = Counter(r for r, _ in self.hand)
        laid = []
        for r, c in counts.items():
            if c >= 4:
                laid.append(r)
                self.hand = [x for x in self.hand if x[0] != r]
                self.sets.append(r)
        return laid


class OpponentBot:
    """Base-class interface for simulated opponents."""
    kind = "generic"

    def choose(self, me: PlayerRuntime, public: "PublicState") -> LegalMove:
        raise NotImplementedError

    def notify(self, turn: dict) -> None:
        """Allow bots to absorb turn history (default: no-op)."""
        pass


# ============================================================================
#  OPPONENT ARCHETYPES (mirroring observed play-styles)
# ============================================================================

class StickyGreedyBot(OpponentBot):
    """vaughn-style: asks for the rank it has the most copies of, sticks
    to the same rank across consecutive turns until it clears or fails."""
    kind = "sticky_greedy"

    def __init__(self) -> None:
        self.locked_rank: Optional[str] = None

    def choose(self, me: PlayerRuntime, public: "PublicState") -> LegalMove:
        # If the locked rank is still in hand, keep pushing for it.
        rank = None
        if self.locked_rank and me.count(self.locked_rank) > 0:
            rank = self.locked_rank
        else:
            counts = Counter(r for r, _ in me.hand)
            if counts:
                rank = counts.most_common(1)[0][0]
            self.locked_rank = rank

        # Pick a random living opponent (or one that asked about this rank).
        targets = [p for p in public.players
                   if p != me.name and public.hand_sizes[p] > 0]
        if not targets:
            targets = [p for p in public.players if p != me.name]
        # Prefer a target that previously asked for this rank (small info use).
        signaled = [p for p in targets
                    if public.asks_by.get(p, Counter()).get(rank, 0) > 0]
        target = random.choice(signaled) if signaled else random.choice(targets)
        return LegalMove(requestee=target, rank=rank)


class TargetLockedBot(OpponentBot):
    """haris/alyser-style: locks onto a single opponent and keeps asking
    them various ranks until they run out of cards or a set is made."""
    kind = "target_locked"

    def __init__(self) -> None:
        self.locked_target: Optional[str] = None

    def choose(self, me: PlayerRuntime, public: "PublicState") -> LegalMove:
        # Maintain target if still has cards.
        others = [p for p in public.players if p != me.name]
        if self.locked_target is None or \
           public.hand_sizes.get(self.locked_target, 0) == 0 or \
           self.locked_target not in others:
            candidates = [p for p in others if public.hand_sizes[p] > 0]
            self.locked_target = random.choice(candidates) if candidates \
                else random.choice(others)

        # Ask for the rank we hold the most of (maximizes hit probability).
        counts = Counter(r for r, _ in me.hand)
        rank = counts.most_common(1)[0][0] if counts else CANONICAL_RANKS[0]
        return LegalMove(requestee=self.locked_target, rank=rank)


class MemoryInformantBot(OpponentBot):
    """steven-style: uses turn history. If any opponent asked for rank R
    (or received R) and I hold R, I ask them."""
    kind = "memory_informant"

    def choose(self, me: PlayerRuntime, public: "PublicState") -> LegalMove:
        my_ranks = me.ranks_in_hand()
        others = [p for p in public.players
                  if p != me.name and public.hand_sizes[p] > 0] or \
                 [p for p in public.players if p != me.name]

        # Look backward through recent turns for a signal about a rank I hold.
        for t in reversed(public.turns):
            rank = t["Rank"]
            if rank not in my_ranks:
                continue
            # Candidate = requester of that ask, or the receiver of that rank.
            cand = t["Requester"]
            if cand != me.name and cand in others:
                return LegalMove(requestee=cand, rank=rank)
            if t["Received"] > 0 and t["Requester"] in others \
                    and t["Requester"] != me.name:
                return LegalMove(requestee=t["Requester"], rank=rank)

        # Fallback: rank we hold the most of, random target.
        counts = Counter(r for r, _ in me.hand)
        rank = counts.most_common(1)[0][0] if counts else my_ranks[0]
        target = random.choice(others)
        return LegalMove(requestee=target, rank=rank)


# ============================================================================
#  OUR BOT, WIRED INTO THE SIMULATOR
# ============================================================================

class CoworkSharkAdapter(OpponentBot):
    """Wraps go_fish_bot.GoFishBot into the same interface as other bots.

    Builds an organizer-style state snapshot on every turn (since the real
    bot expects the same JSON shape) and lets the production evaluator
    pick the move.
    """
    kind = "cowork_shark"

    def __init__(self, mc_samples: int = 16) -> None:
        self.mc_samples = mc_samples

    def choose(self, me: PlayerRuntime, public: "PublicState") -> LegalMove:
        cleaned = {
            "Players": [{"Name": p,
                         "Sets": len(public.sets[p]),
                         "HandSize": public.hand_sizes[p]}
                        for p in public.players],
            "Turns": list(public.turns),
            "OutOfPlay": list(public.out_of_play),
            "Hand": [[r, s] for (r, s) in me.hand],
            "Deck": public.stock_size,
        }
        cleaned = clean_state_dict(cleaned)
        state = build_game_state(cleaned, my_name=me.name)
        bot = GoFishBot(state, mc_samples=self.mc_samples)
        return bot.choose_move()


# ============================================================================
#  PUBLIC STATE SNAPSHOT
# ============================================================================

@dataclass
class PublicState:
    players: List[str]
    hand_sizes: Dict[str, int]
    sets: Dict[str, List[str]]
    out_of_play: List[str]
    stock_size: int
    turns: List[dict]
    asks_by: Dict[str, Counter] = field(default_factory=dict)


# ============================================================================
#  GAME ENGINE
# ============================================================================

class GoFishGame:
    """Legitimate Go Fish rules engine."""

    def __init__(self, player_names: List[str], bots: List[OpponentBot],
                 hand_size: int = 7, rng: Optional[random.Random] = None,
                 verbose: bool = False):
        assert len(player_names) == len(bots)
        self.rng = rng or random.Random()
        self.verbose = verbose
        self.players = [PlayerRuntime(name=n) for n in player_names]
        self.name_to_idx = {p.name: i for i, p in enumerate(self.players)}
        self.bots: Dict[str, OpponentBot] = dict(zip(player_names, bots))
        self.turns: List[dict] = []
        self.out_of_play: List[str] = []

        # Build + shuffle a real 52-card deck.
        deck = [(r, s) for r in CANONICAL_RANKS for s in SUITS]
        self.rng.shuffle(deck)
        for p in self.players:
            p.hand = [deck.pop() for _ in range(hand_size)]
            for rank in p.lay_down_sets():
                self._retire_rank(rank)
        self.stock: List[Tuple[str, str]] = deck
        self.current = 0
        self.max_turns = 2000  # safety cap

    # ---- helpers --------------------------------------------------------

    def _retire_rank(self, rank: str) -> None:
        if rank not in self.out_of_play:
            self.out_of_play.append(rank)

    def _snapshot(self) -> PublicState:
        asks_by: Dict[str, Counter] = defaultdict(Counter)
        for t in self.turns:
            asks_by[t["Requester"]][t["Rank"]] += 1
        return PublicState(
            players=[p.name for p in self.players],
            hand_sizes={p.name: p.hand_size() for p in self.players},
            sets={p.name: list(p.sets) for p in self.players},
            out_of_play=list(self.out_of_play),
            stock_size=len(self.stock),
            turns=list(self.turns),
            asks_by=asks_by,
        )

    def _finished(self) -> bool:
        # Game over when every rank is out of play, OR no player has any
        # cards AND the stock is empty.
        if len(self.out_of_play) == len(CANONICAL_RANKS):
            return True
        any_cards = any(p.hand for p in self.players)
        return (not any_cards) and not self.stock

    # ---- running a turn -------------------------------------------------

    def _take_turn(self, player_idx: int) -> None:
        me = self.players[player_idx]
        # Drawing rule: if the player has no cards and the stock isn't empty,
        # they draw one before acting.
        if not me.hand and self.stock:
            me.hand.append(self.stock.pop())
            for r in me.lay_down_sets():
                self._retire_rank(r)
        if not me.hand:
            return  # they must pass if they genuinely have nothing

        bot = self.bots[me.name]
        snapshot = self._snapshot()
        move = bot.choose(me, snapshot)

        # Validate the move. If the bot proposed an illegal ask (not in hand
        # or self-target), we punish by converting to a random legal ask.
        if move.rank not in me.ranks_in_hand() or move.requestee == me.name:
            fallback_rank = self.rng.choice(me.ranks_in_hand())
            others = [p.name for p in self.players
                      if p.name != me.name and p.hand]
            if not others:
                others = [p.name for p in self.players if p.name != me.name]
            move = LegalMove(requestee=self.rng.choice(others),
                             rank=fallback_rank)

        # Execute.
        target = self.players[self.name_to_idx[move.requestee]]
        taken = target.take_all_of_rank(move.rank)
        got = len(taken)
        me.hand.extend(taken)
        completed_set = ""
        for r in me.lay_down_sets():
            self._retire_rank(r)
            completed_set = r
        self.turns.append({
            "Requester": me.name,
            "Requestee": move.requestee,
            "Rank": move.rank,
            "Received": got,
            "Set": completed_set,
        })
        if self.verbose:
            print(f"    {me.name} -> {move.requestee} ? {move.rank}  "
                  f"got {got}"
                  + (f"  [SET {completed_set}]" if completed_set else ""))

        if got > 0:
            # Successful ask: same player goes again (after possibly
            # drawing into an empty hand).
            self._take_turn(player_idx)
        else:
            # Go fish: draw one from the stock.
            if self.stock:
                drawn = self.stock.pop()
                me.hand.append(drawn)
                for r in me.lay_down_sets():
                    self._retire_rank(r)
                if drawn[0] == move.rank:
                    # Drew the requested rank -> continue turn.
                    if self.verbose:
                        print(f"    {me.name} drew {drawn[0]}"
                              f"{drawn[1]} (hit) -> continue")
                    self._take_turn(player_idx)
                # Otherwise turn ends.

    # ---- main loop ------------------------------------------------------

    def play(self) -> Dict[str, int]:
        steps = 0
        while not self._finished() and steps < self.max_turns:
            idx = self.current
            if self.verbose:
                print(f"  Turn {steps+1}: {self.players[idx].name}")
            self._take_turn(idx)
            steps += 1
            self.current = (self.current + 1) % len(self.players)
        return {p.name: len(p.sets) for p in self.players}


# ============================================================================
#  BATCH SIMULATION + REPORTING
# ============================================================================

def simulate_batch(n_games: int = 1000, seed: int = 0,
                   hand_size: int = 7, verbose: bool = False) -> dict:
    # Fixed player-type mapping -- Cowork Shark seats at index 0.
    # The other three seats rotate through the three opponent archetypes
    # so seat-order bias doesn't skew the result.
    opp_factories = [
        ("sticky_greedy",   lambda: StickyGreedyBot()),
        ("target_locked",   lambda: TargetLockedBot()),
        ("memory_informant", lambda: MemoryInformantBot()),
    ]
    my_label = "cowork_shark"

    wins: Counter = Counter()
    sets_total: defaultdict = defaultdict(list)
    head_to_head_wins: defaultdict = defaultdict(lambda: [0, 0])
    ties = 0

    master_rng = random.Random(seed)
    for g in range(n_games):
        # Rotate opponent seat order for fairness.
        order = master_rng.sample(opp_factories, 3)
        labels = [my_label] + [o[0] for o in order]
        bots = [CoworkSharkAdapter(mc_samples=12)] + [f() for _, f in order]
        names = ["shark", "opp1", "opp2", "opp3"]
        name_to_label = dict(zip(names, labels))

        game_rng = random.Random(master_rng.randint(0, 2**31 - 1))
        if verbose and g < 3:
            print(f"\n=== Game {g+1} (seed child) ===")
            print(f"  seats: " +
                  ", ".join(f"{n}={l}" for n, l in name_to_label.items()))

        game = GoFishGame(names, bots, hand_size=hand_size,
                          rng=game_rng, verbose=(verbose and g < 3))
        sets = game.play()

        # Aggregate.
        for n, s in sets.items():
            sets_total[name_to_label[n]].append(s)
        max_sets = max(sets.values())
        winners = [n for n, s in sets.items() if s == max_sets]
        if len(winners) > 1:
            ties += 1
            for w in winners:
                wins[name_to_label[w]] += 1 / len(winners)
        else:
            wins[name_to_label[winners[0]]] += 1

        # Head-to-head wins (shark vs each opponent type this game).
        shark_sets = sets["shark"]
        for n in names[1:]:
            lab = name_to_label[n]
            games, w = head_to_head_wins[lab]
            head_to_head_wins[lab] = [games + 1, w + (1 if shark_sets >
                                                      sets[n] else 0)]

    # Assemble report.
    report = {
        "n_games": n_games,
        "ties": ties,
        "wins": dict(wins),
        "win_rates": {k: v / n_games for k, v in wins.items()},
        "avg_sets": {k: statistics.mean(v) for k, v in sets_total.items()},
        "stdev_sets": {k: (statistics.pstdev(v) if len(v) > 1 else 0.0)
                       for k, v in sets_total.items()},
        "head_to_head_shark_vs": {
            lab: {"games": g, "wins": w, "rate": (w / g if g else 0.0)}
            for lab, (g, w) in head_to_head_wins.items()
        },
    }
    return report


def print_report(r: dict) -> None:
    n = r["n_games"]
    print("\n==========================================================")
    print(f"  Simulation over {n} games (4 players, 7-card deal)")
    print("==========================================================")
    print(f"  Games with a tied top score: {r['ties']} "
          f"({100*r['ties']/n:.1f}%)")
    print("\n  Win rate by bot type:")
    for name, rate in sorted(r["win_rates"].items(),
                             key=lambda x: -x[1]):
        avg = r["avg_sets"].get(name, 0.0)
        sd = r["stdev_sets"].get(name, 0.0)
        wins = r["wins"].get(name, 0.0)
        marker = " <-- our bot" if name == "cowork_shark" else ""
        print(f"    {name:<18} win_rate={rate*100:6.2f}%  "
              f"wins={wins:6.2f}  avg_sets={avg:4.2f}  sd={sd:4.2f}"
              f"{marker}")
    print("\n  Cowork Shark head-to-head (shark ended with MORE sets):")
    for lab, info in r["head_to_head_shark_vs"].items():
        print(f"    vs {lab:<18} games={info['games']:5}  "
              f"wins={info['wins']:5}  rate={info['rate']*100:6.2f}%")
    print()


def _cli() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--games", type=int, default=1000,
                   help="Number of games to simulate (default 1000).")
    p.add_argument("--seed", type=int, default=0,
                   help="Master RNG seed.")
    p.add_argument("--hand-size", type=int, default=7,
                   help="Hand size (default 7 -- matches provided files).")
    p.add_argument("--verbose", action="store_true",
                   help="Print the first few games turn-by-turn.")
    args = p.parse_args()
    report = simulate_batch(n_games=args.games, seed=args.seed,
                            hand_size=args.hand_size, verbose=args.verbose)
    print_report(report)


if __name__ == "__main__":
    _cli()
