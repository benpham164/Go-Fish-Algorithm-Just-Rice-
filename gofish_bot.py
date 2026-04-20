"""
gofish_bot.py
=============
A competition-grade Go Fish bot designed to maximize completed sets and win
probability in 2, 3, or 4-player matches against other algorithms.

Pipeline (one file, no 3rd-party deps):

    Raw JSON  ->  parse/clean  ->  GameState
                                   +
                                   Inference (replay every turn to build
                                              per-opponent probability models)
                                   +
                                   MoveEvaluator (score every legal ask)
                                   +
                                   MonteCarloSearch (sample hidden worlds,
                                                     1-ply rollout, tiebreak)
                                   ->  {"Requestee": ..., "Request": ...}

USAGE
-----
    # Normal match play: read the organizer's JSON from stdin, print one move.
    cat game_state.json | python gofish_bot.py --name steven

    # Or read from a file directly.
    python gofish_bot.py -f mid-game.json -n steven

    # Run the built-in test harness over every *.json in uploads/
    python gofish_bot.py --test -n steven --debug

    # Fill a placeholder into first-turn-fillable.json (the organizer's sanity
    # test that the bot cleans data properly) and decide a move:
    python gofish_bot.py -f first-turn-fillable.json -n MyBotName

Data format notes
-----------------
Per the organizer's "Game Data Breakdown.boxnote":
  Players: [{Name, Sets(int), HandSize(int)}]
  Turns:   [{Requester, Requestee, Rank, Received(int), Set(str "" if none)}]
  OutOfPlay: [rank, ...]   # completed ranks
  Hand:    [[rank, suit], ...]   # viewer's hand; suit is ignored
  Deck:    int             # stock size

Observed quirks the parser handles safely:
  * "Rank": "1" occurs in several history files (invalid in standard Go Fish
    ranks 2-10, J, Q, K, A).  These turns are dropped during inference so
    garbage doesn't poison the probability model.
  * "game-review.json" uses a different schema (top-level "Sets" list, each
    Players entry has "Hand": [ranks...] and "Sets": [ranks...]).  Detected
    and converted.
  * "T" is accepted as an alias for "10".
  * Player names that are empty strings get filled in with our own name
    (this is how "first-turn-fillable.json" is cleaned).
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
import sys
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RANKS: List[str] = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"]
VALID_RANKS: set = set(RANKS)
DECK_SIZE = 52


# ---------------------------------------------------------------------------
# Parsing / cleaning
# ---------------------------------------------------------------------------

def clean_rank(r) -> Optional[str]:
    """Normalize a rank token to one of the 13 canonical ranks, or None."""
    if r is None:
        return None
    s = str(r).strip().upper()
    if s == "":
        return None
    if s == "T":          # some feeds use T for ten
        s = "10"
    if s == "1":          # invalid — drop silently (observed in history files)
        return None
    return s if s in VALID_RANKS else None


def parse_hand(hand_field) -> List[str]:
    """Accept either [[rank,suit], ...] or [rank, ...] form.  Return ranks."""
    out: List[str] = []
    for card in (hand_field or []):
        if isinstance(card, (list, tuple)) and card:
            r = clean_rank(card[0])
        else:
            r = clean_rank(card)
        if r:
            out.append(r)
    return out


# ---------------------------------------------------------------------------
# Game state
# ---------------------------------------------------------------------------

class GameState:
    """A cleaned, immutable-ish snapshot of the game from our point of view."""

    def __init__(self) -> None:
        self.players: List[str] = []           # turn order
        self.sets_by_player: Dict[str, int] = {}
        self.hand_size: Dict[str, int] = {}
        self.out_of_play: set = set()          # completed ranks
        self.deck: int = 0                     # stock size
        self.my_name: Optional[str] = None
        self.my_hand: List[str] = []           # list of rank strings (our hand)
        self.turns: List[Dict] = []            # cleaned turn log

    # ------------------------------------------------------------------ raw
    @classmethod
    def from_raw(cls, raw: Dict, my_name: Optional[str] = None) -> "GameState":
        gs = cls()
        gs.my_hand = parse_hand(raw.get("Hand"))
        # Fallback for review-format files: the per-player "Hand" lists are
        # present even though the top-level "Hand" is absent.  We'll fill
        # this in *after* my_name is resolved (below).

        # Build player list in the order the organizer provides (this IS turn
        # order — confirmed by inspecting mid-game.json where turns cycle
        # haris->alyser->steven->vaughn).
        for p in raw.get("Players", []):
            name = (p.get("Name") or "").strip()  # may be "" (fillable)
            gs.players.append(name)

            # Sets may be int (live state) or list of ranks (game-review).
            sets_val = p.get("Sets", 0)
            if isinstance(sets_val, list):
                sets_val = len(sets_val)
            gs.sets_by_player[name] = int(sets_val or 0)

            # HandSize may be absent (game-review); infer from "Hand" length.
            hs = p.get("HandSize")
            if hs is None:
                hs = len(p.get("Hand", []) or [])
            gs.hand_size[name] = int(hs)

        # ---- My-name resolution -----------------------------------------
        # Priority: explicit arg > env var > auto-detect via HandSize match.
        if my_name is None:
            my_name = os.environ.get("GOFISH_BOT_NAME")

        # If the fillable file has a blank name slot, inject ours.
        if my_name and "" in gs.players:
            idx = gs.players.index("")
            gs.players[idx] = my_name
            gs.sets_by_player[my_name] = gs.sets_by_player.pop("", 0)
            gs.hand_size[my_name] = gs.hand_size.pop("", len(gs.my_hand))

        if my_name is None:
            hand_len = len(gs.my_hand)
            candidates = [n for n in gs.players
                          if n and gs.hand_size.get(n, -1) == hand_len]
            if len(candidates) == 1:
                my_name = candidates[0]
            elif "steven" in candidates:   # historical default in test files
                my_name = "steven"
            elif candidates:
                my_name = candidates[0]
            elif gs.players:
                my_name = next((n for n in gs.players if n), gs.players[0])

        gs.my_name = my_name

        # Review-format fallback: pick up our hand from Players[name].Hand
        # if the top-level "Hand" was missing.
        if not gs.my_hand and my_name:
            for p in raw.get("Players", []):
                if (p.get("Name") or "").strip() == my_name and p.get("Hand"):
                    gs.my_hand = parse_hand(p.get("Hand"))
                    gs.hand_size[my_name] = len(gs.my_hand)
                    break

        # Deck + out-of-play
        gs.deck = int(raw.get("Deck", 0) or 0)
        oop = raw.get("OutOfPlay")
        if oop is None:
            oop = raw.get("Sets", [])   # game-review schema
        if isinstance(oop, list):
            gs.out_of_play = {clean_rank(r) for r in oop if clean_rank(r)}

        # Clean turn log
        for t in raw.get("Turns", []) or []:
            rank = clean_rank(t.get("Rank"))
            if not rank:
                continue   # drop invalid-rank turns so inference stays sound
            set_done = clean_rank(t.get("Set", "")) or None
            gs.turns.append({
                "requester": t.get("Requester"),
                "requestee": t.get("Requestee"),
                "rank": rank,
                "received": int(t.get("Received", 0) or 0),
                "set": set_done,
            })
        return gs

    # ---------------------------------------------------------------- views
    @property
    def opponents(self) -> List[str]:
        return [p for p in self.players if p != self.my_name]

    def describe(self) -> str:
        return (
            f"me={self.my_name} hand={self.my_hand} "
            f"sets_me={self.sets_by_player.get(self.my_name, 0)} "
            f"deck={self.deck} oop={sorted(self.out_of_play)} "
            f"opps={[(o, self.hand_size[o], self.sets_by_player[o]) for o in self.opponents]}"
        )


# ---------------------------------------------------------------------------
# Opponent model
# ---------------------------------------------------------------------------

class OpponentModel:
    """
    Per-opponent beliefs.

        known[rank]         - lower bound on current count of rank in hand.
                              Zero means "I have no reason to believe they
                              hold this rank" (not "they can't have any").
        cannot_hold[rank]   - rank -> age-in-draws; the opponent was observed
                              to have 0 of rank at that draw-count.  When age
                              grows (opponent draws cards) the constraint
                              fades.  Age = 99 means permanent (set completed).
        interest[rank]      - how many times they've asked for it (proxy for
                              strategic focus).
        asks                - chronological (requestee, rank, received) log.
        style_signals       - free-form counters used to classify the bot
                              as greedy / opportunist / info-gain-seeker.
        draws_since_reset   - how many times they've drawn from the stock
                              since we last updated cannot_hold ages.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self.known: Counter = Counter()
        self.cannot_hold: Dict[str, int] = {}
        self.interest: Counter = Counter()
        self.asks: List[Tuple[str, str, int]] = []
        self.style_signals: Counter = Counter()
        self.draws_since_reset: int = 0

    # --- style classification -------------------------------------------
    @property
    def looks_greedy(self) -> bool:
        # Greedy: repeatedly asking for the same rank or always their biggest
        return self.style_signals["greedy"] >= 2

    @property
    def looks_opportunist(self) -> bool:
        # Opportunist: chases ranks shown in failed asks by others (we can't
        # fully test from one side, but repeated topic-following counts).
        return self.style_signals["follows"] >= 2


# ---------------------------------------------------------------------------
# Inference engine
# ---------------------------------------------------------------------------

class Inference:
    """Replays the turn log to build opponent belief models and probabilities."""

    def __init__(self, gs: GameState) -> None:
        self.gs = gs
        self.models: Dict[str, OpponentModel] = {
            p: OpponentModel(p) for p in gs.players if p
        }
        self._replay()

    # --- replay ---------------------------------------------------------
    def _replay(self) -> None:
        for idx, t in enumerate(self.gs.turns):
            req, tgt = t["requester"], t["requestee"]
            rank, got, setdone = t["rank"], t["received"], t["set"]
            if req not in self.models or tgt not in self.models:
                continue

            # (1) Requester proved they held >=1 of rank at this moment.
            mreq = self.models[req]
            mreq.interest[rank] += 1
            mreq.asks.append((tgt, rank, got))
            mreq.known[rank] = max(mreq.known[rank], 1)
            mreq.cannot_hold.pop(rank, None)

            # (2) Transfer / failure updates.
            if got > 0:
                mreq.known[rank] = mreq.known.get(rank, 0) + got
                self.models[tgt].known[rank] = 0
                self.models[tgt].cannot_hold[rank] = 0
            else:
                # Failed ask: requestee has 0 of rank right now.
                self.models[tgt].known[rank] = 0
                self.models[tgt].cannot_hold[rank] = 0
                # Requester drew 1 card from stock (possibly matching).
                mreq.draws_since_reset += 1
                self._age_constraints(req)

            # (3) Set completion: 4 cards of `setdone` leave play entirely.
            if setdone:
                for m in self.models.values():
                    m.known[setdone] = 0
                    m.cannot_hold[setdone] = 99

            # (4) Style classification (light, only for counter-play).
            self._update_style(req, rank, idx)

    def _age_constraints(self, player: str) -> None:
        m = self.models[player]
        for r in list(m.cannot_hold.keys()):
            if m.cannot_hold[r] < 99:
                m.cannot_hold[r] += 1
                if m.cannot_hold[r] > 4:
                    m.cannot_hold.pop(r, None)

    def _update_style(self, player: str, rank: str, turn_idx: int) -> None:
        m = self.models[player]
        # Greedy: three recent asks on the same rank.
        recent_ranks = [a[1] for a in m.asks[-3:]]
        if len(recent_ranks) == 3 and len(set(recent_ranks)) == 1:
            m.style_signals["greedy"] += 1
        # Follower: their last ask was a rank someone else recently failed on.
        if turn_idx >= 1:
            prev = self.gs.turns[turn_idx - 1]
            if prev["requester"] != player and prev["received"] == 0 \
                    and prev["rank"] == rank:
                m.style_signals["follows"] += 1

    # --- combinatorial primitives --------------------------------------
    def unseen_copies_of(self, rank: str) -> int:
        """Copies of rank not in my hand and not in out-of-play sets and not
        already pinned to somebody's known hand."""
        if rank in self.gs.out_of_play:
            return 0
        in_my = self.gs.my_hand.count(rank)
        pinned = sum(self.models[p].known[rank] for p in self.gs.opponents
                     if p in self.models)
        return max(0, 4 - in_my - pinned)

    def uncommitted_hand_size(self, opp: str) -> int:
        """Cards in opp's hand not already accounted for by known constraints."""
        m = self.models[opp]
        return max(0, self.gs.hand_size.get(opp, 0) - sum(m.known.values()))

    def total_unseen_pool(self) -> int:
        """The pool of cards whose identity we don't know = uncommitted cards
        in opponents' hands + cards in the deck."""
        deck = self.gs.deck
        opp_uncomm = sum(self.uncommitted_hand_size(p) for p in self.gs.opponents
                         if p in self.models)
        return deck + opp_uncomm

    # --- probability queries -------------------------------------------
    def prob_opponent_has(self, opp: str, rank: str) -> float:
        """P(opp currently holds >=1 card of rank)."""
        if rank in self.gs.out_of_play:
            return 0.0
        if opp not in self.models:
            return 0.0
        m = self.models[opp]

        # Hard evidence: we already know they have one.
        if m.known[rank] >= 1:
            return 1.0

        # If a failed-ask constraint is fresh, probability is near zero,
        # tempered by draws taken since.
        if rank in m.cannot_hold:
            age = m.cannot_hold[rank]
            if age >= 99:
                return 0.0
            copies = self.unseen_copies_of(rank)
            pool = self.total_unseen_pool()
            if pool <= 0 or copies <= 0:
                return 0.0
            per_draw = copies / pool
            p_any = 1 - (1 - per_draw) ** max(0, age)
            return min(p_any, 0.30)

        copies = self.unseen_copies_of(rank)
        if copies == 0:
            return 0.0
        uncomm = self.uncommitted_hand_size(opp)
        pool = self.total_unseen_pool()
        if uncomm == 0 or pool == 0:
            return 0.0
        if copies >= pool:
            return 1.0
        try:
            # Hypergeometric: P(no copies in their uncommitted draw).
            p_zero = math.comb(pool - copies, uncomm) / math.comb(pool, uncomm)
            return 1.0 - p_zero
        except (ValueError, ZeroDivisionError):
            return min(1.0, copies * uncomm / pool)

    def expected_copies_from(self, opp: str, rank: str) -> float:
        """E[count of rank in opp's hand]."""
        if rank in self.gs.out_of_play:
            return 0.0
        if opp not in self.models:
            return 0.0
        m = self.models[opp]
        base = float(m.known[rank])
        copies = self.unseen_copies_of(rank)
        if copies <= 0:
            return base
        uncomm = self.uncommitted_hand_size(opp)
        pool = self.total_unseen_pool()
        extra = (uncomm * copies / pool) if pool > 0 else 0.0
        if rank in m.cannot_hold and m.cannot_hold[rank] < 99:
            # Dampen expectation because we know they had zero recently.
            extra *= 0.25
        return base + min(copies, extra)

    def danger_score(self, opp: str) -> float:
        """Heuristic: how close is opp to turning cards into completed sets?"""
        if opp not in self.models:
            return 0.0
        m = self.models[opp]
        score = 2.0 * self.gs.sets_by_player.get(opp, 0)
        for rank, cnt in m.known.items():
            if rank in self.gs.out_of_play or cnt <= 0:
                continue
            score += {1: 0.20, 2: 0.90, 3: 2.25}.get(cnt, 3.0)
        score += 0.04 * self.gs.hand_size.get(opp, 0)
        # If they look greedy or are laser-focused on one rank, that's dangerous.
        score += 0.5 * (1 if m.looks_greedy else 0)
        if m.interest:
            top_rank, top_count = m.interest.most_common(1)[0]
            if top_count >= 3 and top_rank not in self.gs.out_of_play:
                score += 0.8
        return score


# ---------------------------------------------------------------------------
# Move evaluator
# ---------------------------------------------------------------------------

class MoveEvaluator:
    """Generate legal (requestee, rank) pairs and score each one."""

    # Tunable weights — exposed as class attrs so a sweep/tournament can
    # hill-climb them.
    W_COMPLETE_SET     = 12.0   # new_count == 4
    W_FORM_TRIPLE      =  4.5   # new_count == 3
    W_FORM_PAIR        =  1.6   # new_count == 2
    W_JUST_A_CARD      =  0.4   # new_count == 1 (first copy)
    W_CONTINUATION     =  0.75  # bonus for keeping the turn alive
    W_FAILURE_PENALTY  = -0.30  # baseline for a failed ask (lose turn)
    W_INFO_ON_FAILURE  =  0.12  # failure still eliminates a possibility
    W_DENIAL_PER_DANGER=  0.12  # removing a card from a dangerous opp
    W_REVEAL_PENALTY   =  0.22  # each "copy left in hand" is advertised to others
    W_REVEAL_DISTANCE  =  0.70  # decay per turn-order step after us
    W_DEAD_TARGET      = -999.0 # opp has no cards

    def __init__(self, gs: GameState, inf: Inference) -> None:
        self.gs = gs
        self.inf = inf

    # ---------------------------------------------------------- legality
    def legal_moves(self) -> List[Tuple[str, str]]:
        """Return every legal (opp, rank) ask:
              - opp must be another player with >=1 card,
              - rank must currently be in my hand (STRICT rule enforcement),
              - rank must not already be out of play (edge case after our own
                recent completed set)."""
        ranks_i_have = [r for r in set(self.gs.my_hand)
                        if r not in self.gs.out_of_play]
        moves: List[Tuple[str, str]] = []
        for opp in self.gs.opponents:
            if self.gs.hand_size.get(opp, 0) <= 0:
                continue
            for r in ranks_i_have:
                moves.append((opp, r))
        return moves

    # ---------------------------------------------------------- scoring
    def _success_value(self, new_count: float) -> float:
        if new_count >= 4:
            return self.W_COMPLETE_SET
        if new_count >= 3:
            return self.W_FORM_TRIPLE
        if new_count >= 2:
            return self.W_FORM_PAIR
        return self.W_JUST_A_CARD

    def _reveal_penalty(self, rank: str, new_count: float,
                        asked_opp: str, got: float) -> float:
        """Cost of advertising that I hold `rank`.  If I still hold copies
        after the ask, every opponent after me in turn order may target
        them.  Weight by closeness in the turn cycle and by how much I hold."""
        if new_count >= 4:           # set completed, no reveal risk
            return 0.0
        me_idx = self.gs.players.index(self.gs.my_name)
        order = self.gs.players[me_idx + 1:] + self.gs.players[:me_idx]
        penalty = 0.0
        for i, p in enumerate(order):
            if p == self.gs.my_name or self.gs.hand_size.get(p, 0) == 0:
                continue
            dist = self.W_REVEAL_DISTANCE ** i
            # Cards of rank still with me after the ask
            copies_left = new_count if p != asked_opp else max(0.0, new_count - got)
            penalty += dist * self.W_REVEAL_PENALTY * copies_left
        return penalty

    def _denial_bonus(self, opp: str, p_success: float, e_got: float) -> float:
        """Pulling cards from a dangerous opponent is worth extra."""
        danger = self.inf.danger_score(opp)
        return self.W_DENIAL_PER_DANGER * danger * p_success * min(1.0, e_got)

    def score_move(self, opp: str, rank: str) -> Dict[str, float]:
        if self.gs.hand_size.get(opp, 0) == 0:
            return {"total": self.W_DEAD_TARGET, "p_success": 0.0, "e_got": 0.0}

        my_count = self.gs.my_hand.count(rank)
        p_success = self.inf.prob_opponent_has(opp, rank)
        e_got = self.inf.expected_copies_from(opp, rank)

        # Conditional on success, we gain ~e_got / p_success cards.
        cond_got = (e_got / p_success) if p_success > 1e-9 else e_got
        new_count_success = my_count + max(1.0, cond_got)  # >=1 by definition
        new_count_fail = my_count   # no gain; we draw 1 random card

        success_val = self._success_value(new_count_success) + self.W_CONTINUATION
        failure_val = self.W_FAILURE_PENALTY + self.W_INFO_ON_FAILURE * (1 - p_success)
        # When we fail we draw from the stock.  If the stock card equals
        # our rank, we keep going.  Very rough probability: copies/deck-mass.
        if self.gs.deck > 0:
            copies = self.inf.unseen_copies_of(rank)
            pool = self.inf.total_unseen_pool()
            p_draw_match = (copies / pool) if pool > 0 else 0.0
            failure_val += p_draw_match * 0.8   # got a continuation anyway

        expected = p_success * success_val + (1 - p_success) * failure_val
        expected += self._denial_bonus(opp, p_success, cond_got)
        expected -= self._reveal_penalty(rank, new_count_success if p_success > 0.5 else my_count,
                                         opp, cond_got)
        # Counter-greedy / counter-follower: if opp frequently asks `rank`,
        # they almost certainly have >=1 of it (greedy tell).
        m = self.inf.models.get(opp)
        if m and m.interest.get(rank, 0) >= 2 and m.known[rank] == 0:
            expected += 0.4

        return {
            "total": expected,
            "p_success": p_success,
            "e_got": e_got,
            "my_count": my_count,
            "success_val": success_val,
            "failure_val": failure_val,
        }

    def rank_all(self) -> List[Tuple[Tuple[str, str], Dict[str, float]]]:
        scored = [((opp, rk), self.score_move(opp, rk))
                  for (opp, rk) in self.legal_moves()]
        scored.sort(key=lambda x: -x[1]["total"])
        return scored


# ---------------------------------------------------------------------------
# Monte-Carlo tie-breaker
# ---------------------------------------------------------------------------

class MonteCarlo:
    """Sample hidden worlds consistent with known constraints and estimate
    the immediate-plus-one-rollout value of a candidate move."""

    def __init__(self, gs: GameState, inf: Inference,
                 n_samples: int = 96, seed: Optional[int] = None) -> None:
        self.gs = gs
        self.inf = inf
        self.n = n_samples
        self.rng = random.Random(seed)

    # ---------- sampling ----------
    def _build_pool(self) -> List[str]:
        pool: List[str] = []
        for r in RANKS:
            pool.extend([r] * self.inf.unseen_copies_of(r))
        return pool

    def sample_world(self) -> Tuple[Dict[str, List[str]], List[str]]:
        """Return (opp_hands_dict, deck_list) consistent with knowns."""
        pool = self._build_pool()
        self.rng.shuffle(pool)
        # Seed every opponent's hand with their known cards.
        hands: Dict[str, List[str]] = {
            p: list(self.inf.models[p].known.elements())
            for p in self.gs.opponents if p in self.inf.models
        }
        needed = {p: self.inf.uncommitted_hand_size(p) for p in hands}
        idx = 0
        # Best-effort: respect fresh `cannot_hold` constraints.
        for p in hands:
            m = self.inf.models[p]
            count = 0
            tries = 0
            while count < needed[p] and idx < len(pool) and tries < len(pool) * 2:
                card = pool[idx]
                age = m.cannot_hold.get(card)
                forbid = age is not None and age < 2
                if not forbid:
                    hands[p].append(card)
                    pool.pop(idx)
                    count += 1
                else:
                    idx += 1
                tries += 1
            # If we failed (pool exhausted of legal cards), relax constraints.
            while count < needed[p] and pool:
                hands[p].append(pool.pop(0))
                count += 1
            idx = 0
        deck = pool
        return hands, deck

    # ---------- rollout ----------
    def evaluate(self, opp: str, rank: str) -> float:
        """Average reward over samples of a 1-turn rollout."""
        totals = 0.0
        for _ in range(self.n):
            hands, deck = self.sample_world()
            reward = self._rollout(opp, rank, hands, deck)
            totals += reward
        return totals / max(1, self.n)

    def _rollout(self, opp: str, rank: str,
                 hands: Dict[str, List[str]], deck: List[str]) -> float:
        """One-turn-ahead reward from OUR perspective."""
        my_hand = list(self.gs.my_hand)
        reward = 0.0
        got = hands.get(opp, []).count(rank)
        if got > 0:
            # Transfer
            hands[opp] = [c for c in hands[opp] if c != rank]
            my_hand.extend([rank] * got)
            cnt = my_hand.count(rank)
            if cnt >= 4:
                reward += 15.0     # completed set this turn
                my_hand = [c for c in my_hand if c != rank]
            elif cnt == 3:
                reward += 4.0
            elif cnt == 2:
                reward += 1.5
            reward += 0.8          # continuation
            # One more shallow ask: most valuable continuation in my hand.
            if my_hand:
                best_rank = max(set(my_hand), key=my_hand.count)
                best_opp = max(self.gs.opponents,
                               key=lambda o: (hands.get(o, []).count(best_rank),
                                              self.gs.hand_size.get(o, 0)))
                g2 = hands.get(best_opp, []).count(best_rank)
                if g2 > 0:
                    reward += 0.4 * g2 + (1.5 if my_hand.count(best_rank) + g2 >= 4 else 0.0)
        else:
            reward -= 0.25     # lost turn
            if deck:
                drawn = deck[0]
                my_hand.append(drawn)
                cnt = my_hand.count(drawn)
                if cnt >= 4:
                    reward += 15.0
                elif cnt == 3:
                    reward += 2.0
                elif cnt == 2:
                    reward += 0.5
                if drawn == rank:
                    reward += 1.0  # continuation from lucky draw
        # Soft term: penalize world-states where an opponent is near completing
        # a set using ranks we just revealed.
        me_idx = self.gs.players.index(self.gs.my_name)
        for offset in range(1, len(self.gs.players)):
            p = self.gs.players[(me_idx + offset) % len(self.gs.players)]
            if p == self.gs.my_name:
                continue
            for r, cnt in Counter(hands.get(p, [])).items():
                if cnt >= 3 and r in my_hand:
                    reward -= 0.8 / offset
                elif cnt >= 2 and r in my_hand and my_hand.count(r) < 2:
                    reward -= 0.25 / offset
        return reward


# ---------------------------------------------------------------------------
# Top-level decision
# ---------------------------------------------------------------------------

def choose_move(gs: GameState,
                use_mc: bool = True,
                mc_samples: int = 96,
                seed: Optional[int] = None) -> Dict[str, str]:
    """Return the final action as the exact required JSON shape."""
    inf = Inference(gs)
    evaluator = MoveEvaluator(gs, inf)
    scored = evaluator.rank_all()

    # No legal ask: fall back gracefully.
    if not scored:
        opp = next((o for o in gs.opponents if gs.hand_size.get(o, 0) > 0),
                   None)
        if opp and gs.my_hand:
            return {"Requestee": opp, "Request": random.choice(gs.my_hand)}
        # Truly empty — return a syntactically valid blank move.
        return {"Requestee": opp or "", "Request": ""}

    # Monte-Carlo refinement on the top candidates (the heuristic already
    # cuts the search space; the MC just breaks ties using sampled rollouts).
    if use_mc and len(scored) > 1:
        top = [mv for mv, _ in scored[:min(6, len(scored))]]
        mc = MonteCarlo(gs, inf, n_samples=mc_samples, seed=seed)
        # Blend heuristic score with Monte-Carlo estimate (80/20).
        blended = []
        for (opp, rank) in top:
            heur = dict(scored)[(opp, rank)]["total"]
            mcv = mc.evaluate(opp, rank)
            blended.append(((opp, rank), 0.55 * heur + 0.45 * mcv))
        blended.sort(key=lambda x: -x[1])
        best = blended[0][0]
    else:
        best = scored[0][0]

    return {"Requestee": best[0], "Request": best[1]}


# ---------------------------------------------------------------------------
# CLI / test harness
# ---------------------------------------------------------------------------

def run_tests(my_name: Optional[str], debug: bool = False,
              mc_samples: int = 96) -> None:
    """Runs the decision pipeline over every JSON file in uploads/."""
    upload_dir = "/sessions/brave-elegant-clarke/mnt/uploads"
    files = sorted(glob.glob(os.path.join(upload_dir, "*.json")))
    if not files:
        print("No test JSON files found in", upload_dir)
        return
    for path in files:
        try:
            with open(path) as f:
                raw = json.load(f)
        except Exception as e:
            print(f"SKIP {path}: {e}")
            continue
        gs = GameState.from_raw(raw, my_name=my_name)
        move = choose_move(gs, use_mc=True, mc_samples=mc_samples, seed=42)
        tag = os.path.basename(path)
        print(f"=== {tag} ===")
        print("state:", gs.describe())
        print("move :", json.dumps(move))
        if debug:
            inf = Inference(gs)
            ev = MoveEvaluator(gs, inf)
            for (opp, rank), sc in ev.rank_all()[:8]:
                print(f"    {opp:<10} {rank:<3}  score={sc['total']:+.2f} "
                      f"p_succ={sc['p_success']:.3f}  E[cards]={sc['e_got']:.2f}  "
                      f"my_count={sc['my_count']}")
        print()


def main() -> None:
    ap = argparse.ArgumentParser(description="Competitive Go Fish bot.")
    ap.add_argument("--file", "-f", default=None,
                    help="path to JSON game-state; defaults to stdin")
    ap.add_argument("--name", "-n", default=None,
                    help="this bot's player name (or set GOFISH_BOT_NAME)")
    ap.add_argument("--no-mc", action="store_true",
                    help="skip the Monte-Carlo tie-breaker")
    ap.add_argument("--mc-samples", type=int, default=96)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--debug", action="store_true",
                    help="print inference details to stderr")
    ap.add_argument("--test", action="store_true",
                    help="run built-in harness over uploaded test JSONs")
    args = ap.parse_args()

    if args.test:
        run_tests(args.name, debug=args.debug, mc_samples=args.mc_samples)
        return

    if args.file:
        with open(args.file) as f:
            raw = json.load(f)
    else:
        raw = json.load(sys.stdin)

    gs = GameState.from_raw(raw, my_name=args.name)
    move = choose_move(gs, use_mc=not args.no_mc,
                       mc_samples=args.mc_samples, seed=args.seed)

    if args.debug:
        sys.stderr.write(gs.describe() + "\n")

    # Exactly the required JSON shape, nothing else on stdout.
    print(json.dumps(move))


if __name__ == "__main__":
    main()
