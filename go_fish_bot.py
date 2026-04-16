#!/usr/bin/env python3
"""
===============================================================================
  COMPETITION GO FISH BOT  --  "Cowork Shark"
===============================================================================

A competition-ready Go Fish player that combines:

    (1) Strict data cleaning of organizer-provided JSON (including the
        `"1"` vs `"10"` inconsistency observed in the history files).
    (2) A full probabilistic game-state tracker that updates after every
        historical turn (asks, transfers, failed asks, completed sets).
    (3) Per-opponent inference: rank-by-rank holding probabilities,
        "known holdings" (from asks + transfers), "known non-holdings"
        (from recent failed asks), and "aggressiveness" signals.
    (4) An adaptive opponent model that classifies each opponent as
        greedy / sticky / information-gain / unknown, and adjusts the
        decision weights to exploit their tendencies.
    (5) A move generator that enumerates only *legal* (requestee, rank)
        pairs -- i.e. ranks the bot actually holds.
    (6) A probability-weighted evaluator plus a shallow Monte-Carlo
        rollout over sampled hidden hands, so the bot can see past the
        next ask and think about turn momentum, set completion,
        and opponent damage.
    (7) Strict output in the exact required JSON structure:

            {"Requestee":"player name", "Request":"rank"}

The code is structured as independent, testable modules in one file so
it can be dropped into a competition harness as a single artifact.

-------------------------------------------------------------------------------
STRATEGY SUMMARY
-------------------------------------------------------------------------------

Goal: win by ending with the most completed sets of 4.

High-level reasoning order applied to every move:

    1. COMPLETE-A-SET FIRST
       If we already hold 3 of some rank R, a successful ask finishes a
       set immediately AND guarantees a continuation turn. This is worth
       MORE than any speculative long-term play.

    2. EXPLOIT PUBLIC INFORMATION
       If any living opponent has asked for rank R, or received rank R
       in a transfer since the last time they laid down R's set, they
       almost certainly still hold at least one R. If we hold R, asking
       *them* is by far the best bet.

    3. BAYES OVER REMAINING RANKS
       For every legal (player, rank) pair, compute
           P(player holds >= 1 of rank) * expected_cards_gained
       using the hypergeometric distribution over the "unseen pool"
       for that opponent, conditioned on everything we've observed.

    4. CONTINUATION / MOMENTUM
       Successes keep our turn alive and give us another information
       draw. Weight moves by their success probability in addition to
       the raw card-count expectation.

    5. COUNTER-STRATEGY
       * Against greedy bots that repeatedly ask for the same rank,
         target *them* with that rank so we steal the collection they
         just built up.
       * Against "most-likely-player" bots that always ask us for the
         most recently seen rank, we avoid asking for a rank when the
         expected post-move information leak is catastrophic (e.g.
         revealing that we have 3 of a rank to a dangerous opponent).
       * Against information-poor bots, we prefer moves that reveal
         the *least* true information about our hand.

    6. SHALLOW MONTE CARLO
       For ambiguous top-scoring moves we sample plausible hidden
       hands consistent with observed history and play out a few turns
       using the opponent models to break ties on expected set-count
       gain.

-------------------------------------------------------------------------------
HANDLING PLAYER COUNTS  (2, 3, 4)
-------------------------------------------------------------------------------

The algorithm is player-count agnostic:

    * Hand sizes are read from the state (not hard-coded), so both
      5-card and 7-card openings work identically.
    * Inference iterates over *all* opponents listed in `Players`.
    * In 2-player games, all inference collapses to a single opponent
      and the hypergeometric simplifies (the "unseen pool" is just
      stock + opponent's hand).
    * In 3- and 4-player games, the bot performs cross-opponent
      reasoning -- e.g. if opponent A shows rank X and we later see A
      transfer X to B, B is now the prime target for X.

-------------------------------------------------------------------------------
OPPONENT MODELING
-------------------------------------------------------------------------------

Each opponent gets an `OpponentProfile` that tracks:

    * asks issued (count per rank)
    * asks received (and whether they succeeded)
    * cards transferred IN and OUT
    * sets completed
    * repeat-rank asks (signals "sticky" / greedy)
    * whether they tend to re-ask the same target (signals predictable)

These signals adjust move-scoring weights. For instance:

    * A "sticky greedy" opponent with three recent asks for rank 7 is
      treated as holding 7s with very high probability -- so if WE
      have a 7, we prioritize asking them.
    * An opponent who has completed 3+ sets is tagged "dangerous" and
      we deprioritize asks from them that would leak information.

-------------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ============================================================================
# CONSTANTS
# ============================================================================

# All legal ranks in a standard 52-card deck.
# Note: the ORGANIZER'S history files store "10" as a one-character rank "1"
# inside Turns.Rank, but as "10" everywhere else (Hand, Set, OutOfPlay).
# We normalize everything to "10" in this codebase.
CANONICAL_RANKS: List[str] = ["2", "3", "4", "5", "6", "7", "8", "9",
                              "10", "J", "Q", "K", "A"]

# One-character alias used in the organizer's Turns.Rank field for the 10.
TEN_ALIAS = "1"

# Total copies of every rank in a standard deck.
COPIES_PER_RANK = 4


# ============================================================================
#  SECTION 1.  FILE PARSING AND DATA CLEANING
# ============================================================================

def normalize_rank(raw: str) -> str:
    """
    Normalize any rank string to the canonical form used throughout the bot.

    In the organizer's JSON, the rank 10 is sometimes stored as the single
    character "1" in Turns.Rank (but as "10" in Hand / Set / OutOfPlay).
    We collapse both forms to "10".

    Unknown / blank input is returned unchanged (caller may filter).
    """
    if raw is None:
        return ""
    s = str(raw).strip().upper()
    if s == TEN_ALIAS:
        return "10"
    # Tolerate stray lowercase, trailing whitespace, etc.
    if s in CANONICAL_RANKS:
        return s
    # A few defensive aliases.
    aliases = {"T": "10", "JACK": "J", "QUEEN": "Q", "KING": "K", "ACE": "A"}
    return aliases.get(s, s)


def parse_state_file(path: str) -> dict:
    """
    Load a competition state file from disk, clean its data, and return
    a normalized dict. Robust to partially-filled ("fillable") files.
    """
    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    return clean_state_dict(raw)


def clean_state_dict(raw: dict) -> dict:
    """
    Normalize a state dict in-place (returning a new copy):
        * Rank strings normalized via `normalize_rank`
        * Missing keys filled with safe defaults
        * Whitespace / blank player names stripped
        * Hand entries coerced to [rank, suit] pairs
    """
    state = copy.deepcopy(raw)

    # Players list: at minimum Name / Sets / HandSize.
    players = state.get("Players", []) or []
    cleaned_players: List[dict] = []
    for p in players:
        if not isinstance(p, dict):
            continue
        name = str(p.get("Name", "")).strip()
        if not name:
            # Skip blank name entries (e.g. unfilled slots) but log them.
            continue
        cleaned_players.append({
            "Name": name,
            "Sets": int(p.get("Sets", 0) or 0),
            "HandSize": int(p.get("HandSize", 0) or 0),
        })
    state["Players"] = cleaned_players

    # Turns: normalize rank fields.
    turns = state.get("Turns", []) or []
    cleaned_turns: List[dict] = []
    for t in turns:
        if not isinstance(t, dict):
            continue
        cleaned_turns.append({
            "Requester": str(t.get("Requester", "")).strip(),
            "Requestee": str(t.get("Requestee", "")).strip(),
            "Rank": normalize_rank(t.get("Rank", "")),
            "Received": int(t.get("Received", 0) or 0),
            "Set": normalize_rank(t.get("Set", "")) if t.get("Set") else "",
        })
    state["Turns"] = cleaned_turns

    # OutOfPlay: normalize.
    state["OutOfPlay"] = [normalize_rank(r) for r in
                         state.get("OutOfPlay", []) or [] if r]

    # Hand: list of [rank, suit]. Tolerate weird formats.
    hand: List[List[str]] = []
    for entry in state.get("Hand", []) or []:
        if isinstance(entry, list) and len(entry) >= 1:
            rank = normalize_rank(entry[0])
            suit = str(entry[1]).strip().upper() if len(entry) >= 2 else "?"
            hand.append([rank, suit])
        elif isinstance(entry, str):
            hand.append([normalize_rank(entry), "?"])
    state["Hand"] = hand

    # Deck size may be labeled "Deck" or "Stock".
    state["Deck"] = int(state.get("Deck", state.get("Stock", 0)) or 0)

    return state


# ============================================================================
#  SECTION 2.  GAME STATE MODEL
# ============================================================================

@dataclass
class PlayerInfo:
    """Everything publicly known about a single player."""
    name: str
    sets: int = 0
    hand_size: int = 0

    # ----- inference bookkeeping (filled by InferenceEngine) -----

    # Lower-bound count of cards of each rank this player definitely holds
    # (from transfers received or from an ask they issued that never
    # completed into a set). Never exceeds 4 - out_of_play adjustments.
    known_min: Dict[str, int] = field(default_factory=lambda: defaultdict(int))

    # Ranks they are known NOT to hold as of a given time. Value = index of
    # most recent turn at which they said "go fish" for that rank. If they
    # later draw from the stock, this gets weaker.
    known_zero_as_of_turn: Dict[str, int] = field(default_factory=dict)

    # Per-rank asks / asks-received counters (for opponent profiling).
    asks_issued: Counter = field(default_factory=Counter)
    asks_received: Counter = field(default_factory=Counter)
    asks_success: Counter = field(default_factory=Counter)

    # Cards that are believed to have left the hand via giving them to
    # someone else or completing a set. Useful for tight accounting.
    cards_given_out: Counter = field(default_factory=Counter)

    def __repr__(self) -> str:  # pragma: no cover - debug only
        return (f"PlayerInfo({self.name} sets={self.sets} "
                f"hand={self.hand_size} known_min="
                f"{dict(self.known_min)})")


@dataclass
class GameState:
    """The fully-reconstructed public game state."""

    my_name: str
    my_hand: List[Tuple[str, str]]          # (rank, suit)
    stock_size: int
    out_of_play: List[str]
    players: Dict[str, PlayerInfo]          # all players keyed by name
    turn_order: List[str]                   # order in the Players list
    turns: List[dict]                       # cleaned turn history

    # ---------- convenience helpers ----------

    @property
    def opponents(self) -> List[str]:
        return [n for n in self.turn_order if n != self.my_name]

    def rank_count_in_my_hand(self, rank: str) -> int:
        return sum(1 for r, _s in self.my_hand if r == rank)

    def my_rank_set(self) -> List[str]:
        """Distinct ranks in my own hand (legal-ask candidates)."""
        return sorted({r for r, _ in self.my_hand})

    def total_sets_completed(self) -> int:
        return sum(p.sets for p in self.players.values())

    def cards_unseen_to_me(self) -> int:
        """Cards not in my hand and not in a completed set."""
        return 52 - len(self.my_hand) - COPIES_PER_RANK * len(self.out_of_play)

    def copies_still_in_play(self, rank: str) -> int:
        """How many copies of `rank` are somewhere other than completed sets."""
        if rank in self.out_of_play:
            return 0
        return COPIES_PER_RANK


# ----------------------------------------------------------------------------
# Building a GameState from a cleaned JSON dict
# ----------------------------------------------------------------------------

def build_game_state(cleaned: dict, my_name: Optional[str] = None) -> GameState:
    """
    Build a GameState from a cleaned state dict.

    `my_name` is passed in by the competition runner (command line / env /
    explicit argument). If omitted, we fall back to a heuristic: find the
    single player whose declared HandSize matches our actual Hand length
    -- and if multiple match, use the first one that matches turn history
    consistency (i.e. never appeared as the requester of an ask that this
    hand could not possibly have issued).
    """
    turn_order = [p["Name"] for p in cleaned["Players"]]

    if my_name is None or my_name not in turn_order:
        my_name = _infer_self_name(cleaned, fallback=my_name)

    players = {
        p["Name"]: PlayerInfo(name=p["Name"],
                              sets=p["Sets"],
                              hand_size=p["HandSize"])
        for p in cleaned["Players"]
    }

    state = GameState(
        my_name=my_name,
        my_hand=[(c[0], c[1]) for c in cleaned["Hand"]],
        stock_size=cleaned["Deck"],
        out_of_play=list(cleaned["OutOfPlay"]),
        players=players,
        turn_order=turn_order,
        turns=list(cleaned["Turns"]),
    )
    return state


def _infer_self_name(cleaned: dict, fallback: Optional[str]) -> str:
    """
    Best-effort "who am I?" inference when the runner didn't tell us.

    Strategy:
        * If exactly one Players entry has HandSize == len(Hand), use it.
        * Otherwise, among those that match, pick the one whose ask
          history is most consistent with the hand (they issued asks for
          ranks we still hold).
        * Final fallback: first player in the list.
    """
    my_hand_size = len(cleaned["Hand"])
    my_ranks = Counter(c[0] for c in cleaned["Hand"])
    candidates = [p["Name"] for p in cleaned["Players"]
                  if p["HandSize"] == my_hand_size]
    if len(candidates) == 1:
        return candidates[0]

    if candidates:
        best, best_score = None, -1
        for name in candidates:
            score = 0
            for t in cleaned["Turns"]:
                if t["Requester"] == name and my_ranks.get(t["Rank"], 0) > 0:
                    # They asked for a rank we still hold -> consistent.
                    score += 1
            if score > best_score:
                best, best_score = name, score
        if best:
            return best

    if fallback:
        return fallback
    # Last-ditch fallback: first player.
    return cleaned["Players"][0]["Name"] if cleaned["Players"] else ""


# ============================================================================
#  SECTION 3.  INFERENCE ENGINE
# ============================================================================

class InferenceEngine:
    """
    Walks through the turn history and updates every PlayerInfo with:
        * lower bounds on what each opponent still holds,
        * per-rank "known-zero-as-of-turn" markers,
        * counters that drive the opponent model.
    """

    def __init__(self, state: GameState):
        self.state = state

    # ---- main entry point ----

    def run(self) -> None:
        s = self.state
        for idx, turn in enumerate(s.turns):
            self._apply_turn(idx, turn)
        # After replaying every historical turn, retire ranks that are
        # fully out of play from every lower-bound counter.
        for p in s.players.values():
            for rank in list(p.known_min.keys()):
                if rank in s.out_of_play:
                    p.known_min[rank] = 0
                    p.known_zero_as_of_turn.pop(rank, None)

    # ---- per-turn bookkeeping ----

    def _apply_turn(self, idx: int, turn: dict) -> None:
        s = self.state
        r_name, e_name = turn["Requester"], turn["Requestee"]
        rank = turn["Rank"]
        received = turn["Received"]
        completed_set_rank = turn["Set"]

        if r_name not in s.players or e_name not in s.players:
            return  # malformed row

        requester = s.players[r_name]
        requestee = s.players[e_name]

        # (a) An ask for rank R means the requester held >= 1 of R at
        #     that moment. We record this as a LOWER BOUND that will
        #     only be decremented later by explicit give-aways or by
        #     the set being completed.
        requester.asks_issued[rank] += 1
        requestee.asks_received[rank] += 1
        if rank not in s.out_of_play:
            requester.known_min[rank] = max(requester.known_min[rank], 1)
            # This ask supersedes any earlier "known-zero" marker.
            requester.known_zero_as_of_turn.pop(rank, None)

        # (b) The requestee now gives `received` cards of `rank`.
        if received > 0:
            requester.asks_success[rank] += 1
            # Requestee had at least `received` of that rank -- but
            # they just gave them ALL to requester, so the lower bound
            # on requestee DROPS to 0 for that rank.
            requestee.known_min[rank] = 0
            requestee.known_zero_as_of_turn[rank] = idx
            requestee.cards_given_out[rank] += received
            # Requester's lower bound goes up by the received count
            # -- unless this turn also laid down the set, in which
            # case the rank is now out-of-play and nobody holds it.
            if completed_set_rank == rank:
                requester.known_min[rank] = 0
                requester.known_zero_as_of_turn[rank] = idx
            else:
                requester.known_min[rank] = \
                    requester.known_min.get(rank, 0) + received
        else:
            # Failed ask: requestee had zero of that rank AT THIS TIME.
            # (If they later draw from the stock or receive a transfer
            # we may have to weaken this, so we store the turn index.)
            requestee.known_min[rank] = 0
            requestee.known_zero_as_of_turn[rank] = idx

        # (c) If a set was laid down this turn, the requester's
        #     internal count of that rank goes to zero AND the rank
        #     is out-of-play for everyone.
        if completed_set_rank:
            for p in s.players.values():
                p.known_min[completed_set_rank] = 0
                p.known_zero_as_of_turn[completed_set_rank] = idx
            if completed_set_rank not in s.out_of_play:
                s.out_of_play.append(completed_set_rank)


# ============================================================================
#  SECTION 4.  OPPONENT MODELING
# ============================================================================

@dataclass
class OpponentProfile:
    """Cheap behavioral features used to classify each opponent."""
    name: str
    greedy_score: float = 0.0
    sticky_rank_score: float = 0.0
    target_repeat_score: float = 0.0
    near_complete_danger: float = 0.0
    sets: int = 0
    classification: str = "unknown"


class OpponentModel:
    """
    Classify each opponent and expose a small API the evaluator can use:

        * prob_still_holds(name, rank)      -> [0,1]
        * expected_count(name, rank)        -> float
        * danger_score(name)                -> [0,1]   (near-win pressure)
    """

    def __init__(self, state: GameState):
        self.state = state
        self.profiles: Dict[str, OpponentProfile] = {
            name: OpponentProfile(name=name, sets=pi.sets)
            for name, pi in state.players.items()
        }
        self._classify_all()

    # ---------------- classification ----------------

    def _classify_all(self) -> None:
        s = self.state
        for name in s.players:
            pi = s.players[name]
            prof = self.profiles[name]

            # Sticky rank: do they repeatedly ask for the same rank?
            if pi.asks_issued:
                max_asks_for_one_rank = max(pi.asks_issued.values())
                total_asks = sum(pi.asks_issued.values())
                prof.sticky_rank_score = (max_asks_for_one_rank / total_asks
                                          if total_asks else 0.0)

            # Greedy (ask for most-held rank): we can't see their hand,
            # but we can see whether their asks align with their known
            # lower-bound holdings. If known_min[rank] > 0 for their
            # asks, that's a "greedy" fingerprint.
            greedy_hits = 0
            for t in s.turns:
                if t["Requester"] != name:
                    continue
                if pi.known_min.get(t["Rank"], 0) > 0:
                    greedy_hits += 1
            total_asks = sum(pi.asks_issued.values())
            prof.greedy_score = (greedy_hits / total_asks
                                 if total_asks else 0.0)

            # Target-repeat: do they keep asking the same player?
            target_counter = Counter(t["Requestee"] for t in s.turns
                                     if t["Requester"] == name)
            if target_counter:
                max_targets = max(target_counter.values())
                total_targets = sum(target_counter.values())
                prof.target_repeat_score = (max_targets / total_targets
                                            if total_targets else 0.0)

            # Danger: sets-completed + how close they appear to a new set.
            prof.near_complete_danger = min(1.0, pi.sets / 4.0 + 0.1 *
                                            sum(1 for v in pi.known_min.values()
                                                if v >= 2))

            # Classification with simple thresholds.
            if prof.sticky_rank_score >= 0.5 and total_asks >= 2:
                prof.classification = "sticky_greedy"
            elif prof.target_repeat_score >= 0.6 and total_asks >= 3:
                prof.classification = "target_locked"
            elif total_asks >= 2 and prof.greedy_score >= 0.7:
                prof.classification = "greedy"
            else:
                prof.classification = "balanced_or_unknown"

    # ---------------- public probability API ----------------

    def prob_still_holds(self, name: str, rank: str) -> float:
        """P(player `name` holds >=1 of `rank` right now)."""
        s = self.state
        if rank in s.out_of_play or name == s.my_name:
            return 0.0

        pi = s.players[name]

        # Rock-solid lower bound: they DEFINITELY hold >=1.
        if pi.known_min.get(rank, 0) >= 1:
            return 1.0

        # Recent failed ask with no subsequent transfer/drawback: 0.
        if rank in pi.known_zero_as_of_turn:
            last_zero_turn = pi.known_zero_as_of_turn[rank]
            # Check whether they could have acquired one since.
            if not self._could_have_acquired_since(name, rank, last_zero_turn):
                return 0.0

        # Otherwise: hypergeometric over the "unseen pool" from our POV.
        unseen = self._unseen_by_me()
        copies_remaining = self._copies_remaining_outside_me(rank)
        if copies_remaining <= 0 or unseen <= 0:
            return 0.0

        # Prob that NONE of their `hand_size` cards are of this rank:
        # C(unseen - copies, hand_size) / C(unseen, hand_size)
        hand = pi.hand_size
        if hand <= 0:
            return 0.0
        p_none = _hypergeom_p_zero(unseen, copies_remaining, hand)
        return max(0.0, min(1.0, 1.0 - p_none))

    def expected_count(self, name: str, rank: str) -> float:
        """E[number of `rank` cards in `name`'s hand]."""
        s = self.state
        if rank in s.out_of_play or name == s.my_name:
            return 0.0
        pi = s.players[name]

        min_held = pi.known_min.get(rank, 0)
        unseen = self._unseen_by_me()
        copies_remaining = self._copies_remaining_outside_me(rank) - min_held
        if copies_remaining <= 0 or unseen <= 0:
            return float(min_held)
        # The remaining hand slots (after the known minimum) draw from
        # a reduced unseen pool.
        remaining_hand = pi.hand_size - min_held
        if remaining_hand <= 0:
            return float(min_held)
        expected_extra = remaining_hand * (copies_remaining /
                                           max(unseen - min_held, 1))
        return min_held + expected_extra

    def danger_score(self, name: str) -> float:
        return self.profiles[name].near_complete_danger

    # ---------------- helpers ----------------

    def _unseen_by_me(self) -> int:
        """
        Size of the pool of cards I do NOT see: all opponents' hands
        plus the stock.
        """
        s = self.state
        return (sum(pi.hand_size for name, pi in s.players.items()
                    if name != s.my_name) + s.stock_size)

    def _copies_remaining_outside_me(self, rank: str) -> int:
        """Total copies of `rank` outside my hand and not out-of-play."""
        s = self.state
        if rank in s.out_of_play:
            return 0
        in_my_hand = s.rank_count_in_my_hand(rank)
        return COPIES_PER_RANK - in_my_hand

    def _could_have_acquired_since(self, name: str, rank: str,
                                   last_zero_turn: int) -> bool:
        """
        Between turn `last_zero_turn` and the current moment, could
        `name` have drawn a `rank` from the stock or received it from
        a transfer we didn't see?
        """
        s = self.state
        # Look at every turn after last_zero_turn.
        for t in s.turns[last_zero_turn + 1:]:
            # They received cards of that rank from another player.
            if t["Requester"] == name and t["Rank"] == rank and t["Received"] > 0:
                return True
            # Their hand grew via a failed ask -> they drew from stock.
            # (The draw is hidden, but it COULD be of this rank.)
            if t["Requester"] == name and t["Received"] == 0:
                return True
        return False


# ============================================================================
#  SECTION 5.  MOVE GENERATION
# ============================================================================

@dataclass
class LegalMove:
    """A fully-specified legal ask."""
    requestee: str
    rank: str


def generate_legal_moves(state: GameState) -> List[LegalMove]:
    """
    Enumerate every LEGAL ask given the rules:
        * must request a rank the bot currently holds,
        * must target a living opponent that is not self.
    """
    ranks_in_hand = state.my_rank_set()
    # Only opponents with any cards left are useful targets. A player
    # whose hand_size == 0 cannot give cards (they'd immediately draw,
    # but from our turn's POV the ask fails no matter what).
    valid_targets = [o for o in state.opponents
                     if state.players[o].hand_size > 0]
    # If every opponent is empty (edge-case), still emit moves so the
    # runner doesn't crash -- the game engine will drive from there.
    if not valid_targets:
        valid_targets = state.opponents

    return [LegalMove(requestee=o, rank=r)
            for o in valid_targets for r in ranks_in_hand]


# ============================================================================
#  SECTION 6.  MOVE EVALUATION
# ============================================================================

class MoveEvaluator:
    """
    Score every legal move on a single composite utility.

    Utility =
        +  W_SET_COMPLETE  * I(success completes a 4-set)
        +  W_TRIPLE_BUILD  * I(success builds a triple)
        +  W_PAIR_BUILD    * I(success builds a pair)
        +  W_EXPECTED      * E[cards gained]
        +  W_SUCCESS_PROB  * P(success)
        +  W_CONTINUE      * P(success) * average_followup_utility
        +  W_EXPLOIT       * exploit_bonus_for_counter_strategy
        -  W_INFO_LEAK     * leak_penalty(rank, requestee)
        -  W_DANGER        * danger_score(requestee) * assists_requestee
    """

    # ---------- tunable weights ----------
    # Completing a set is the ONLY thing the game actually scores on,
    # so W_SET_COMPLETE dominates. Triple-building is the second most
    # valuable outcome because it moves us one ask away from a set,
    # AND it lets us capitalize on the "who still has rank R" info
    # we already exposed.
    W_SET_COMPLETE = 18.0
    W_TRIPLE_BUILD = 4.5
    W_PAIR_BUILD = 1.4
    W_EXPECTED = 2.0
    W_SUCCESS_PROB = 1.5
    W_CONTINUE = 1.2
    W_EXPLOIT = 3.0
    W_INFO_LEAK = 0.8
    W_DANGER = 1.5

    def __init__(self, state: GameState, model: OpponentModel):
        self.state = state
        self.model = model

    def score(self, move: LegalMove) -> Tuple[float, Dict[str, float]]:
        s = self.state
        r, target = move.rank, move.requestee
        profile = self.model.profiles.get(target)
        if profile is None:
            return -1e9, {}

        my_count = s.rank_count_in_my_hand(r)
        p_success = self.model.prob_still_holds(target, r)
        exp_count = self.model.expected_count(target, r)

        # Outcome bonuses (conditional on success).
        completes_set = (my_count == 3)          # we already have 3
        builds_triple = (my_count == 2)
        builds_pair = (my_count == 1)

        # Exploit bonus: sticky/greedy opponents holding onto a rank are
        # the single best target for that rank.
        exploit_bonus = 0.0
        asks_for_rank = self.state.players[target].asks_issued.get(r, 0)
        if asks_for_rank >= 1:
            # They asked for it, meaning they held it; they probably still do.
            exploit_bonus += 1.0 * asks_for_rank
        if profile.classification == "sticky_greedy":
            exploit_bonus += 0.5

        # Information-leak penalty.
        # Asking reveals that we hold >=1 of that rank. If we hold 3,
        # we LOUDLY signal "one away from completing". HOWEVER this
        # only hurts us when the ask FAILS: if it succeeds and we
        # complete the set, the rank immediately goes out-of-play
        # and the leak is irrelevant. So we weight leak by (1-p_success).
        leak = 0.0
        if my_count == 3:
            leak += 1.5
        elif my_count == 2:
            leak += 0.6
        else:
            leak += 0.2
        leak *= (1.0 - p_success)
        # The leak is worse if aimed at a dangerous opponent.
        leak *= (1.0 + self.model.danger_score(target))

        # Danger-assist penalty: if the ask FAILS, we tell the target
        # that we hold at least one of rank r. If the target already
        # looks dangerous (many sets, big hand) and they ALSO happen
        # to hold other copies of r, we just handed them a great next
        # ask. Mitigated if we expect high success anyway.
        danger_assist = (1.0 - p_success) * self.model.danger_score(target)

        # Continuation term: if we succeed, we keep our turn. Estimate
        # a one-ply lookahead for average follow-up utility: (a) if the
        # ask completes a set, we get to ask again from a now 3+ card
        # richer table; (b) if it builds a triple, we may complete
        # it next ask.
        followup = 0.0
        if completes_set:
            followup += 2.0
        elif builds_triple:
            followup += 1.2
        else:
            followup += 0.4

        # Greedy E[cards] term: expected cards we'd actually get = E[count]
        # conditional on they hold any. Approximate = exp_count.
        components = {
            "p_success": p_success,
            "exp_count": exp_count,
            "completes_set": float(completes_set),
            "builds_triple": float(builds_triple),
            "builds_pair": float(builds_pair),
            "exploit": exploit_bonus,
            "leak": leak,
            "danger_assist": danger_assist,
            "followup": followup,
            "my_count": my_count,
        }

        util = (
            self.W_SET_COMPLETE * completes_set * p_success
            + self.W_TRIPLE_BUILD * builds_triple * p_success
            + self.W_PAIR_BUILD * builds_pair * p_success
            + self.W_EXPECTED * exp_count
            + self.W_SUCCESS_PROB * p_success
            + self.W_CONTINUE * p_success * followup
            + self.W_EXPLOIT * exploit_bonus
            - self.W_INFO_LEAK * leak
            - self.W_DANGER * danger_assist
        )
        return util, components


# ============================================================================
#  SECTION 7.  MONTE-CARLO TIE-BREAKER (shallow rollout)
# ============================================================================

class MonteCarloTieBreaker:
    """
    When two or more legal moves have near-identical evaluator scores,
    we sample plausible hidden hands consistent with everything observed
    and play out a 2-ply rollout to estimate expected set gain.

    This is a *shallow* search, intentionally, so the full bot stays
    under tight latency budgets.
    """

    def __init__(self, state: GameState, model: OpponentModel,
                 samples: int = 24, seed: int = 17):
        self.state = state
        self.model = model
        self.samples = samples
        self.rng = random.Random(seed)

    def break_tie(self, top_moves: List[Tuple[float, LegalMove, dict]]
                  ) -> LegalMove:
        if not top_moves:
            raise ValueError("Cannot tie-break empty move list.")
        best_move, best_ev = top_moves[0][1], -math.inf
        for base_score, move, _ in top_moves:
            ev = self._estimate(move)
            # Combine with deterministic score to avoid pure-MC noise
            # flipping between equally-good moves.
            combined = 0.6 * base_score + 0.4 * ev
            if combined > best_ev:
                best_move, best_ev = move, combined
        return best_move

    def _estimate(self, move: LegalMove) -> float:
        """Average 2-ply payoff across `self.samples` imagined worlds."""
        total = 0.0
        for _ in range(self.samples):
            total += self._one_rollout(move)
        return total / max(self.samples, 1)

    def _one_rollout(self, move: LegalMove) -> float:
        """
        Deal plausible hidden hands, then evaluate:
            * +1 if the sampled requestee happens to hold the rank
              (proxy for 'success' -> continuation + cards)
            * +2 if the ask would complete a set (we hold 3)
            * +0.5 bonus per extra copy gained.
        """
        s = self.state
        my_count = s.rank_count_in_my_hand(move.rank)
        sampled = self._sample_hidden()
        target_hand = sampled.get(move.requestee, Counter())
        received = target_hand.get(move.rank, 0)
        reward = 0.0
        if received > 0:
            reward += 1.0
            reward += 0.5 * received
            if my_count + received >= 4:
                reward += 2.0   # set-completion bonus
        return reward

    def _sample_hidden(self) -> Dict[str, Counter]:
        """
        Construct a random assignment of the unseen cards to the
        opponents (respecting their declared HandSize and the already-
        inferred lower bounds).

        Stock cards are *not* assigned to any player, they just
        consume pool capacity.
        """
        s = self.state
        # Build the pool of unseen cards (rank list, ignoring suits).
        pool: List[str] = []
        for rank in CANONICAL_RANKS:
            if rank in s.out_of_play:
                continue
            remaining = COPIES_PER_RANK - s.rank_count_in_my_hand(rank)
            pool.extend([rank] * remaining)

        # First, give each opponent their lower-bound known cards.
        assigned: Dict[str, Counter] = {o: Counter() for o in s.opponents}
        for o in s.opponents:
            pi = s.players[o]
            for rank, n in pi.known_min.items():
                if n <= 0 or rank in s.out_of_play:
                    continue
                take = min(n, pool.count(rank), pi.hand_size)
                for _ in range(take):
                    pool.remove(rank)
                    assigned[o][rank] += 1

        # Shuffle and distribute the rest: stock eats capacity first.
        self.rng.shuffle(pool)
        stock_capacity = s.stock_size
        idx = 0
        # Stock consumes the first `stock_capacity` cards of the pool.
        idx += min(stock_capacity, len(pool))

        for o in s.opponents:
            pi = s.players[o]
            needed = pi.hand_size - sum(assigned[o].values())
            for _ in range(max(0, needed)):
                if idx >= len(pool):
                    break
                assigned[o][pool[idx]] += 1
                idx += 1

        return assigned


# ============================================================================
#  SECTION 8.  ACTION SELECTION
# ============================================================================

class GoFishBot:
    """Glue class that wires everything together and picks the final move."""

    def __init__(self, state: GameState, mc_samples: int = 24,
                 seed: int = 17):
        self.state = state
        InferenceEngine(state).run()
        self.model = OpponentModel(state)
        self.evaluator = MoveEvaluator(state, self.model)
        self.tie_breaker = MonteCarloTieBreaker(state, self.model,
                                                samples=mc_samples, seed=seed)

    def choose_move(self) -> LegalMove:
        moves = generate_legal_moves(self.state)
        if not moves:
            # We hold NO cards. This shouldn't happen mid-turn, but if it
            # does, we emit a "safe" dummy pointing to the first opponent
            # with any ranks the table has seen -- the engine will make
            # us draw from the stock.
            opp = next(iter(self.state.opponents), "")
            # Any legal rank we could be dealt: pick from canonical list.
            rank = CANONICAL_RANKS[0]
            return LegalMove(requestee=opp, rank=rank)

        scored: List[Tuple[float, LegalMove, dict]] = []
        for m in moves:
            score, comps = self.evaluator.score(m)
            scored.append((score, m, comps))

        scored.sort(key=lambda x: x[0], reverse=True)
        top_score = scored[0][0]
        near_top = [(sc, mv, c) for (sc, mv, c) in scored
                    if top_score - sc < 0.5]

        if len(near_top) == 1:
            return near_top[0][1]
        return self.tie_breaker.break_tie(near_top)

    # -------- serialization to the competition's exact JSON format --------

    @staticmethod
    def to_output_json(move: LegalMove, rank_style: str = "single") -> str:
        """
        Emit EXACTLY:
            {"Requestee":"player name", "Request":"rank"}
        with no trailing whitespace, no pretty-printing, and no extra keys.

        Parameters
        ----------
        rank_style:
            "single" -> the 10 is emitted as "1"  (matches the organizer's
                        Turns.Rank convention, which the game doc defines
                        as "a character (2-9, J, Q, K, A)"). This is the
                        default and the safe choice for this competition.
            "full"   -> the 10 is emitted as "10" (human-readable).
        """
        r = move.rank
        if rank_style == "single" and r == "10":
            r = "1"
        # We intentionally build the JSON manually to guarantee key order
        # and key capitalization match the organizer's spec exactly.
        return ('{"Requestee":%s, "Request":%s}'
                % (json.dumps(move.requestee), json.dumps(r)))


# ============================================================================
#  SECTION 9.  HYPERGEOMETRIC HELPERS
# ============================================================================

def _hypergeom_p_zero(pool: int, hits: int, draws: int) -> float:
    """
    P(0 hits when drawing `draws` cards without replacement from a pool
    of `pool` cards that contains `hits` target cards).

    = C(pool - hits, draws) / C(pool, draws)
    """
    if draws <= 0 or hits <= 0:
        return 1.0 if hits <= 0 else 1.0  # no target cards -> P(0)=1
    if draws > pool:
        return 0.0
    # Numerically stable product form:
    prob = 1.0
    for i in range(draws):
        num = (pool - hits - i)
        den = (pool - i)
        if den <= 0:
            return 0.0
        if num <= 0:
            return 0.0
        prob *= num / den
    return prob


# ============================================================================
#  SECTION 10.  CLI / TEST HARNESS
# ============================================================================

def run_on_file(path: str, my_name: Optional[str], quiet: bool = False,
                mc_samples: int = 24, rank_style: str = "single") -> str:
    """
    Load a state file, play one turn, print/return the JSON move.

    If `my_name` is None, auto-discover (see _infer_self_name).
    """
    cleaned = parse_state_file(path)
    state = build_game_state(cleaned, my_name=my_name)
    bot = GoFishBot(state, mc_samples=mc_samples)
    move = bot.choose_move()
    output = GoFishBot.to_output_json(move, rank_style=rank_style)
    if not quiet:
        print(output)
    return output


def _demo_all(base_dir: str) -> None:
    """Run the bot on every provided history file for quick verification."""
    scenarios = [
        ("first-turn.json",           None,
         "Opening turn, 4 players @ 7 cards, deck=24."),
        ("first-turn-fillable.json",  None,
         "Opening turn with a different hand (fillable variant)."),
        ("mid-game.json",             None,
         "Mid-game with a rich turn history; several sets already done."),
        ("end-of-game.json",          None,
         "Endgame with deck almost empty and heavy inference signal."),
    ]
    for fname, my, desc in scenarios:
        full = os.path.join(base_dir, fname)
        if not os.path.exists(full):
            continue
        print(f"\n===== {fname} =====")
        print(desc)
        cleaned = parse_state_file(full)
        state = build_game_state(cleaned, my_name=my)
        print(f"    my_name inferred as: {state.my_name!r}")
        print(f"    my_hand: "
              f"{[r for r, _ in state.my_hand]}")
        print(f"    out_of_play: {state.out_of_play}")
        bot = GoFishBot(state, mc_samples=16)
        # Show opponent profiles for transparency.
        for name in state.opponents:
            prof = bot.model.profiles[name]
            pi = state.players[name]
            print(f"    opp '{name}': class={prof.classification} "
                  f"hand={pi.hand_size} sets={pi.sets} "
                  f"known_min={dict(pi.known_min)}")
        move = bot.choose_move()
        print(f"    MOVE: {GoFishBot.to_output_json(move)}")


def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="Competition Go Fish bot: emit one legal ask.")
    parser.add_argument("--state", "-s", help="Path to the state JSON file.")
    parser.add_argument("--name", "-n", default=None,
                        help="This program's player name (recommended).")
    parser.add_argument("--demo", action="store_true",
                        help="Run on every bundled demo file.")
    parser.add_argument("--demo-dir", default=None,
                        help="Directory containing demo files "
                             "(defaults to the script's directory).")
    parser.add_argument("--samples", type=int, default=24,
                        help="Monte-Carlo samples for tie-breaking.")
    parser.add_argument("--rank-style", choices=["single", "full"],
                        default="single",
                        help="Output format for the 10 rank ('single' "
                             "emits \"1\" per the organizer's Turns.Rank "
                             "convention; 'full' emits \"10\").")
    args = parser.parse_args()

    # Environment-variable fallback for competition runners.
    name_from_env = os.environ.get("GO_FISH_BOT_NAME")
    my_name = args.name or name_from_env

    if args.demo:
        base = args.demo_dir or os.path.dirname(os.path.abspath(__file__))
        _demo_all(base)
        return

    if not args.state:
        parser.error("Either --state PATH or --demo is required.")
    run_on_file(args.state, my_name, quiet=False, mc_samples=args.samples,
                rank_style=args.rank_style)


if __name__ == "__main__":
    _cli()
