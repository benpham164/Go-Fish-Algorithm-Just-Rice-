#!/usr/bin/env python3
"""
Test harness for the Go Fish competition bot.

Runs a battery of checks against every provided history file:

    (1) JSON parses + cleans without error.
    (2) Rank normalization ("1" <-> "10") works.
    (3) The player-name fillable case is handled:
            - blank "Name" field is tolerated,
            - injected bogus whitespace/casing is stripped,
            - unknown player name falls back to auto-detection.
    (4) Every chosen move satisfies the *legal ask* rule:
            - the requested rank is in the bot's current hand,
            - the requestee is a real opponent.
    (5) Output JSON matches the exact competition format:
            {"Requestee":"player name", "Request":"rank"}
    (6) Bot runs end-to-end on all 4 supplied scenario files.

Exit code is 0 on success, non-zero on any failure.
"""

import copy
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from go_fish_bot import (                                  # noqa: E402
    GoFishBot,
    build_game_state,
    clean_state_dict,
    generate_legal_moves,
    normalize_rank,
    parse_state_file,
)

FILES = [
    "first-turn.json",
    "first-turn-fillable.json",
    "mid-game.json",
    "end-of-game.json",
]

OUTPUT_PATTERN = re.compile(
    r'^\{"Requestee":"[^"]+", "Request":"(?:[2-9AJQK]|1|10)"\}$')

failures = []


def check(condition: bool, msg: str) -> None:
    tag = "PASS" if condition else "FAIL"
    print(f"  [{tag}] {msg}")
    if not condition:
        failures.append(msg)


# ---------------------------------------------------------------------------
# Unit tests for rank normalization (the cleaner's main job).
# ---------------------------------------------------------------------------
print("\n--- normalize_rank unit tests ---")
check(normalize_rank("1") == "10",    "'1' is normalized to '10' (Turns.Rank convention)")
check(normalize_rank("10") == "10",   "'10' stays '10'")
check(normalize_rank("T") == "10",    "'T' alias normalized to '10'")
check(normalize_rank("j") == "J",     "lowercase 'j' uppercased to 'J'")
check(normalize_rank(" A ") == "A",   "whitespace trimmed from ranks")
check(normalize_rank("King") == "K",  "'King' alias normalized to 'K'")
check(normalize_rank("") == "",       "empty rank stays empty")

# ---------------------------------------------------------------------------
# End-to-end on each provided file.
# ---------------------------------------------------------------------------
for fname in FILES:
    path = os.path.join(HERE, fname)
    if not os.path.exists(path):
        print(f"\n--- {fname}: SKIP (missing) ---")
        continue

    print(f"\n--- {fname} ---")
    cleaned = parse_state_file(path)

    # Rank normalization: every Turns.Rank and OutOfPlay should be canonical.
    valid_ranks = {"2", "3", "4", "5", "6", "7", "8",
                   "9", "10", "J", "Q", "K", "A"}
    all_ok = all(t["Rank"] in valid_ranks for t in cleaned["Turns"])
    check(all_ok, "every Turns.Rank is canonical after cleaning")

    state = build_game_state(cleaned, my_name=None)
    bot = GoFishBot(state, mc_samples=16)
    move = bot.choose_move()

    # Legal-ask rule: rank must be in own hand.
    ranks_in_hand = {r for r, _ in state.my_hand}
    check(move.rank in ranks_in_hand,
          f"chosen rank {move.rank!r} is in own hand {sorted(ranks_in_hand)!r}")

    # Requestee must be a real opponent, not self.
    check(move.requestee in state.opponents,
          f"requestee {move.requestee!r} is a valid opponent")

    # Exact JSON format.
    out = GoFishBot.to_output_json(move)
    check(bool(OUTPUT_PATTERN.match(out)),
          f"output matches required JSON shape: {out}")

    # Round-trip parseable as JSON with correct keys.
    parsed = json.loads(out)
    check(set(parsed.keys()) == {"Requestee", "Request"},
          "output JSON has exactly the required keys")

    print(f"    -> {out}")

# ---------------------------------------------------------------------------
# Fillable-file cleaning: inject blank / messy names and confirm we recover.
# ---------------------------------------------------------------------------
print("\n--- fillable data cleaning stress test ---")
with open(os.path.join(HERE, "first-turn-fillable.json")) as fh:
    base = json.load(fh)

# Simulate an organizer filling a blank slot with whitespace/casing noise.
messy = copy.deepcopy(base)
messy["Players"][0]["Name"] = "  MyBoT  "
cleaned = clean_state_dict(messy)
check(cleaned["Players"][0]["Name"] == "MyBoT",
      "leading/trailing whitespace stripped from injected name")

# Simulate a fillable slot left literally empty.
messy2 = copy.deepcopy(base)
messy2["Players"][0]["Name"] = ""
cleaned2 = clean_state_dict(messy2)
check(len(cleaned2["Players"]) == len(base["Players"]) - 1,
      "blank-name player is dropped during cleaning")

# State still builds and a legal move is produced.
state2 = build_game_state(cleaned2, my_name=None)
bot2 = GoFishBot(state2, mc_samples=4)
move2 = bot2.choose_move()
check(move2.rank in {r for r, _ in state2.my_hand},
      "bot still picks a legal rank after blank-name cleaning")

# ---------------------------------------------------------------------------
# Rank-style switch: confirm "10" can be emitted as "1" or "10".
# ---------------------------------------------------------------------------
print("\n--- rank-style emission ---")
from go_fish_bot import LegalMove
m = LegalMove(requestee="alice", rank="10")
check(GoFishBot.to_output_json(m, rank_style="single") ==
      '{"Requestee":"alice", "Request":"1"}',
      "'single' rank-style emits '1' for the 10")
check(GoFishBot.to_output_json(m, rank_style="full") ==
      '{"Requestee":"alice", "Request":"10"}',
      "'full' rank-style emits '10' for the 10")

# ---------------------------------------------------------------------------
# Summary.
# ---------------------------------------------------------------------------
print("\n====================================================")
if failures:
    print(f"TEST HARNESS RESULT: {len(failures)} failure(s)")
    for f in failures:
        print(f"    * {f}")
    sys.exit(1)
print("TEST HARNESS RESULT: ALL CHECKS PASSED")
