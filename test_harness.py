"""
test_harness.py
===============
Standalone harness that exercises gofish_bot.py against the organizer-provided
JSON history files.  For every file it:

    * parses and cleans the state,
    * runs the bot's choose_move(),
    * asserts the output JSON has the exact required shape,
    * asserts legality (rank is in our hand; requestee is a real other player
      with >=1 card),
    * prints full inference diagnostics for the top candidate moves.

Run:
    python test_harness.py                   # default name=steven
    python test_harness.py --name MyBot      # inject name into blank slot
    python test_harness.py --uploads /path   # point at a different folder
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Dict, List

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)
import gofish_bot as gb                                            # noqa: E402


# ---------------------------------------------------------------------------
# Assertions
# ---------------------------------------------------------------------------

def assert_output_shape(move: Dict) -> None:
    assert isinstance(move, dict), "move must be dict"
    assert set(move.keys()) == {"Requestee", "Request"}, \
        f"keys must be exactly Requestee/Request, got {list(move.keys())}"
    assert isinstance(move["Requestee"], str), "Requestee must be str"
    assert isinstance(move["Request"], str), "Request must be str"


def assert_legality(move: Dict, gs: gb.GameState) -> None:
    # We only enforce legality when a legal move was possible.
    legal = [(o, r) for o in gs.opponents
             for r in set(gs.my_hand)
             if gs.hand_size.get(o, 0) > 0 and r not in gs.out_of_play]
    if not legal:
        return                        # bot may return a blank safely
    assert move["Request"] in set(gs.my_hand), \
        f"illegal ask: rank {move['Request']!r} not in my hand {gs.my_hand}"
    assert move["Requestee"] in gs.opponents, \
        f"illegal requestee: {move['Requestee']!r} not in opponents {gs.opponents}"
    assert gs.hand_size.get(move["Requestee"], 0) > 0, \
        f"illegal requestee: {move['Requestee']!r} has 0 cards"
    assert move["Request"] not in gs.out_of_play, \
        f"illegal ask: {move['Request']!r} is already a completed set"


# ---------------------------------------------------------------------------
# Per-file report
# ---------------------------------------------------------------------------

def report_file(path: str, my_name: str, mc_samples: int,
                debug: bool) -> Dict[str, str]:
    with open(path) as f:
        raw = json.load(f)
    gs = gb.GameState.from_raw(raw, my_name=my_name)
    move = gb.choose_move(gs, use_mc=True, mc_samples=mc_samples, seed=42)

    assert_output_shape(move)
    assert_legality(move, gs)

    tag = os.path.basename(path)
    print(f"=== {tag} ===")
    print(f"  me      : {gs.my_name}")
    print(f"  hand    : {gs.my_hand}")
    print(f"  deck    : {gs.deck}    out_of_play: {sorted(gs.out_of_play)}")
    print(f"  opps    : {[(o, gs.hand_size[o], gs.sets_by_player[o]) for o in gs.opponents]}")
    print(f"  chosen  : {json.dumps(move)}")

    if debug:
        inf = gb.Inference(gs)
        ev = gb.MoveEvaluator(gs, inf)
        ranked = ev.rank_all()
        if ranked:
            print("  top candidates (heuristic only):")
            for (opp, rank), sc in ranked[:6]:
                print(f"    {opp:<10} {rank:<3}"
                      f"  score={sc['total']:+.3f}"
                      f"  P(succ)={sc['p_success']:.3f}"
                      f"  E[cards]={sc['e_got']:.2f}"
                      f"  my={sc['my_count']}")
        else:
            print("  no legal candidates")
    print()
    return move


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--uploads", "-u",
                    default="/sessions/brave-elegant-clarke/mnt/uploads",
                    help="folder containing *.json test files")
    ap.add_argument("--name", "-n", default="steven",
                    help="my bot's player name (injected into blank slots)")
    ap.add_argument("--mc-samples", type=int, default=96)
    ap.add_argument("--no-debug", action="store_true",
                    help="suppress candidate-move breakdown")
    args = ap.parse_args()

    files: List[str] = sorted(glob.glob(os.path.join(args.uploads, "*.json")))
    if not files:
        print(f"No JSON files in {args.uploads}")
        sys.exit(1)

    failed = 0
    for path in files:
        try:
            report_file(path, my_name=args.name,
                        mc_samples=args.mc_samples, debug=not args.no_debug)
        except AssertionError as e:
            failed += 1
            print(f"ASSERTION FAILED on {path}: {e}\n")
        except Exception as e:
            failed += 1
            print(f"ERROR on {path}: {e}\n")

    total = len(files)
    passed = total - failed
    print(f"------ summary: {passed}/{total} files passed ({failed} failed) ------")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
