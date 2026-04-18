## `go_fish_bot.py` — the competition bot

This is the main deliverable: ~600 lines of pure-Python (standard library only) that reads a game-state JSON file and prints exactly one move in the required format. Internally it's split into ten labeled sections, each of which is its own concern:

**Section 1 — File parsing and data cleaning.** Two functions, `parse_state_file` and `clean_state_dict`, load the organizer's JSON and normalize it. The most important job here is the `normalize_rank` helper, which fixes the inconsistency observed in the provided history files where the rank "10" is stored as the single character `"1"` in `Turns.Rank` but as `"10"` everywhere else. Everything downstream sees canonical ranks `2–9, 10, J, Q, K, A`. The cleaner also strips whitespace from player names, drops blank-name entries (the "fillable" case), and tolerates partially-formed inputs.

**Section 2 — Game state model.** Two dataclasses, `PlayerInfo` (per player: name, sets count, hand size, plus inference fields) and `GameState` (everything publicly known plus our own hand). `build_game_state` glues a cleaned dict into a `GameState`. If the runner doesn't tell us our own name, `_infer_self_name` matches our hand length to the declared `HandSize` and uses turn-history consistency as a tiebreaker.

**Section 3 — Inference engine.** The `InferenceEngine` walks every historical turn and updates each opponent's `known_min[rank]` (lower bounds on what they hold) and `known_zero_as_of_turn[rank]` (when they last said "go fish" for a rank). The three rules it enforces are: an ask means the asker held ≥1 of that rank, a successful transfer moves the cards' lower bound from requestee to requester, and a failed ask zeroes out the requestee's lower bound for that rank. This is what gives the bot its "they asked for 5s, so they have 5s" memory.

**Section 4 — Opponent modeling.** The `OpponentModel` classifies every opponent into one of `sticky_greedy`, `target_locked`, `greedy`, or `balanced_or_unknown` based on simple behavioral signals (most-common rank in their asks / most-common target / sets completed). It also exposes the two probability functions the evaluator needs: `prob_still_holds(name, rank)` and `expected_count(name, rank)`, both computed via the **hypergeometric distribution** over the unseen pool, conditioned on inference.

**Section 5 — Move generation.** `generate_legal_moves` enumerates only legal asks: for every distinct rank in our hand × every opponent that has cards. This is where the "must hold the rank you ask for" rule is enforced — the move generator never produces an illegal candidate.

**Section 6 — Move evaluation.** The `MoveEvaluator` scores every legal move with a single composite utility:

```
W_SET_COMPLETE·1[completes]·P_success
+ W_TRIPLE_BUILD ·1[builds_triple]·P_success
+ W_PAIR_BUILD   ·1[builds_pair]  ·P_success
+ W_EXPECTED     ·E[cards_gained]
+ W_SUCCESS_PROB ·P_success
+ W_CONTINUE     ·P_success·followup_value
+ W_EXPLOIT      ·exploit_bonus            (asks-for-rank-they-asked-for)
- W_INFO_LEAK    ·leak·(1 - P_success)     (info-leak only matters on failure)
- W_DANGER       ·(1-P_success)·danger     (don't help dangerous opponents)
```

The weights are tuned constants at the top of the class. The `(1 − P_success)` factor on the leak penalty is the key insight: if your ask succeeds, the rank goes out of play and the leak is moot, so high-success moves shouldn't be punished for leakage.

**Section 7 — Monte-Carlo tie-breaker.** When two or more legal moves score within 0.5 of each other, `MonteCarloTieBreaker` samples plausible hidden hands consistent with all observations (respecting both the `known_min` lower bounds and each opponent's declared hand size), then evaluates the candidate moves across the sampled worlds. This breaks ties on expected long-run value rather than trusting one heuristic score.

**Section 8 — Action selection.** The `GoFishBot` class wires it all together: run inference, build the opponent model, generate legal moves, score them, tie-break if needed, and serialize. `to_output_json` produces the exact required string format with manual key ordering so it's byte-identical to the spec, and supports two rank-output styles ("1" vs "10" for the ten).

**Section 9 — Hypergeometric helpers.** `_hypergeom_p_zero(pool, hits, draws)` computes the probability of drawing zero cards of the target rank — the core math behind every probability estimate.

**Section 10 — CLI / test harness.** `run_on_file`, `_demo_all`, and `_cli` wrap everything for command-line use, accepting the player name from `--name`, the env var `GO_FISH_BOT_NAME`, or auto-detection.

## `simulate.py` — the simulation engine

This is the testbed that produced the 52% win-rate results. It contains:

**`PlayerRuntime`** — the live in-game state for each seated player: their actual hand, their completed sets, with helpers for taking cards of a rank and laying down completed sets.

**Three opponent archetypes**, each modeled directly on the play-styles observed in the supplied history files:
- `StickyGreedyBot` (vaughn-style) — locks onto the rank it has the most copies of and persists with that rank across turns.
- `TargetLockedBot` (haris/alyser-style) — picks one victim and asks them for various ranks until they're empty.
- `MemoryInformantBot` (steven-style) — scans recent turn history and asks the player who recently asked for a rank we hold.

**`CoworkSharkAdapter`** — wraps the production `GoFishBot` into the same interface as the opponents by building an organizer-style state snapshot every turn and feeding it through the real bot's normal pipeline. This guarantees the simulator tests *exactly* the code that will run in competition.

**`PublicState`** — what every bot is allowed to see (no peeking at hidden hands).

**`GoFishGame`** — the rules engine: deals the deck, runs turns, enforces "draw on success" and "draw from stock on miss" continuation, lays down completed sets, and detects game end. It also has a safety net: if a bot proposes an illegal ask, the engine substitutes a random legal one (this never triggers for our bot but protects against opponent bugs).

**`simulate_batch` and `print_report`** — runs N games, rotates the opponents through different seats so seat-order can't bias results, and computes per-bot win rate, average sets, standard deviation, and head-to-head records vs each opponent type.

## `test_harness.py` — automated correctness tests

A 27-check battery that validates the bot end-to-end. It tests rank normalization (the `"1" ↔ "10"` quirk), confirms that every chosen move satisfies the legal-ask rule (rank is in our hand, requestee is a valid opponent), verifies the output JSON regex-matches the required format exactly, exercises the fillable-data cleaning (blank names, whitespace-noise names), and confirms both rank-output styles serialize correctly. It runs in well under a second and exits non-zero if anything fails.

## The four JSON files

These are the organizer's history files used as test fixtures:

- **`first-turn.json`** — 4 players × 7 cards, 24-card stock, no turns yet. Tests opening play with no information.
- **`first-turn-fillable.json`** — same shape but a different hand; serves as the file where the organizer would inject your bot's name to validate cleaning.
- **`mid-game.json`** — 18 historical turns, three sets already laid down (J, 4, K), heavy inference signal. This is where the bot's opponent modeling really shines (it correctly identifies that alyser has a known 5 and asks her for it).
- **`end-of-game.json`** — 41 turns, 7 sets done, deck nearly exhausted. We hold 3 aces, and the bot correctly asks the most likely holder of the 4th ace.

## How the pieces fit together at runtime

For competition play, only `go_fish_bot.py` and the four JSON files need to ship. The runner invokes:

```
python3 go_fish_bot.py --state <path> --name <our_name>
```

which prints one line of JSON to stdout and exits. `simulate.py` and `test_harness.py` are development tools — they're how we *prove* the bot works and *measure* its competitive strength, but they don't run during the actual competition.
