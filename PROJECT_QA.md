Here's a walkthrough of the full pipeline:

---

**Stage 1: Parsing & Cleaning**

Before anything else, the bot normalizes all raw data. Every rank token in the JSON passes through `clean_rank()`, which uppercases it, maps `"T"` to `"10"`, and silently drops `"1"` (an invalid rank that appears in some history files). `parse_hand()` handles both `[[rank, suit], ...]` and `[rank, ...]` card formats. Any turn in the history with an invalid rank gets dropped entirely so it can't corrupt the probability model later.

---

**Stage 2: GameState**

`GameState.from_raw()` builds a cleaned snapshot of everything the bot knows: our hand, every player's hand size and set count, the stock size, and the full turn log. One subtle job here is resolving our own name — it tries the CLI `--name` arg first, then the `GOFISH_BOT_NAME` env var, and finally auto-detects by finding the player whose recorded hand size matches the length of our actual hand. If the JSON has a blank name slot (the organizer's "fillable" test format), it gets replaced with our name. Player list order is preserved as-is because it matches turn order.

---

**Stage 3: Inference**

This is where the bot builds a belief model for each opponent by replaying the entire turn log from scratch. For every turn, it extracts three facts: the requester must have held at least one of the rank they asked for (you can only ask for what you have), a successful transfer tells us exactly how many cards changed hands, and a failed ask proves the requestee had zero of that rank at that moment.

These facts populate two key data structures per opponent. `known[rank]` is a lower-bound count — if we saw someone receive 2 Queens, we know they hold at least 2. `cannot_hold[rank]` is an aging constraint — after a failed ask, the opponent is marked as holding zero of that rank, but the constraint fades as they draw cards (incremented by 1 per draw, expired after 4 draws) since they might have drawn that rank since. A permanent tombstone (age 99) is set when a rank is completed and all 4 cards leave play.

The engine also tracks `interest[rank]` (how often an opponent has asked for it, a tell for what they're collecting) and does light style classification — labeling opponents as "greedy" (same rank three asks in a row) or "follower" (chasing ranks others just failed on).

For probability queries, `prob_opponent_has()` works through three cases in priority order: if `known[rank] >= 1`, return 1.0 immediately. If there's a fresh `cannot_hold` constraint, return a dampened probability (capped at 0.30) based on how many draws have happened since. Otherwise, use the hypergeometric distribution — given how many unseen copies of the rank exist and how many uncommitted card slots the opponent has, what's the probability at least one slot holds that rank?

---

**Stage 4: MoveEvaluator**

Every legal `(opponent, rank)` pair is scored by expected value: `p × success_value + (1 − p) × failure_value`, then adjusted with two additional terms.

The success value is tiered — completing a set of 4 is worth 12 points, forming a triple 4.5, a pair 1.6, a single 0.4 — plus a flat continuation bonus for keeping your turn. The failure value isn't purely negative: it includes an information credit (you learned the opponent doesn't have it), and if the stock is non-empty, a small bonus for the probability you lucky-draw the rank you just asked for.

On top of the base EV, two corrections are applied. The denial bonus rewards pulling cards from a dangerous opponent — one who already has many known cards and is close to completing sets. The reveal penalty accounts for the fact that asking advertises you hold a rank: every opponent after you in turn order can exploit this before you play again. The penalty is weighted by turn-order distance with geometric decay (0.70 per step), multiplied by how many copies of the rank you'd still hold after the ask. If the ask completes a set, the penalty is zero since the cards leave play.

All legal moves are scored this way and sorted descending.

---

**Stage 5: Monte Carlo**

The top 6 candidates from the heuristic evaluator are refined by running 96 sampled rollouts. `sample_world()` builds a plausible hidden game state: it constructs a pool of all cards not in our hand or out-of-play, shuffles it, seeds each opponent's hand with their `known` cards, then fills their remaining uncommitted slots from the pool while respecting fresh `cannot_hold` constraints (cards the opponent was recently proven not to hold).

For each sampled world, `_rollout()` simulates one turn ahead: if the opponent has the rank, cards transfer and reward accumulates using the same tier structure. On success it also takes one more greedy step — finding the best follow-up ask in that world for a shallow second-ply estimate. On failure, the bot draws from stock and checks if it got lucky. A soft defensive penalty fires if the ask would leave cards in hand that an opponent is close to collecting — decayed by turn-order distance.

After 96 samples, the average reward is the MC estimate for that move.

---

**Final decision**

The heuristic score and MC estimate are blended 55/45 and the highest blended score wins. The result is printed as `{"Requestee": "playerName", "Request": "rank"}` — the only thing written to stdout.
