# Go Fish Bot — Project Q&A

A write-up of the design, problem, solution, algorithm, and troubleshooting process for the competition Go Fish bot.

---

## 1. How did we come up with the solution?

We started by reading every file the organizer provided — the Box note that documents the JSON schema, and the five state snapshots (`first-turn.json`, `first-turn-fillable.json`, `mid-game.json`, `late-game.json`, `game-review.json`). That let us reverse-engineer the exact data format instead of guessing: `Players[]`, `Turns[]`, `OutOfPlay`, `Hand`, `Deck`.

From the turn history we also noticed real-world quirks (e.g., invalid `"Rank": "1"` entries) that forced us to write defensive cleaning rather than a strict parser. Once the data structure was pinned down, the strategy design followed the objective: maximize completed sets, which means every decision reduces to *"which ask gives the highest expected value toward finishing books, net of what it reveals."*

---

## 2. What is the problem?

Go Fish is a hidden-information, multi-agent game where a bot must choose both:

- **whom to ask**, and
- **what rank to ask for**

…every turn, under the strict legality rule that you can only ask for ranks you already hold.

A weak bot ignores everything except its own hand. A strong bot must:

- track history,
- infer hidden holdings across 2–4 opponents, and
- — most importantly in a bot-vs-bot setting — account for the fact that every ask leaks information the other bots will exploit.

The competition also requires clean handling of partial states and slightly dirty data.

---

## 3. How did we solve that?

We built a four-stage pipeline in one Python module:

1. **Parse and clean** the JSON — dropping invalid ranks, filling blank-name slots, handling two different file schemas.
2. **Build a full game-state model** of everything we know (our hand, deck size, completed sets, opponent hand sizes, turn history).
3. **Run an inference engine** that replays every historical turn and maintains per-opponent "knowledge":
   - `known[rank]` — lower-bound card counts,
   - `cannot_hold[rank]` with an age counter that decays as opponents draw,
   - `interest[rank]` — how often they've asked for it,
   - style flags (greedy, follower).
4. **Score every legal `(requestee, rank)` pair** with a utility function that rewards sets, triples, and pairs, penalizes information reveal weighted by turn-order distance, and adds a denial bonus for pulling cards from dangerous opponents.
5. **Tiebreak with a Monte Carlo sampler** that deals hidden cards consistent with constraints, simulates one turn of play, and blends that estimate with the analytical score.

---

## 4. How does the algorithm work?

On each turn the bot does four things:

### Step 1 — Replay the turn log

Every ask proves the asker held ≥1 of that rank. Every success moves a known count and zeroes the giver's count for that rank. Every failure zeros the requestee and flags a draw that ages prior constraints. Every completed set permanently removes all four copies from consideration.

### Step 2 — Compute hypergeometric probabilities

The "unseen pool" = opponents' uncommitted hand cards + deck.

`P(opp has rank) = 1 − P(zero of the remaining copies of that rank landed in their uncommitted slots)`

### Step 3 — Enumerate and score every legal ask

Only ranks currently in our hand, only live opponents:

```
score = P(success) × value_of_success
      + P(fail)    × (penalty + info_value)
      + denial_bonus
      − reveal_penalty
```

- `value_of_success` jumps hard at pair, triple, and especially at completing a set.
- `reveal_penalty` decays with turn-order distance (`0.70^i`), so leaking information right before a dangerous player costs more than leaking it right after.

### Step 4 — Monte Carlo re-rank

The top candidates are re-ranked by Monte Carlo: sample full hidden states, play one turn forward, average the reward. The best blended move is emitted as:

```json
{"Requestee": "...", "Request": "..."}
```

---

## 5. How did we troubleshoot the issue?

We built two independent test drivers and iterated against their output.

### `test_harness.py`

Runs the bot on every uploaded JSON and asserts legality:

- rank is in hand,
- requestee is a real live opponent,
- rank isn't already out of play.

This caught early data-cleaning bugs like the `"1"` rank and the game-review schema having hands inside `Players[]` instead of at the top level.

### `simulate_gofish.py`

A full 4-player Go Fish engine with baseline opponents (Random, Greedy, Tracker, Probabilistic). Running **1,000-game tournaments** against each gave reliable win-rate numbers and exposed a real weakness — a table of three Tracker bots dropped our win rate to ~37%.

### Diagnosis and sweep

We diagnosed it as the reveal-penalty coefficient being too small for a pure-tracker field, then swept that weight across five settings to confirm no simple single-knob fix existed (it stayed in the 29–38% range), which told us the fix needs a targeted stealth mode rather than a parameter tweak.

### Tunable design

Throughout, we kept the tuning constants exposed as class attributes so future fixes are a one-line change.

---

## Summary of measured performance

1,000 games per matchup (6,000 games total):

| Opponents                                | Win rate  | Avg sets (of 13) |
|------------------------------------------|-----------|------------------|
| 3 × Random                               | 83.6%     | 6.11             |
| 3 × Greedy                               | 80.5%     | 6.12             |
| 3 × Probabilistic                        | 69.3%     | 5.39             |
| 3 × Tracker                              | 36.7%     | 3.96             |
| Mixed (Greedy + Tracker + Random)        | 75.7%     | 5.65             |
| Strong mix (Greedy + Tracker + Prob.)    | 68.8%     | 5.42             |

**Overall average: ~69% win rate in a 4-player game** (vs. the 25% a random seat would earn).
