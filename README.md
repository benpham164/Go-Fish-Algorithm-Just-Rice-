# Go Fish Bot — README

A competition-ready Go Fish bot written in Python. It reads the organizer's game-state JSON, thinks about the best move, and prints one line in the required format:

```
{"Requestee": "player name", "Request": "rank"}
```

## What's in this folder

| File | What it does |
|------|--------------|
| `gofish_bot.py` | The actual bot. This is the file the competition runs. |
| `test_harness.py` | Runs the bot against every JSON test file and checks that each move is valid. |
| `simulate_gofish.py` | A full 4-player Go Fish game engine that plays our bot against simpler bots so we can measure how strong it is. |
| `README.md` | This file. |

## How to run it

Play one move from a game-state file:

```bash
python gofish_bot.py -f mid-game.json -n steven
```

Or feed it through stdin (how the competition will call it):

```bash
cat state.json | python gofish_bot.py --name MyBotName
```

Check that the bot handles all the provided test files correctly:

```bash
python test_harness.py --name steven
```

Play a full tournament against other bots:

```bash
python simulate_gofish.py --games 200
```

## How the bot thinks (in plain English)

Go Fish is a game of memory and guessing. The bot does the same three things a very focused human player would do, just faster.

### 1. It reads the room

Every turn in the history tells the bot something. If Alice asks Bob for "7s", Alice must have at least one 7. If Bob hands over three 7s, Alice now has four 7s on her way to a book — and Bob has none. The bot replays every past turn and keeps a running mental note of what each player definitely has, definitely does not have, and what they've seemed interested in.

### 2. It does the math

For every legal move (every combination of "who to ask" and "which rank in my hand to ask for"), the bot estimates:

- **How likely the ask succeeds** — using probability based on what's still unknown.
- **How good success would be** — completing a set of 4 is worth a lot; making a triple is worth some; just picking up one card is worth a little.
- **How bad failure would be** — you lose your turn and draw a random card.
- **What asking reveals** — by asking for "7s" you just told the whole table you have some. The bot subtracts points for that, more if a dangerous opponent is about to play right after you.
- **Whether it denies a rival** — pulling cards away from a player who is close to completing a set is a bonus.

The move with the highest score wins.

### 3. It double-checks with simulation

For the top few candidate moves, the bot imagines many possible ways the hidden cards could be distributed (Monte Carlo sampling), plays the next turn out in each one, and averages the results. This catches situations where the math-only score picks a move that happens to go badly in most realistic worlds.

## How it handles different situations

- **2, 3, or 4 players**: The bot reads the player list from the file instead of assuming a count. Everything scales automatically.
- **Partly played games**: The bot fast-forwards through the turn history so starting mid-game works the same as starting at turn 1.
- **Messy data**: Invalid ranks like "1" get dropped. Empty name slots in the test file get filled in with your bot's name. The two different file formats (live game vs. post-game review) are both supported.
- **Different opponent styles**: The bot watches how each opponent plays. If somebody keeps asking for the same rank, it flags them as "greedy" and adapts. If somebody follows up on other players' failed asks, it notices that too.

## How strong is it?

We ran a 4-player tournament with the bot against three families of baseline opponents:

- **Random** — picks any legal move.
- **Greedy** — always asks for whatever it has the most of, from the player with the biggest hand.
- **Tracker** — watches history and targets players who signaled interest in a rank.

Results over many games:

| Opponents | Our bot's win rate |
|-----------|--------------------|
| 3 Random  | ~80% |
| 3 Greedy  | ~82% |
| Mixed (Greedy + Tracker + Random) | ~74% |

When starting from the actual partially-played states in the uploaded test files, the bot wins **70–88%** of games depending on the state.

The one scenario where it's weaker is a table full of three Tracker bots, because every reveal gets exploited three ways. This is a known tuning target — see "Tuning" below.

## Tuning

If you want to make the bot stronger for a specific competition, the easiest dials are at the top of the `MoveEvaluator` class in `gofish_bot.py`:

- `W_COMPLETE_SET` — how much finishing a book is worth
- `W_FORM_TRIPLE`, `W_FORM_PAIR` — smaller wins
- `W_REVEAL_PENALTY` — how much you care about tipping your hand
- `W_DENIAL_PER_DANGER` — how much you care about pulling cards from leaders

Bump them up or down and re-run `simulate_gofish.py` to see the effect.

## Output format

The bot always prints exactly one line of JSON to standard output:

```
{"Requestee": "haris", "Request": "2"}
```

No extra logging, no trailing text — safe to pipe directly into the competition harness.
