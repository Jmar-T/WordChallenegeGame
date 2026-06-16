# Wordle DQN Agent

A Deep Q-Network (DQN) agent trained to solve Wordle across word lengths 5–10 using reinforcement learning.

The agent learns purely from gameplay experience — no hardcoded word strategies, no pre-labelled data. It discovers which words to guess by maximising cumulative reward across thousands of training games.

---

## Project Structure

```
wordle-dqn-agent/
│
├── agents/                  # Trained model weights (.pth), one per word length
│   ├── wordle_dqn_len5.pth
│   ├── wordle_dqn_len6.pth
│   └── ...
│
├── original/                # Early-stage scripts from the initial prototype
│                            # Kept for contrast — shows how the project evolved
│
├── words.txt                # Word list (~175k words, lengths 2–28)
│
├── wordle_env.py            # Wordle game environment (state, rewards, masking)
├── dqn_agent.py             # Network architecture, embeddings, training logic
│
├── train.py                 # Train a single word length
├── train_all.py             # Batch train lengths 5–10
├── evaluate.py              # Benchmark trained agents
│
├── play.py                  # Play Wordle yourself in the terminal
└── test_agent.py            # Watch the agent solve a specific word
```

---
## Requirements

```
python >= 3.10
torch
numpy
```

Install dependencies:
```bash
pip install -r requirements.txt
```

## Quickstart

**Watch the agent solve a word:**
```bash
python test_agent.py
# Enter a word to test (5–10 letters): crane
```

**Play Wordle yourself:**
```bash
python play.py
```

**Benchmark all trained models:**
```bash
python evaluate.py
```

---

## How It Works

### Environment (`wordle_env.py`)

The game state is represented as a `(26 × word_length)` matrix — one row per letter of the alphabet, one column per position. Each cell holds a hint value:

| Value | Meaning |
|-------|---------|
| `0`   | Unknown |
| `1`   | Yellow — letter present, wrong position |
| `2`   | Green — letter correct, correct position |
| `-1`  | Grey — letter eliminated |

Hint assignment uses a two-pass algorithm (greens first, then yellows/greys) to correctly handle words with repeated letters.

The action mask enforces Hard Mode rules — every guess must be consistent with all previously revealed hints, and previously guessed words are excluded.

### Agent (`dqn_agent.py`)

**Scoring Network architecture**

Rather than a flat output layer with one node per vocabulary word, the agent uses a scoring network that takes a concatenated `[board_state | word_embedding]` vector and outputs a single Q-value:

```
[state (26×wlen) | word_embedding (158+ dims)] → FC(256) → FC(256) → FC(128) → Q-value
```

At decision time all valid (masked) words are scored in one batched forward pass and the highest scorer is selected. This allows the learned Q-value signal to generalise across structurally similar words rather than treating each vocabulary entry as an isolated index.

**Word embeddings**

Each word is represented by a fixed 158-dimensional feature vector (for 5-letter words; scales with length) computed once at startup from the training dictionary:

| Feature | Dims | Description |
|---------|------|-------------|
| Letter presence | 26 | Binary — does this letter appear in the word? |
| Letter positions | 26 × word_len | Binary — which letter is at which slot? |
| Corpus frequency | 1 | Sum of per-letter frequencies across the full dictionary |
| Uniqueness | 1 | Distinct letters / word_length |

**Double DQN**

Two networks are maintained — the main network selects actions, a time-delayed target network evaluates their value. The target is synced every 100 episodes. This prevents the overestimation bias that arises when a single network both selects and scores future actions.

**Opener selection**

The top-20 opening words are selected automatically from the dictionary at training time using corpus frequency and uniqueness scores, with no hardcoded word lists. This means opener selection is fully length-agnostic — a 9-letter training run finds the best 9-letter openers from that vocabulary automatically.

---

## Training

**Single length:**
```bash
# Edit WORD_LENGTH in train.py, then:
python train.py
```

**All lengths 5–10 (recommended):**
```bash
python train_all.py
```

Each length runs two rounds of 5,500 episodes with a warm start between rounds. The warm start preserves the full replay buffer so round 2 inherits 40,000+ diverse transitions rather than rebuilding from empty.

Training produces two files per length — a `.pth` weights file and a `.pkl` buffer file. The buffer is deleted automatically after training completes; only the `.pth` is needed for inference.

Move the `.pth` files into the `agents/` directory before running evaluation or test scripts.

---

## Results

Evaluated over 1,000 seeded games per length in fully greedy mode (ε=0):

| Word Length | Vocabulary Size | Win Rate | Avg Turns to Win |
|-------------|----------------|----------|-----------------|
| 5 letters   | 8,904          | 75.9%    | 4.57            |
| 6 letters   | 15,232         | 75.0%    | 4.20            |
| 7 letters   | 23,109         | 75.2%    | 3.95            |
| 8 letters   | 28,420         | 73.9%    | 3.45            |
| 9 letters   | 24,873         | 70.9%    | 3.48            |
| 10 letters  | 20,300         | 64.5%    | 2.96            |

The 5-letter model improved from a 38.6% baseline to 75.0% through a series of architectural and training improvements described below.

---

## Development Journey

The project went through several major iterations, each with a measurable impact on performance:

| Change | Win Rate | Key Insight |
|--------|----------|-------------|
| 128-node baseline | 38.6% | Starting point |
| 256-node network | 43.1% | More capacity needed for large action space |
| Vectorised replay | 43.7% | Stable gradients, ~15× faster training |
| Scoring network + word embeddings | 66.0% | Structural jump — signal now generalises across similar words |
| Warm start with buffer preservation | 73.8% | Inheriting diverse replay buffer eliminated re-learning cost |
| Full convergence (ε → 0.05) | 75.0% | Stable exploitation of mature weights |

The biggest single improvement was switching from a flat output head to the scoring network architecture — a 22-point jump. The second largest was fixing the warm start to preserve the replay buffer between training rounds, which recovered 7 points that had been lost to buffer-rebuild overhead.

---

## Limitations

**Words not in the training dictionary cannot be solved reliably.** The scoring network learns Q-values for specific vocabulary entries. A word absent from training has a near-random Q-value and will not be reliably selected by the greedy policy, even if its letters would otherwise point toward it.

**The action mask does most of the early-game work.** During high-epsilon training, the agent wins ~60% of games purely from random valid guesses. The network's contribution is primarily in late-game disambiguation when the valid word set has narrowed significantly.

**Win rate declines with vocabulary size.** Larger vocabularies (lengths 7–10) present a harder credit assignment problem — each word is visited proportionally less often during training, so Q-value estimates are noisier.

---

## Future Improvements

- **Letter-targeting output head** — replace per-word Q-values with a learned letter-desirability vector. The agent would output which letters to probe at which positions, and a separate scorer would rank valid words by how well they match. This would allow genuine generalisation to unseen words.
- **Entropy-based endgame** — switch from Q-value selection to information-theoretic scoring when the valid word set narrows below a threshold (e.g. 10 words remaining). This is how the strongest Wordle solvers handle disambiguation.
- **Positional letter frequency embeddings** — extend corpus frequency to track how often each letter appears at each specific position, rather than globally. Richer features would push the performance ceiling higher.

---

