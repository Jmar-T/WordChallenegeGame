"""
evaluate.py

Tournament benchmarker — evaluates trained agents across word lengths.

Runs each saved model over a fixed set of 1,000 seeded games in fully
greedy mode (ε=0) and reports win rate and average turns to win.
Both random seeds are fixed so every model faces the exact same sequence
of secret words, making results directly comparable across runs.

Usage
-----
    python evaluate.py

To evaluate a subset of lengths, edit TARGET_LENGTHS below.
"""

import os
import random

import numpy as np
import torch

from dqn_agent import DQNAgent
from wordle_env import WordleEnv

# ======================================================================
# CONFIGURATION
# ======================================================================

DICTIONARY_PATH = "words.txt"
AGENTS_DIR      = "agents"
TARGET_LENGTHS  = [5, 6, 7, 8, 9, 10]
NUM_GAMES       = 1_000
RANDOM_SEED     = 42

# ======================================================================


def load_agent(model_path: str, dictionary: list[str], word_len: int) -> DQNAgent:
    """
    Loads a trained agent from a checkpoint file.

    Handles both the current checkpoint format (dict with 'model_state' key)
    and bare state_dicts from older torch.save() calls, so historical models
    can still be evaluated without conversion.
    """
    agent = DQNAgent(state_size=26 * word_len, dictionary=dictionary)

    checkpoint = torch.load(model_path, weights_only=True)
    state_dict = (
        checkpoint["model_state"]
        if isinstance(checkpoint, dict) and "model_state" in checkpoint
        else checkpoint
    )
    agent.model.load_state_dict(state_dict)
    agent.model.eval()
    agent.epsilon = 0.0  # fully greedy — no exploration
    return agent


def evaluate_model(model_path: str, word_len: int, num_games: int) -> tuple[float, float]:
    """
    Evaluates a single model over num_games seeded games.

    Parameters
    ----------
    model_path : str  — path to the .pth checkpoint file
    word_len   : int  — word length this model was trained on
    num_games  : int  — number of evaluation games to run

    Returns
    -------
    win_rate  : float — percentage of games won (0–100)
    avg_turns : float — average turns taken on winning games
    """
    env   = WordleEnv(dictionary_path=DICTIONARY_PATH, word_length=word_len)
    agent = load_agent(model_path, env.dictionary, word_len)

    # Fix both generators so env.reset() (random.choice) and numpy calls
    # produce the same sequence for every model
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    wins        = 0
    total_turns = 0

    for _ in range(num_games):
        state = env.reset()
        done  = False

        while not done:
            mask       = env.get_action_mask()
            action_idx = agent.act(state, action_mask=mask)
            state, _, done = env.step(action_idx)

        if env.dictionary[action_idx] == env.secret_word:
            wins        += 1
            total_turns += env.current_round

    win_rate  = (wins / num_games) * 100
    avg_turns = total_turns / wins if wins > 0 else float(WordleEnv.MAX_ROUNDS)
    return win_rate, avg_turns


def main() -> None:
    print("=" * 60)
    print("  Wordle DQN Agent — Evaluation Tournament")
    print(f"  {NUM_GAMES:,} games per model | seed={RANDOM_SEED}")
    print("=" * 60)

    results = []

    for word_len in TARGET_LENGTHS:
        model_path = os.path.join(AGENTS_DIR, f"wordle_dqn_len{word_len}.pth")

        if not os.path.exists(model_path):
            print(f"  len{word_len} → [skipped — {model_path} not found]")
            continue

        win_rate, avg_turns = evaluate_model(model_path, word_len, NUM_GAMES)
        results.append((word_len, win_rate, avg_turns))
        print(
            f"  {word_len}-letter words → "
            f"Win Rate: {win_rate:.1f}% | "
            f"Avg Turns to Win: {avg_turns:.2f}"
        )

    if results:
        print("\n  Summary")
        print("  " + "-" * 44)
        best = max(results, key=lambda r: r[1])
        print(f"  Best win rate : {best[0]}-letter words at {best[1]:.1f}%")
        avg_across = sum(r[1] for r in results) / len(results)
        print(f"  Avg win rate  : {avg_across:.1f}% across {len(results)} lengths")


if __name__ == "__main__":
    main()