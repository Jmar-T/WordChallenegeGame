"""
test_agent.py

Tests a trained DQN agent against a specific word chosen by the user.

The agent runs in fully greedy mode (ε=0) — every guess is the highest
Q-value word among those still valid under the current board constraints.
The game trace is printed turn by turn so the agent's reasoning can be
inspected visually.

Limitations
-----------
The agent can only solve words that exist in its training dictionary.
Structurally, the scoring network learns Q-values for specific vocabulary
entries — a word that was never a training target has a near-random Q-value
and will not be reliably selected. See README.md for details.

Usage
-----
    python test_agent.py

Or call test_agent() directly from another script:

    from test_agent import test_agent
    test_agent("crane")
"""

import os

import numpy as np
import torch

from dqn_agent import DQNAgent
from wordle_env import WordleEnv

# ======================================================================
# CONFIGURATION
# ======================================================================

DICTIONARY_PATH = "words.txt"
AGENTS_DIR      = "agents"
MIN_WORD_LENGTH = 5
MAX_WORD_LENGTH = 10

# ======================================================================


def _load_agent(word_length: int, dictionary: list[str]) -> DQNAgent:
    """
    Loads the trained model for the given word length.

    Raises FileNotFoundError if no checkpoint exists for this length.
    """
    model_path = os.path.join(AGENTS_DIR, f"wordle_dqn_len{word_length}.pth")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No trained model found for {word_length}-letter words.\n"
            f"Expected: '{model_path}'\n"
            f"Run train_all.py to train models for all supported lengths."
        )

    agent = DQNAgent(state_size=26 * word_length, dictionary=dictionary)
    checkpoint = torch.load(model_path, weights_only=True)
    state_dict = (
        checkpoint["model_state"]
        if isinstance(checkpoint, dict) and "model_state" in checkpoint
        else checkpoint
    )
    agent.model.load_state_dict(state_dict)
    agent.model.eval()
    agent.epsilon = 0.00  # fully greedy
    return agent


def test_agent(word: str) -> None:
    """
    Runs the trained agent against a specific secret word and prints
    the full game trace.

    Parameters
    ----------
    word : str — the secret word to solve (must be in the training dictionary)
    """
    word = word.strip().lower()

    # Input validation
    if not word.isalpha():
        print(f"Invalid input: '{word}' contains non-alphabetic characters.")
        return

    word_length = len(word)
    if not (MIN_WORD_LENGTH <= word_length <= MAX_WORD_LENGTH):
        print(
            f"'{word}' is {word_length} letters. "
            f"Supported lengths: {MIN_WORD_LENGTH}–{MAX_WORD_LENGTH}."
        )
        return

    env = WordleEnv(dictionary_path=DICTIONARY_PATH, word_length=word_length)

    if word not in env.dictionary:
        print(f"'{word}' is not in the {word_length}-letter dictionary.")
        print("The agent can only solve words it was trained on.")
        print("Check that the word appears in words.txt.")
        return

    try:
        agent = _load_agent(word_length, env.dictionary)
    except FileNotFoundError as e:
        print(e)
        return

    # Run the game
    state         = env.reset(word=word)
    done          = False
    words_guessed = []

    print(f"\nSecret word : {'*' * word_length}  ({word_length} letters)")
    print(f"Max guesses : {WordleEnv.MAX_ROUNDS}")
    print("-" * 30)

    while not done:
        mask       = env.get_action_mask()
        action_idx = agent.act(state, action_mask=mask)
        state, _, done = env.step(action_idx)
        words_guessed.append(env.dictionary[action_idx])

        guess   = words_guessed[-1]
        correct = guess == env.secret_word
        marker  = "🟩 SOLVED" if correct else ""
        print(f"  {len(words_guessed)}. {guess}  {marker}")

    won = words_guessed[-1] == env.secret_word

    print("-" * 30)
    if won:
        print(f"✅ Solved '{env.secret_word}' in {len(words_guessed)} guess{'es' if len(words_guessed) != 1 else ''}.")
    else:
        print(f"❌ Failed to solve '{env.secret_word}' in {WordleEnv.MAX_ROUNDS} guesses.")


def main() -> None:
    word = input(
        f"Enter a word for the agent to solve "
        f"({MIN_WORD_LENGTH}–{MAX_WORD_LENGTH} letters): "
    ).strip()
    test_agent(word)


if __name__ == "__main__":
    main()