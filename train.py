"""
train.py

Single-length training script for the Wordle DQN agent.

Trains a scoring-network Double DQN agent on words of a fixed length.
Supports warm starts from a previous checkpoint so training can be
resumed or extended without relearning the random exploration phase.

Usage
-----
Configure the constants in the CONFIGURATION block below, then run:

    python train.py

For batch training across lengths 5–10 use train_all.py instead.

Warm start workflow
-------------------
Round 1  WARM_START = False  →  saves wordle_dqn_len5.pth + _buffer.pkl
Round 2  WARM_START = True   →  loads round 1 checkpoint, continues training
         SAVE_WEIGHTS / SAVE_BUFFER should use a new filename each round
         to avoid overwriting the previous checkpoint.
"""

import os
import random

import numpy as np
import torch

from dqn_agent import DQNAgent, get_best_openers
from wordle_env import WordleEnv

# ======================================================================
# CONFIGURATION
# ======================================================================

WORD_LENGTH = 8

# Opener strategy
# USE_FORCED_OPENER = True  : sample from the top-N embedding-scored openers
#                             each episode. Length-agnostic — works for any
#                             word length without maintaining a hardcoded list.
# USE_FORCED_OPENER = False : agent selects its own opener freely.
USE_FORCED_OPENER = True
OPENER_POOL_SIZE  = 20

# Warm start — set True to resume from an existing checkpoint
WARM_START         = True
CHECKPOINT_WEIGHTS = f"wordle_dqn_len{WORD_LENGTH}.pth"
CHECKPOINT_BUFFER  = f"wordle_dqn_len{WORD_LENGTH}_buffer.pkl"

# Output paths for this run
SAVE_WEIGHTS = f"wordle_dqn_len{WORD_LENGTH}.pth"
SAVE_BUFFER  = f"wordle_dqn_len{WORD_LENGTH}_buffer.pkl"

EPISODES   = 5_500
BATCH_SIZE = 32

# ======================================================================


def run_training(
    env: WordleEnv,
    agent: DQNAgent,
    episodes: int,
    opener_words: list[str],
    batch_size: int,
) -> None:
    """Runs the main training loop for a fixed number of episodes."""
    rolling_wins   = 0
    rolling_reward = 0.0

    for e in range(1, episodes + 1):
        state = env.reset()
        done  = False

        while not done:
            mask = env.get_action_mask()

            if USE_FORCED_OPENER and env.current_round == 0:
                start_word = random.choice(opener_words)
                action_idx = env.dictionary.index(start_word)
            else:
                action_idx = agent.act(state, action_mask=mask)

            next_state, reward, done = env.step(action_idx)

            next_valid = (
                [i for i, v in enumerate(env.get_action_mask()) if v]
                if not done else []
            )

            agent.remember(state, action_idx, reward, next_state, done, next_valid)
            state = next_state

        if env.dictionary[action_idx] == env.secret_word:
            rolling_wins += 1
        rolling_reward += reward

        agent.replay(batch_size)

        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay

        if e % 100 == 0:
            agent.update_target_network()

        if e % 100 == 0:
            print(
                f"  Episode {e:>5}/{episodes} | "
                f"Win Rate: {rolling_wins:.0f}% | "
                f"Avg Reward: {rolling_reward / 100:>7.1f} | "
                f"ε: {agent.epsilon:.3f}"
            )
            rolling_wins   = 0
            rolling_reward = 0.0


def main() -> None:
    env        = WordleEnv(dictionary_path="words.txt", word_length=WORD_LENGTH)
    state_size = 26 * WORD_LENGTH

    agent = DQNAgent(state_size=state_size, dictionary=env.dictionary)

    if USE_FORCED_OPENER:
        opener_pool  = get_best_openers(env.dictionary, agent.word_embeddings, top_n=OPENER_POOL_SIZE)
        opener_words = [w for w, _ in opener_pool]
        print(f"Top openers: {', '.join(opener_words[:5])} ...")
    else:
        opener_words = []

    if WARM_START:
        for path in [CHECKPOINT_WEIGHTS, CHECKPOINT_BUFFER]:
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Warm start enabled but checkpoint not found: '{path}'"
                )
        agent.load_checkpoint(CHECKPOINT_WEIGHTS, CHECKPOINT_BUFFER)
    else:
        print(f"Fresh run | ε=1.0 | {len(env.dictionary):,} words | {EPISODES:,} episodes\n")

    run_training(env, agent, EPISODES, opener_words, BATCH_SIZE)
    agent.save_checkpoint(SAVE_WEIGHTS, SAVE_BUFFER)
    print("Training complete.")


if __name__ == "__main__":
    main()