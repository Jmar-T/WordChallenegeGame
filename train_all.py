"""
train_all.py

Batch trainer — trains one DQN agent per word length for lengths 5 through 10.

Each length runs two rounds of 5,500 episodes with a warm start between
rounds, equivalent to 11,000 total episodes of continuous training. The
replay buffer is preserved between rounds so the warm start inherits the
full diversity of the previous run's experience rather than rebuilding
from an empty buffer at near-greedy epsilon.

Output
------
One .pth file per length written to the current directory:
    wordle_dqn_len5.pth  through  wordle_dqn_len10.pth

Intermediate .pkl buffer files are deleted automatically after each length
completes — they are not needed for inference or evaluation.

Hyperparameter scaling
----------------------
Buffer size and epsilon decay are scaled with word length to account for:
  - longer words generating more transitions per game (more turns allowed)
  - larger vocabularies requiring more exploration before exploitation

Usage
-----
    python train_all.py
"""

import os
import random
import time

import torch

from dqn_agent import DQNAgent, get_best_openers
from wordle_env import WordleEnv

# ======================================================================
# CONFIGURATION
# ======================================================================

TARGET_LENGTHS   = [6, 7, 8, 9, 10]
EPISODES_ROUND_1 = 5_500
EPISODES_ROUND_2 = 5_500
BATCH_SIZE       = 32
OPENER_POOL_SIZE = 20


def buffer_size(word_len: int) -> int:
    """Scale replay buffer with word length. +8k per extra letter above 5."""
    return 40_000 + (word_len - 5) * 8_000


def epsilon_decay(word_len: int) -> float:
    """Slightly slower decay for larger vocabularies. +0.00003 per extra letter."""
    return 0.9997 + (word_len - 5) * 0.00003


def weights_path(word_len: int) -> str:
    return f"wordle_dqn_len{word_len}.pth"


def buffer_path(word_len: int) -> str:
    return f"wordle_dqn_len{word_len}_buffer.pkl"


# ======================================================================


def train_one_round(
    env: WordleEnv,
    agent: DQNAgent,
    episodes: int,
    opener_words: list[str],
    word_len: int,
    round_num: int,
) -> None:
    """Runs one training round, printing rolling 100-episode metrics."""
    rolling_wins   = 0
    rolling_reward = 0.0

    for e in range(1, episodes + 1):
        state = env.reset()
        done  = False

        while not done:
            mask       = env.get_action_mask()
            action_idx = (
                env.dictionary.index(random.choice(opener_words))
                if env.current_round == 0
                else agent.act(state, action_mask=mask)
            )

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

        agent.replay(BATCH_SIZE)

        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay

        if e % 100 == 0:
            agent.update_target_network()

        if e % 100 == 0:
            print(
                f"  [len{word_len} R{round_num}] Ep {e:>5}/{episodes} | "
                f"Win: {rolling_wins:>3}% | "
                f"Avg Reward: {rolling_reward / 100:>7.1f} | "
                f"ε: {agent.epsilon:.3f}"
            )
            rolling_wins   = 0
            rolling_reward = 0.0


def train_length(word_len: int) -> None:
    """Trains a full two-round agent for a single word length."""
    print(f"\n{'=' * 60}")
    print(f"  Word length : {word_len}")
    print(f"{'=' * 60}")

    env   = WordleEnv(dictionary_path="words.txt", word_length=word_len)
    buf   = buffer_size(word_len)
    decay = epsilon_decay(word_len)

    agent               = DQNAgent(state_size=26 * word_len, dictionary=env.dictionary, buffer_size=buf)
    agent.epsilon_decay = decay

    opener_pool  = get_best_openers(env.dictionary, agent.word_embeddings, top_n=OPENER_POOL_SIZE)
    opener_words = [w for w, _ in opener_pool]

    print(f"  Vocab: {len(env.dictionary):,} | Buffer: {buf:,} | Decay: {decay}")
    print(f"  Top openers: {', '.join(opener_words[:5])} ...\n")

    # Round 1 — fresh training
    print(f"  Round 1 — {EPISODES_ROUND_1:,} episodes")
    t0 = time.time()
    train_one_round(env, agent, EPISODES_ROUND_1, opener_words, word_len, round_num=1)
    agent.save_checkpoint(weights_path(word_len), buffer_path(word_len))
    print(f"  Round 1 done in {(time.time() - t0) / 60:.1f} min | ε={agent.epsilon:.3f}\n")

    # Round 2 — warm start from round 1
    print(f"  Round 2 — {EPISODES_ROUND_2:,} episodes (warm start)")
    agent.load_checkpoint(weights_path(word_len), buffer_path(word_len))
    t0 = time.time()
    train_one_round(env, agent, EPISODES_ROUND_2, opener_words, word_len, round_num=2)
    agent.save_checkpoint(weights_path(word_len), buffer_path(word_len))
    print(f"  Round 2 done in {(time.time() - t0) / 60:.1f} min | ε={agent.epsilon:.3f}")

    # Remove buffer — only the .pth is needed after training
    if os.path.exists(buffer_path(word_len)):
        os.remove(buffer_path(word_len))
        print(f"  Buffer deleted: {buffer_path(word_len)}")

    print(f"  ✅ len{word_len} complete → {weights_path(word_len)}")


def main() -> None:
    total_start = time.time()

    for wlen in TARGET_LENGTHS:
        train_length(wlen)

    elapsed = (time.time() - total_start) / 60
    print(f"\n{'=' * 60}")
    print(f"All lengths complete in {elapsed:.1f} minutes.")
    print(f"Models: {[weights_path(l) for l in TARGET_LENGTHS]}")


if __name__ == "__main__":
    main()