"""
dqn_agent.py

Double DQN agent with a scoring network architecture and word embeddings.

Architecture overview
---------------------
Rather than a flat output head with one node per vocabulary word, the agent
uses a ScoringNetwork that takes a concatenated [board_state | word_embedding]
vector and outputs a single Q-value scalar. At decision time every valid
(masked) word is scored in one batched forward pass and the highest scorer
is selected.

This allows learned Q-value signal to generalise across structurally similar
words via their shared embedding features, rather than treating each word as
an isolated anonymous index.

Word embedding features (per word, 26 + 26*word_len + 1 + 1 dims)
------------------------------------------------------------------
  Letter presence   — binary 26-vec; 1 if letter appears anywhere in word.
  Letter positions  — binary (26 x word_len) matrix, flattened; encodes
                      which letter sits at which slot. Mirrors the state
                      representation so the network can directly compare
                      board hints against candidate word structure.
  Corpus frequency  — scalar; sum of per-letter frequencies computed from
                      the training dictionary. Rewards guesses that probe
                      common letters in this specific vocabulary.
  Uniqueness        — scalar; distinct_letters / word_len. Rewards words
                      that probe the maximum number of new positions.

Double DQN
----------
Two networks (main + target) are maintained. The main network selects the
best next action; the target network evaluates its value. The target network
is synced to the main network every N episodes, preventing overestimation
bias from the agent chasing its own inflated Q-value predictions.
"""

import pickle
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# ======================================================================
# WORD EMBEDDING BUILDER
# ======================================================================

def build_word_embeddings(dictionary: list[str]) -> tuple[np.ndarray, int]:
    """
    Computes a fixed (vocab_size, embed_dim) feature matrix at startup.
    Embeddings are static — they are never updated during training.

    Parameters
    ----------
    dictionary : list[str]
        Sorted vocabulary list. Index i corresponds to word i.

    Returns
    -------
    embeddings : np.ndarray, shape (vocab_size, embed_dim)
    embed_dim  : int
    """
    vocab_size = len(dictionary)
    word_len   = len(dictionary[0])
    embed_dim  = 26 + (26 * word_len) + 2  # presence + position + freq + uniqueness

    # Corpus letter frequencies — derived from this specific dictionary
    letter_counts = np.zeros(26, dtype=np.float32)
    for word in dictionary:
        for ch in word:
            letter_counts[ord(ch) - ord('a')] += 1
    letter_freq = letter_counts / letter_counts.sum()

    freq_offset = 26 + 26 * word_len   # index of scalar frequency feature
    embeddings  = np.zeros((vocab_size, embed_dim), dtype=np.float32)

    for i, word in enumerate(dictionary):
        vec        = np.zeros(embed_dim, dtype=np.float32)
        freq_score = 0.0

        for j, ch in enumerate(word):
            r = ord(ch) - ord('a')
            vec[r]                     = 1.0   # letter presence
            vec[26 + r * word_len + j] = 1.0   # letter at position
            freq_score                += letter_freq[r]

        vec[freq_offset]     = freq_score
        vec[freq_offset + 1] = len(set(word)) / word_len  # uniqueness

        embeddings[i] = vec

    return embeddings, embed_dim


# ======================================================================
# OPENER SELECTOR
# ======================================================================

def get_best_openers(
    dictionary: list[str],
    word_embeddings: torch.Tensor,
    top_n: int = 20,
) -> list[tuple[str, float]]:
    """
    Ranks all words by opener quality using only static embedding features.
    No trained network weights are required — works before or after training.

    At turn 1 the board state is all zeros, so opener quality is determined
    entirely by how much information a word is likely to reveal:

      corpus_frequency (weight 0.6) — probes common letters, maximising the
          chance of hitting greens and yellows on the first guess.
      uniqueness (weight 0.4)       — avoids repeated letters, probing the
          maximum number of distinct positions.

    The top_n words are sampled from randomly during training to expose the
    agent to a variety of strong openers rather than a single fixed word.

    Parameters
    ----------
    dictionary     : list[str]  — sorted vocabulary
    word_embeddings: torch.Tensor, shape (vocab_size, embed_dim)
    top_n          : int        — pool size to return

    Returns
    -------
    list of (word, score) tuples, sorted descending by score
    """
    word_len    = len(dictionary[0])
    freq_offset = 26 + 26 * word_len

    freq_scores   = word_embeddings[:, freq_offset].numpy()
    unique_scores = word_embeddings[:, freq_offset + 1].numpy()

    # Normalise each feature to [0, 1] before combining
    def normalise(arr):
        lo, hi = arr.min(), arr.max()
        return (arr - lo) / (hi - lo + 1e-8)

    combined    = 0.6 * normalise(freq_scores) + 0.4 * normalise(unique_scores)
    top_indices = np.argsort(combined)[::-1][:top_n]

    return [(dictionary[i], float(combined[i])) for i in top_indices]


# ======================================================================
# SCORING NETWORK
# ======================================================================

class ScoringNetwork(nn.Module):
    """
    Maps a concatenated [board_state | word_embedding] vector to a scalar
    Q-value for that (state, word) pair.

    The shared weight structure means signal from any training example
    updates the scoring function for all structurally similar words,
    providing far denser gradient signal than a flat per-word output head.

    Input  : state_size + embed_dim  (e.g. 130 + 158 = 288 for 5-letter)
    Hidden : 256 → 256 → 128
    Output : 1 scalar Q-value
    """

    def __init__(self, state_size: int, embed_dim: int):
        super().__init__()
        input_dim = state_size + embed_dim
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256,       256)
        self.fc3 = nn.Linear(256,       128)
        self.out = nn.Linear(128,         1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.out(x).squeeze(-1)  # (batch,)


# ======================================================================
# DQN AGENT
# ======================================================================

class DQNAgent:
    """
    Double DQN agent for Wordle.

    Parameters
    ----------
    state_size  : int        — flattened board state dimension (26 * word_len)
    dictionary  : list[str]  — sorted vocabulary from WordleEnv
    buffer_size : int        — replay buffer capacity (default 40 000)
    """

    def __init__(
        self,
        state_size: int,
        dictionary: list[str],
        buffer_size: int = 40_000,
    ):
        self.state_size  = state_size
        self.action_size = len(dictionary)
        self.memory      = deque(maxlen=buffer_size)

        # Hyperparameters
        self.gamma         = 0.95
        self.epsilon       = 1.0
        self.epsilon_decay = 0.9997
        self.epsilon_min   = 0.05
        self.learning_rate = 0.0005

        # Build static word embedding matrix
        embeddings_np, self.embed_dim = build_word_embeddings(dictionary)
        self.word_embeddings = torch.from_numpy(embeddings_np).float()

        # Double DQN — main network selects actions, target network evaluates them
        self.model        = ScoringNetwork(state_size, self.embed_dim)
        self.target_model = ScoringNetwork(state_size, self.embed_dim)
        self.update_target_network()

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    # ------------------------------------------------------------------
    # CHECKPOINT
    # ------------------------------------------------------------------

    def save_checkpoint(self, weights_path: str, buffer_path: str) -> None:
        """
        Saves a full training checkpoint to two files.

        weights_path : .pth file — model weights, target weights,
                                   optimizer state, and current epsilon.
        buffer_path  : .pkl file — full replay buffer (delete after training;
                                   not needed for inference).

        Saving the optimizer state preserves Adam's moment accumulators,
        avoiding a re-stabilisation period at the start of a warm run.
        Saving the target network separately preserves the stabilising lag
        between main and target rather than collapsing it to zero on reload.
        """
        torch.save({
            "model_state":        self.model.state_dict(),
            "target_model_state": self.target_model.state_dict(),
            "optimizer_state":    self.optimizer.state_dict(),
            "epsilon":            self.epsilon,
        }, weights_path)

        with open(buffer_path, "wb") as f:
            pickle.dump(self.memory, f)

        print(
            f"Checkpoint saved → weights: '{weights_path}' | "
            f"buffer: '{buffer_path}' ({len(self.memory):,} transitions)"
        )

    def load_checkpoint(self, weights_path: str, buffer_path: str) -> None:
        """
        Restores a full training checkpoint from the files written by
        save_checkpoint(). Target network lag and optimizer state are both
        preserved exactly as they were at the end of the previous run.
        """
        checkpoint = torch.load(weights_path, weights_only=True)
        self.model.load_state_dict(checkpoint["model_state"])
        self.target_model.load_state_dict(checkpoint["target_model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.epsilon = checkpoint["epsilon"]

        with open(buffer_path, "rb") as f:
            self.memory = pickle.load(f)

        print(
            f"Checkpoint loaded ← '{weights_path}' | "
            f"{len(self.memory):,} transitions | ε={self.epsilon:.3f}"
        )

    # ------------------------------------------------------------------
    # CORE AGENT METHODS
    # ------------------------------------------------------------------

    def update_target_network(self) -> None:
        """Hard-copies main network weights to the target network."""
        self.target_model.load_state_dict(self.model.state_dict())

    def act(self, state: np.ndarray, action_mask: list[bool] = None) -> int:
        """
        Selects an action using an epsilon-greedy policy.

        During exploration (random) the agent picks uniformly from valid words.
        During exploitation (greedy) all valid words are scored in a single
        batched forward pass and the highest scorer is returned.

        Parameters
        ----------
        state       : np.ndarray — current board state, shape (26, word_len)
        action_mask : list[bool] — True for each valid word index; if None,
                                   all words are considered valid.

        Returns
        -------
        int — index into the dictionary
        """
        valid_indices = (
            [i for i, v in enumerate(action_mask) if v]
            if action_mask is not None
            else list(range(self.action_size))
        ) or list(range(self.action_size))

        if np.random.rand() <= self.epsilon:
            return random.choice(valid_indices)

        self.model.eval()
        state_tensor = torch.from_numpy(state.flatten()).float()
        with torch.no_grad():
            scores = self._score_words(self.model, state_tensor, valid_indices)
        return valid_indices[torch.argmax(scores).item()]

    def remember(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        next_valid_indices: list[int],
    ) -> None:
        """Stores a transition in the replay buffer."""
        self.memory.append((state, action, reward, next_state, done, next_valid_indices))

    def replay(self, batch_size: int) -> None:
        """
        Samples a mini-batch from the replay buffer and performs one
        vectorised Double DQN update.

        Double DQN Bellman target
        -------------------------
        1. Main model selects the best next action (argmax over valid words).
        2. Target model evaluates that action's value.
        3. Target = reward + γ * target_Q(next_state, best_action)

        Step 1 uses the main model to decouple action selection from
        evaluation, which prevents the overestimation bias that arises
        when the same network both selects and scores the next action.
        """
        if len(self.memory) < batch_size:
            return

        batch  = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones, next_valids = zip(*batch)

        rewards_t = torch.tensor(rewards, dtype=torch.float)
        dones_t   = torch.tensor(dones,   dtype=torch.float)

        self.model.train()

        # Current Q for each taken action — one batched forward pass
        states_t      = torch.from_numpy(np.array([s.flatten() for s in states])).float()
        action_embeds = self.word_embeddings[torch.tensor(actions, dtype=torch.long)]
        current_q     = self.model(torch.cat([states_t, action_embeds], dim=1))

        # Double DQN targets — per sample (valid word sets differ each turn)
        targets = []
        with torch.no_grad():
            for i in range(batch_size):
                if dones[i]:
                    targets.append(rewards_t[i].item())
                    continue

                ns  = torch.from_numpy(next_states[i].flatten()).float()
                nv  = next_valids[i] or list(range(self.action_size))

                best_action = nv[torch.argmax(self._score_words(self.model, ns, nv)).item()]
                target_q    = self._score_words(self.target_model, ns, [best_action])[0].item()
                targets.append(rewards_t[i].item() + self.gamma * target_q)

        loss = self.criterion(current_q, torch.tensor(targets, dtype=torch.float))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    # ------------------------------------------------------------------
    # INTERNAL HELPERS
    # ------------------------------------------------------------------

    def _score_words(
        self,
        model: ScoringNetwork,
        state_flat: torch.Tensor,
        word_indices: list[int],
    ) -> torch.Tensor:
        """
        Scores a set of candidate words in a single batched forward pass.

        Builds input by tiling the state vector alongside each word's
        embedding, then passes the combined matrix through the network.

        Parameters
        ----------
        model       : ScoringNetwork to use for scoring
        state_flat  : 1-D tensor of shape (state_size,)
        word_indices: list of vocabulary indices to score

        Returns
        -------
        torch.Tensor of shape (len(word_indices),) — one Q-value per word
        """
        n           = len(word_indices)
        state_tiled = state_flat.unsqueeze(0).expand(n, -1)
        embeds      = self.word_embeddings[torch.tensor(word_indices, dtype=torch.long)]
        return model(torch.cat([state_tiled, embeds], dim=1))