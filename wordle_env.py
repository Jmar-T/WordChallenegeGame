"""
wordle_env.py

Custom Wordle simulation environment for reinforcement learning.

The environment manages game state as a (26 x word_length) matrix where each
cell tracks the hint status of a letter at a position:
    0  = unknown
    1  = yellow (letter exists but wrong position)
    2  = green  (letter correct and in correct position)
   -1  = grey   (letter eliminated at this position)

Constraints are enforced in Hard Mode: every guess must be consistent with
all previously revealed hints.
"""

import random
import sys
import numpy as np


class WordleEnv:
    """
    Wordle game environment compatible with a DQN agent.

    Parameters
    ----------
    dictionary_path : str
        Path to a plain-text word list, one word per line.
    word_length : int
        Only words of this exact length are loaded into the dictionary.
    """

    # Hint value constants
    UNKNOWN = 0
    YELLOW  = 1
    GREEN   = 2
    GREY    = -1

    MAX_ROUNDS = 6

    def __init__(self, dictionary_path: str, word_length: int = 5):
        self.word_length  = word_length
        self.dictionary   = self._load_dict(dictionary_path, word_length)
        self.secret_word  = ""
        self.current_round = 0
        self.guessed_words = set()
        self.state        = np.zeros((26, word_length), dtype=np.int8)

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    def reset(self, word: str = None) -> np.ndarray:
        """
        Resets the game state and selects a new secret word.

        Parameters
        ----------
        word : str, optional
            Pin a specific secret word instead of choosing randomly.
            Must already exist in the dictionary.

        Returns
        -------
        np.ndarray
            Initial zeroed state matrix of shape (26, word_length).
        """
        self.secret_word   = word if word else random.choice(self.dictionary)
        self.current_round = 0
        self.guessed_words = set()
        self.state         = np.zeros((26, self.word_length), dtype=np.int8)
        return self.state.copy()

    def step(self, action_index: int) -> tuple[np.ndarray, float, bool]:
        """
        Advances the game by one guess.

        Parameters
        ----------
        action_index : int
            Index into self.dictionary identifying the guessed word.

        Returns
        -------
        next_state : np.ndarray  — updated board state (26, word_length)
        reward     : float       — shaped reward signal for this turn
        done       : bool        — True if the game has ended
        """
        guessed_word = self.dictionary[action_index]
        self.guessed_words.add(guessed_word)
        self.current_round += 1

        next_state = self.state.copy()
        reward     = -5.0  # base per-turn penalty

        secret_claimed  = [False] * self.word_length
        guess_handled   = [False] * self.word_length

        # --------------------------------------------------------------
        # PASS 1 — Lock green (exact) matches
        # --------------------------------------------------------------
        for i, char in enumerate(guessed_word):
            if char == self.secret_word[i]:
                row = ord(char) - ord('a')
                if next_state[row, i] != self.GREEN:
                    next_state[row, i] = self.GREEN
                    reward += 2.0          # bonus for each newly confirmed slot
                secret_claimed[i] = True
                guess_handled[i]  = True

        # --------------------------------------------------------------
        # PASS 2 — Evaluate yellows and greys for unresolved positions
        # --------------------------------------------------------------
        for i, char in enumerate(guessed_word):
            if guess_handled[i]:
                continue

            row          = ord(char) - ord('a')
            yellow_found = False

            for s_idx in range(self.word_length):
                if self.secret_word[s_idx] == char and not secret_claimed[s_idx]:
                    yellow_found        = True
                    secret_claimed[s_idx] = True
                    break

            if yellow_found:
                if next_state[row, i] != self.GREEN:
                    next_state[row, i] = self.GREY   # wrong position for this column

                for c in range(self.word_length):
                    if c != i and next_state[row, c] == self.UNKNOWN:
                        next_state[row, c] = self.YELLOW
                        reward += 0.5
            else:
                for c in range(self.word_length):
                    if next_state[row, c] != self.GREEN:
                        next_state[row, c] = self.GREY

        # --------------------------------------------------------------
        # Win / loss rewards
        # --------------------------------------------------------------
        if guessed_word == self.secret_word:
            reward += 100.0
        elif self.current_round == self.MAX_ROUNDS:
            reward -= 50.0

        done       = guessed_word == self.secret_word or self.current_round == self.MAX_ROUNDS
        self.state = next_state
        return next_state.copy(), reward, done

    def get_action_mask(self) -> list[bool]:
        """
        Returns a boolean list of length len(self.dictionary).

        True  — word is consistent with all current hints (legal guess).
        False — word violates at least one known constraint.

        Constraints enforced
        --------------------
        A. No completely grey letters (letter eliminated from secret entirely).
        B. All green slots must match exactly.
        C. All yellow letters must appear somewhere in the word.
        D. Words already guessed this game are excluded.

        If all words are filtered out (edge case with unusual secret words),
        the fallback returns all True so the agent is never stuck.
        """
        grey_letters      = set()
        yellow_letters    = set()
        green_constraints = {}  # position -> required character

        for r in range(26):
            char      = chr(r + ord('a'))
            row_state = self.state[r, :]

            if self.GREEN in row_state:
                for c, val in enumerate(row_state):
                    if val == self.GREEN:
                        green_constraints[c] = char
            if self.YELLOW in row_state:
                yellow_letters.add(char)
            if self.GREY in row_state and self.GREEN not in row_state and self.YELLOW not in row_state:
                grey_letters.add(char)

        mask = []
        for word in self.dictionary:
            # Rule A — no grey letters
            if any(ch in grey_letters for ch in word):
                mask.append(False)
                continue

            # Rule B — green slots must match exactly
            if any(word[idx] != ch for idx, ch in green_constraints.items()):
                mask.append(False)
                continue

            # Rule C — all yellow letters must be present
            if not all(ch in word for ch in yellow_letters):
                mask.append(False)
                continue

            # Rule D — no repeated guesses
            if word in self.guessed_words:
                mask.append(False)
                continue

            mask.append(True)

        if not any(mask):
            return [True] * len(self.dictionary)

        return mask

    # ------------------------------------------------------------------
    # INTERNAL HELPERS
    # ------------------------------------------------------------------

    def _load_dict(self, file_path: str, word_length: int) -> list[str]:
        """
        Loads words of exactly word_length alphabetic characters from file_path.
        Returns a sorted, deduplicated list for deterministic index mapping.
        """
        valid_words = set()
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    word = line.strip().lower()
                    if len(word) == word_length and word.isalpha():
                        valid_words.add(word)
        except FileNotFoundError:
            print(f"Error: dictionary file '{file_path}' not found.")
            sys.exit(1)

        if not valid_words:
            print(f"Error: no {word_length}-letter words found in '{file_path}'.")
            sys.exit(1)

        return sorted(valid_words)