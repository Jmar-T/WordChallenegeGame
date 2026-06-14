"""
play.py

Human-playable Wordle game.

A terminal implementation of Wordle supporting any word length available
in the dictionary. The game selects a secret word at random, accepts
guesses from the player, and provides colour-coded feedback after each
attempt in the style of the original game:

    UPPERCASE  — correct letter, correct position  (green)
    lowercase  — correct letter, wrong position    (yellow)
    -          — letter not in the word            (grey)

Usage
-----
    python play.py
"""

import random
import sys


# ======================================================================
# DICTIONARY
# ======================================================================

def load_dictionary(file_path: str, word_length: int) -> list[str]:
    """
    Loads words of exactly word_length alphabetic characters.
    Returns a sorted, deduplicated list.
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
        print(f"No {word_length}-letter words found in '{file_path}'.")
        sys.exit(1)

    return sorted(valid_words)


# ======================================================================
# FEEDBACK RENDERER
# ======================================================================

def build_feedback(guess: str, secret: str) -> tuple[list[str], set[str], set[str]]:
    """
    Compares a guess against the secret word using the same two-pass
    algorithm as WordleEnv to ensure consistent behaviour.

    Returns
    -------
    display   : list[str] — one token per position (UPPER, lower, or '-')
    yellows   : set[str]  — letters present but in wrong position this turn
    greys     : set[str]  — letters confirmed absent this turn
    """
    word_len        = len(secret)
    secret_claimed  = [False] * word_len
    guess_handled   = [False] * word_len
    display         = ["-"] * word_len

    # Pass 1 — lock greens
    for i, ch in enumerate(guess):
        if ch == secret[i]:
            display[i]         = ch.upper()
            secret_claimed[i]  = True
            guess_handled[i]   = True

    # Pass 2 — evaluate yellows and greys
    yellows: set[str] = set()
    greys:   set[str] = set()

    for i, ch in enumerate(guess):
        if guess_handled[i]:
            continue

        yellow_found = False
        for s_idx in range(word_len):
            if secret[s_idx] == ch and not secret_claimed[s_idx]:
                yellow_found        = True
                secret_claimed[s_idx] = True
                break

        if yellow_found:
            display[i] = ch.lower()
            yellows.add(ch)
        else:
            greys.add(ch)

    return display, yellows, greys


# ======================================================================
# GAME LOOP
# ======================================================================

def play_wordle(dictionary_path: str = "words.txt") -> None:
    """Runs one complete game of Wordle in the terminal."""

    # Word length selection
    try:
        word_length = int(input("Word length (e.g. 5): ").strip())
    except ValueError:
        print("Please enter a valid integer.")
        return

    if word_length < 2:
        print("Word length must be at least 2.")
        return

    print(f"\nLoading dictionary...")
    dictionary  = load_dictionary(dictionary_path, word_length)
    secret_word = random.choice(dictionary)
    max_rounds  = 6

    print(f"Game started! Guess the {word_length}-letter word. You have {max_rounds} attempts.")
    print("  UPPERCASE = correct position | lowercase = wrong position | - = not in word")
    print("-" * 50)

    all_greys   : set[str] = set()
    all_yellows : set[str] = set()
    guessed_words: set[str] = set()
    solved = False

    for attempt in range(1, max_rounds + 1):
        # Input validation loop
        while True:
            raw = input(f"\nAttempt {attempt}/{max_rounds}: ").strip().lower()

            if not raw.isalpha():
                print("  Letters only — no numbers or symbols.")
                continue
            if len(raw) != word_length:
                print(f"  Must be exactly {word_length} letters.")
                continue
            if raw not in dictionary:
                print(f"  '{raw}' is not in the dictionary.")
                continue
            if raw in guessed_words:
                print(f"  You already guessed '{raw}'.")
                continue
            break

        guessed_words.add(raw)
        display, yellows, greys = build_feedback(raw, secret_word)
        all_yellows |= yellows
        all_greys   |= greys

        print(f"  Result  : {' '.join(display)}")
        if all_greys:
            print(f"  Absent  : {' '.join(sorted(all_greys))}")
        if all_yellows:
            print(f"  Misplaced: {' '.join(sorted(all_yellows))}")

        if raw == secret_word:
            print(f"\n🎉 Solved in {attempt} {'attempt' if attempt == 1 else 'attempts'}!")
            solved = True
            break

    if not solved:
        print(f"\n❌ The word was: {secret_word.upper()}")


if __name__ == "__main__":
    play_wordle()