import csv
import random
import math

import matplotlib.pyplot as plt
from tqdm import tqdm

random.seed(0)                 # Seed for reproducibility
EPS = 1e-10                    # Laplace smoothing constant

def load_inputs():
    with open('data/alphabet.csv') as f:
        alphabet = next(csv.reader(f))

    with open('data/letter_probabilities.csv') as f:
        P = [float(p) + EPS for p in next(csv.reader(f))]
        s = sum(P);  P = [p/s for p in P]              # renormalise

    with open('data/letter_transition_matrix.csv') as f:
        reader = csv.reader(f)
        M = [[float(p) + EPS for p in row] for row in reader]
        # renormalise each column (j is previous symbol)
        for j in range(len(M)):
            col_sum = sum(row[j] for row in M)
            for i in range(len(M)):
                M[i][j] /= col_sum
    return alphabet, P, M

ALPHABET, LETTER_PROBS, LETTER_TRANSITION_MATRIX = load_inputs()
LETTER_TO_INDEX = {letter: idx for idx, letter in enumerate(ALPHABET)}
INDEX_TO_LETTER = {idx: letter for idx, letter in enumerate(ALPHABET)}

def text_to_numbers(text: str) -> list[int]:
    """
    The default representation of this algorithm is to break
    down strings into lists of numbers corresponding to which 
    order the letter appears in the alphabet. This function 
    takes 
    
    The alphabet is give in `/data/alphabet.csv` as a csv of 
    each letter in the alphabet. 
    
    For example:
        'abcd' --> [0,1,2,3]
        'bob' --> [0,14,0]
    """
    return [LETTER_TO_INDEX[char] for char in text]

def numbers_to_text(numbers: list[int]) -> str:
    """
    This is the inverse of the function "text_to_numbers". See
    the "text_to_numbers" documentation for more information.
    """
    return ''.join(INDEX_TO_LETTER[num] for num in numbers)

def swap(permutation: list[int]) -> list[int]:
    """
    The function takes a permutation of a list of numbers 
    from 0 to (len(ALPHABET) - 1) and randomly picks two 
    numbers to exchange. It then returns the permutation 
    with those two numbers exchanged.
    """

    i, j = random.sample(range(len(permutation)), 2)
    new_permutation = permutation.copy() 
    new_permutation[i], new_permutation[j] = new_permutation[j], new_permutation[i]
    return new_permutation

def log_liklihood(data: list[int], permutation: list[int]):
    """
    Given a permutation, this computes the log liklihood
    of that permutation occuring
    """
    log_liklihood_of_transitions = [
        math.log(LETTER_TRANSITION_MATRIX
            [
                permutation.index(data[k]) # applied the inverse permutation
            ][
                permutation.index(data[k-1]) # applied the inverse permutation
            ]
        )
        for k in range(1, len(data))
    ]

    return math.log(LETTER_PROBS[permutation.index(data[0])]) + sum(log_liklihood_of_transitions)


def decode(
    ciphertext: str, 
    has_breakpoint: bool, 
    n_iter: int = 1000, 
    true_plaintext: list[int] = None,
) -> str:

    assert has_breakpoint == False #only support no breakpoints for now

    # Lists for tracking quantities to plot later:
    tracking = {
        "log_liklihood": [],
        "decoding_accuracy": [],
        "total_accepted_transitions": [],
    }

    numerized_ciphertext = text_to_numbers(ciphertext)
    numerized_true_plaintext = text_to_numbers(true_plaintext) if true_plaintext is not None else None
    accepted_transitions = 0
    
    F = [i for i in range(len(ALPHABET))]

    for t in tqdm(range(n_iter)):

        F_p = swap(F)

        log_alpha = min(
            0,
            log_liklihood(numerized_ciphertext, F_p) - log_liklihood(numerized_ciphertext, F)
        )

        r= random.random()

        if r <= math.exp(log_alpha): #accept
            F = F_p
            accepted_transitions+= 1

        numerized_plaintext = [F.index(y) for y in numerized_ciphertext]
        tracking["log_liklihood"].append(log_liklihood(numerized_ciphertext, F))
        if numerized_true_plaintext is not None:
            tracking["decoding_accuracy"].append(sum(a == b for a, b in zip(numerized_plaintext, numerized_true_plaintext)) / len(numerized_plaintext))
        tracking["total_accepted_transitions"].append(accepted_transitions)
    
    plaintext = numbers_to_text(numerized_plaintext)

    return plaintext, tracking

def plot_tracking_information(ciphertext, tracking, window_size = 100):

    import numpy as np
    import os

    os.makedirs("plots", exist_ok=True)

    iters = np.arange(1, len(tracking["log_liklihood"]) + 1)

    # ------------------------------------------------
    # (a)  Log-likelihood trace
    # ------------------------------------------------
    plt.figure(figsize=(7, 4))
    plt.plot(iters, tracking["log_liklihood"], lw=1)
    plt.title("Log-Likelihood of Accepted State vs. Iteration")
    plt.xlabel("Iteration")
    plt.ylabel("log p(y|f)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("plots/log_likelihood.png", dpi=200)

    # ------------------------------------------------
    # (b)  Sliding-window acceptance rate
    # ------------------------------------------------
    T = window_size        # ← already defined above (e.g. 20 or 100)

    accepts = np.array(tracking["total_accepted_transitions"])
    accept_rate = (accepts[T-1:] - accepts[:-T+1]) / T        # vectorised diff

    plt.figure(figsize=(7, 4))
    plt.plot(iters[T-1:], accept_rate, lw=1)
    plt.title(f"Sliding-Window Acceptance Rate (window = {T})")
    plt.xlabel("Iteration")
    plt.ylabel("Acceptance rate")
    plt.ylim(0, 1)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"plots/accept_rate_window{T}.png", dpi=200)

    # ------------------------------------------------
    # (c)  Decoding accuracy
    # ------------------------------------------------
    if tracking["decoding_accuracy"]:          # only if true plaintext supplied
        plt.figure(figsize=(7, 4))
        plt.plot(iters, tracking["decoding_accuracy"], lw=1)
        plt.title("Decoding Accuracy vs. Iteration")
        plt.xlabel("Iteration")
        plt.ylabel("Accuracy fraction")
        plt.ylim(0, 1)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig("plots/decoding_accuracy.png", dpi=200)

    # ------------------------------------------------
    # (d)  Log-likelihood per symbol in **bits**
    # ------------------------------------------------
    ll = np.array(tracking["log_liklihood"])
    bits_per_symbol = -(ll / len(ciphertext)) / np.log(2)   # −log₂ p(x)/n

    plt.figure(figsize=(7, 4))
    plt.plot(iters, bits_per_symbol, lw=1, color="purple")
    plt.title("Log-Likelihood per Symbol in Bits")
    plt.xlabel("Iteration")
    plt.ylabel("− log₂ p(x)  (bits per symbol)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("plots/loglike_bits_per_symbol.png", dpi=200)

    plt.show()          # optional: pop all four windows

def plot_tracking_information_of_many_experiments(trackings, labels):

    import numpy as np
    import os

    os.makedirs("plots", exist_ok=True)

    iters = np.arange(1, len(trackings[0]["log_liklihood"]) + 1)

    plt.figure(figsize=(7, 4))
    for tracking, label in zip(trackings, labels):
        plt.plot(iters, tracking["decoding_accuracy"], lw=1, label = label)
    plt.title("Decoding Accuracy vs. Iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Accuracy fraction")
    plt.ylim(0, 1)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.legend()
    plt.savefig("plots/decoding_accuracy_of_different_segments.png", dpi=200)

if __name__ == "__main__":

    with open("data/sample/ciphertext.txt", 'r') as file:
        ciphertext = file.read()
    with open("data/sample/plaintext.txt", 'r') as file:
        true_plaintext = file.read()
    
    print("INFO : starting decoding of large segment")
    plaintext, tracking = decode(ciphertext, False, n_iter=2000, true_plaintext=true_plaintext)

    plot_tracking_information(
        ciphertext=ciphertext,
        tracking = tracking,
        window_size=100
    )

    print("INFO : starting decoding of multiple segment lengths")
    seg_lengths = [10, 100, 1000, 4000]
    labels = [f"Segment length = {seg_len}" for seg_len in seg_lengths]
    ciper_segments = [ciphertext[:seg_length] for seg_length in seg_lengths]
    true_plain_segments = [true_plaintext[:seg_length] for seg_length in seg_lengths]
    trackings = []
    for ciper_segment,true_plain_segment  in zip(ciper_segments,true_plain_segments ):
        _, tracking = decode(ciper_segment, False, n_iter=2000, true_plaintext=true_plain_segment)
        trackings.append(tracking)

    print(trackings)
    
    plot_tracking_information_of_many_experiments(trackings, labels)



