import matplotlib.pyplot as plt
import numpy as np

from src.decode import decode

import os

def plot_tracking_information(ciphertext, tracking, window_size = 100):

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
    
    plot_tracking_information_of_many_experiments(trackings, labels)