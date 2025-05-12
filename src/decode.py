import csv
import random
import math


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
    n_iter: int = 10000, 
    true_plaintext: list[int] = None,
) -> str:

    assert has_breakpoint == False #only support no breakpoints for now

    # Lists for tracking quantities to plot later:
    tracking = None
    if true_plaintext is not None:
        tracking = {
            "log_liklihood": [],
            "decoding_accuracy": [],
            "total_accepted_transitions": [],
        }

    numerized_ciphertext = text_to_numbers(ciphertext)
    numerized_true_plaintext = text_to_numbers(true_plaintext) if true_plaintext is not None else None
    accepted_transitions = 0
    
    F = [i for i in range(len(ALPHABET))]
    current_log_liklihood = log_liklihood(numerized_ciphertext, F)
    n_iter_since_liklihood_improve = 0

    for t in range(n_iter):

        F_p = swap(F)

        swapped_log_liklihood = log_liklihood(numerized_ciphertext, F_p)

        log_alpha = min(
            0,
            swapped_log_liklihood - current_log_liklihood
        )

        r= random.random()

        if r <= math.exp(log_alpha): #accept
            F = F_p
            current_log_liklihood = swapped_log_liklihood
            n_iter_since_liklihood_improve = 0
            accepted_transitions+= 1
        else:
            n_iter_since_liklihood_improve

        numerized_plaintext = [F.index(y) for y in numerized_ciphertext]

        # Early stopping
        if n_iter_since_liklihood_improve > 100:
            break

        # Tracking information (only needed when plotting liklihoods, etc.)
        if tracking is not None:
            tracking["log_liklihood"].append(current_log_liklihood)
            if numerized_true_plaintext is not None:
                tracking["decoding_accuracy"].append(sum(a == b for a, b in zip(numerized_plaintext, numerized_true_plaintext)) / len(numerized_plaintext))
            tracking["total_accepted_transitions"].append(accepted_transitions)
    
    plaintext = numbers_to_text(numerized_plaintext)

    if tracking is not None:    
        return plaintext, tracking # return tracking if we're tracking
    return plaintext


if __name__ == "__main__":

    with open("data/sample/ciphertext.txt", 'r') as file:
        ciphertext = file.read()
    with open("data/sample/plaintext.txt", 'r') as file:
        true_plaintext = file.read()
    
    print("INFO : starting decoding of large segment")
    plaintext = decode(ciphertext, False, n_iter=2000, true_plaintext=None)





