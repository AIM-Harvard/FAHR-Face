# moved from training/train_fahr_face_survivel to training/train_fahr_facesurvival
def c_index(T, E, scores):
    """
    Calculate C-index for survival analysis predictions.

    Parameters:
    T (array): Array of observed event times.
    E (array): Array indicating whether the event occurred (1) or was censored (0).
    scores (array): Predicted risk scores.

    Returns:
    c_index (float): C-index value.
    """
    n = len(T)
    concordant = 0
    num_pairs = 0

    for i in range(n):
        if E[i] == 1:
            for j in range(n):
                if T[j] > T[i]:
                    num_pairs += 1
                    if scores[i] > scores[j]:
                        concordant += 1

    c_index = concordant / num_pairs if num_pairs > 0 else 0.0
    return c_index
