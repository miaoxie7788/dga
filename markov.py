import numpy as np


def markov_fit(seqs):
    """
        Fit a 1st order Markov chain to a list of iterables.
    :param seqs: a list of iterables
    :return:     fitted Markov model
    """
    seqs = [np.asarray(seq, dtype=str) for seq in seqs]

    seq_sizes = [seq.size for seq in seqs]

    state_space, inv, cnt_vector = np.unique(np.concatenate(seqs), return_inverse=True, return_counts=True)

    seq_invs = np.split(inv, np.cumsum(seq_sizes)[:-1])

    cnt_matrix = np.zeros((state_space.size, state_space.size))
    for seq_size, seq_inv in zip(seq_sizes, seq_invs):
        if seq_size > 1:
            for k in range(seq_size - 1):
                cnt_matrix[seq_inv[k]][seq_inv[k + 1]] += 1

    start_prob_vector = np.true_divide(cnt_vector, sum(seq_sizes))

    # Suppress warning for zero/zero.
    with np.errstate(divide='ignore', invalid='ignore'):
        trans_prob_matrix = np.true_divide(cnt_matrix, cnt_matrix.sum(axis=1, keepdims=True))

    # Set NaNs from zero/zero to zero.
    # Transition probability should be zero in this case.
    trans_prob_matrix = np.nan_to_num(trans_prob_matrix)

    markov_model = {"s": state_space, "p0": start_prob_vector, "p1": trans_prob_matrix}
    return markov_model


def markov_apply(seq, markov_model, n=3, method='avg', is_log=False):
    """
        Compute a sequence's probability based on the given 1st order markov chain.
    :param seq:             iterable
    :param markov_model:    fitted Markov model
    :param n:               n-gram
    :param method:          'avg', 'sum', 'max', 'min', 'med'
    :param is_log:          the logarithm of a probability
    :return:                sequence probability
    """
    seq = np.asarray(seq, dtype=str)

    # n-grams
    if len(seq) < n:
        seqs = [seq]
    else:
        seqs = [seq[k: k + n] for k in range(len(seq) - n + 1)]

    # Initialise the 1st Markov chain.
    state_space, start_prob_vector, trans_prob_matrix = markov_model['s'], markov_model['p0'], markov_model['p1']

    # Replace log(0) with constant log(1e-100).
    log0 = -230.26

    state_size = len(state_space)

    log_prs = list()

    for seq in seqs:
        seq_size = min(n, len(seq))
        seq_inv = -1 * np.ones(seq_size, dtype=int)
        for k, state in zip(range(state_size), state_space):
            seq_inv[np.where(seq == state)] = k

        if seq_inv[0] == -1:
            log_prs.append(log0)
            continue
        else:
            log_pr = np.log(start_prob_vector[seq_inv[0]])

            for k in range(seq_size - 1):
                t0, t1 = seq_inv[k], seq_inv[k + 1]

                if t0 != -1 and t1 != -1:
                    pr = trans_prob_matrix[t0][t1]
                    if pr > 0:
                        log_pr += np.log(pr)
                    else:
                        log_pr = log0
                        break
                else:
                    log_pr = log0
                    break
            log_prs.append(log_pr)

    # avg, sum, max, min, med
    func_dict = {'avg': np.mean, 'sum': np.sum, 'max': np.max, 'min': np.min, 'med': np.median}

    seq_pr = func_dict[method](log_prs)

    # probability or log-probability
    if not is_log:
        seq_pr = np.exp(seq_pr)

    return seq_pr
