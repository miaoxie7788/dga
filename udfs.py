"""
    User defined functions (udfs).
"""
from math import log2


def transform_domain_len_udf(s):
    """
    Compute the length of characters for a domain.
    :param s:   domain,     e.g.,   000directory
    :return:    length,     e.g.,   12
    """
    if s:
        n = len(s)
    else:
        n = 0

    return n


def transform_domain_num_len_udf(s):
    """
    Compute the length of numerical characters for a domain.
    :param s:   domain,     e.g.,   000directory
    :return:    length,     e.g.,   3
    """
    if s:
        n = sum(c.isdigit() for c in s)
    else:
        n = 0

    return n


def transform_domain_sym_len_udf(s):
    """
    Compute the length of non-alpha and non-digit characters for a domain.
    :param s:   domain,     e.g.,   000directory
    :return:    length,     e.g.,   3
    """
    if s:
        n = sum(not (c.isalpha() or c.isdigit()) for c in s)
    else:
        n = 0

    return n


def transform_domain_vow_len_udf(s):
    """
    Compute the length of vowels for a domain.
    :param s:   domain,     e.g.,   000directory
    :return:    length,     e.g.,   4
    """
    vowels = ['a', 'e', 'i', 'o', 'u', 'y']
    if s:
        n = sum(c.lower() in vowels for c in s)
    else:
        n = 0

    return n


def transform_domain_uniq_count_udf(s):
    """
    Compute the count of unique characters for a domain.
    :param s:   domain,     e.g.,   000directory
    :return:    length,     e.g.,   12
    """
    if s:
        n = len(set(s))
    else:
        n = 0

    return n


def transform_domain_norm_ent_udf(s):
    """
    Compute the normalised entropy for a domain.
        reference: Inline DGA Detection with Deep Networks
    :param s:   domain,     e.g.,   000directory
    :return:    entropy,    e.g.,   0.843
    """
    if s:
        n = len(s)
        freq = {c: s.count(c) for c in set(s)}
        dist = {c: freq[c] / n for c in freq}

        if n == 1:
            ent = 0
        else:
            # normalised entropy
            ent = -1 * sum(dist[c] * log2(dist[c]) for c in dist) / log2(n)
    else:
        ent = 0

    return ent


def transform_domain_gini_idx_udf(s):
    """
    Compute the Gini index for a domain.
        reference: Inline DGA Detection with Deep Networks
    :param s:   domain,     e.g.,   000directory
    :return:    length,     e.g.,   0.861
    """
    if s:
        n = len(s)
        freq = {c: s.count(c) for c in set(s)}
        dist = {c: freq[c] / n for c in freq}

        # Gini index
        gni = 1 - sum(dist[c] ** 2 for c in dist)
    else:
        gni = 1

    return gni


def transform_domain_class_err_udf(s):
    """
    Compute the classification error for a domain.
        reference: Inline DGA Detection with Deep Networks
    :param s:   domain,     e.g.,   000directory
    :return:    length,     e.g.,   3\
    """
    if s:
        n = len(s)
        freq = {c: s.count(c) for c in set(s)}
        dist = {c: freq[c] / n for c in freq}

        # classification error
        cer = 1 - max(dist.values())
    else:
        cer = 1

    return cer


