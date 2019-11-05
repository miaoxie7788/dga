"""
    User defined functions (udfs).
"""


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