def format_label_unit(tuple):
    """Formats a (label, unit) tuple into a "label (unit)" string.

    Parameters
    ----------
    tuple : tuple of str
        A tuple containing a label and a unit.

    Returns
    -------
    str
        A formatted string "label (unit)".
    """    
    return f"{tuple[0]} ({tuple[1]})"

def word_error_rate(insertions, deletions, substitutions, words):
    """Calculate the Word Error Rate (WER) defined as:
         
        ((insertions + deletions + substitutions) / words) * 100

    Parameters
    ----------
    insertions : float
        The number of insertions
    deletions : float
        The number of deletions
    substitutions : float
        The number of substitutions
    words : float
        The total number of words

    Returns
    -------
    float
        The Word Error Rate (WER) 
    """    
    return ((insertions + deletions + substitutions) / words) * 100
