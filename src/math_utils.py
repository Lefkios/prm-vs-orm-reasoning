"""
Answer extraction and equivalence checking for MATH-500.

MATH answers are LaTeX expressions (fractions, intervals, tuples, sqrt, etc.),
unlike GSM8K's plain numeric "#### 42" format. Two answers can be the same
value written differently (e.g. "1/2" vs "\\frac{1}{2}"), so a string match
isn't enough — we normalize first, then fall back to sympy for anything that
still doesn't match as a string.

Based on the normalization approach from the original Hendrycks MATH eval
script, since full sympy LaTeX parsing alone fails on a lot of real MATH
answers (tuples, intervals, text annotations, degree symbols, etc).
"""

import re
from sympy.parsing.latex import parse_latex
from sympy import simplify


def extract_boxed_answer(text):
    """Pull the contents of the last \\boxed{...} in the text, handling
    nested braces (e.g. \\boxed{\\frac{1}{2}})."""
    idx = text.rfind("\\boxed")
    if idx == -1:
        return None

    start = text.find("{", idx)
    if start == -1:
        return None

    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1:i].strip()
    return None


def _normalize(expr):
    if expr is None:
        return None

    expr = expr.strip()

    # strip outer $ ... $ or $$ ... $$
    expr = expr.strip("$").strip()

    # remove \left, \right (sympy/string compare doesn't care about these)
    expr = expr.replace("\\left", "").replace("\\right", "")

    # remove \text{...} and \mbox{...} annotations, keep inner content
    expr = re.sub(r'\\(?:text|mbox)\{([^}]*)\}', r'\1', expr)

    # remove \! \, \; spacing commands and plain whitespace
    expr = re.sub(r'\\[!,;]', '', expr)
    expr = expr.replace(" ", "")

    # normalize \dfrac and \tfrac to \frac
    expr = expr.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")

    # drop a trailing period (common in MATH solutions)
    if expr.endswith("."):
        expr = expr[:-1]

    # normalize percent sign spacing
    expr = expr.replace("\\%", "%")

    return expr


def _sympy_equal(a, b):
    """Try parsing both sides as LaTeX math and check symbolic equivalence.
    Returns None (not True/False) if parsing fails, so callers can fall
    back to a plain string comparison instead of treating a parse error
    as "not equal"."""
    try:
        a_parsed = parse_latex(a)
        b_parsed = parse_latex(b)
        diff = simplify(a_parsed - b_parsed)
        return diff == 0
    except Exception:
        return None


def is_equivalent(predicted, true_answer):
    if predicted is None or true_answer is None:
        return False

    pred_norm = _normalize(predicted)
    true_norm = _normalize(true_answer)

    if pred_norm == true_norm:
        return True

    # sympy comparison handles cases like "1/2" vs "\frac{1}{2}" vs "0.5",
    # but doesn't handle tuples/intervals well, so it's a fallback, not primary
    result = _sympy_equal(pred_norm, true_norm)
    if result is not None:
        return result

    return False
