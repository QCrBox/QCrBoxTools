import re

from ..base import DataQuality

data_quality_to_css_name = {
    DataQuality.BAD: "data-quality-bad",
    DataQuality.BADISH: "data-quality-badish",
    DataQuality.MARGINAL: "data-quality-marginal",
    DataQuality.GOODISH: "data-quality-goodish",
    DataQuality.GOOD: "data-quality-good",
    DataQuality.INFORMATION: "data-quality-information",
}


def convert_to_mathjax(text):
    """
    Converts math expressions in the text to MathJax format.

    - Rewrites $...$ to \( ... \) for inline math.
    - Leaves $$...$$ intact for display math.
    - Rewrites \$ to $.

    Parameters
    ----------
    text : str
        Input text with formulas.

    Returns
    -------
    str
        Text with MathJax-compatible formulas.
    """
    display_math_placeholder = "__DISPLAY_MATH__"
    text = re.sub(r"\$\$(.*?)\$\$", lambda m: display_math_placeholder + m.group(1) + display_math_placeholder, text)
    text = re.sub(r"(?<!\\)\$(.*?)(?<!\\)\$", r"\\(\1\\)", text)
    text = text.replace(display_math_placeholder, "$$")
    text = text.replace(r"\$", "$")
    return text
