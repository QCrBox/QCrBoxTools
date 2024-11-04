from textwrap import dedent

import pytest

from qcrboxtools.analyse.quality.base import DataQuality
from qcrboxtools.analyse.quality.html import convert_to_mathjax
from qcrboxtools.analyse.quality.html.quality_box import QualityIndicatorBox, quality_box_div, quality_div_group


@pytest.mark.parametrize(
    "input_text, expected_output",
    [
        # Basic tests
        (r"This is a formula $wR_2(F^2)$.", r"This is a formula \(wR_2(F^2)\)."),
        (r"This is a display formula $$wR_2(F^2)$$.", r"This is a display formula $$wR_2(F^2)$$."),
        # Mixed inline and display formulas
        (r"Inline: $wR_2(F^2)$, and display: $$wR_2(F^2)$$.", r"Inline: \(wR_2(F^2)\), and display: $$wR_2(F^2)$$."),
        # Tests for escaped dollar signs
        (r"The cost is \$5, and the formula is $x + y$.", r"The cost is $5, and the formula is \(x + y\)."),
        (
            r"The total cost is \$5, the discounted price is \$3, and the formula is $x^2 + y^2$.",
            r"The total cost is $5, the discounted price is $3, and the formula is \(x^2 + y^2\).",
        ),
        # Multiple inline formulas
        (
            r"First formula: $x + y$, second formula: $a = b + c$, and third: $m^2 = n^2 + 2mn$.",
            r"First formula: \(x + y\), second formula: \(a = b + c\), and third: \(m^2 = n^2 + 2mn\).",
        ),
        # Formulas with numbers and escaped dollar signs
        (r"The result is $E = mc^2$, and the amount is \$100.", r"The result is \(E = mc^2\), and the amount is $100."),
        # Inline inside sentence
        (
            r"Use the formula $x = y + z$ to find the value of x, and also consider $a = b + c$.",
            r"Use the formula \(x = y + z\) to find the value of x, and also consider \(a = b + c\).",
        ),
        # Mixed complex cases
        (
            r"Display math: $$a^2 + b^2 = c^2$$, cost: \$20, and inline: $e = mc^2$ and $y = mx + b$.",
            r"Display math: $$a^2 + b^2 = c^2$$, cost: $20, and inline: \(e = mc^2\) and \(y = mx + b\).",
        ),
        # Nested dollar signs
        (
            r"Consider $x$ is a variable, and for the cost \$5, use $a = b + c$.",
            r"Consider \(x\) is a variable, and for the cost $5, use \(a = b + c\).",
        ),
        # Edge case with empty inline formula
        (
            r"An empty inline formula: $$Display$$ and $$$$. What happens?",
            r"An empty inline formula: $$Display$$ and $$$$. What happens?",
        ),
        # Mathjax compatible Angstrom symbol
        (
            r'The C-O bond length is 1.46 $\AA$',
            r'The C-O bond length is 1.46 \(\unicode[.8,0]{x212B}\)',
        )
    ],
)
def test_convert_to_mathjax(input_text, expected_output):
    assert convert_to_mathjax(input_text) == expected_output


def test_create_quality_box_div():
    # Define input values
    indicator = QualityIndicatorBox(name="Accuracy $x + y$", value="98", unit="%", quality_level=DataQuality.GOOD)

    # Expected HTML output
    expected_html = dedent(r"""
        <div class="indicator data-quality-good">
            <div class="name">Accuracy \(x + y\)</div>
            <div class="value">98</div>
            <div class="unit">%</div>
        </div>
    """).strip()

    result = quality_box_div(indicator)
    assert result == expected_html


@pytest.mark.parametrize(
    "indicator, expected_html",
    [
        (
            QualityIndicatorBox(name="Accuracy $x$", value="98", unit="%", quality_level=DataQuality.GOOD),
            dedent(r"""
                <div class="indicator data-quality-good">
                    <div class="name">Accuracy \(x\)</div>
                    <div class="value">98</div>
                    <div class="unit">%</div>
                </div>
            """).strip(),
        ),
        (
            QualityIndicatorBox(name="Consistency", value="70", unit="ms", quality_level=DataQuality.MARGINAL),
            dedent("""
                <div class="indicator data-quality-marginal">
                    <div class="name">Consistency</div>
                    <div class="value">70</div>
                    <div class="unit">ms</div>
                </div>
            """).strip(),
        ),
    ],
)
def test_quality_box_div(indicator, expected_html):
    """Test the HTML structure generation for a single quality indicator box."""
    assert quality_box_div(indicator) == expected_html


def test_quality_div_group():
    """Test the HTML generation for a group of quality indicator boxes."""
    indicators = [
        QualityIndicatorBox(name="Accuracy", value="98", unit="%", quality_level=DataQuality.GOOD),
        QualityIndicatorBox(name="Consistency", value="70", unit="ms", quality_level=DataQuality.MARGINAL),
        QualityIndicatorBox(name="Completeness", value="50", unit="%", quality_level=DataQuality.BAD),
    ]

    expected_html = dedent("""
        <div class="container">
            <div class="indicator data-quality-good">
                <div class="name">Accuracy</div>
                <div class="value">98</div>
                <div class="unit">%</div>
            </div>
            <div class="indicator data-quality-marginal">
                <div class="name">Consistency</div>
                <div class="value">70</div>
                <div class="unit">ms</div>
            </div>
            <div class="indicator data-quality-bad">
                <div class="name">Completeness</div>
                <div class="value">50</div>
                <div class="unit">%</div>
            </div>
        </div>
    """).strip()

    assert quality_div_group(indicators) == expected_html
