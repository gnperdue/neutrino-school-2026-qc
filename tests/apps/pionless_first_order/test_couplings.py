"""Tests for the pionless-EFT reference couplings (Watson+ Tab. 3, page 17)."""

import pytest

from qec_utility_estimation.apps.pionless_first_order.couplings import (
    reference_couplings_at_spacing,
)

# Constants for the independent h = 1/(2 M_N a_L^2) cross-check (Watson+ Eq. 38);
# hbar*c converts the fm spacing into the natural units the formula assumes.
_NUCLEON_MASS_MEV = 938.0
_HBAR_C_MEV_FM = 197.327


@pytest.mark.parametrize("spacing_fm", [1.4, 2.2])
def test_tabulated_hopping_scale_matches_inverse_mass_spacing_formula(
    spacing_fm: float,
) -> None:
    """Tabulated h equals 1/(2 M_N a_L^2) at each spacing (Watson+ Eq. 38).

    h is not an independent constant -- Eq. 38 defines h = 1/(2 M_N a_L^2). This
    cross-checks the table's h against that formula by an independent
    computation, catching a transcription error in h that a bare value
    assertion could not.
    """
    expected_h = _HBAR_C_MEV_FM**2 / (2.0 * _NUCLEON_MASS_MEV * spacing_fm**2)

    couplings = reference_couplings_at_spacing(spacing_fm)

    assert couplings.hopping_scale_mev == pytest.approx(expected_h, rel=5e-3)


@pytest.mark.parametrize(
    ("spacing_fm", "two_body_mev", "three_body_mev"),
    [(1.4, -98.23, 127.84), (2.2, -40.19, 42.51)],
)
def test_tabulated_contact_couplings_match_published_values(
    spacing_fm: float, two_body_mev: float, three_body_mev: float
) -> None:
    """Regression pin on the fitted contact constants C, D (Watson+ Tab. 3).

    C and D are fitted low-energy constants (from Refs. [50]/[136]) with no
    closed form, so this only guards against an accidental edit to the
    transcribed table. Their physics validation is the PDF cross-check and the
    downstream Trotter-bracket / headline reproduction, not this assertion.
    """
    couplings = reference_couplings_at_spacing(spacing_fm)

    assert couplings.two_body_mev == pytest.approx(two_body_mev)
    assert couplings.three_body_mev == pytest.approx(three_body_mev)


def test_reference_couplings_rejects_unknown_spacing() -> None:
    """An untabulated spacing raises ValueError listing the available spacings.

    Only a_L in {1.4, 2.2} fm is tabulated (Watson+ Tab. 3); any other spacing
    must fail loudly -- naming the supported values -- rather than return a
    default or wrong coupling set.
    """
    with pytest.raises(ValueError, match="available"):
        reference_couplings_at_spacing(1.0)
