"""Pionless-EFT reference couplings at fixed lattice spacings (Watson+ Tab. 3).

Watson+ 2023 (arXiv:2312.05344) take the 1.4 fm row from Roggero+ 2019
(arXiv:1911.06368) Tab. I, t = 10.5794, C0 = -98.2265511 and D0 = 127.839693 MeV,
rounded to two decimals, and the 2.2 fm row from Rokash+ (arXiv:1308.3386, J. Phys. G
41, 015105).
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class PionlessCouplings:
    """Pionless-EFT couplings at a fixed lattice spacing.

    Notation follows Watson+ 2023 Sec. 4.1 (Eqs. 38-40):
        - hopping_scale_mev: h = 1 / (2 M_N a_L^2), in MeV.
        - two_body_mev: C_pi-less (contact two-body coupling), in MeV.
        - three_body_mev: D_pi-less (contact three-body coupling), in MeV.

    Attributes:
        hopping_scale_mev: h.
        two_body_mev: C_pi-less.
        three_body_mev: D_pi-less.
    """

    hopping_scale_mev: float
    two_body_mev: float
    three_body_mev: float


_REFERENCE_TABLE = {
    1.4: PionlessCouplings(
        hopping_scale_mev=10.58, two_body_mev=-98.23, three_body_mev=127.84
    ),
    2.2: PionlessCouplings(
        hopping_scale_mev=4.29, two_body_mev=-40.19, three_body_mev=42.51
    ),
}


def reference_couplings_at_spacing(spacing_fm: float) -> PionlessCouplings:
    """Tabulated reference couplings from Watson+ 2023 Tab. 3 (page 17).

    Args:
        spacing_fm: Lattice spacing in fm; must be one of the tabulated literals
            (the lookup is exact-match, not arithmetic-derived).

    Returns:
        PionlessCouplings populated from Tab. 3.

    Raises:
        ValueError: If spacing_fm is not a tabulated spacing; the message lists
            the available spacings.
    """
    try:
        return _REFERENCE_TABLE[spacing_fm]
    except KeyError:
        available = ", ".join(str(s) for s in sorted(_REFERENCE_TABLE))
        raise ValueError(
            f"no tabulated pionless couplings for {spacing_fm} fm; "
            f"available (fm): {available}."
        ) from None
