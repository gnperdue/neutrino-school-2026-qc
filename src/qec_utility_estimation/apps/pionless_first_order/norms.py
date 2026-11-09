"""Pionless-EFT Hamiltonian norm inputs for qubitization.

The qubitization cost primitives consume two physical norms of the leading-order
pionless Hamiltonian (Roggero+ 2019, arXiv:1911.06368): the LCU 1-norm lambda,
which sets the qubiterate-QPE register width and application count (Eq. 21), and
the energy spread Delta-H = E_max - E_min, which sets the QSVT time-evolution
polynomial degree (Eq. 12). Both are derived from the tabulated contact couplings.
"""

import math
from dataclasses import dataclass

from qec_utility_estimation.apps.pionless_first_order.couplings import (
    PionlessCouplings,
)

# Nuclear binding energy per nucleon at saturation density, b_max, in MeV: the
# standard nuclear-matter value, the default of Roggero+ 2019 Eq. 12's A b_max.
_DEFAULT_SATURATION_BINDING_MEV = 16.0


@dataclass(frozen=True, kw_only=True)
class PauliCoefficients:
    """The coefficient of each kind of Pauli string in the Hamiltonian, in MeV."""

    hopping_mev: float  # each of a site's 24 hopping strings
    identity_mev: float  # once per site
    single_z_mev: float  # each of a site's 4 Z strings
    pair_zz_mev: float  # each of a site's 6 ZZ strings
    triple_zzz_mev: float  # each of a site's 4 ZZZ strings


def pauli_coefficients(couplings: PionlessCouplings) -> PauliCoefficients:
    """The Pauli strings' coefficients of Roggero+ 2019 Eq. 1's Hamiltonian, in MeV.

    Eq. 1 has -t per bond for each species, C0 per pair of nucleons on a site, D0 per
    triple, and the constant 2 D t A. Watson+ 2023 (arXiv:2312.05344) Eqs. 38-40 write
    the same Hamiltonian, and the contacts' source, Rokash+ 2013 (arXiv:1308.3386)
    Eq. 20, has the same (D / 6) :rho^3:. With n = (1 - Z) / 2 and

        a_i^dagger a_j + a_j^dagger a_i = (X_i Z...Z X_j + Y_i Z...Z Y_j) / 2,

    each of a bond's two strings carries -t / 2, and the contacts collect to

        identity:  3/2 C0 + 1/2 D0
        each Z:    -3/4 (C0 + D0 / 2)
        each ZZ:   (C0 + D0) / 4
        each ZZZ:  -D0 / 8

    If you compare with the paper, its printed Pauli forms differ from its Eq. 1, and
    this library follows Eq. 1. Eq. 4's second line and Eq. 5 sum over every site and
    all its neighbors, which counts each bond twice and gives the 24 M |t| of Eq. B6.
    Eqs. 6 and 7 double every D0, which gives a triply occupied site 3 C0 + 2 D0, where
    Eq. 1 and Eq. B9 have 3 C0 + D0.

    Args:
        couplings: Pionless contact couplings (t, C0, D0) at a fixed spacing.

    Returns:
        The coefficient of each kind of string, in the couplings' energy unit (MeV).
    """
    t = couplings.hopping_scale_mev
    c0 = couplings.two_body_mev
    d0 = couplings.three_body_mev
    return PauliCoefficients(
        hopping_mev=-t / 2.0,
        identity_mev=1.5 * c0 + 0.5 * d0,
        single_z_mev=-0.75 * (c0 + 0.5 * d0),
        pair_zz_mev=0.25 * (c0 + d0),
        triple_zzz_mev=-0.125 * d0,
    )


def hamiltonian_one_norm(site_count: int, couplings: PionlessCouplings) -> float:
    """LCU 1-norm lambda = sum_j lambda_j of the pionless Hamiltonian (Eq. C4).

    By Roggero+'s definition (Eqs. 19, C2, C4), lambda is the sum of |coefficient|
    over the Pauli strings actually block-encoded, the Gamma = 38 M strings of
    pauli_coefficients:

        lambda = M (12 |t|               # 24 hopping strings of |t| / 2
                    + 3 |C0 + D0 / 2|    # 4 Z strings
                    + 1.5 |C0 + D0|      # 6 ZZ strings
                    + 0.5 |D0|)          # 4 ZZZ strings

    The opposite signs of C0 < 0 and D0 > 0 cancel within these contact strings, so
    the sum is tighter than adding the sizes in the number operators' basis,
    M (6 |C0| + 4 |D0|). The identity term is a classical energy offset, excluded
    from lambda.

    Args:
        site_count: Number of lattice sites / orbitals M (at least 1).
        couplings: Pionless contact couplings (t, C0, D0) at a fixed spacing.

    Returns:
        The LCU 1-norm lambda in the couplings' energy unit (MeV).

    Raises:
        ValueError: If site_count < 1.
    """
    if site_count < 1:
        raise ValueError(
            f"hamiltonian_one_norm: site_count must be >= 1; got {site_count}."
        )
    strings = pauli_coefficients(couplings)
    one_site = (
        24 * abs(strings.hopping_mev)
        + 4 * abs(strings.single_z_mev)
        + 6 * abs(strings.pair_zz_mev)
        + 4 * abs(strings.triple_zzz_mev)
    )
    return site_count * one_site


def energy_spread_bound(
    nucleon_count: int,
    couplings: PionlessCouplings,
    saturation_binding_mev: float = _DEFAULT_SATURATION_BINDING_MEV,
) -> float:
    """Energy spread Delta-H = E_max - E_min of the pionless Hamiltonian (Eq. 12).

    Roggero+ Eq. 12 bounds the spread by the physical kinetic and potential norms
    plus the extensive binding energy:

        Delta-H <= ||K||_phys + ||V||_phys + A * b_max
        ||K||_phys = 3 A |t| pi**2                                # Eq. B7 (D = 3)
        ||V||_phys = max(|C0| floor(A/2),
                         |D0+3 C0| floor(A/3),
                         |4 D0+6 C0| floor(A/4))                  # Eq. B9

    with A = nucleon_count and b_max = saturation_binding_mev (so |E_min| <=
    A b_max). The tighter Eq. B9 potential exploits the opposite-sign C0/D0
    cancellation (the same physics as the Pauli-basis lambda), so the three
    branches are compared and the max taken; it is not a fixed branch.

    Units: returns Delta-H in MeV (every input coupling and b_max is in MeV). The
    QSVT block-encoding norm is the derived, *centered* operator norm Delta-H/2;
    pass that (in MeV) to ``qsvt_evolution_degree`` together with a MeV^-1 time
    horizon tau_max = 1/sigma. Delta-H bounds the A-particle sector and is
    independent of the site count M (no site_count parameter).

    Args:
        nucleon_count: Number of nucleons A (at least 1).
        couplings: Pionless contact couplings (t, C0, D0) at a fixed spacing.
        saturation_binding_mev: Binding energy per nucleon at saturation density
            b_max, in MeV (default 16.0, the standard nuclear-matter value); must
            be a finite number >= 0.

    Returns:
        The energy spread Delta-H = E_max - E_min in MeV.

    Raises:
        ValueError: If nucleon_count < 1, or saturation_binding_mev is not a
            finite number >= 0 (nan and inf raise).
    """
    if nucleon_count < 1:
        raise ValueError(
            f"energy_spread_bound: nucleon_count must be >= 1; got {nucleon_count}."
        )
    if not 0 <= saturation_binding_mev < math.inf:
        raise ValueError(
            "energy_spread_bound: saturation_binding_mev must be a finite number >= 0; "
            f"got {saturation_binding_mev}."
        )
    hopping = couplings.hopping_scale_mev
    two_body = couplings.two_body_mev
    three_body = couplings.three_body_mev
    kinetic = 3 * nucleon_count * abs(hopping) * math.pi**2
    potential = max(
        abs(two_body) * (nucleon_count // 2),
        abs(three_body + 3 * two_body) * (nucleon_count // 3),
        abs(4 * three_body + 6 * two_body) * (nucleon_count // 4),
    )
    return kinetic + potential + nucleon_count * saturation_binding_mev


def identity_coefficient(site_count: int, couplings: PionlessCouplings) -> float:
    """Coefficient of the identity in the pionless potential's Pauli expansion.

    Collecting the potential's Pauli strings on M sites (pauli_coefficients) leaves
    the identity with M (3/2 C0 + 1/2 D0), a classical energy offset that
    hamiltonian_one_norm excludes and the block encoding leaves out. It is one
    of the two constants the block encoding drops, the other being the kinetic
    term's 2 D t A (Roggero+ 2019 Eq. 1, D = 3), so the energies the phase estimation
    reads are shifted by minus this coefficient and by minus 2 D t A together
    (phase_estimation_energy_bound). It is negative at both tabulated spacings:
    -83.425 MeV per site at 1.4 fm, -2,252.475 MeV on the 27-site 4He lattice.

    Args:
        site_count: Number of lattice sites M; an integer >= 1.
        couplings: Pionless contact couplings (t, C0, D0) at a fixed spacing.

    Returns:
        The coefficient in the couplings' energy unit (MeV).

    Raises:
        ValueError: If site_count is not an integer >= 1 (a float raises, whole
            or not).
    """
    if type(site_count) is not int or site_count < 1:
        raise ValueError(
            "identity_coefficient: site_count must be an integer >= 1 (a lattice has "
            f"a whole number of sites, at least one); got {site_count!r}."
        )
    return site_count * pauli_coefficients(couplings).identity_mev


def potential_per_nucleon_bound(couplings: PionlessCouplings) -> float:
    """The largest size of potential energy one nucleon can carry, in MeV.

    Contacts act only among nucleons on one site, and a site holds at most four. A
    pair costs C0, a triple 3 C0 + D0 and a full site 6 C0 + 4 D0 (Roggero+ 2019 Eq. 1),
    so a nucleon's share of any arrangement is at most the largest of

        |C0| / 2,  |3 C0 + D0| / 3,  |6 C0 + 4 D0| / 4.

    Unlike the floors of Eq. B9, this also covers mixed arrangements, such as a pair
    beside a triple. Triples set it at both tabulated spacings: 55.617 MeV at 1.4 fm.

    Args:
        couplings: Pionless contact couplings (t, C0, D0) at a fixed spacing.

    Returns:
        The bound in the couplings' energy unit (MeV).
    """
    c0 = couplings.two_body_mev
    d0 = couplings.three_body_mev
    return max(
        abs(c0) / 2.0,
        abs(3.0 * c0 + d0) / 3.0,
        abs(6.0 * c0 + 4.0 * d0) / 4.0,
    )


def phase_estimation_energy_bound(
    nucleon_count: int,
    site_count: int,
    couplings: PionlessCouplings,
) -> float:
    """Bound on |E| over the energies the phase estimation reads, in MeV.

    The block encoding presents the pionless Hamiltonian without its two
    constants, the kinetic term's 2 D t A (Roggero+ 2019 Eq. 1, D = 3) and the
    potential's identity M (3/2 C0 + 1/2 D0) (identity_coefficient): its
    strings are the hopping terms and the potential's Z, ZZ and ZZZ strings, so
    the phase estimation reads E - 2 D t A - identity_coefficient(M) for each
    A-nucleon energy E of Eq. 1 (E + 1,998.6 MeV for 4He on 27 sites at a = 1.4 fm,
    where 2 D t A = 253.9 MeV). On A nucleons the triangle inequality bounds what is
    read:

        |E_read| <= A (6 |t| + v) + |identity_coefficient(M)|.

    A nucleon's hop is -2 t (cos k_x + cos k_y + cos k_z), at most 6 |t| in size, and
    its share of the potential is at most v = potential_per_nucleon_bound. The hop's
    bound is this lattice's own. The continuum kinetic norm of Eq. B7, 3 pi^2 |t| per
    nucleon, would put the bound above lambda on the smallest lattices, where it says
    nothing, though every energy lies within lambda. For 4He on 27 sites at a = 1.4 fm
    (A = 4, M = 27) the value is 476.39 + 2,252.475 = 2,728.86 MeV. It is the
    spectral_norm_bound of lookup_keep_probability_bits under the library's reading, the
    largest |E| among the energies read rather than the norm over the whole space.

    Args:
        nucleon_count: Number of nucleons A; an integer >= 1.
        site_count: Number of lattice sites M; an integer >= 1.
        couplings: Pionless contact couplings (t, C0, D0) at a fixed spacing.

    Returns:
        The bound in MeV.

    Raises:
        ValueError: If nucleon_count or site_count is not an integer >= 1 (a
            float raises, whole or not).
    """
    if type(nucleon_count) is not int or nucleon_count < 1:
        raise ValueError(
            "phase_estimation_energy_bound: nucleon_count must be an integer >= 1 (a "
            f"nucleus has a whole number of nucleons, at least one); got "
            f"{nucleon_count!r}."
        )
    if type(site_count) is not int or site_count < 1:
        raise ValueError(
            "phase_estimation_energy_bound: site_count must be an integer >= 1 (a "
            f"lattice has a whole number of sites, at least one); got {site_count!r}."
        )
    hop = 6.0 * abs(couplings.hopping_scale_mev)
    share = potential_per_nucleon_bound(couplings)
    shift = abs(identity_coefficient(site_count, couplings))
    return nucleon_count * (hop + share) + shift
