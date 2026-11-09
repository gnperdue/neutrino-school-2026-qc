"""Tests for the pionless-EFT Hamiltonian norm inputs (Roggero+ 2019)."""

import itertools
import math

import pytest

from qec_utility_estimation.apps.pionless_first_order.couplings import (
    PionlessCouplings,
    reference_couplings_at_spacing,
)
from qec_utility_estimation.apps.pionless_first_order.norms import (
    energy_spread_bound,
    hamiltonian_one_norm,
    identity_coefficient,
    pauli_coefficients,
    phase_estimation_energy_bound,
    potential_per_nucleon_bound,
)


def contact_energy(occupations: tuple[int, ...], couplings: PionlessCouplings) -> float:
    """Roggero+ Eq. 1's contact energy of a site: C0 a pair, D0 a triple, in MeV."""
    nucleons = sum(occupations)
    return (
        math.comb(nucleons, 2) * couplings.two_body_mev
        + math.comb(nucleons, 3) * couplings.three_body_mev
    )


@pytest.mark.parametrize("spacing_fm", [1.4, 2.2])
def test_the_pauli_strings_reproduce_the_contact_energy_at_every_occupation(
    spacing_fm: float,
) -> None:
    """The Z strings must give Eq. 1's energy on all 16 occupations of a site's modes.

    Eq. 1 charges C0 for each pair of nucleons on a site and D0 for each triple, so
    k nucleons cost C(k,2) C0 + C(k,3) D0: 3 C0 + D0 on a triply occupied site, 6 C0
    + 4 D0 on a full one. Each mode's Z is 1 - 2 n. This is the physics a coefficient
    must answer to, whatever form it is printed in, so the test computes the energy
    from the occupations and never from the coefficients' formula.
    """
    couplings = reference_couplings_at_spacing(spacing_fm)
    strings = pauli_coefficients(couplings)

    for occupations in itertools.product((0, 1), repeat=4):
        z = []
        for occupation in occupations:
            z.append(1 - 2 * occupation)
        total = strings.identity_mev
        for first in range(4):
            total += strings.single_z_mev * z[first]
        for first, second in itertools.combinations(range(4), 2):
            total += strings.pair_zz_mev * z[first] * z[second]
        for first, second, third in itertools.combinations(range(4), 3):
            total += strings.triple_zzz_mev * z[first] * z[second] * z[third]

        assert total == pytest.approx(contact_energy(occupations, couplings))


@pytest.mark.parametrize("spacing_fm", [1.4, 2.2])
def test_a_bonds_two_strings_give_a_nucleon_its_kinetic_energy(
    spacing_fm: float,
) -> None:
    """A nucleon at rest has no kinetic energy, and a slow one has t k^2.

    Here k is in lattice units.
    On one bond, X Z...Z X and Y Z...Z Y each carry a nucleon across with amplitude 1,
    so a coefficient c on both is a hop of 2 c. With Eq. 1's constant 2 D t per
    nucleon, a plane wave then has the energy

        E(k) = 6 t + 2 (2 c) (cos k_x + cos k_y + cos k_z).

    The physics fixes c: E(0) = 0, and E(k) tends to t k^2 = hbar^2 k^2 / (2 M), the
    energy of a nucleon of mass M, since t = hbar^2 / (2 M a^2). Strings of size |t|
    would fail both, as a nucleon of half the mass.
    """
    couplings = reference_couplings_at_spacing(spacing_fm)
    t = couplings.hopping_scale_mev
    hop = 2.0 * pauli_coefficients(couplings).hopping_mev

    at_rest = 6.0 * t + 2.0 * hop * 3.0
    small_k = 0.01
    slow = 6.0 * t + 2.0 * hop * (math.cos(small_k) + 2.0)

    assert at_rest == pytest.approx(0.0, abs=1e-12)
    assert slow == pytest.approx(t * small_k**2, rel=1e-4)


def test_one_norm_sums_the_sizes_of_the_pauli_strings() -> None:
    """lambda = M (24 |h| + 4 |z| + 6 |zz| + 4 |zzz|), 338.225 MeV a site at 1.4 fm.

    Written out, lambda = M (12 |t| + 3 |C0 + D0/2| + 1.5 |C0 + D0| + 0.5 |D0|). It is
    checked at both tabulated spacings, which have different kinetic to potential
    ratios, so a factor error that one total could hide shows in the pair: lambda(20)
    = 6,764.5 MeV at 1.4 fm and 2,660.4 MeV at 2.2 fm. The last assertion pins the
    basis. Adding the sizes in the number operators' basis, 12 M |t| + M (6 |C0| +
    4 |D0|) = 24,554.0 MeV, forgoes the cancellation between C0 < 0 and D0 > 0, so
    lambda must not equal it.
    """
    couplings_14 = reference_couplings_at_spacing(1.4)
    couplings_22 = reference_couplings_at_spacing(2.2)
    strings = pauli_coefficients(couplings_14)

    one_norm_14 = hamiltonian_one_norm(20, couplings_14)
    one_norm_22 = hamiltonian_one_norm(20, couplings_22)

    assert one_norm_14 == pytest.approx(
        20
        * (
            24 * abs(strings.hopping_mev)
            + 4 * abs(strings.single_z_mev)
            + 6 * abs(strings.pair_zz_mev)
            + 4 * abs(strings.triple_zzz_mev)
        )
    )
    assert one_norm_14 == pytest.approx(6764.5)
    assert one_norm_22 == pytest.approx(2660.4)

    number_basis_one_norm = 12 * 20 * abs(couplings_14.hopping_scale_mev) + 20 * (
        6 * abs(couplings_14.two_body_mev) + 4 * abs(couplings_14.three_body_mev)
    )
    assert number_basis_one_norm == pytest.approx(24554.0)
    assert one_norm_14 != pytest.approx(number_basis_one_norm)


def test_one_norm_scales_linearly_in_site_count() -> None:
    """lambda is linear in M: every block-encoded weight scales with the count.

    Both the Gamma = 38 M Pauli strings and their per-site coefficients are
    M-independent, so doubling the site count doubles lambda exactly:
    lambda(40) = 2 lambda(20). The 40Ar lattice point lambda(1000) = 338225 MeV
    (the value fed to the QPE counter in the end-to-end test) is triangulated here
    at the unit level, guarding the linear scaling between M = 20 and the M = 1000
    lattice point against any spurious constant offset.
    """
    couplings = reference_couplings_at_spacing(1.4)

    one_norm_20 = hamiltonian_one_norm(20, couplings)
    one_norm_40 = hamiltonian_one_norm(40, couplings)
    one_norm_1000 = hamiltonian_one_norm(1000, couplings)

    assert one_norm_40 == pytest.approx(2 * one_norm_20)
    assert one_norm_1000 == pytest.approx(338225.0)


def test_energy_spread_bound_sums_physical_norms_and_binding() -> None:
    """Delta-H = ||K||_phys + ||V||_phys + A*b_max for 40Ar at a = 1.4 fm.

    The spread bound (Eq. 12) sums the physical kinetic norm 3 A |t| pi**2 (Eq.
    B7), the tighter physical potential norm max{n2, n3, n4} (Eq. B9), and the
    extensive binding A*b_max. At A = 40, a = 1.4 fm with the default b_max = 16
    MeV the kinetic is 3*40*10.58*pi**2 = 12530.45, the potential max selects
    n3 = |D0+3C0|*floor(40/3) = 166.85*13 = 2169.05, and the binding is 40*16 =
    640, so Delta-H = 15339.5 MeV. A single hand-derived value, independent of the
    implementation's expression, with the tolerance absorbing the pi**2 precision;
    the n3 branch selection is pinned separately by the max-branch test below.
    """
    couplings = reference_couplings_at_spacing(1.4)

    assert energy_spread_bound(40, couplings) == pytest.approx(15339.5, abs=0.1)


@pytest.mark.parametrize(
    ("nucleon_count", "expected_potential_norm"),
    [(4, 196.46), (40, 2169.05)],
)
def test_energy_spread_potential_selects_tighter_max_branch(
    nucleon_count: int, expected_potential_norm: float
) -> None:
    """||V||_phys is the max over the three Eq. B9 branches, not a fixed one.

    Eq. B9 bounds the physical potential norm by max{n2, n3, n4} with
    n2=|C0|floor(A/2), n3=|D0+3C0|floor(A/3), n4=|4D0+6C0|floor(A/4); which branch
    is tightest depends on A. At A=4 the floors are (2,1,1) and n2=98.23*2=196.46
    wins; at A=40 the floors are (20,13,10) and n3=166.85*13=2169.05 wins.
    Recovered by subtracting the kinetic and binding from the spread, the two
    nucleon numbers prove a genuine max() rather than a hardcoded branch. (n4
    never wins for these couplings.)
    """
    couplings = reference_couplings_at_spacing(1.4)
    kinetic = 3 * nucleon_count * couplings.hopping_scale_mev * math.pi**2
    binding = nucleon_count * 16.0

    spread = energy_spread_bound(nucleon_count, couplings)

    assert spread - kinetic - binding == pytest.approx(expected_potential_norm)


def test_norm_functions_reject_nonphysical_inputs() -> None:
    """Site/nucleon counts below 1 and negative binding fail fast.

    Both norms are extensive in a positive particle count: M >= 1 sites for the
    1-norm and A >= 1 nucleons for the spread, so counts below 1 are unphysical
    and must raise. The saturation binding b_max is a nonnegative energy; a
    negative value would push the spread below its kinetic+potential floor and
    must also raise.
    """
    couplings = reference_couplings_at_spacing(1.4)

    with pytest.raises(ValueError, match="site_count"):
        hamiltonian_one_norm(0, couplings)
    with pytest.raises(ValueError, match="site_count"):
        hamiltonian_one_norm(-1, couplings)
    with pytest.raises(ValueError, match="nucleon_count"):
        energy_spread_bound(0, couplings)
    with pytest.raises(ValueError, match="nucleon_count"):
        energy_spread_bound(-3, couplings)
    with pytest.raises(ValueError, match="saturation_binding_mev"):
        energy_spread_bound(40, couplings, saturation_binding_mev=-1.0)


def test_identity_coefficient_is_the_contact_energy_averaged_over_the_occupations() -> (
    None
):
    """The identity term is the average of a site's contact energy, M (3/2 C0 + 1/2 D0).

    Every Z string averages to zero over a site's 16 occupations, so the identity's
    coefficient, a classical energy offset the block encoding leaves out, is what is
    left of Eq. 1's contact energy. The test computes that average directly: at 1.4 fm,
    -83.425 MeV per site and -2,252.475 MeV on the 27-site 4He lattice; at 2.2 fm,
    -39.03 MeV per site. The coefficient is linear in M.
    """
    couplings_14 = reference_couplings_at_spacing(1.4)
    couplings_22 = reference_couplings_at_spacing(2.2)

    total_14 = 0.0
    for occupations in itertools.product((0, 1), repeat=4):
        total_14 += contact_energy(occupations, couplings_14)

    per_site_14 = identity_coefficient(1, couplings_14)
    per_site_22 = identity_coefficient(1, couplings_22)

    assert per_site_14 == pytest.approx(total_14 / 16)
    assert per_site_14 == pytest.approx(-83.425)
    assert identity_coefficient(27, couplings_14) == pytest.approx(-2252.475)
    assert per_site_22 == pytest.approx(-39.03)
    assert identity_coefficient(40, couplings_22) == pytest.approx(40 * per_site_22)


def test_energy_bound_sums_the_hop_the_contacts_and_the_shift() -> None:
    """|E| over the energies the phase estimation reads is at most A (6 t + v) + |I|.

    The block encoding drops the Hamiltonian's two constants, the kinetic term's
    2 D t A and the potential's identity I = M (3/2 C0 + 1/2 D0), so it reads the
    hopping plus the potential minus I. On A nucleons the triangle inequality bounds
    each: a nucleon's hop is -2 t sum_d cos k_d, at most 6 t in size; its share of the
    potential is at most v, the largest of |C0| / 2, |3 C0 + D0| / 3 and
    |6 C0 + 4 D0| / 4. For 4He on 27 sites at a = 1.4 fm (A = 4, M = 27) that is
    4 (63.48 + 55.617) + 2,252.475 = 2,728.86 MeV, the spectral bound the keep-bit
    count takes. The bound grows with A (A = 12 at M = 27 is larger) and with M through
    the shift (A = 4 at M = 64 is larger by the 37 extra sites' 83.425 MeV each).
    """
    couplings = reference_couplings_at_spacing(1.4)

    helium = phase_estimation_energy_bound(4, 27, couplings)

    assert helium == pytest.approx(2728.86, abs=0.01)
    assert phase_estimation_energy_bound(12, 27, couplings) > helium
    assert phase_estimation_energy_bound(4, 64, couplings) == pytest.approx(
        helium + 37 * 83.425
    )


@pytest.mark.parametrize("spacing_fm", [1.4, 2.2])
def test_no_arrangement_of_nucleons_exceeds_the_potential_per_nucleon(
    spacing_fm: float,
) -> None:
    """Every way of sharing up to 16 nucleons among sites stays within v per nucleon.

    An arrangement's potential energy is the sum of its sites' contact energies, with
    at most four nucleons on a site. The test walks every arrangement, by the counts of
    sites holding two, three and four nucleons, the rest sitting alone at no cost. v
    must cover them all, including mixed ones such as a pair beside a triple at A = 5,
    which the floors of Roggero+ Eq. B9 miss, and some arrangement must reach it, so
    that v is no looser than it has to be.
    """
    couplings = reference_couplings_at_spacing(spacing_fm)
    pair_energy = contact_energy((1, 1), couplings)
    triple_energy = contact_energy((1, 1, 1), couplings)
    full_energy = contact_energy((1, 1, 1, 1), couplings)
    per_nucleon = potential_per_nucleon_bound(couplings)

    largest_share = 0.0
    for nucleons in range(1, 17):
        for pairs in range(nucleons // 2 + 1):
            for triples in range((nucleons - 2 * pairs) // 3 + 1):
                for fulls in range((nucleons - 2 * pairs - 3 * triples) // 4 + 1):
                    energy = (
                        pairs * pair_energy
                        + triples * triple_energy
                        + fulls * full_energy
                    )
                    share = abs(energy) / nucleons
                    assert share <= per_nucleon + 1e-9
                    largest_share = max(largest_share, share)

    assert largest_share == pytest.approx(per_nucleon)


@pytest.mark.parametrize(
    ("nucleon_count", "site_count"), [(4, 8), (12, 8), (12, 27), (40, 27), (40, 64)]
)
def test_the_energy_bound_stays_below_lambda_on_small_lattices(
    nucleon_count: int, site_count: int
) -> None:
    """A bound on what the walk can read must stay below lambda, however small the box.

    Every energy of a Hamiltonian lies within its one-norm, and the keep-bit formula
    diverges at ||H|| = lambda, so a bound at or above lambda says nothing and stops the
    calculation. These are the smallest lattices that hold helium, carbon and argon.
    """
    couplings = reference_couplings_at_spacing(1.4)

    bound = phase_estimation_energy_bound(nucleon_count, site_count, couplings)

    assert bound < hamiltonian_one_norm(site_count, couplings)


@pytest.mark.parametrize(
    ("nucleon_count", "site_count", "message"),
    [
        (4, 0, "site_count must be an integer >= 1"),
        (4, 27.0, "site_count must be an integer >= 1"),
        (0, 27, "nucleon_count must be an integer >= 1"),
        (4.0, 27, "nucleon_count must be an integer >= 1"),
    ],
)
def test_energy_bound_rejects_unusable_counts(
    nucleon_count: float, site_count: float, message: str
) -> None:
    """Nucleon and site counts are Python integers of at least one.

    A zero count describes no system, and a whole-valued float is not an integer
    and raises, so an integer taken from a numpy array cannot pass silently. The
    identity coefficient checks the site count the same way. Each error names the
    argument.
    """
    couplings = reference_couplings_at_spacing(1.4)

    with pytest.raises(ValueError, match=message):
        phase_estimation_energy_bound(
            nucleon_count,  # type: ignore[arg-type]
            site_count,  # type: ignore[arg-type]
            couplings,
        )
    if "site_count" in message:
        with pytest.raises(ValueError, match=message):
            identity_coefficient(site_count, couplings)  # type: ignore[arg-type]


@pytest.mark.parametrize("binding", [-1.0, math.nan, math.inf])
def test_energy_spread_rejects_an_unusable_saturation_binding(binding: float) -> None:
    """b_max is a finite number >= 0.

    A negative, nan or infinite saturation binding raises: nan passes a plain "< 0"
    test and would come back as a nan spread. The error names the argument.
    """
    couplings = reference_couplings_at_spacing(1.4)

    with pytest.raises(
        ValueError, match="saturation_binding_mev must be a finite number >= 0"
    ):
        energy_spread_bound(4, couplings, binding)
