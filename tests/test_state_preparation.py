"""Tests for the sparse-state load of Fomichev+ 2023."""

import itertools
import math

import pytest

from qec_utility_estimation.state_preparation import (
    SparseStateLoadCost,
    TranslationSymmetrizationCost,
    sparse_state_load_cost,
    translation_symmetrization_cost,
)


def test_complete_load_reproduces_the_counts_at_three_ground_state_sizes() -> None:
    """The complete load at a small table's size and at the two largest 4He sizes.

    Fomichev+ 2023 (arXiv:2310.18410, PRX Quantum 5, 040339 (2024)) load D
    configurations through an enumeration register of L = ceil(log2 D) qubits, for
    (2L - 2) D + 2**(L + 1) + D Toffolis (their Eq. 17), plus (c - 3) L for the L
    rotations of c-bit angles (their App. D). The ancillas are the 5 L - 3 of their
    Eq. 18 and the c-qubit phase-gradient state those rotations assume; the c-bit
    angle register fits in the identification register, 2L - 1 qubits idle until
    step 3, at all three sizes. With 16-bit angles:

              D    L     leading  rotation  ancillas
            600   10      13,448       130        63
         19,683   15     636,343       195        88
        531,441   20  22,823,351       260       113

    the leading counts being 18 * 600 + 2048 + 600, 28 * 19,683 + 65,536 + 19,683
    and 38 * 531,441 + 2,097,152 + 531,441.

    Each D is the number of amplitudes loaded. D = 600 stands for a small sparse
    table. The other two are sizes of the 4He ground state as four distinguishable
    nucleons on a 27-site lattice:
    531,441 = 27**4 is their whole product space, and 19,683 = 27**3 is the
    dimension of its zero-momentum block. That dimension is not the number of site
    configurations the block's states occupy: each basis state of the block is an
    equal-weight sum over the 27 translates of one configuration, so in the site
    basis the block's ground state is spread over all 531,441. A load from 19,683
    amplitudes therefore needs a symmetrization over the 27 translations, which
    this count leaves out.

    None of the three sizes is a power of two, so the ceiling of log2 D is in play
    at every row. The whole result is compared at each size, and the total at the
    middle one: 636,538 Toffolis.
    """
    assert sparse_state_load_cost(600, rotation_bits=16) == SparseStateLoadCost(
        leading_toffoli_count=13_448, rotation_toffoli_count=130, ancilla_qubits=63
    )
    middle = sparse_state_load_cost(19_683, rotation_bits=16)
    assert middle == SparseStateLoadCost(
        leading_toffoli_count=636_343, rotation_toffoli_count=195, ancilla_qubits=88
    )
    assert middle.toffoli_count == 636_538
    assert sparse_state_load_cost(531_441, rotation_bits=16) == SparseStateLoadCost(
        leading_toffoli_count=22_823_351, rotation_toffoli_count=260, ancilla_qubits=113
    )


@pytest.mark.parametrize(
    ("num_configurations", "leading", "rotation", "ancillas"),
    [
        (2, 6, 1, 10),
        (1024, 21_504, 10, 51),
        (2**20, 42_991_616, 20, 101),
    ],
)
def test_complete_load_equals_two_l_plus_one_per_configuration_at_powers_of_two(
    num_configurations: int, leading: int, rotation: int, ancillas: int
) -> None:
    """At D = 2**L the leading count reduces to (2L + 1) D exactly.

    With D a power of two, 2**(L + 1) = 2 D, so Eq. (17)'s (2L - 2) D + 2**(L + 1)
    + D is (2L + 1) D: 3 * 2 = 6 at D = 2 (L = 1), 21 * 1024 = 21,504 at L = 10,
    and 41 * 2**20 = 42,991,616 at L = 20. These are the only sizes at which the
    register width ceil(log2 D) differs from floor(log2 D) + 1, so they pin the
    rounding. At rotation_bits = 4, the smallest accepted, the rotation count is
    (4 - 3) L = L. The ancillas are Eq. (18)'s 5 L - 3 and the 4-qubit
    phase-gradient state, 47 + 4 = 51 and 97 + 4 = 101; at D = 2 the
    identification register is a single qubit, too small for the 4-bit angle
    register, which adds 4 more: 2 + 4 + 4 = 10. D = 2 is the smallest load the
    function accepts. The whole result is compared.
    """
    assert sparse_state_load_cost(
        num_configurations, rotation_bits=4
    ) == SparseStateLoadCost(
        leading_toffoli_count=leading,
        rotation_toffoli_count=rotation,
        ancilla_qubits=ancillas,
    )


def test_complete_load_grows_with_each_configuration_below_the_sources_bound() -> None:
    """The leading count across D = 2 to 5000, and where it jumps.

    For every D from 2 to 5000 the leading count lies strictly between (2L - 1) D,
    which the formula implies (its 2**(L + 1) term is positive), and the source's
    own bound (2L + 3) D (that term is below 4 D, because 2**(L - 1) < D), and it
    grows with every configuration added. Where the enumeration register grows,
    from D = 1024 (L = 10) to 1025 (L = 11), it jumps from 21 * 1024 = 21,504 to
    20 * 1025 + 4096 + 1025 = 25,621. That jump is the one check that separates the
    source's 2**(L + 1) from the 2 D it equals at every power of two. A sweep, so a
    rounding or sign error at a register boundary cannot hide between pinned values.
    """
    sizes = range(2, 5001)
    counts = [
        sparse_state_load_cost(size, rotation_bits=16).leading_toffoli_count
        for size in sizes
    ]

    for size, count in zip(sizes, counts, strict=True):
        register = math.ceil(math.log2(size))
        assert (2 * register - 1) * size < count < (2 * register + 3) * size
    assert all(later > earlier for earlier, later in itertools.pairwise(counts))
    assert counts[1024 - 2] == 21_504
    assert counts[1025 - 2] == 25_621


def test_complete_load_adds_one_toffoli_per_register_qubit_per_angle_bit() -> None:
    """Each extra bit of rotation angle adds one Toffoli per enumeration qubit.

    Step 1 applies L = ceil(log2 D) rotations, one for each qubit of the
    enumeration register, each a phase-gradient addition of a c-bit angle at c - 3
    Toffolis (Fomichev+ App. D). At D = 19,683 (L = 15) the rotation count is 15,
    30 and 195 at 4, 5 and 16 bits, 15 per bit. The leading count does not depend
    on the angles' precision. The ancillas grow by one qubit per bit, the
    phase-gradient state's, while the angle register fits in the 29-qubit
    identification register: 72 + 4, 72 + 5 and 72 + 16.
    """
    costs = [sparse_state_load_cost(19_683, rotation_bits=bits) for bits in (4, 5, 16)]

    assert [cost.rotation_toffoli_count for cost in costs] == [15, 30, 195]
    assert {cost.leading_toffoli_count for cost in costs} == {636_343}
    assert [cost.ancilla_qubits for cost in costs] == [76, 77, 88]


def test_complete_load_borrows_the_idle_identification_register_for_its_angles() -> (
    None
):
    """Step 1's angle register sits in the identification register while it fits.

    Each step-1 rotation adds a c-bit angle into the phase-gradient state, and the
    angle needs c qubits of its own (Fomichev+ App. E: the qubits "to store the
    phase"). The identification register, 2L - 1 qubits, holds nothing until step
    3, so the angle sits there while c <= 2L - 1 and costs no qubit; beyond that
    it takes c qubits. At D = 19,683 (L = 15, an identification register of 29
    qubits), 29-bit angles fit, 72 + 29 = 101 ancillas with the phase-gradient
    state, and 30-bit angles do not, 72 + 30 + 30 = 132.
    """
    assert sparse_state_load_cost(19_683, rotation_bits=29).ancilla_qubits == 101
    assert sparse_state_load_cost(19_683, rotation_bits=30).ancilla_qubits == 132


def test_complete_load_rejects_unusable_inputs() -> None:
    """Fewer than two configurations, or angles of fewer than four bits, fail fast.

    One configuration is a single bitstring, written by NOT gates; the closed forms
    there give 1 Toffoli and -3 ancillas, and below one there is nothing to load. A
    count that is not an integer -- a whole-valued float, a NaN, or True, a bool
    equal to 1 -- raises as well. Below four bits the rotation term (c - 3) L is
    zero or negative, and a float is not a number of bits. Each error names its
    argument. The angle precision has no default: a call without it raises
    TypeError naming rotation_bits, because the library does not choose it. A bare
    second number would not say at the call site that it is a number of bits, so
    passing the precision by position raises TypeError too.
    """
    with pytest.raises(ValueError, match="num_configurations must be an integer"):
        sparse_state_load_cost(1, rotation_bits=16)
    with pytest.raises(ValueError, match="num_configurations must be an integer"):
        sparse_state_load_cost(0, rotation_bits=16)
    with pytest.raises(ValueError, match="num_configurations must be an integer"):
        sparse_state_load_cost(-1, rotation_bits=16)
    with pytest.raises(ValueError, match="num_configurations must be an integer"):
        sparse_state_load_cost(19683.0, rotation_bits=16)
    with pytest.raises(ValueError, match="num_configurations must be an integer"):
        sparse_state_load_cost(math.nan, rotation_bits=16)
    with pytest.raises(ValueError, match="num_configurations must be an integer"):
        sparse_state_load_cost(True, rotation_bits=16)
    with pytest.raises(ValueError, match="rotation_bits must be an integer"):
        sparse_state_load_cost(19_683, rotation_bits=3)
    with pytest.raises(ValueError, match="rotation_bits must be an integer"):
        sparse_state_load_cost(19_683, rotation_bits=0)
    with pytest.raises(ValueError, match="rotation_bits must be an integer"):
        sparse_state_load_cost(19_683, rotation_bits=16.0)
    with pytest.raises(TypeError, match="rotation_bits"):
        sparse_state_load_cost(19_683)
    with pytest.raises(TypeError, match="positional"):
        sparse_state_load_cost(19_683, 16)


def test_pinned_load_and_spreading_undercut_the_direct_load_for_helium() -> None:
    """Loading 4He with one nucleon pinned, then spreading it, beats the direct load.

    The zero-momentum ground state of 4He on the 3**3 lattice has 27**4 = 531,441
    amplitudes in the site basis, but only 27**3 = 19,683 of them are independent:
    the state is the sum of the 27 lattice translates of the state with one
    nucleon pinned at the origin. The pinned state can be loaded by the complete
    load and then spread over the translations (translation_symmetrization_cost on
    a periodic 3**3 lattice with four modes per site), which makes the
    zero-momentum state exactly. At 16 angle bits, a stated input, the pinned
    load and its spreading cost under 3% of the direct load of all 531,441
    amplitudes in Toffolis, and the symmetrization under 0.1% of the pinned load it
    completes. Both functions count Toffolis in the same field, so the ratios are
    read from their composition.
    """
    pinned_load = sparse_state_load_cost(19_683, rotation_bits=16).toffoli_count
    direct_load = sparse_state_load_cost(531_441, rotation_bits=16).toffoli_count
    symmetrization = translation_symmetrization_cost(3, 4).toffoli_count

    assert symmetrization < 0.001 * pinned_load
    assert pinned_load + symmetrization < 0.03 * direct_load


def test_helium_symmetrization_is_swaps_signs_and_three_superpositions() -> None:
    """Spreading the pinned 4He state over the 3**3 lattice costs 470 Toffolis.

    On a periodic 3**3 lattice of four Jordan-Wigner modes per site, the shift
    register holds three two-qubit axis registers, each in an equal superposition
    over three values (two rotations and four AND computations apiece, twelve
    ANDs and six rotations in all). Each axis register's two bits control a cyclic
    shift of the lattice by one and by two sites along that axis. A shift is a
    qubit permutation of 2 x 36 = 72 controlled swaps per axis (n - gcd(n, k) = 2
    block transpositions on each of the lines, the lines holding n**2 m = 36
    modes between them), 432 over three axes and two shifts, plus a fermionic sign
    per line and shift, the product of two occupation parities applied as one
    controlled-controlled-Z: 9 + 3 + 1 = 13 lines, 26 over the two shifts. The
    register's erasure is Clifford. Qubits: six for the register and two for the
    parities.
    """
    cost = translation_symmetrization_cost(3, 4)

    assert cost == TranslationSymmetrizationCost(
        swap_toffoli_count=432,
        sign_toffoli_count=26,
        superposition_toffoli_count=12,
        rotation_count=6,
        ancilla_qubits=8,
    )
    assert cost.toffoli_count == 470


@pytest.mark.parametrize(
    ("sites_per_axis", "swaps", "signs", "superposition", "rotations", "qubits"),
    [
        (2, 12, 7, 0, 0, 5),
        (3, 108, 26, 12, 6, 8),
        (4, 240, 42, 0, 0, 8),
        (5, 900, 93, 18, 6, 11),
        (6, 1_404, 129, 12, 6, 11),
    ],
)
def test_symmetrization_follows_the_cyclic_shifts_of_each_lattice(
    *,
    sites_per_axis: int,
    swaps: int,
    signs: int,
    superposition: int,
    rotations: int,
    qubits: int,
) -> None:
    """Each lattice shape reaches a branch of the count, at one mode per site.

    An axis register of ceil(log2 n) bits controls shifts by 1, 2, 4, ... sites;
    a shift by k is n - gcd(n, k) block transpositions on each line, the lines of
    an axis holding n**2 modes between them, so the swaps are
    3 n**2 sum_k (n - gcd(n, k)), the signs (n**2 + n + 1) per shift, and the
    superpositions three copies of Babbush+ Fig. 12 over n. n = 2 has one bit and
    a power-of-two superposition; n = 4 has gcd(4, 2) = 2, which halves the
    second shift's transpositions; n = 5 has three bits and an odd part of 5;
    n = 6 is the one shape that combines an odd part above one with gcd(6, 2) =
    gcd(6, 4) = 2. n = 3 at one mode per site, against helium's 470 at four,
    shows that only the swaps scale with the modes per site: 108 against 432,
    with the signs and the superpositions unchanged.
    """
    cost = translation_symmetrization_cost(sites_per_axis, 1)

    assert cost == TranslationSymmetrizationCost(
        swap_toffoli_count=swaps,
        sign_toffoli_count=signs,
        superposition_toffoli_count=superposition,
        rotation_count=rotations,
        ancilla_qubits=qubits,
    )


def test_symmetrization_of_a_single_site_per_axis_costs_nothing() -> None:
    """A lattice of one site per axis has no translations to spread over.

    The zero-momentum state and the pinned state are the same state, so nothing
    is shifted, no sign is applied, no register is prepared, and no qubit is
    held, the two parity qubits included: they serve sign gates that never run.
    """
    assert translation_symmetrization_cost(1, 4) == TranslationSymmetrizationCost(
        swap_toffoli_count=0,
        sign_toffoli_count=0,
        superposition_toffoli_count=0,
        rotation_count=0,
        ancilla_qubits=0,
    )


@pytest.mark.parametrize(
    ("sites_per_axis", "modes_per_site", "message"),
    [
        (0, 4, "sites_per_axis must be an integer >= 1"),
        (-1, 4, "sites_per_axis must be an integer >= 1"),
        (3.0, 4, "sites_per_axis must be an integer >= 1"),
        (True, 4, "sites_per_axis must be an integer >= 1"),
        (3, 0, "modes_per_site must be an integer >= 1"),
        (3, 4.0, "modes_per_site must be an integer >= 1"),
    ],
)
def test_symmetrization_rejects_unusable_inputs(
    sites_per_axis: float, modes_per_site: float, message: str
) -> None:
    """A lattice has a whole number of sites per axis and of modes per site.

    Zero or negative counts describe no lattice; a whole-valued float is not an
    integer and raises before it reaches the gcd, and so does True, a bool that
    equals 1. Each error names the argument.
    """
    with pytest.raises(ValueError, match=message):
        translation_symmetrization_cost(
            sites_per_axis,  # type: ignore[arg-type]
            modes_per_site,  # type: ignore[arg-type]
        )
