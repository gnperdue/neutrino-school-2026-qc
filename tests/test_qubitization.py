"""Tests for the Roggero+ 2019 App. C controlled-qubiterate gate-cost model."""

import math

import pytest

from qec_utility_estimation.apps.pionless_first_order.couplings import (
    reference_couplings_at_spacing,
)
from qec_utility_estimation.apps.pionless_first_order.norms import (
    energy_spread_bound,
    hamiltonian_one_norm,
    phase_estimation_energy_bound,
)
from qec_utility_estimation.qubitization import (
    EqualSuperpositionCost,
    LookupPrepareCost,
    PerShotCost,
    QubiterateCost,
    QubiterateQpeCost,
    controlled_qubiterate_cost,
    controlled_reflection_toffoli_count,
    equal_superposition_cost,
    jacobi_anger_truncation_error,
    lcu_ancilla_register_size,
    lookup_keep_probability_bits,
    per_shot_t_cost,
    prepare_rotation_count,
    qsvt_evolution_degree,
    qubiterate_qpe_ancilla_count,
    reflection_toffoli_count,
    table_lookup_prepare_t_count,
    translation_invariant_lookup_prepare_cost,
    two_sided_qpe_application_count,
    two_sided_qpe_rms_energy_error,
    unary_iteration_select_toffoli_count,
)
from qec_utility_estimation.synthesis import (
    coherent_synthesis_precision,
    repeat_until_success_t_count,
)


@pytest.mark.parametrize(
    ("num_terms", "expected_register_size"),
    [
        (1, 0),
        (2, 1),
        (3, 2),
        (4, 2),
        (512, 9),
        (513, 10),
        (1024, 10),
        (1025, 11),
    ],
)
def test_ancilla_register_size_ceils_log2_at_domain_floor_and_boundaries(
    num_terms: int, expected_register_size: int
) -> None:
    """The LCU index register is ceil(log2(num_terms)), exact across the domain.

    The qubiterate's whole gate-count ledger keys on N_A = ceil(log2(Gamma))
    through the 2**(N_A-1) leading term, so an off-by-one at an exact power of
    two corrupts every count. A float log2 can round 1024 to 1023.999... or
    1024.000...1; the boundaries 512->9, 513->10, 1024->10, 1025->11 pin the
    exact integer ceiling. The small-Gamma values 1->0, 2->1, 3->2, 4->2 pin the
    returns at the domain floor (where this helper is exercised only indirectly
    via the cost-function reject path).
    """
    assert lcu_ancilla_register_size(num_terms) == expected_register_size


@pytest.mark.parametrize("num_terms", [5, 760, 1026, 38000, 40_000_000])
def test_qubiterate_counts_equal_the_sum_of_prepare_reflection_and_select(
    num_terms: int,
) -> None:
    """The App. C.1 T and CNOT totals are the sums of their listed components.

    The source prices one controlled qubiterate as one prepare, one controlled
    reflection and one select, and prints the component counts beside the totals.
    Rebuilding a total from the component lines checks one line of the source
    against another. The CNOT line is the only such check of a leading
    coefficient: the source writes the select's leading term as an implicit
    product, 15 x 2**(N_A-1), and a misreading of it changes the CNOT sum (prepare
    adds 2 x 2**(N_A-1)) but not the printed CNOT total, 17 x 2**(N_A-1). The T
    line checks only the sub-leading terms, because the select line and the T
    total print the same leading term. Components: select T = 15 * 2**(N_A-1) +
    6 N_A - 28 and CNOT = 15 * 2**(N_A-1) + 6 N_A - 26, on N_A - 1 work qubits;
    reflection T = 8 N_A - 9 and CNOT = 6 N_A - 6, reusing the select's work
    qubits; prepare CNOT = 2**N_A - 2 N_A - 2.

    The other two fields are restated from the source rather than summed. The
    source counts "only two-qubit Clifford gates and rotations or T gates", so the
    single-qubit total is the prepare's 2**N_A - N_A - 2 arbitrary one-qubit gates
    alone (the Hadamard, Phase and NOT gates of the select and the reflection are
    left out), and the ancilla total is the N_A index register plus the select's
    work qubits.

    Probed at the term counts of 12C's orbital basis (760), 4He on 27 sites (1026)
    and 40Ar on 10**3 sites (38000), and into the dense regime (N_A = 26), starting
    from N_A = 3. At the accepted floor N_A = 2 the prepare's CNOT term is -2; the
    counts there are pinned by the validity-floor test.
    """
    register_size = lcu_ancilla_register_size(num_terms)
    half_capacity = 2 ** (register_size - 1)

    select_t = 15 * half_capacity + 6 * register_size - 28
    select_cnot = 15 * half_capacity + 6 * register_size - 26
    select_work_qubits = register_size - 1
    reflection_t = 8 * register_size - 9
    reflection_cnot = 6 * register_size - 6
    prepare_cnot = 2**register_size - 2 * register_size - 2
    prepare_single_qubit = 2**register_size - register_size - 2

    assert controlled_qubiterate_cost(num_terms) == QubiterateCost(
        select_t_count=select_t,
        reflection_t_count=reflection_t,
        cnot_count=prepare_cnot + reflection_cnot + select_cnot,
        single_qubit_count=prepare_single_qubit,
        ancilla_count=register_size + select_work_qubits,
    )


def test_qubiterate_cost_at_the_pionless_12c_orbital_point() -> None:
    """Literal App. C.1 counts for the pionless model at M = 20, a small basis for 12C.

    For the leading-order pionless model with four fermion species, Roggero+
    App. C.1 gives Gamma = 24 M + 14 M = 38 M LCU terms (kinetic plus potential;
    the 14 M is Eq. 8's count of potential terms). In the paper M counts lattice
    sites; using M = 20 for the active orbital basis of 12C is this project's
    convention. Then Gamma = 760 and N_A = ceil(log2 760) = 10, so the App. C.1
    components give
        T    = 7712 (select) + 71 (reflection)                  = 7783
        CNOT = 7714 (select) + 54 (reflection) + 1002 (prepare) = 8770
        U(2) = 2**10 - 10 - 2                                   = 1012
        anc  = 10 (index register) + 9 (select work qubits)     = 19,
    with select T = 15*2**9 + 6*10 - 28 and select CNOT = 15*2**9 + 6*10 - 26. A
    pinned regression value for 12C, readable without running code; the
    component-sum test is the check of the App. C.1 transcription.
    """
    assert controlled_qubiterate_cost(760) == QubiterateCost(
        select_t_count=7712,
        reflection_t_count=71,
        cnot_count=8770,
        single_qubit_count=1012,
        ancilla_count=19,
    )


def test_qubiterate_cost_triangulates_at_the_40Ar_lattice_point() -> None:
    """Literal App. C.1 counts at Roggero+'s 40Ar point, a far-separated N_A.

    Roggero+ cost the 40Ar inclusive response on an M = 10**3 site lattice, so
    Gamma = 38 M = 38000 and N_A = ceil(log2 38000) = 16. The App. C.1 components
    give
        T    = 491588 (select) + 119 (reflection)                  = 491707
        CNOT = 491590 (select) + 90 (reflection) + 65502 (prepare) = 557182
        U(2) = 2**16 - 16 - 2                                      = 65518
        anc  = 16 (index register) + 15 (select work qubits)       = 31,
    with select T = 15*2**15 + 6*16 - 28 and select CNOT = 15*2**15 + 6*16 - 26.
    A second pinned regression value that triangulates the 12C one: a far-separated
    basis size (50x Gamma, N_A 10->16) that a value fitted to the 12C point cannot
    satisfy. It also shows the exact-vs-shorthand gap: the exact T (4.9e5) is
    ~1.7x the 7.5*Gamma shorthand (2.85e5).
    """
    assert controlled_qubiterate_cost(38000) == QubiterateCost(
        select_t_count=491588,
        reflection_t_count=119,
        cnot_count=557182,
        single_qubit_count=65518,
        ancilla_count=31,
    )


def test_qubiterate_cost_splits_its_t_column_into_select_and_reflection() -> None:
    """The App. C.1 T column, read as the two components the source prints.

    The source prices the select at 15 * 2**(N_A - 1) + 6 N_A - 28 T and the
    controlled reflection at 8 N_A - 9 T, and its T total is their sum. For 4He on
    27 sites, Gamma = 38 * 27 = 1026 and N_A = 11, that is
    15 * 2**10 + 6 * 11 - 28 = 15398 T of select and 8 * 11 - 9 = 79 T of
    reflection, a T column of 15477.

    A caller that prices the select another way needs the reflection alone, and
    one that keeps the appendix's select needs the select alone, so both are
    fields. The T column is computed from them and cannot disagree with them. The
    literal tests at Gamma = 3, 760 and 38000 compare both fields as part of the
    whole result, and dataclass equality does not reach the computed column, so
    this test pins the one size those tests do not cover and the column itself.
    """
    cost = controlled_qubiterate_cost(1026)

    assert cost.select_t_count == 15398
    assert cost.reflection_t_count == 79
    assert cost.t_count == 15477


def test_unary_iteration_select_counts_terms_where_the_appendix_counts_slots() -> None:
    """Babbush+ 2018's unary-iteration count, and how it differs from App. C.1's.

    A controlled select over L terms by unary iteration is L - 1 AND computations
    and nothing else that consumes T gates (Babbush+ 2018, Phys. Rev. X 8, 041015,
    Sec. III A). Its worked case is L = 11: 10 AND computations, 40 T at 4 T each
    (their Fig. 7). A single term needs none: its Pauli string is applied under the
    control alone, which is Clifford.

    From Gamma = 1024 to 1025 the index register grows from 10 to 11 qubits. The
    unary count grows by one, from 1023 to 1024 AND computations (4092 to 4096 T),
    because it is sized by the terms. App. C.1's select is sized by the register's
    2**N_A slots and doubles, from 7712 to 15398 T.
    """
    assert unary_iteration_select_toffoli_count(11) == 10
    assert unary_iteration_select_toffoli_count(1) == 0

    assert unary_iteration_select_toffoli_count(1024) == 1023
    assert unary_iteration_select_toffoli_count(1025) == 1024
    assert controlled_qubiterate_cost(1024).select_t_count == 7712
    assert controlled_qubiterate_cost(1025).select_t_count == 15398


@pytest.mark.parametrize("num_terms", [0, -1, math.nan, math.inf, 1026.0, True])
def test_unary_iteration_select_rejects_term_counts_that_are_not_positive_integers(
    num_terms: float,
) -> None:
    """A select runs over a whole number of terms, at least one.

    At zero terms the count would be -1, and below zero more negative still, so
    both raise. A NaN, an infinity and a whole-valued float are not integers and
    raise, and so does True, a bool that equals 1. Each error names the term
    count.
    """
    with pytest.raises(ValueError, match="num_terms must be an integer >= 1"):
        unary_iteration_select_toffoli_count(num_terms)


def test_controlled_reflection_takes_one_and_per_qubit_beyond_the_first() -> None:
    """A controlled reflection about the zero state of n qubits is n - 1 ANDs.

    Up to NOT gates and a Clifford phase on the control, the reflection is a Z on
    one of the n qubits, controlled on the other n - 1 and on the control. Its n
    controls expand into n - 1 nested AND computations, with a CZ under the last
    (Babbush+ 2018, Phys. Rev. X 8, 041015, Sec. III A and Figs. 4 and 5). A
    single qubit therefore costs none: its reflection under the control is a CZ.

    At the index register of 4He on 27 sites, N_A = 11, the count is 10 AND
    computations (40 T at 4 T each), where App. C.1 charges the same gate
    8 * 11 - 9 = 79 T.
    The count is linear in the width. Babbush+'s O(log L) for the reflection
    (their Fig. 1) is linear in it too, since log L is the width of their |l>
    register.
    """
    assert controlled_reflection_toffoli_count(1) == 0
    assert controlled_reflection_toffoli_count(11) == 10


@pytest.mark.parametrize("num_qubits", [0, -1, math.nan, math.inf, 51.0, True])
def test_controlled_reflection_rejects_widths_that_are_not_positive_integers(
    num_qubits: float,
) -> None:
    """A reflection acts on a whole number of qubits, at least one.

    At zero qubits the count would be -1, and below zero more negative still, so
    both raise. A NaN, an infinity and a whole-valued float are not integers and
    raise, and so does True, a bool that equals 1. Each error names the width.
    """
    with pytest.raises(ValueError, match="num_qubits must be an integer >= 1"):
        controlled_reflection_toffoli_count(num_qubits)


@pytest.mark.parametrize(
    ("num_qubits", "expected_toffolis"),
    [(1, 0), (2, 0), (3, 1), (7, 5), (40, 38), (130, 128)],
)
def test_reflection_about_zero_takes_one_and_fewer_than_its_controlled_form(
    num_qubits: int, expected_toffolis: int
) -> None:
    """An uncontrolled reflection about the zero state of n qubits is n - 2 ANDs.

    Up to NOT gates and a global phase, the reflection is a Z on one of the n
    qubits controlled on the other n - 1, and those controls expand into n - 2
    nested AND computations with a CZ under the last (Babbush+ 2018, Phys. Rev.
    X 8, 041015, Sec. III A and Figs. 4 and 5). The controlled reflection's
    extra control is one more AND computation, so from two qubits up the count
    is one fewer than controlled_reflection_toffoli_count: a CZ against a CCZ at
    two qubits. One qubit costs none either way (a Z against a CZ), so the count
    stops at zero there rather than going negative.

    The widths are those of 4He on 27 sites with 11 energy bits: 7 is the
    amplification's index register (the good-subspace reflection, 5 ANDs), 130 the
    system, index and enumeration registers together (the reflection about the
    state the preparation starts from, 128), and 40 the walk's one-site prepare at
    mu = 11, where the controlled count is 39.
    """
    assert reflection_toffoli_count(num_qubits) == expected_toffolis
    if num_qubits >= 2:
        assert (
            reflection_toffoli_count(num_qubits)
            == controlled_reflection_toffoli_count(num_qubits) - 1
        )


@pytest.mark.parametrize("num_qubits", [0, -1, 6.0, True])
def test_reflection_about_zero_rejects_unusable_widths(num_qubits: float) -> None:
    """A reflection acts on a whole number of qubits, at least one.

    At zero qubits and below there is nothing to reflect, and the count's floor
    at zero would otherwise hide the error, so both raise. A whole-valued float
    is not a width (6.0 would otherwise pass as the index register's 6), and
    neither is True, a bool that equals 1. Each error names the width.
    """
    with pytest.raises(ValueError, match="num_qubits must be an integer >= 1"):
        reflection_toffoli_count(num_qubits)


def test_cost_formula_validity_floor_excludes_negative_counts() -> None:
    """Below N_A = 2 the App. C.1 closed forms go negative and are rejected.

    At N_A = 1 (num_terms = 2) three of the four totals are negative: T =
    15 + 14 - 37 = -8, CNOT = 17 + 10 - 34 = -7 and single-qubit = 2 - 1 - 2 = -1.
    controlled_qubiterate_cost must raise ValueError there, naming the reason,
    rather than return them. N_A = 2 (num_terms = 3) is the accepted floor, the
    smallest register at which every total is non-negative, and the counts there
    are pinned from the components:
        T    = 14 (select) + 7 (reflection)               = 21
        CNOT = 16 (select) + 6 (reflection) - 2 (prepare) = 20
        U(2) = 2**2 - 2 - 2                               = 0
        anc  = 2 (index register) + 1 (select work qubit) = 3.
    The prepare's CNOT term, 2**N_A - 2 N_A - 2, is -2 at this register size: the
    closed forms are large-register expressions, and at the floor the CNOT total
    is the formula's value, not a physical bound.
    """
    with pytest.raises(ValueError, match="T, CNOT and single-qubit"):
        controlled_qubiterate_cost(2)
    assert controlled_qubiterate_cost(3) == QubiterateCost(
        select_t_count=14,
        reflection_t_count=7,
        cnot_count=20,
        single_qubit_count=0,
        ancilla_count=3,
    )


@pytest.mark.parametrize("num_terms", [0, -1])
def test_register_size_rejects_nonpositive_term_counts(num_terms: int) -> None:
    """An LCU needs at least one term; non-positive counts fail fast.

    For num_terms <= 0, (num_terms - 1).bit_length() would return a spurious
    positive width (e.g. 0 -> 1) instead of an error, so lcu_ancilla_register_size
    -- and controlled_qubiterate_cost, which calls it -- must raise ValueError.
    """
    with pytest.raises(ValueError):
        lcu_ancilla_register_size(num_terms)
    with pytest.raises(ValueError):
        controlled_qubiterate_cost(num_terms)


@pytest.mark.parametrize("num_terms", [math.nan, math.inf, 1026.0])
def test_register_size_rejects_term_counts_that_are_not_integers(
    num_terms: float,
) -> None:
    """A term count is a whole number of terms and is passed as an integer.

    The width comes from integer arithmetic, which a float would break with an
    error that does not name the argument. A NaN, an infinity and a whole-valued
    float each raise ValueError and name the term count, from
    lcu_ancilla_register_size and from controlled_qubiterate_cost, which sizes its
    register through it.
    """
    with pytest.raises(ValueError, match="num_terms must be an integer"):
        lcu_ancilla_register_size(num_terms)
    with pytest.raises(ValueError, match="num_terms must be an integer"):
        controlled_qubiterate_cost(num_terms)


@pytest.mark.parametrize("num_terms", [760, 5000, 38000, 1_000_000, 40_000_000])
def test_leading_t_term_brackets_the_per_term_shorthand(num_terms: int) -> None:
    """The exact T-count's leading term lies between 7.5*Gamma and 15*Gamma.

    The "7.5*Gamma" shorthand (a reading of App. C.1 of Roggero+ 2019,
    arXiv:1911.06368) sets 2**N_A ~ Gamma. Exactly, the App. C.1 leading term
    15*2**(N_A-1) = 7.5*2**N_A is bracketed by
    [7.5*Gamma, 15*Gamma] because 2**(N_A-1) < Gamma <= 2**N_A -- so the shorthand
    is a lower estimate and the exact count is up to ~2x larger (the power-of-two
    rounding). That is 7.5 to 15 T gates per LCU term. The bracket is asserted on
    twice the leading term to stay in integers. The full t_count carries an
    additional 14*N_A - 37 correction, removed here to isolate the leading term:
    the raw t_count breaks the upper bound at the low edge of a plateau, where it
    exceeds 15*Gamma by 14*N_A - 52. Spanning Gamma to 4e7 (N_A = 26) also
    exercises the dense-orbital regime used downstream.
    """
    register_size = lcu_ancilla_register_size(num_terms)
    leading_t_term = controlled_qubiterate_cost(num_terms).t_count - (
        14 * register_size - 37
    )
    assert 15 * num_terms <= 2 * leading_t_term <= 30 * num_terms


def test_qubiterate_cost_is_nondecreasing_in_term_count() -> None:
    """Every count is non-decreasing in Gamma, flat across each power-of-two plateau.

    N_A = ceil(log2 Gamma) is a step function, constant on (2**(N_A-1), 2**N_A],
    so all four counts are constant within a plateau and jump only at a
    power-of-two boundary: cost(512) < cost(513) = cost(1024) < cost(1025)
    component-wise. A monotonicity invariant that catches a sign or rounding error
    a single-point check would miss.
    """
    low = controlled_qubiterate_cost(512)
    plateau_start = controlled_qubiterate_cost(513)
    plateau_end = controlled_qubiterate_cost(1024)
    high = controlled_qubiterate_cost(1025)

    assert plateau_start == plateau_end
    for field in ("t_count", "cnot_count", "single_qubit_count", "ancilla_count"):
        assert getattr(low, field) < getattr(plateau_start, field)
        assert getattr(plateau_end, field) < getattr(high, field)


def test_argon_one_norm_drives_qubitization_qpe_ancilla_scale() -> None:
    """The 40Ar LCU 1-norm sets a 16-qubit QPE register and 2**16-1 walks.

    Composes the two qubitization primitives end to end: the pionless
    1-norm lambda(M=1000) @ a=1.4 fm feeds the qubiterate-QPE counter at the
    inclusive-response 10 MeV resolution. lambda = 338225 MeV gives
    lambda/Delta-omega = 33822.5, so W_q = ceil(log2 33822.5) = 16 and textbook
    phase estimation runs 2**16 - 1 = 65535 walk steps. Roggero+ Fig. 4 shows 18
    qubitization-QPE ancillas at this resolution, two more than ours. The paper's
    absolute norms (Eqs. B3 and B6: 24 M |t| + M(6|C0| + 8|D0|) = 1866020 MeV)
    reproduce that value, ceil(log2 186602) = 18; the Pauli 1-norm of its Eq. 1 used
    here is smaller. Pins the units contract -- lambda and resolution share an
    energy unit -- and the end-to-end composition.
    """
    one_norm = hamiltonian_one_norm(1000, reference_couplings_at_spacing(1.4))

    cost = qubiterate_qpe_ancilla_count(one_norm, 10.0)

    assert cost == QubiterateQpeCost(ancilla_qubits=16, application_count=65535)


@pytest.mark.parametrize(
    ("one_norm", "resolution", "expected_width"),
    [
        (1024.0, 1.0, 10),
        (1025.0, 1.0, 11),
        (3000.0, 1.0, 12),
        (15000.0, 10.0, 11),
        (2048.0000001, 1.0, 12),
    ],
)
def test_qpe_ancilla_count_ceils_log2_of_norm_resolution_ratio(
    one_norm: float, resolution: float, expected_width: int
) -> None:
    """W_q = ceil(log2(one_norm / resolution)), float-robust at the boundary.

    The register width is the integer ceiling of log2 of the norm-to-resolution
    ratio (Roggero+ Eq. 21). The exact power of two 1024 must give 10 (not 11),
    and 1025 must give 11; 3000 (non-power) gives 12 and the ratio 1500 (via a
    non-unit resolution) gives 11. The trap input 2048.0000001 sits a hair above
    2**11, so it must round *up* to 12: a naive ``math.ceil(math.log2(...))`` that
    lost the tiny excess to float rounding would return 11. This drives the
    power-of-two correction, not a bare float ceil-log2.
    """
    assert (
        qubiterate_qpe_ancilla_count(one_norm, resolution).ancilla_qubits
        == expected_width
    )


@pytest.mark.parametrize(
    ("one_norm", "resolution"),
    [(338225.0, 10.0), (1024.0, 1.0), (1025.0, 1.0), (3000.0, 1.0)],
)
def test_qpe_application_count_brackets_norm_resolution_ratio(
    one_norm: float, resolution: float
) -> None:
    """Applications = 2**W_q - 1, with 2**W_q bracketing lambda/Delta-omega.

    Textbook phase estimation runs exactly 2**W_q - 1 controlled walks (the geometric
    sum sum_{j} 2**j), Roggero+'s ~lambda/Delta-omega up to power-of-two rounding.
    Because W_q is the *tight* ceiling, the register capacity 2**W_q is at least
    the ratio but strictly less than twice it: ratio <= 2**W_q < 2*ratio. This
    distinguishes the exact count from the order-of-magnitude estimate and pins
    that the width is neither one too small (capacity below the ratio) nor one too
    large (capacity at least double).
    """
    cost = qubiterate_qpe_ancilla_count(one_norm, resolution)
    ratio = one_norm / resolution
    register_capacity = 2**cost.ancilla_qubits

    assert cost.application_count == register_capacity - 1
    assert ratio <= register_capacity < 2 * ratio


def test_qpe_ancilla_count_rejects_subdomain_inputs() -> None:
    """Inputs with lambda/Delta-omega <= 1 or nonpositive norms fail fast.

    W_q = ceil(log2(lambda/Delta-omega)) is only meaningful for a ratio > 1; at
    ratio == 1 it collapses to W_q = 0 (zero-qubit register, zero applications)
    and below 1 it goes negative, so equal or sub-resolution norms must raise
    rather than return a degenerate cost. Nonpositive resolution or norm are
    outside log2's domain and must also raise. The smallest valid input -- a
    ratio a hair above 1 -- yields the minimal nondegenerate output W_q = 1 with
    a single application, fixing the lower edge of the valid domain.
    """
    with pytest.raises(ValueError, match="resolution"):
        qubiterate_qpe_ancilla_count(100.0, 0.0)
    with pytest.raises(ValueError, match="resolution"):
        qubiterate_qpe_ancilla_count(100.0, -1.0)
    with pytest.raises(ValueError, match="one_norm"):
        qubiterate_qpe_ancilla_count(0.0, 10.0)
    with pytest.raises(ValueError, match="one_norm"):
        qubiterate_qpe_ancilla_count(-5.0, 10.0)
    with pytest.raises(ValueError, match="exceed"):
        qubiterate_qpe_ancilla_count(10.0, 10.0)
    with pytest.raises(ValueError, match="exceed"):
        qubiterate_qpe_ancilla_count(5.0, 10.0)

    smallest = qubiterate_qpe_ancilla_count(10.0000001, 10.0)

    assert smallest == QubiterateQpeCost(ancilla_qubits=1, application_count=1)


def test_qpe_ancilla_count_rejects_non_finite_inputs() -> None:
    """A 1-norm or a resolution that is not a finite number raises.

    A NaN compares false with every number, so a guard written as "at most zero"
    would let it through, and the exact ceiling of the logarithm would then return
    a register of width 0 with zero applications: a degenerate result that looks
    like an answer. An infinite 1-norm would do the same. An infinite resolution
    would be caught only by the ordering check, with a message about the ratio.
    Each of the four must raise and name the argument that is not finite. Two
    finite arguments can still have a ratio that overflows to infinity, 1e200 MeV
    over 1e-200 MeV, and that gives the same width 0; it raises and names the
    ratio.
    """
    with pytest.raises(ValueError, match="one_norm must be a finite number"):
        qubiterate_qpe_ancilla_count(math.nan, 10.0)
    with pytest.raises(ValueError, match="one_norm must be a finite number"):
        qubiterate_qpe_ancilla_count(math.inf, 10.0)
    with pytest.raises(ValueError, match="resolution must be a finite number"):
        qubiterate_qpe_ancilla_count(100.0, math.nan)
    with pytest.raises(ValueError, match="resolution must be a finite number"):
        qubiterate_qpe_ancilla_count(100.0, math.inf)
    with pytest.raises(ValueError, match="ratio one_norm / resolution"):
        qubiterate_qpe_ancilla_count(1e200, 1e-200)


def test_twice_the_two_sided_walk_steps_is_one_more_than_the_textbook_count() -> None:
    """With the inverse walk on the 0 branch, 2**(W_q - 1) steps replace 2**W_q - 1.

    Babbush+ 2018 (Sec. II B, Eq. 16, Fig. 2) apply W**n on the 1 branch of a
    control qubit and (W^dagger)**n on the 0 branch, which doubles the phase each
    application writes. A register of W_q qubits then takes 2**(W_q - 1) walk
    steps where textbook phase estimation takes 2**W_q - 1 at the same resolution,
    so twice the new count is one more than the textbook's: 1024 against 2047 for
    4He on 27 sites, where W_q = 11.

    The textbook count is read from qubiterate_qpe_ancilla_count at ratios
    lambda / Delta-omega = 2**W_q, exact powers of two, so each register width
    from 1 to 60 comes up once.
    """
    for width in range(1, 61):
        textbook = qubiterate_qpe_ancilla_count(2.0**width, 1.0)
        assert textbook.ancilla_qubits == width
        assert (
            2 * two_sided_qpe_application_count(textbook.ancilla_qubits)
            == textbook.application_count + 1
        )


@pytest.mark.parametrize("ancilla_qubits", [0, -1, 12.0, True])
def test_two_sided_walk_steps_reject_widths_that_are_not_positive_integers(
    ancilla_qubits: float,
) -> None:
    """An energy register has a whole number of qubits, at least one.

    At zero qubits the count would be a half and below zero smaller still, so both
    raise. A whole-valued float is not an integer and raises, and so does True, a
    bool that equals 1. Each error names the width.
    """
    with pytest.raises(ValueError, match="ancilla_qubits must be an integer >= 1"):
        two_sided_qpe_application_count(ancilla_qubits)  # type: ignore[arg-type]


def test_argon_energy_spread_drives_qsvt_evolution_degree() -> None:
    """The centered 40Ar spread sets a degree-767 QSVT evolution polynomial.

    Composes the spread bound with the QSVT degree primitive in our centered,
    MeV-native convention. energy_spread_bound(40) @ 1.4 fm is Delta-H = 15339.5
    MeV, so the centered operator norm is ||H||_op = Delta-H/2 = 7669.75 MeV. The
    single 10 MeV resolution knob fixes the QSVT horizon tau_max = 1/sigma =
    0.1 MeV^-1 (the reciprocal of the 10 MeV QPE resolution), giving the
    heuristic degree ceil(7669.75 * 0.1) = 767. Pins the centering convention
    (operator norm = Delta-H/2, not the raw spread) and the MeV / MeV^-1
    reciprocal-units contract -- a GeV^-1 time here would inflate the degree by
    10**3.
    """
    spread = energy_spread_bound(40, reference_couplings_at_spacing(1.4))
    resolution_mev = 10.0

    degree = qsvt_evolution_degree(spread / 2, 1 / resolution_mev, 1e-3)

    assert degree.heuristic == 767


@pytest.mark.parametrize(
    ("block_encoding_norm", "total_time", "expected_degree"),
    [(5.0, 100.0, 500), (5.0, 100.5, 503)],
)
def test_heuristic_degree_is_norm_time_product(
    block_encoding_norm: float, total_time: float, expected_degree: int
) -> None:
    """heuristic = ceil(alpha * tau), the d ~ tau*||H|| scaling.

    The first case is a chiral bandwidth: ||H|| = 5 GeV with
    tau_max = 100 GeV^-1 gives degree 500 (a consistent GeV / GeV^-1 unit pair).
    The second forces a non-integer product 5 * 100.5 = 502.5 that must round *up*
    to 503 -- pinning a ceil, distinct from round-half-to-even (502) and floor
    (502).
    """
    degree = qsvt_evolution_degree(block_encoding_norm, total_time, 1e-3)

    assert degree.heuristic == expected_degree


def test_jacobi_anger_degree_meets_truncation_contract() -> None:
    """jacobi_anger is the smallest degree whose Bessel tail meets epsilon.

    The rigorous degree must be exactly the minimal truncation satisfying the
    Jacobi-Anger error bound 2*sum_{k>d}|J_k(z)| <= epsilon at z = alpha*tau: the
    returned degree d meets it and d - 1 exceeds it. Checked against the
    independent tail oracle jacobi_anger_truncation_error, so the assertion pins
    the minimality contract regardless of how the degree finder searches.
    """
    block_encoding_norm, total_time, epsilon = 5.0, 100.0, 1e-3
    z = block_encoding_norm * total_time

    degree = qsvt_evolution_degree(
        block_encoding_norm, total_time, epsilon
    ).jacobi_anger

    assert jacobi_anger_truncation_error(z, degree) <= epsilon
    assert jacobi_anger_truncation_error(z, degree - 1) > epsilon


def test_jacobi_anger_overhead_over_heuristic_is_small_and_sublinear() -> None:
    """jacobi_anger exceeds the heuristic by a small, sublinear additive overhead.

    The rigorous constructive degree is the leading z plus an O(z**(1/3) +
    log(1/epsilon)) correction, so it sits just above the heuristic and the ratio
    approaches 1 as z grows -- the opposite of a multiplicative (e/2)*z law, which
    would hold the ratio constant near 1.36. Pinned across two decades of z: the
    overhead at z = 500 is a few dozen, and the z = 5000 ratio is strictly smaller
    (closer to 1). An implementation-independent guard on the *shape* of the
    overhead, refuting the multiplicative reading.
    """
    near = qsvt_evolution_degree(5.0, 100.0, 1e-3)  # z = 500
    far = qsvt_evolution_degree(5.0, 1000.0, 1e-3)  # z = 5000

    assert near.heuristic < near.jacobi_anger <= near.heuristic + 40
    assert 1.0 < far.jacobi_anger / far.heuristic < near.jacobi_anger / near.heuristic


def test_jacobi_anger_degree_grows_as_epsilon_tightens() -> None:
    """Tightening epsilon raises jacobi_anger via the additive O(log 1/eps) term.

    The rigorous degree carries a +O(log(1/epsilon)) correction over the leading
    z, so demanding a smaller approximation error strictly increases the degree.
    Pinned across the error scale at fixed z = 500: each tighter epsilon yields a
    strictly larger jacobi_anger.
    """
    norm, time = 5.0, 100.0

    coarse = qsvt_evolution_degree(norm, time, 1e-2).jacobi_anger
    medium = qsvt_evolution_degree(norm, time, 1e-5).jacobi_anger
    fine = qsvt_evolution_degree(norm, time, 1e-9).jacobi_anger

    assert coarse < medium < fine


@pytest.mark.parametrize(
    ("block_encoding_norm", "total_time", "epsilon"),
    [
        (0.0, 100.0, 1e-3),
        (-1.0, 100.0, 1e-3),
        (5.0, 0.0, 1e-3),
        (5.0, -1.0, 1e-3),
        (5.0, 100.0, 0.0),
        (5.0, 100.0, -1e-3),
        (5.0, 100.0, 1.0),
        (5.0, 100.0, 1.5),
    ],
)
def test_qsvt_degree_rejects_nonphysical_inputs(
    block_encoding_norm: float, total_time: float, epsilon: float
) -> None:
    """Nonpositive norm/time or epsilon outside (0, 1) fail fast.

    The degree d ~ alpha*tau is meaningless for a nonpositive block-encoding norm
    or evolution time, and the approximation error epsilon must be a genuine
    probability in the open interval (0, 1): epsilon <= 0 is unachievable and
    epsilon >= 1 (no accuracy demanded) would collapse the degree. Each violation
    must raise rather than return a degenerate or silently-looping result.
    """
    with pytest.raises(ValueError):
        qsvt_evolution_degree(block_encoding_norm, total_time, epsilon)


def test_rotation_count_brackets_the_rz_line_of_the_source_figure() -> None:
    """The figure's R_z line sits between the appendix's smallest and largest counts.

    Roggero+ Fig. 5 plots the total qubitization R_z count over all of phase
    estimation's walk steps for 40Ar on the M = 10**3 lattice (Gamma = 38000,
    N_A = 16), counting each T gate as one rotation. Read off the vector figure,
    with one pixel worth about 1.4%, the upper of its two qubitization lines is
    613,000 times the 2**18 - 1 applications of the paper's own register
    (W_q = 18 in its Fig. 4, one more than this library computes for the point),
    about 1.61e11 at 10 MeV. So 613,000 is the figure's count per controlled
    qubiterate. The App. C.1 T column there is 491,707 and one prepare has 65,518
    one-qubit gates, so charging 1, 2 or 3 rotations per gate gives 557,225,
    622,743 and 688,261.

    The equality checks the counting function against the appendix's closed form.
    The two inequalities compare those three sums with the figure's line: the
    first is 9% below it and the third is 12% above it. They follow from the
    equality and cannot fail on their own. They record that the appendix's numbers
    and the figure, two statements of the cost by the source, agree within the
    range of the convention. What the figure charged per gate is not established:
    2 per gate is 1.6% above the line, and other readings come as close.
    """
    figure_line = 613_000
    t_column = controlled_qubiterate_cost(38000).t_count

    totals = [
        t_column
        + prepare_rotation_count(
            38000, rotations_per_gate=rotations, prepares_per_qubiterate=1
        )
        for rotations in (1, 2, 3)
    ]

    assert totals == [557_225, 622_743, 688_261]
    assert totals[0] < 0.92 * figure_line
    assert totals[2] > 1.08 * figure_line


@pytest.mark.parametrize(
    ("rotations_per_gate", "prepares_per_qubiterate", "expected_rotations"),
    [
        (1, 1, 1012),
        (2, 1, 2024),
        (3, 1, 3036),
        (1, 2, 2024),
        (2, 2, 4048),
        (3, 2, 6072),
    ],
)
def test_prepare_rotation_count_at_the_12c_point_for_every_convention(
    rotations_per_gate: int, prepares_per_qubiterate: int, expected_rotations: int
) -> None:
    """Literal rotation counts for the pionless model at M = 20, Gamma = 760.

    At Gamma = 760 (N_A = 10) one prepare has 2**10 - 10 - 2 = 1012 one-qubit
    gates. Charging r rotations per gate and p prepares per qubiterate gives
    p * 1012 * r: 1012, 2024 and 3036 for one prepare, and 2024, 4048 and 6072 for
    two. Readable without running code.
    """
    assert (
        prepare_rotation_count(
            760,
            rotations_per_gate=rotations_per_gate,
            prepares_per_qubiterate=prepares_per_qubiterate,
        )
        == expected_rotations
    )


def test_prepare_rotation_count_rejects_conventions_outside_their_ranges() -> None:
    """Each convention is accepted only in its possible range, as an exact integer.

    A one-qubit unitary is at most three Euler rotations, and a nontrivial gate
    needs at least one, so rotations_per_gate of 0, 4 or -1 must raise. A
    controlled qubiterate applies two prepares and the source counts one, so
    prepares_per_qubiterate of 0 or 3 must raise. Each message names its argument.
    The two arguments multiply, so their ranges are the only thing that tells them
    apart: 3 rotations is accepted and 3 prepares is not. Each convention is an
    exact integer. True equals 1 and 2.0 equals 2, so a test of the value alone
    would accept both, and the count would come back as a float for the second;
    both raise, for either argument.
    """
    with pytest.raises(ValueError, match="rotations_per_gate"):
        prepare_rotation_count(760, rotations_per_gate=0, prepares_per_qubiterate=1)
    with pytest.raises(ValueError, match="rotations_per_gate"):
        prepare_rotation_count(760, rotations_per_gate=4, prepares_per_qubiterate=1)
    with pytest.raises(ValueError, match="rotations_per_gate"):
        prepare_rotation_count(760, rotations_per_gate=-1, prepares_per_qubiterate=1)
    with pytest.raises(ValueError, match="prepares_per_qubiterate"):
        prepare_rotation_count(760, rotations_per_gate=3, prepares_per_qubiterate=0)
    with pytest.raises(ValueError, match="prepares_per_qubiterate"):
        prepare_rotation_count(760, rotations_per_gate=3, prepares_per_qubiterate=3)
    with pytest.raises(ValueError, match="rotations_per_gate"):
        prepare_rotation_count(760, rotations_per_gate=True, prepares_per_qubiterate=1)
    with pytest.raises(ValueError, match="rotations_per_gate"):
        prepare_rotation_count(760, rotations_per_gate=2.0, prepares_per_qubiterate=1)
    with pytest.raises(ValueError, match="prepares_per_qubiterate"):
        prepare_rotation_count(760, rotations_per_gate=3, prepares_per_qubiterate=True)
    with pytest.raises(ValueError, match="prepares_per_qubiterate"):
        prepare_rotation_count(760, rotations_per_gate=3, prepares_per_qubiterate=2.0)


def test_prepare_rotation_count_refuses_a_register_with_no_gates_to_count() -> None:
    """Three or four terms raise, because the closed form would price them at zero.

    App. C.1's count of one-qubit gates, 2**N_A - N_A - 2, is a large-register
    expression that reaches 0 at N_A = 2 (num_terms = 3 or 4). A synthesis price
    built from it would be zero, and a zero price is something the caller writes:
    a prepare of three or four amplitudes needs two or three rotations. Both term
    counts raise, and the message names the smallest accepted count. A term count
    below the gate counts' own floor (num_terms = 2) raises as it does there. At
    num_terms = 5 (N_A = 3) the closed form has 3 gates, so two prepares at three
    rotations per gate are 2 * 3 * 3 = 18 rotations.
    """
    with pytest.raises(ValueError, match="num_terms"):
        prepare_rotation_count(2, rotations_per_gate=3, prepares_per_qubiterate=2)
    with pytest.raises(ValueError, match="num_terms >= 5"):
        prepare_rotation_count(3, rotations_per_gate=3, prepares_per_qubiterate=2)
    with pytest.raises(ValueError, match="num_terms >= 5"):
        prepare_rotation_count(4, rotations_per_gate=3, prepares_per_qubiterate=2)

    smallest_accepted = prepare_rotation_count(
        5, rotations_per_gate=3, prepares_per_qubiterate=2
    )

    assert smallest_accepted == 18


def test_rotation_count_cannot_be_asked_without_both_conventions_by_name() -> None:
    """Neither counting convention has a default or can be passed by position.

    The library chooses neither convention. A call that leaves out the rotations
    per gate raises TypeError and names them, and so does a call that leaves out
    the prepares per qubiterate; a call with neither is then covered. Passed by
    position, a bare 3 and a bare 2 would not say at the call site which is the
    rotations per gate and which the prepares, so that raises too.
    """
    with pytest.raises(TypeError, match="rotations_per_gate"):
        prepare_rotation_count(760, prepares_per_qubiterate=2)
    with pytest.raises(TypeError, match="prepares_per_qubiterate"):
        prepare_rotation_count(760, rotations_per_gate=3)
    with pytest.raises(TypeError, match="positional"):
        prepare_rotation_count(760, 3, 2)


def test_table_lookup_prepare_is_the_rate_times_the_term_count() -> None:
    """The lookup price is linear in the caller's rate and in the coefficient count.

    At Gamma = 1026, 4He on 27 sites, rates of 8 and 12 T per coefficient give
    8208 and 12312 T per controlled qubiterate. The two literals rule out a sum
    and fix the slope in the rate at this Gamma (12312 / 8208 = 12 / 8), and
    doubling the rate to 16 is a third point on the same line. Both literals are
    taken at one Gamma, so the dependence on Gamma is what they leave open:
    doubling Gamma at a fixed rate must double the price.
    """
    base = table_lookup_prepare_t_count(1026, t_per_coefficient=8.0)

    assert base == 8208
    assert table_lookup_prepare_t_count(1026, t_per_coefficient=12.0) == 12312
    assert table_lookup_prepare_t_count(1026, t_per_coefficient=16.0) == 2 * base
    assert table_lookup_prepare_t_count(2052, t_per_coefficient=8.0) == 2 * base


def test_table_lookup_prepare_rejects_unusable_rates_and_empty_tables() -> None:
    """A lookup has a finite, positive cost per coefficient and at least one of them.

    A rate of 0 or below would price the prepare at nothing or less, which is the
    silent omission this function exists to prevent. A rate that is not a number,
    or is infinite, would pass into a total without complaint. Each raises and
    names the rate. A table of zero coefficients raises and names the term count.
    So does a term count that is not an integer: a NaN or an infinite one would
    come back as a price that is not a number, and a table holds a whole number of
    coefficients, so a float raises, whole or not.
    One coefficient is accepted: the App. C.1 floor of three terms belongs to the
    rotation tree's closed forms and does not bind a lookup.
    """
    with pytest.raises(ValueError, match="t_per_coefficient"):
        table_lookup_prepare_t_count(1026, t_per_coefficient=0.0)
    with pytest.raises(ValueError, match="t_per_coefficient"):
        table_lookup_prepare_t_count(1026, t_per_coefficient=-8.0)
    with pytest.raises(ValueError, match="t_per_coefficient"):
        table_lookup_prepare_t_count(1026, t_per_coefficient=math.nan)
    with pytest.raises(ValueError, match="t_per_coefficient"):
        table_lookup_prepare_t_count(1026, t_per_coefficient=math.inf)
    with pytest.raises(ValueError, match="num_terms"):
        table_lookup_prepare_t_count(0, t_per_coefficient=8.0)
    with pytest.raises(ValueError, match="num_terms"):
        table_lookup_prepare_t_count(math.nan, t_per_coefficient=8.0)
    with pytest.raises(ValueError, match="num_terms"):
        table_lookup_prepare_t_count(math.inf, t_per_coefficient=8.0)
    with pytest.raises(ValueError, match="num_terms"):
        table_lookup_prepare_t_count(1026.0, t_per_coefficient=8.0)

    assert table_lookup_prepare_t_count(1, t_per_coefficient=8.0) == 8


def test_table_lookup_prepare_cannot_be_asked_without_a_rate_by_name() -> None:
    """The lookup's rate has no default and cannot be passed by position.

    The library asserts no T count per stored coefficient, so a call with the term
    count alone raises TypeError and names the rate. A bare second number would
    not say at the call site that it is a rate, so that raises too.
    """
    with pytest.raises(TypeError, match="t_per_coefficient"):
        table_lookup_prepare_t_count(1026)
    with pytest.raises(TypeError, match="positional"):
        table_lookup_prepare_t_count(1026, 8.0)


def test_keep_bits_gain_one_bit_per_halving_of_the_resolution() -> None:
    """Each halving of the target error costs one more bit of keep probability.

    Babbush+ 2018 Eq. (36) sets mu = ceil(log2(2 sqrt(2) lambda / Delta-E) + ...),
    so halving Delta-E adds one bit to its leading term. For 4He on 27 sites at
    a = 1.4 fm the 1-norm is 9,132.075 MeV. ||H|| is bounded on the 4-nucleon
    energies the phase estimation reads, in the frame of the block encoding, which
    drops the kinetic constant 2 D t A and the potential's identity
    M (3/2 C0 + 1/2 D0) = -2,252.475 MeV (Roggero+ 2019 Eq. 1): the triangle
    inequality bounds them by 2,728.86 MeV, the bound the library states. The sum
    is then 12.470 bits at 5 MeV, and at 5 / 2**k MeV it is 12.470 + k, 0.530 bits
    below each ceiling: mu = 13 + k for k = 0 to 10.
    """
    couplings = reference_couplings_at_spacing(1.4)
    one_norm = hamiltonian_one_norm(27, couplings)
    four_nucleon_bound = phase_estimation_energy_bound(4, 27, couplings)

    bits = [
        lookup_keep_probability_bits(
            one_norm, 5.0 / 2**halvings, spectral_norm_bound=four_nucleon_bound
        )
        for halvings in range(11)
    ]

    assert bits == list(range(13, 24))


def test_keep_bits_for_helium_hardly_depend_on_the_norm_bound() -> None:
    """For 4He on 27 sites, mu = 13 for any bound on ||H|| below 0.6078 lambda.

    The third term of Babbush+ 2018 Eq. (36), -log2(1 - ||H||**2 / lambda**2),
    grows without bound as ||H|| nears lambda. Here (a = 1.4 fm, lambda =
    9,132.075 MeV, Delta-E = 5 MeV) the first two terms sum to 12.335 bits, so
    mu stays 13 until the third reaches 0.665 bits, at ||H|| = 0.6078 lambda,
    5,550 MeV. The bound on the 4-nucleon energies, 2,728.86 MeV (the halving
    test above), is twice below that, and its third term is 0.135 bits. At 0.6077
    lambda mu = 13, and at 0.6079 lambda mu = 14. App. A's norm over the whole space
    stays below it here too, so the two readings agree: the hopping is at most
    96 t in size on the 3^3 lattice, and the potential without its identity at
    most 83.425 MeV per site, which is 3,268.2 MeV.
    """
    couplings = reference_couplings_at_spacing(1.4)
    one_norm = hamiltonian_one_norm(27, couplings)

    def bits(norm_bound: float) -> int:
        return lookup_keep_probability_bits(
            one_norm, 5.0, spectral_norm_bound=norm_bound
        )

    assert bits(phase_estimation_energy_bound(4, 27, couplings)) == 13
    assert bits(96 * couplings.hopping_scale_mev + 27 * 83.425) == 13
    assert bits(0.6077 * one_norm) == 13
    assert bits(0.6079 * one_norm) == 14


def test_keep_bits_count_the_resolution_term_when_the_resolution_nears_the_norm() -> (
    None
):
    """The second term of Eq. (36) adds a bit when Delta-E comes close to lambda.

    log2(1 + Delta-E**2 / (8 lambda**2)) is 1e-8 bits for 4He on 27 sites, too
    small to move a ceiling there, but it reaches 0.17 bits as Delta-E nears
    lambda. At lambda = 1 MeV, Delta-E = 0.75 MeV and ||H|| = 1e-6 MeV, the
    leading term is 1.915 bits and the second 0.098, so the sum, 2.013 bits,
    passes the ceiling at 2: mu = 3 where the leading term alone gives 2.
    """
    assert lookup_keep_probability_bits(1.0, 0.75, spectral_norm_bound=1e-6) == 3


def test_keep_bits_take_large_energies_whose_ratio_is_moderate() -> None:
    """Energies near the float limit give a count when their ratio is moderate.

    Eq. (36) depends on the energies through their ratios alone, so 1e308 MeV over
    1e307 MeV is a resolution ten times below the 1-norm, like any other. The
    leading factor is 2 sqrt(2) * 10 = 28.28 and the second 1.00125, a log2 of
    4.82, so mu = 5. Only a ratio that overflows raises (the validation test below).
    """
    assert lookup_keep_probability_bits(1e308, 1e307, spectral_norm_bound=1.0) == 5


def test_keep_bits_ceiling_is_exact_at_a_power_of_two() -> None:
    """An argument of exactly 2**30 needs 30 bits, and one just above it 31.

    Eq. (35) is a non-strict inequality, so where the log2 argument of Eq. (36) is
    2**j, mu = j suffices. With lambda = 1 MeV, a norm bound of 1e-9 MeV and
    Delta-E = 2 sqrt(2) / 2**30 MeV, the second and third factors round to 1 and
    the argument is 2**30 to the last bit; with sqrt(2) taken exactly it is
    2**30 (1 - 7e-17), whose ceiling is also 30. At the next smaller Delta-E the
    argument is 2**30 (1 + 2**-52), 2**30 (1 + 9e-17) with sqrt(2) exact, and
    mu = 31. A ceiling of a float log2 would give 30 there, since log2 rounds
    30 + 3e-16 down to 30.0.
    """
    resolution = 2 * math.sqrt(2) / 2**30
    just_below = math.nextafter(resolution, 0.0)

    assert lookup_keep_probability_bits(1.0, resolution, spectral_norm_bound=1e-9) == 30
    assert lookup_keep_probability_bits(1.0, just_below, spectral_norm_bound=1e-9) == 31


@pytest.mark.parametrize(
    ("one_norm", "resolution", "spectral_norm_bound", "message"),
    [
        (0.0, 5.0, 1.0, "one_norm must be a finite number > 0"),
        (-1.0, 5.0, 1.0, "one_norm must be a finite number > 0"),
        (math.nan, 5.0, 1.0, "one_norm must be a finite number > 0"),
        (math.inf, 5.0, 1.0, "one_norm must be a finite number > 0"),
        (100.0, 0.0, 1.0, "resolution must be a finite number > 0"),
        (100.0, -1.0, 1.0, "resolution must be a finite number > 0"),
        (100.0, math.nan, 1.0, "resolution must be a finite number > 0"),
        (100.0, math.inf, 1.0, "resolution must be a finite number > 0"),
        (100.0, 5.0, 0.0, "spectral_norm_bound must be a finite number > 0"),
        (100.0, 5.0, -1.0, "spectral_norm_bound must be a finite number > 0"),
        (100.0, 5.0, math.nan, "spectral_norm_bound must be a finite number > 0"),
        (100.0, 5.0, math.inf, "spectral_norm_bound must be a finite number > 0"),
        (100.0, 100.0, 1.0, "resolution must be below one_norm"),
        (100.0, 5.0, 100.0, "spectral_norm_bound must be below one_norm"),
        (1e200, 1e-200, 1.0, "overflows"),
    ],
)
def test_keep_bits_reject_inputs_outside_the_source_domain(
    one_norm: float, resolution: float, spectral_norm_bound: float, message: str
) -> None:
    """Eq. (36) takes three finite positive energies, ordered as the source has them.

    A zero, negative, NaN or infinite energy has no logarithm to take, or gives a
    count that only looks like one, since a NaN compares false with every number.
    Babbush+ "do not take Delta-E larger than lambda". Rejecting Delta-E = lambda
    as well is this library's choice: every energy lies within lambda of zero, so
    an estimate of zero meets that error. The third term diverges as ||H||
    reaches lambda. So a resolution or a norm bound equal to the 1-norm raises.
    Two finite energies can still have a ratio that overflows, 1e200 MeV over
    1e-200 MeV, and that raises too. Each error names what is at fault.
    """
    with pytest.raises(ValueError, match=message):
        lookup_keep_probability_bits(
            one_norm, resolution, spectral_norm_bound=spectral_norm_bound
        )


def test_keep_bits_cannot_be_asked_without_the_norm_bound_by_name() -> None:
    """The bound on ||H|| has no default and cannot be passed by position.

    mu depends on the bound, and the library chooses none, so a call without it
    raises TypeError and names it. The three arguments are energies in one unit,
    and a positional call that swapped the resolution and the bound would return a
    plausible count, so a bound passed by position raises too.
    """
    with pytest.raises(TypeError, match="spectral_norm_bound"):
        lookup_keep_probability_bits(9_132.075, 5.0)
    with pytest.raises(TypeError, match="positional"):
        lookup_keep_probability_bits(9_132.075, 5.0, 2_728.86)


def test_one_table_on_one_site_is_the_generic_lookup_prepare() -> None:
    """On one site the prepare is Babbush+'s generic one, inside the band it replaces.

    With M = 1 there is no site to put in superposition, and the cost is that of
    Babbush+ 2018 Fig. 11 for L unique amplitudes, 4(L + mu) + O(log L) T. With
    the 1026 terms of 4He on 27 sites in one table and mu = 13 (Eq. 36's count at
    5 MeV) it is 1025 (lookup) + 13 (comparison) + 11 (swap) + 2 * 10 (the equal
    superposition over 1026 = 2 * 513, whose odd part takes ten qubits) = 1069 AND
    computations and 2 rotations, on 1 + 26 + 22 = 49 register qubits, the width
    the unary-select test below states for the whole table. The prepare and its
    inverse, 2 * 4 * 1069 = 8552 T before their rotations, sit inside the 8 to 12
    T per coefficient that priced the whole table before (8208 to 12,312 T), 4.2%
    above its low end.
    """
    assert translation_invariant_lookup_prepare_cost(
        1026, 1, keep_bits=13
    ) == LookupPrepareCost(toffoli_count=1069, rotation_count=2, register_qubits=49)


def test_one_site_prepare_follows_the_table_not_the_lattice() -> None:
    """The prepare grows with one site's coefficients, and barely with the sites.

    The pionless Hamiltonian has the same 38 strings on every site of a periodic
    lattice, so the table holds 38 coefficients and an equal superposition picks
    the site. At mu = 13 and the 27 sites of 4He's lattice the prepare is 37
    (lookup) + 13 (comparison) + 6 (swap) + 10 (the table's equal superposition,
    odd part 19) + 10 (the sites', odd part 27) = 76 AND computations, 4
    rotations, and 1 + 26 + 12 + 5 = 44 register qubits, against 1069 and 49 for
    the whole 1026-term table in the test above. At 1000 sites the sites'
    superposition takes 2 * 7 (odd part 125) and the site register 10 qubits: 80
    and 49. At 76 coefficients on the 27 sites the lookup grows by 38 and the swap
    by one qubit, the odd part staying 19: 115 and 46. The cost follows the table,
    not the lattice. Each keep bit adds one AND computation to the comparison and
    a qubit each to the keep and sigma registers: at mu = 14, 77 and 46.
    """
    assert translation_invariant_lookup_prepare_cost(
        38, 27, keep_bits=13
    ) == LookupPrepareCost(toffoli_count=76, rotation_count=4, register_qubits=44)
    assert translation_invariant_lookup_prepare_cost(
        38, 27, keep_bits=14
    ) == LookupPrepareCost(toffoli_count=77, rotation_count=4, register_qubits=46)
    assert translation_invariant_lookup_prepare_cost(
        38, 1000, keep_bits=13
    ) == LookupPrepareCost(toffoli_count=80, rotation_count=4, register_qubits=49)
    assert translation_invariant_lookup_prepare_cost(
        76, 27, keep_bits=13
    ) == LookupPrepareCost(toffoli_count=115, rotation_count=4, register_qubits=46)


def test_one_site_prepare_needs_rotations_only_for_odd_parts() -> None:
    """A power-of-two size is put in equal superposition by Hadamards alone.

    Babbush+ 2018 Fig. 12 prepares an equal superposition over 2**k n values, n
    odd, with amplitude amplification on the odd part and two R_z rotations; at
    n = 1 "the R_z rotations are not needed". So each superposition costs its two
    rotations only when its size has an odd factor above 1. At mu = 14:

    - 32 coefficients on 64 sites: no rotations, 31 + 14 + 5 = 50 AND
      computations, 1 + 28 + 10 + 6 = 45 register qubits;
    - the smallest table, 2 coefficients on one site: no rotations, 1 + 14 + 1 =
      16 AND computations, 1 + 28 + 2 = 31 qubits;
    - 38 coefficients on one site: the table's 2 rotations alone;
    - 32 coefficients on 27 sites: the sites' 2 rotations alone.
    """
    assert translation_invariant_lookup_prepare_cost(
        32, 64, keep_bits=14
    ) == LookupPrepareCost(toffoli_count=50, rotation_count=0, register_qubits=45)
    assert translation_invariant_lookup_prepare_cost(
        2, 1, keep_bits=14
    ) == LookupPrepareCost(toffoli_count=16, rotation_count=0, register_qubits=31)
    assert translation_invariant_lookup_prepare_cost(
        38, 1, keep_bits=14
    ) == LookupPrepareCost(toffoli_count=67, rotation_count=2, register_qubits=41)
    assert translation_invariant_lookup_prepare_cost(
        32, 27, keep_bits=14
    ) == LookupPrepareCost(toffoli_count=60, rotation_count=2, register_qubits=44)


@pytest.mark.parametrize(
    ("coefficients_per_site", "site_count", "keep_bits", "message"),
    [
        (1, 27, 14, "coefficients_per_site must be an integer >= 2"),
        (0, 27, 14, "coefficients_per_site must be an integer >= 2"),
        (38.0, 27, 14, "coefficients_per_site must be an integer >= 2"),
        (True, 27, 14, "coefficients_per_site must be an integer >= 2"),
        (38, 0, 14, "site_count must be an integer >= 1"),
        (38, 27.0, 14, "site_count must be an integer >= 1"),
        (38, True, 14, "site_count must be an integer >= 1"),
        (38, 27, 0, "keep_bits must be an integer >= 1"),
        (38, 27, 14.0, "keep_bits must be an integer >= 1"),
        (38, 27, True, "keep_bits must be an integer >= 1"),
    ],
)
def test_one_site_prepare_rejects_sizes_that_are_not_whole_numbers_in_range(
    coefficients_per_site: int, site_count: int, keep_bits: int, message: str
) -> None:
    """A table, a lattice and a keep register each have a whole size in range.

    One coefficient needs no table, since the prepare is then the sites'
    superposition alone, which this cost does not describe, so fewer than two
    raise. A lattice has at least one site, and a keep register at least one bit.
    A whole-valued float is not an integer and raises, and so does True, a bool
    that equals 1. Each error names the argument.
    """
    with pytest.raises(ValueError, match=message):
        translation_invariant_lookup_prepare_cost(
            coefficients_per_site, site_count, keep_bits=keep_bits
        )


def test_one_site_prepare_cannot_be_asked_without_the_keep_bits_by_name() -> None:
    """The keep bits have no default and cannot be passed by position.

    mu follows from the caller's resolution and bound on ||H||
    (lookup_keep_probability_bits), so the prepare takes no default for it, and a
    call without it raises TypeError and names it. The three arguments are whole
    numbers, and a bare third one would not say at the call site that it is a bit
    count and not a size, so that raises too.
    """
    with pytest.raises(TypeError, match="keep_bits"):
        translation_invariant_lookup_prepare_cost(38, 27)
    with pytest.raises(TypeError, match="positional"):
        translation_invariant_lookup_prepare_cost(38, 27, 14)


def test_helium_walk_cost_follows_the_stated_prepare_price() -> None:
    """One 4He walk at three stated prepare prices.

    The case is 4He on a 3**3 lattice at a = 1.4 fm: Gamma = 38 * 27 = 1026
    (N_A = 11), and the pionless 1-norm of 9,132 MeV at 5 MeV resolution gives
    W_q = 11, textbook phase estimation's 2047 controlled qubiterates. The
    select and the controlled reflection are stated as the appendix's,
    15 * 2**10 + 6 * 11 - 28 = 15398 T and 8 * 11 - 9 = 79 T, so the App. C.1 T
    column is 15477 per qubiterate.

    A table-lookup prepare at 8 and at 12 T per coefficient adds 8208 and 12312
    per qubiterate: 2047 * (15477 + 8208) = 48,483,195 and
    2047 * (15477 + 12312) = 56,884,083 T gates.

    The appendix's rotation tree, counted as the circuit applies it (two
    prepares) and as the appendix states it (three rotations per gate), has
    2 * 2035 * 3 = 12210 rotations per qubiterate. Synthesized at a precision of
    1e-5 each, 30 T by the repeat-until-success count, that is 366,300 T per
    qubiterate and 2047 * (15477 + 366300) = 781,497,519 T gates, sixteen times
    the lookup at 8. The precision 1e-5 is usable only if the squared errors add,
    as for randomized synthesis: the walk's 2.5e7 rotations then sum to 2.5e-3.
    Added linearly, as coherent errors do, the same rotations sum to 250, and a
    total of 2.5e-3 would need 1e-10 per rotation.

    The 2033 target's envelope is 1e9 T gates, so the stated price decides which
    side of it this walk falls on. That comparison belongs to the analysis that
    sets the price and is not asserted here.

    Sourcing the walk steps from the live 1-norm -> W_q chain pins the whole
    composition: lattice to term count, norm to walk steps, and the caller's three
    prices added before the multiply. The split between the three parts is compared
    as well, because a total alone cannot show which part a wrong number came from.
    """
    site_count = 27
    num_terms = 38 * site_count
    one_norm = hamiltonian_one_norm(site_count, reference_couplings_at_spacing(1.4))
    query_count = qubiterate_qpe_ancilla_count(one_norm, 5.0).application_count
    appendix = controlled_qubiterate_cost(num_terms)

    lookup_low = per_shot_t_cost(
        query_count,
        select_t_per_qubiterate=appendix.select_t_count,
        reflection_t_per_qubiterate=appendix.reflection_t_count,
        prepare_t_per_qubiterate=table_lookup_prepare_t_count(
            num_terms, t_per_coefficient=8.0
        ),
    )
    lookup_high = per_shot_t_cost(
        query_count,
        select_t_per_qubiterate=appendix.select_t_count,
        reflection_t_per_qubiterate=appendix.reflection_t_count,
        prepare_t_per_qubiterate=table_lookup_prepare_t_count(
            num_terms, t_per_coefficient=12.0
        ),
    )
    rotation_count = prepare_rotation_count(
        num_terms, rotations_per_gate=3, prepares_per_qubiterate=2
    )
    synthesized = per_shot_t_cost(
        query_count,
        select_t_per_qubiterate=appendix.select_t_count,
        reflection_t_per_qubiterate=appendix.reflection_t_count,
        prepare_t_per_qubiterate=rotation_count * repeat_until_success_t_count(1e-5),
    )

    assert rotation_count == 12210
    assert lookup_low == PerShotCost(
        query_count=2047,
        select_t_per_qubiterate=15398,
        reflection_t_per_qubiterate=79,
        prepare_t_per_qubiterate=8208,
    )
    assert lookup_low.t_count == 48_483_195
    assert lookup_high == PerShotCost(
        query_count=2047,
        select_t_per_qubiterate=15398,
        reflection_t_per_qubiterate=79,
        prepare_t_per_qubiterate=12312,
    )
    assert lookup_high.t_count == 56_884_083
    assert synthesized == PerShotCost(
        query_count=2047,
        select_t_per_qubiterate=15398,
        reflection_t_per_qubiterate=79,
        prepare_t_per_qubiterate=366_300,
    )
    assert synthesized.t_count == 781_497_519


def test_unary_select_walk_for_helium_prices_each_part_of_a_step() -> None:
    """The 4He walk, composed with a select by unary iteration.

    The case is 4He on a 3**3 lattice at a = 1.4 fm: Gamma = 38 * 27 = 1026
    (N_A = 11), and the pionless 1-norm of 9,132 MeV at 5 MeV resolution gives
    W_q = 11, textbook phase estimation's 2047 controlled qubiterates. Each step
    costs three parts:

    - the select, by unary iteration: 1025 AND computations at 4 T each, 4100 T;
    - the controlled reflection, about the zero state of the 49 qubits the prepare
      acts on: 48 AND computations at 4 T each, 192 T;
    - a table-lookup prepare at 8 or at 12 T per coefficient, 8208 or 12312 T.

    The lookup prepare acts on 1 + 2 mu + 2 N_A qubits (Babbush+ 2018, Phys. Rev.
    X 8, 041015, Eq. 34), with mu the bits of its keep probabilities. Their Eq. 36
    sets mu = 13 here. Its leading term is log2(2 sqrt(2) * 9,132.075 / 5) = 12.33,
    and the other two keep the ceiling at 13 while ||H|| < 0.61 lambda; the bound
    on the 4-nucleon energies, 2,728.86 MeV (mostly the 2,252.475 MeV that removing
    the identity term raises them by), is 0.30 lambda. The prepare therefore spans
    1 + 26 + 22 = 49 qubits, where App. C.1's spans the 11 index qubits.

    A shot costs 2047 * 12500 = 25,587,500 or 2047 * 16604 = 33,988,388 T gates,
    0.9% and 0.7% above the 25,356,189 and 33,757,077 with App. C.1's reflection,
    8 * 11 - 9 = 79 T, and against 48,483,195 and 56,884,083 with App. C.1's
    select and reflection (the walk-cost test above).

    The walk steps come from the live 1-norm -> W_q chain and every part from its own
    function, so the test pins the composition as an analysis would call it. The
    whole result is compared, because a total alone cannot show which part a wrong
    number came from. Whether the walk fits a hardware envelope is a conclusion of
    the analysis that sets the prices, and is not asserted here.
    """
    site_count = 27
    num_terms = 38 * site_count
    couplings = reference_couplings_at_spacing(1.4)
    one_norm = hamiltonian_one_norm(site_count, couplings)
    query_count = qubiterate_qpe_ancilla_count(one_norm, 5.0).application_count
    t_per_and_computation = 4
    select_t = t_per_and_computation * unary_iteration_select_toffoli_count(num_terms)
    keep_bits = lookup_keep_probability_bits(
        one_norm,
        5.0,
        spectral_norm_bound=phase_estimation_energy_bound(4, site_count, couplings),
    )
    prepare_width = 1 + 2 * keep_bits + 2 * lcu_ancilla_register_size(num_terms)
    reflection_t = t_per_and_computation * controlled_reflection_toffoli_count(
        prepare_width
    )

    low = per_shot_t_cost(
        query_count,
        select_t_per_qubiterate=select_t,
        reflection_t_per_qubiterate=reflection_t,
        prepare_t_per_qubiterate=table_lookup_prepare_t_count(
            num_terms, t_per_coefficient=8.0
        ),
    )
    high = per_shot_t_cost(
        query_count,
        select_t_per_qubiterate=select_t,
        reflection_t_per_qubiterate=reflection_t,
        prepare_t_per_qubiterate=table_lookup_prepare_t_count(
            num_terms, t_per_coefficient=12.0
        ),
    )

    assert low == PerShotCost(
        query_count=2047,
        select_t_per_qubiterate=4100,
        reflection_t_per_qubiterate=192,
        prepare_t_per_qubiterate=8208,
    )
    assert low.t_count == 25_587_500
    assert high == PerShotCost(
        query_count=2047,
        select_t_per_qubiterate=4100,
        reflection_t_per_qubiterate=192,
        prepare_t_per_qubiterate=12312,
    )
    assert high.t_count == 33_988_388


def test_two_sided_walk_for_helium_with_a_one_site_prepare() -> None:
    """The 4He walk on a one-site table, read by two-sided phase estimation.

    The case is 4He on a 3**3 lattice at a = 1.4 fm: Gamma = 38 * 27 = 1026
    terms, with the pionless 1-norm lambda = 9,132.075 MeV, read at 5 MeV.
    Composed as an analysis would call it:

    - the walk steps: W_q = 11, and phase estimation that runs the inverse walk
      on the 0 branch takes 2**10 = 1024 of them, where textbook phase
      estimation takes 2047;
    - the keep bits: mu = 13 (Babbush+ 2018, Eq. 36), with ||H|| bounded on the
      4-nucleon energies, 2,728.86 MeV: mostly the 2,252.475 MeV that removing the
      identity term raises them by;
    - the prepare: the 38 coefficients of one site in the table and the 27 sites
      in equal superposition, for 76 AND computations, 4 rotations, and 44
      qubits held through the select;
    - the rotations: 1024 steps * 2 prepares * 4 = 8,192 in the walk, held
      within a stated synthesis budget of 0.01 with errors adding linearly, so
      1.2207e-6 each, 33 T by repeat-until-success;
    - per step, at 4 T per AND computation: the select by unary iteration,
      4 * 1025 = 4100 T; the controlled reflection over the 44 qubits,
      4 * 43 = 172 T; the prepare and its inverse, 2 * (4 * 76 + 4 * 33) = 872 T.

    A shot costs 1024 * 5144 = 5,267,456 T gates, against 25,587,500 to
    33,988,388 with textbook phase estimation and the whole 1026-term table at 8 to
    12 T per coefficient (the unary-select test above): 4.9 to 6.5 times fewer. The
    budget is this test's statement, not an analysis's, and covers the walk's
    rotations alone.

    The parts and the whole are compared, so a wrong number shows which part it
    came from.
    """
    site_count = 27
    coefficients_per_site = 38
    couplings = reference_couplings_at_spacing(1.4)
    one_norm = hamiltonian_one_norm(site_count, couplings)
    resolution = 5.0
    qpe = qubiterate_qpe_ancilla_count(one_norm, resolution)
    steps = two_sided_qpe_application_count(qpe.ancilla_qubits)
    four_nucleon_bound = phase_estimation_energy_bound(4, site_count, couplings)
    keep_bits = lookup_keep_probability_bits(
        one_norm, resolution, spectral_norm_bound=four_nucleon_bound
    )
    prepare = translation_invariant_lookup_prepare_cost(
        coefficients_per_site, site_count, keep_bits=keep_bits
    )
    walk_rotations = steps * 2 * prepare.rotation_count
    t_per_rotation = repeat_until_success_t_count(
        coherent_synthesis_precision(0.01, walk_rotations)
    )
    t_per_and_computation = 4
    shot = per_shot_t_cost(
        steps,
        select_t_per_qubiterate=t_per_and_computation
        * unary_iteration_select_toffoli_count(coefficients_per_site * site_count),
        reflection_t_per_qubiterate=t_per_and_computation
        * controlled_reflection_toffoli_count(prepare.register_qubits),
        prepare_t_per_qubiterate=2
        * (
            t_per_and_computation * prepare.toffoli_count
            + prepare.rotation_count * t_per_rotation
        ),
    )

    assert prepare == LookupPrepareCost(
        toffoli_count=76, rotation_count=4, register_qubits=44
    )
    assert t_per_rotation == 33
    assert shot == PerShotCost(
        query_count=1024,
        select_t_per_qubiterate=4100,
        reflection_t_per_qubiterate=172,
        prepare_t_per_qubiterate=872,
    )
    assert shot.t_count == 5_267_456


def test_zero_prepare_price_returns_the_app_c1_t_column_per_shot() -> None:
    """A zero price is the explicit way to ask for the App. C.1 T column of a shot.

    End-to-end at Roggero+'s 40Ar lattice point: the pionless 1-norm
    lambda(M=1000) @ a=1.4 fm at 10 MeV resolution gives 2**16 - 1 = 65535
    controlled qubiterates, and Gamma = 38 M = 38000 (N_A = 16) gives a T column
    of 491707, the select's 15*2**15 + 6*16 - 28 = 491588 plus the controlled
    reflection's 8*16 - 9 = 119. A shot with the prepare priced at zero therefore
    costs
        65535 * 491707 = 32224018245 ~ 3.2e10 T gates,
    a product past the 32-bit integer range. The source's own T total is this
    column, with the prepare's rotations reported beside it, so it has to stay
    reachable. It is reachable only by stating the appendix's select and reflection
    and writing the zero for the prepare. The query count comes from the live
    1-norm -> W_q -> walk steps chain, so the test also pins that composition for
    40Ar; the component-sum test is the check of the App. C.1 transcription.
    """
    one_norm = hamiltonian_one_norm(1000, reference_couplings_at_spacing(1.4))
    query_count = qubiterate_qpe_ancilla_count(one_norm, 10.0).application_count

    cost = per_shot_t_cost(
        query_count,
        select_t_per_qubiterate=491588,
        reflection_t_per_qubiterate=119,
        prepare_t_per_qubiterate=0,
    )

    assert cost == PerShotCost(
        query_count=65535,
        select_t_per_qubiterate=491588,
        reflection_t_per_qubiterate=119,
        prepare_t_per_qubiterate=0,
    )
    assert cost.t_count == 32224018245


def test_per_shot_cost_cannot_be_asked_without_each_price_by_name() -> None:
    """No price, of the select, the reflection or the prepare, has a default.

    A call that leaves out one price raises TypeError and names it:
    select_t_per_qubiterate, reflection_t_per_qubiterate or
    prepare_t_per_qubiterate. Prices passed by position raise as well. The first
    three are the contract of this function: no caller gets a per-shot number
    without stating what each part costs. The fourth keeps the statement readable
    at the call site, where a bare number would not say what it is.
    """
    with pytest.raises(TypeError, match="select_t_per_qubiterate"):
        per_shot_t_cost(
            2047, reflection_t_per_qubiterate=71, prepare_t_per_qubiterate=0
        )
    with pytest.raises(TypeError, match="reflection_t_per_qubiterate"):
        per_shot_t_cost(2047, select_t_per_qubiterate=7712, prepare_t_per_qubiterate=0)
    with pytest.raises(TypeError, match="prepare_t_per_qubiterate"):
        per_shot_t_cost(
            2047, select_t_per_qubiterate=7712, reflection_t_per_qubiterate=71
        )
    with pytest.raises(TypeError, match="positional"):
        per_shot_t_cost(2047, 7712, 71, 0)


def test_per_shot_cost_adds_the_prepare_price_before_multiplying() -> None:
    """The per-shot total is the query count times the sum of the three parts.

    At the 12C orbital point (Gamma = 760, N_A = 10) the App. C.1 T column is 7783
    per controlled qubiterate (7712 for the select and 71 for the controlled
    reflection, both stated here), and a lookup prepare at 8 T per coefficient is
    6080, so a shot of 2047 controlled qubiterates, the 2**11 - 1 walk steps of
    textbook phase estimation, costs 2047 * (7783 + 6080) = 28377561 T gates. The
    query count and the prices are passed directly, decoupled from the
    1-norm -> W_q chain and from the pricing functions that the end-to-end tests
    exercise, so a regression in the composition is localized here. The whole
    result is compared, which pins the split, and the total is compared beside it.
    """
    cost = per_shot_t_cost(
        2047,
        select_t_per_qubiterate=7712,
        reflection_t_per_qubiterate=71,
        prepare_t_per_qubiterate=6080,
    )

    assert cost == PerShotCost(
        query_count=2047,
        select_t_per_qubiterate=7712,
        reflection_t_per_qubiterate=71,
        prepare_t_per_qubiterate=6080,
    )
    assert cost.t_count == 28377561


def test_per_shot_cost_scales_linearly_in_query_count() -> None:
    """Doubling the query count doubles the T cost; the per-qubiterate parts are fixed.

    A shot of 2q identical controlled qubiterates costs exactly twice a shot of q,
    because the cost of one qubiterate depends only on the stated prices, not on
    how many times the walk is applied: the total doubles and the three
    per-qubiterate fields are unchanged across the two calls. The cross-call
    invariant catches a transposition of the query count with a per-qubiterate
    part, or a per-qubiterate part that wrongly varies with the query count --
    failures the single-point tests (each fixing one query count and one set of
    prices) cannot isolate. The prepare's price is nonzero, 6080 T per qubiterate,
    so that the prepare takes part in the doubling.
    """
    base = per_shot_t_cost(
        2047,
        select_t_per_qubiterate=7712,
        reflection_t_per_qubiterate=71,
        prepare_t_per_qubiterate=6080,
    )
    doubled = per_shot_t_cost(
        4094,
        select_t_per_qubiterate=7712,
        reflection_t_per_qubiterate=71,
        prepare_t_per_qubiterate=6080,
    )

    assert doubled.t_count == 2 * base.t_count
    assert doubled.select_t_per_qubiterate == base.select_t_per_qubiterate
    assert doubled.reflection_t_per_qubiterate == base.reflection_t_per_qubiterate
    assert doubled.prepare_t_per_qubiterate == base.prepare_t_per_qubiterate


@pytest.mark.parametrize(
    ("query_count", "num_terms", "prepare_price"),
    [
        (1, 380, 0),
        (1023, 380, 3040),
        (7, 5000, 40000.25),
        (250000, 1_000_000, 12_000_000.5),
    ],
)
def test_appendix_select_and_reflection_keep_every_per_shot_total_of_the_t_column(
    query_count: int, num_terms: int, prepare_price: float
) -> None:
    """Stating App. C.1's select and reflection keeps every per-shot total as it was.

    A caller that keeps the appendix's parts states them as
    controlled_qubiterate_cost(num_terms).select_t_count and .reflection_t_count,
    so the total is the query count times the T column plus the prepare, as before
    the select and the reflection became stated prices. The query count and the
    three prices are echoed back.

    A select price a quarter T above the appendix's then moves the total by a
    quarter T per application, and so does a reflection price a quarter T above
    the appendix's. Each stated price enters the total as given, fraction included,
    and is neither replaced by the appendix's nor rounded.

    Probed at term counts disjoint from the pinned ones (760, 1026, 38000) used
    elsewhere, so a special-cased formula cannot hide. The prepare prices are zero,
    a whole number and two binary fractions, because a price is an expected count;
    the fractions are exact in binary, so every comparison is exact. Gamma = 380
    is the pionless model at M = 10; 5000 and 1000000 span into the lattice regime.
    """
    appendix = controlled_qubiterate_cost(num_terms)

    cost = per_shot_t_cost(
        query_count,
        select_t_per_qubiterate=appendix.select_t_count,
        reflection_t_per_qubiterate=appendix.reflection_t_count,
        prepare_t_per_qubiterate=prepare_price,
    )
    shifted_select = per_shot_t_cost(
        query_count,
        select_t_per_qubiterate=appendix.select_t_count + 0.25,
        reflection_t_per_qubiterate=appendix.reflection_t_count,
        prepare_t_per_qubiterate=prepare_price,
    )
    shifted_reflection = per_shot_t_cost(
        query_count,
        select_t_per_qubiterate=appendix.select_t_count,
        reflection_t_per_qubiterate=appendix.reflection_t_count + 0.25,
        prepare_t_per_qubiterate=prepare_price,
    )

    assert cost.query_count == query_count
    assert cost.select_t_per_qubiterate == appendix.select_t_count
    assert cost.reflection_t_per_qubiterate == appendix.reflection_t_count
    assert cost.prepare_t_per_qubiterate == prepare_price
    assert cost.t_count == query_count * (appendix.t_count + prepare_price)
    assert shifted_select.t_count - cost.t_count == query_count * 0.25
    assert shifted_reflection.t_count - cost.t_count == query_count * 0.25


def test_per_shot_cost_rejects_subdomain_inputs() -> None:
    """Unusable query counts and unusable prices fail fast.

    A shot applies at least one controlled qubiterate, so query_count <= 0 is
    nonsensical and must raise. The count is a whole number of applications and is
    passed as an integer: a NaN or an infinite one would pass into a total without
    complaint, and a float, whole or not, would come back in a field that says
    integer. Each raises and names the query count. A negative price, for the
    select, the reflection or the prepare, would lower a total below what the other
    parts cost, and a price that is not a number, or is infinite, would pass into a
    total without complaint; each raises and names the price.

    The tightest valid corner -- query_count = 1 (the minimal shot, one
    application) with App. C.1's parts at its floor, num_terms = 3 (a select of 14 T and
    a reflection of 7 T), and a zero prepare -- succeeds and returns the floor's T
    column of 21, fixing the lower edge of the valid domain. With all three prices
    zero the same shot returns 0: zero is a price a caller may state, and the
    library adds nothing the caller did not state.
    """
    prices = {
        "select_t_per_qubiterate": 7712,
        "reflection_t_per_qubiterate": 71,
        "prepare_t_per_qubiterate": 0,
    }
    with pytest.raises(ValueError, match="query_count"):
        per_shot_t_cost(0, **prices)
    with pytest.raises(ValueError, match="query_count"):
        per_shot_t_cost(-1, **prices)
    with pytest.raises(ValueError, match="query_count"):
        per_shot_t_cost(math.nan, **prices)
    with pytest.raises(ValueError, match="query_count"):
        per_shot_t_cost(math.inf, **prices)
    with pytest.raises(ValueError, match="query_count"):
        per_shot_t_cost(2047.0, **prices)
    with pytest.raises(ValueError, match="prepare_t_per_qubiterate"):
        per_shot_t_cost(2047, **(prices | {"prepare_t_per_qubiterate": -1.0}))
    with pytest.raises(ValueError, match="prepare_t_per_qubiterate"):
        per_shot_t_cost(2047, **(prices | {"prepare_t_per_qubiterate": math.nan}))
    with pytest.raises(ValueError, match="prepare_t_per_qubiterate"):
        per_shot_t_cost(2047, **(prices | {"prepare_t_per_qubiterate": math.inf}))
    with pytest.raises(ValueError, match="select_t_per_qubiterate"):
        per_shot_t_cost(2047, **(prices | {"select_t_per_qubiterate": -1.0}))
    with pytest.raises(ValueError, match="select_t_per_qubiterate"):
        per_shot_t_cost(2047, **(prices | {"select_t_per_qubiterate": math.nan}))
    with pytest.raises(ValueError, match="select_t_per_qubiterate"):
        per_shot_t_cost(2047, **(prices | {"select_t_per_qubiterate": math.inf}))
    with pytest.raises(ValueError, match="reflection_t_per_qubiterate"):
        per_shot_t_cost(2047, **(prices | {"reflection_t_per_qubiterate": -1.0}))
    with pytest.raises(ValueError, match="reflection_t_per_qubiterate"):
        per_shot_t_cost(2047, **(prices | {"reflection_t_per_qubiterate": math.nan}))
    with pytest.raises(ValueError, match="reflection_t_per_qubiterate"):
        per_shot_t_cost(2047, **(prices | {"reflection_t_per_qubiterate": math.inf}))

    floor = per_shot_t_cost(
        1,
        select_t_per_qubiterate=14,
        reflection_t_per_qubiterate=7,
        prepare_t_per_qubiterate=0,
    )
    all_zero = per_shot_t_cost(
        1,
        select_t_per_qubiterate=0,
        reflection_t_per_qubiterate=0,
        prepare_t_per_qubiterate=0,
    )

    assert floor == PerShotCost(
        query_count=1,
        select_t_per_qubiterate=14,
        reflection_t_per_qubiterate=7,
        prepare_t_per_qubiterate=0,
    )
    assert floor.t_count == 21
    assert all_zero.t_count == 0


def test_equal_superposition_matches_the_fig_12_caption_at_odd_and_even_sizes() -> None:
    """Two rotations and 2 ceil(log2 L) ANDs for an odd part L, none for a power of two.

    Babbush+ 2018 Fig. 12 prepares (1/sqrt(2**k L)) sum |l> for L odd: the k
    factors of two take Hadamards alone, and the odd part takes one round of
    amplitude amplification with two R_z rotations and 8 log L T uncontrolled,
    2 ceil(log2 L) AND computations at 4 T each; "If L = 1, the Rz rotations are
    not needed" and the count is zero. Four sizes separate the parts: 54 values and
    the one-site table's 38 entries (odd parts 27 and 19:
    two rotations, ten ANDs, six register qubits each), the walk's 27 sites (two,
    ten, five), a power of two, 64 (none, none, six), and one value (all zero, the
    register's zero state).
    """
    assert equal_superposition_cost(54) == EqualSuperpositionCost(
        toffoli_count=10, rotation_count=2, register_qubits=6
    )
    assert equal_superposition_cost(38) == EqualSuperpositionCost(
        toffoli_count=10, rotation_count=2, register_qubits=6
    )
    assert equal_superposition_cost(27) == EqualSuperpositionCost(
        toffoli_count=10, rotation_count=2, register_qubits=5
    )
    assert equal_superposition_cost(64) == EqualSuperpositionCost(
        toffoli_count=0, rotation_count=0, register_qubits=6
    )
    assert equal_superposition_cost(1) == EqualSuperpositionCost(
        toffoli_count=0, rotation_count=0, register_qubits=0
    )


@pytest.mark.parametrize("size", [0, -1, 54.0, True])
def test_equal_superposition_rejects_sizes_that_are_not_positive_integers(
    size: float,
) -> None:
    """A superposition spans a whole number of values, at least one.

    Zero and negative sizes have no state to prepare; a whole-valued float is not
    an integer and raises, and so does True, a bool that equals 1. Each error
    names the argument.
    """
    with pytest.raises(ValueError, match="size must be an integer >= 1"):
        equal_superposition_cost(size)  # type: ignore[arg-type]


def test_keep_bits_for_helium_follow_the_delivered_error() -> None:
    """Helium's keep bits are 11 when mu takes the error the walk steps deliver.

    Three library numbers compose into the keep bits of the walk's prepare for 4He
    on 27 sites at a = 1.4 fm, with an 11-qubit energy register: lambda from
    hamiltonian_one_norm, the bound on the energies the phase estimation reads
    from phase_estimation_energy_bound (in the block-encoded frame), and the error
    from two_sided_qpe_rms_energy_error, which the 2**10 walk steps of two-sided
    phase estimation deliver. All three are in MeV, and lookup_keep_probability_bits
    gives 11, against 13 at the 5 MeV sizing parameter the earlier tests pass.
    """
    couplings = reference_couplings_at_spacing(1.4)
    one_norm = hamiltonian_one_norm(27, couplings)

    bits = lookup_keep_probability_bits(
        one_norm,
        two_sided_qpe_rms_energy_error(one_norm, 11),
        spectral_norm_bound=phase_estimation_energy_bound(4, 27, couplings),
    )

    assert bits == 11


def test_two_sided_phase_estimation_resolves_helium_to_twenty_mev() -> None:
    """1,024 applications of the 4He walk estimate the energy to 19.8 MeV RMS.

    Babbush+ 2018 (Eqs. 22-26) pair 2**m walk applications with a root-mean-square
    energy error of sqrt(2) pi lambda / 2**(m + 1): the Holevo error of a control
    register prepared in their state chi_m (Eq. 17), with the circuit's systematic
    phase error set equal to it and added in quadrature. Two-sided phase estimation
    on a W_q-qubit register applies the walk 2**(W_q - 1) times, so the error is
    sqrt(2) pi lambda / 2**W_q: for 4He on 27 sites, lambda =
    hamiltonian_one_norm(27) at 1.4 fm, 9,132.1 MeV, and W_q = 11, that is
    19.81 MeV. It is the register's mid-band energy bin,
    2 pi lambda / 2**W_q = 28.02 MeV, over sqrt(2), and each added register qubit
    halves it.
    """
    couplings = reference_couplings_at_spacing(1.4)
    one_norm = hamiltonian_one_norm(27, couplings)
    mid_band_bin = 2.0 * math.pi * one_norm / 2.0**11

    error = two_sided_qpe_rms_energy_error(one_norm, 11)

    assert one_norm == pytest.approx(9_132.1, abs=0.1)
    assert error == pytest.approx(19.81, abs=0.01)
    assert error == pytest.approx(mid_band_bin / math.sqrt(2.0))
    assert two_sided_qpe_rms_energy_error(one_norm, 12) == pytest.approx(error / 2)


@pytest.mark.parametrize(
    ("one_norm", "ancilla_qubits", "message"),
    [
        (0.0, 12, "one_norm must be a finite number > 0"),
        (-1.0, 12, "one_norm must be a finite number > 0"),
        (math.inf, 12, "one_norm must be a finite number > 0"),
        (math.nan, 12, "one_norm must be a finite number > 0"),
        (9_132.1, 0, "ancilla_qubits must be an integer >= 1"),
        (9_132.1, 12.0, "ancilla_qubits must be an integer >= 1"),
        (9_132.1, True, "ancilla_qubits must be an integer >= 1"),
    ],
)
def test_two_sided_rms_error_rejects_unusable_inputs(
    one_norm: float, ancilla_qubits: float, message: str
) -> None:
    """A norm is a finite positive energy and a register width a whole count.

    Zero, negative, infinite and nan norms would give a zero, negative, infinite
    or nan error that a keep-bit count downstream could not reject; a register
    of no qubits estimates nothing; a whole-valued float is not an integer and
    raises, and so does True, a bool that equals 1. Each error names the
    argument.
    """
    with pytest.raises(ValueError, match=message):
        two_sided_qpe_rms_energy_error(one_norm, ancilla_qubits)  # type: ignore[arg-type]


def test_two_sided_rms_error_survives_very_wide_registers() -> None:
    """A register wider than a float's exponent still gets a finite, positive error.

    No calculation proposes a 1,024-qubit energy register, but the formula must
    not raise OverflowError there, an exception outside the ValueError its
    docstring names: dividing by 2**1024 as a float overflows, while scaling the
    exponent does not. The error stays positive and finite, and halves with the
    extra qubit, as at every width.
    """
    one_norm = 9_132.1

    wide = two_sided_qpe_rms_energy_error(one_norm, 1024)
    wider = two_sided_qpe_rms_energy_error(one_norm, 1025)

    assert 0.0 < wide < math.inf
    assert wider == pytest.approx(wide / 2)
