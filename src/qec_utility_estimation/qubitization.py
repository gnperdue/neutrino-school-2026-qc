"""Roggero+ 2019 App. C controlled-qubiterate gate-cost model.

Gate counts for the qubitization walk operator ("qubiterate") of a
Linear-Combination-of-Unitaries (LCU) block-encoding, after Roggero+ 2019
(arXiv:1911.06368) Appendix C.1. The appendix's closed-form lower bounds for one
controlled qubiterate are functions of the LCU term count Gamma alone, through
the index-register width N_A = ceil(log2(Gamma)).

The rest of the module builds on those counts and depends on more than Gamma.
The select, the controlled reflection and the prepare are priced under
conventions the caller states. The select is the appendix's, sized by the 2**N_A
slots of the index register, or one by unary iteration (Babbush+ 2018), linear in
Gamma itself. The controlled reflection covers every qubit the prepare acts on:
the appendix's is sized for its own prepare on the N_A index qubits, and an AND
chain (Babbush+ 2018) for the width of the caller's prepare. The prepare, which
the appendix counts in one-qubit gates, is a rotation count, sized like the
appendix's select by the 2**N_A slots, or a table lookup: at the caller's rate,
linear in Gamma, or itemized from Babbush+ 2018 for a Hamiltonian that repeats on
every site, linear in one site's coefficients, with keep probabilities of the
bits their Eq. 36 sets. The applications in one shot are the walk steps of
textbook phase estimation, those of phase estimation that runs the inverse walk on
the 0 branch (Babbush+ 2018), about half as many, or a QSVT evolution degree. The
per-shot T count composes the three stated parts, and is the appendix's lower
bound only when its select and its reflection are stated and the prepare is
priced at zero.
"""

import math
from dataclasses import dataclass

from scipy.special import jv


def lcu_ancilla_register_size(num_terms: int) -> int:
    """N_A = ceil(log2(num_terms)), the LCU index-register width (App. C.1).

    Computed as ``(num_terms - 1).bit_length()`` -- exact integer arithmetic that
    avoids the float-log2 rounding error at exact powers of two.

    Args:
        num_terms: Number of LCU terms Gamma; an integer >= 1.

    Returns:
        The index-register width N_A in qubits.

    Raises:
        ValueError: If num_terms is not an integer >= 1 (an LCU has a whole
            number of terms, at least one; a float raises, whole or not).
    """
    if type(num_terms) is not int or num_terms < 1:
        raise ValueError(
            "lcu_ancilla_register_size: num_terms must be an integer >= 1; got "
            f"{num_terms!r}."
        )
    return (num_terms - 1).bit_length()


@dataclass(frozen=True)
class QubiterateCost:
    """Controlled-qubiterate lower-bound gate counts (Roggero+ App. C.1).

    The source counts "only two-qubit Clifford gates and rotations or T gates", so
    the Hadamard, Phase and NOT gates of the select and the reflection are in none
    of these fields.

    Attributes:
        select_t_count: Number of T / T-dagger gates of the select,
            15 * 2**(N_A - 1) + 6 N_A - 28. The count is sized by the 2**N_A slots
            of the index register, not by the Gamma terms.
        reflection_t_count: Number of T / T-dagger gates of the controlled
            reflection, 8 N_A - 9: a Z with N_A controls, sized for the appendix's
            prepare, which acts on the N_A index qubits alone.
        cnot_count: Number of CNOT gates (prepare, controlled reflection, select).
        single_qubit_count: Number of arbitrary one-qubit unitaries U(2), all from
            the prepare. The source charges three z-rotations for each; converting
            them to T gates is left to the caller.
        ancilla_count: Number of ancilla qubits: the N_A index register plus the
            select's N_A - 1 work qubits, which the reflection reuses.
    """

    select_t_count: int
    reflection_t_count: int
    cnot_count: int
    single_qubit_count: int
    ancilla_count: int

    @property
    def t_count(self) -> int:
        """The App. C.1 T column: the select plus the controlled reflection.

        It is not the full non-Clifford cost of a qubiterate: the prepare's
        rotations are in ``single_qubit_count`` and carry no T.
        """
        return self.select_t_count + self.reflection_t_count


def controlled_qubiterate_cost(num_terms: int) -> QubiterateCost:
    """The App. C.1 lower-bound counts for one controlled qubiterate.

    With N_A = ``lcu_ancilla_register_size(num_terms)``:

        select_t_count     = 15 * 2**(N_A - 1) + 6 * N_A - 28
        reflection_t_count = 8 * N_A - 9
        cnot_count         = 17 * 2**(N_A - 1) + 10 * N_A - 34
        single_qubit_count = 2**N_A - N_A - 2
        ancilla_count      = 2 * N_A - 1

    and the T column, ``t_count``, is the sum of the first two,
    15 * 2**(N_A - 1) + 14 * N_A - 37.

    The source writes the leading terms as an implicit product, "15 2^(N_A-1)"
    and "17 2^(N_A-1)", which typesets like 152 and 172 raised to a power. They
    are 15 and 17 times a power of two: App. C.1 prices the qubiterate as one
    prepare (2**N_A - 2 N_A - 2 CNOT), one controlled reflection (8 N_A - 9 T,
    6 N_A - 6 CNOT) and one select (15 * 2**(N_A - 1) + 6 N_A - 28 T,
    15 * 2**(N_A - 1) + 6 N_A - 26 CNOT), and the T and CNOT totals above are
    those sums.

    What the counts leave out. ``t_count`` is the select plus the controlled
    reflection: the prepare's 2**N_A - N_A - 2 arbitrary one-qubit unitaries are
    counted in ``single_qubit_count`` and carry no T, so ``t_count`` is not the
    full non-Clifford cost. At the source's three z-rotations per unitary and a
    few tens of T gates per synthesized rotation, the prepare is an order of
    magnitude more than ``t_count``; a caller that needs T gates must price it
    (prepare_rotation_count and table_lookup_prepare_t_count give the two ways,
    and per_shot_t_cost takes the price). The source also counts one prepare per
    qubiterate although the walk applies two, and neglects the Gamma controlled
    unitaries of the select, which are Clifford. Once per shot, and so outside
    these per-qubiterate counts, it neglects an initial controlled prepare as
    well.

    The closed forms are large-register expressions and are rejected below
    N_A = 2 (num_terms = 3), the smallest register size at which all four totals
    are non-negative: at N_A = 1 (num_terms = 2) the T, CNOT and single-qubit
    totals are -8, -7 and -1, so this raises rather than return them. At the floor
    the totals are 21 T, 20 CNOT, 0 single-qubit and 3 ancillas; the single-qubit
    count of exactly 0 is the degenerate edge of the asymptotic, and
    prepare_rotation_count refuses to price a prepare from it. The CNOT total
    there still contains the prepare's term 2**N_A - 2 N_A - 2 = -2 (0 at N_A = 3,
    positive beyond), so near the floor the totals are the formulas' values and
    not a physical bound.

    Args:
        num_terms: Number of LCU terms Gamma; an integer >= 3.

    Returns:
        The controlled-qubiterate gate counts.

    Raises:
        ValueError: If num_terms is not an integer (propagated from
            lcu_ancilla_register_size), or if the index register N_A < 2
            (num_terms < 3), where the App. C.1 lower bound breaks down.
    """
    register_size = lcu_ancilla_register_size(num_terms)
    if register_size < 2:
        raise ValueError(
            "controlled_qubiterate_cost: the App. C.1 lower bound is valid only "
            f"for index register N_A >= 2 (num_terms >= 3); got num_terms={num_terms} "
            f"(N_A={register_size}), where the T, CNOT and single-qubit counts "
            "would be negative."
        )
    leading = 2 ** (register_size - 1)
    return QubiterateCost(
        select_t_count=15 * leading + 6 * register_size - 28,
        reflection_t_count=8 * register_size - 9,
        cnot_count=17 * leading + 10 * register_size - 34,
        single_qubit_count=2**register_size - register_size - 2,
        ancilla_count=2 * register_size - 1,
    )


def unary_iteration_select_toffoli_count(num_terms: int) -> int:
    """Toffoli gates of a controlled select over num_terms terms, by unary iteration.

    Babbush+ 2018 (arXiv:1805.03662, Phys. Rev. X 8, 041015) Sec. III A builds a
    controlled indexed operation over L items (their Eq. 29) from L - 1 AND
    computations and nothing else that consumes T gates. Each AND computation is a
    Toffoli onto a fresh qubit that costs 4 T gates to compute and none to
    uncompute, which is done by measurement, so the source's count is 4L - 4 T
    gates (their Fig. 7, at L = 11: 10 AND computations, 40 T). The control is part
    of the count. The count assumes each selected term is a Pauli string, with its
    sign, which is Clifford once controlled on the unary flag (their Sec. III B
    extends Eq. 29 from X_l to any set of Pauli operators). This returns

        num_terms - 1,

    the count of AND computations, in Toffoli gates. The caller converts to T gates
    at its stated rate; the source's is 4 per AND computation.

    The count follows the number of terms. App. C.1's select,
    QubiterateCost.select_t_count, follows the 2**N_A slots of the index register
    instead: at Gamma = 1024 it is 7712 T against 1023 AND computations here (4092
    T at the source's rate), and at 1025, one term more, it is 15398 T against 1024
    (4096 T).

    Args:
        num_terms: Number of LCU terms Gamma; an integer >= 1.

    Returns:
        The Toffoli count of one controlled select; 0 for a single term.

    Raises:
        ValueError: If num_terms is not an integer >= 1 (a float raises, whole or
            not).
    """
    if type(num_terms) is not int or num_terms < 1:
        raise ValueError(
            "unary_iteration_select_toffoli_count: num_terms must be an integer >= 1 "
            "(a select runs over a whole number of terms, at least one); got "
            f"{num_terms!r}."
        )
    return num_terms - 1


def controlled_reflection_toffoli_count(num_qubits: int) -> int:
    """Toffoli gates of a reflection about the zero state of num_qubits, controlled.

    The walk's reflection 2|0><0| - 1 acts on every qubit the prepare acts on and,
    in the controlled walk, is controlled on the phase-estimation ancilla
    (Babbush+ 2018, arXiv:1805.03662, Phys. Rev. X 8, 041015, Fig. 1). Up to NOT
    gates and a Clifford phase on the control, it is a Z on one of the n qubits
    whose n controls are the other n - 1 and the control. The controls expand into
    n - 1 nested AND computations (their Sec. III A, Figs. 4 and 5), each a Toffoli
    onto a fresh qubit at 4 T to compute and none to uncompute, and the Z under the
    one remaining control is a CZ, a Clifford gate. The AND chain holds n - 1 work
    qubits at its peak. This returns

        num_qubits - 1,

    in Toffoli gates, like unary_iteration_select_toffoli_count; the caller
    converts to T gates at its stated rate.

    The width is the caller's, and it follows the prepare. App. C.1's prepare acts
    on the N_A index qubits alone, and the appendix charges the same gate
    QubiterateCost.reflection_t_count = 8 N_A - 9 T: 79 T at N_A = 11, against 10
    AND computations here (40 T at 4 T each). A table-lookup prepare of all Gamma
    coefficients acts on 1 + 2 mu + 2 N_A qubits (Babbush+ Eq. 34 and Sec. III D),
    with mu the bits of its keep probabilities: 49 at Gamma = 1026 (N_A = 11) and
    mu = 13, for 48 AND computations. One that stores a single site's coefficients
    spans LookupPrepareCost.register_qubits
    (translation_invariant_lookup_prepare_cost): 44 at 38 coefficients per site,
    27 sites and mu = 13, for 43.

    Args:
        num_qubits: Width n of the register reflected about its zero state, in
            qubits, the control not included; an integer >= 1.

    Returns:
        The Toffoli count of one controlled reflection; 0 for a single qubit.

    Raises:
        ValueError: If num_qubits is not an integer >= 1 (a float raises, whole or
            not).
    """
    if type(num_qubits) is not int or num_qubits < 1:
        raise ValueError(
            "controlled_reflection_toffoli_count: num_qubits must be an integer >= 1 "
            "(a reflection acts on a whole number of qubits, at least one); got "
            f"{num_qubits!r}."
        )
    return num_qubits - 1


def reflection_toffoli_count(num_qubits: int) -> int:
    """Toffoli gates of a reflection about the zero state of num_qubits, uncontrolled.

    The reflection 2|0><0| - 1 on n qubits is, up to NOT gates and a global phase,
    a Z on one qubit controlled on the other n - 1. The controls expand into n - 2
    nested AND computations (Babbush+ 2018, arXiv:1805.03662, Phys. Rev. X 8,
    041015, Sec. III A, Figs. 4 and 5), each a Toffoli onto a fresh qubit at 4 T
    to compute and none to uncompute, and the Z under the one remaining control is
    a CZ, a Clifford gate; the AND chain holds n - 2 work qubits at its peak. This
    returns

        max(num_qubits - 2, 0),

    one fewer than controlled_reflection_toffoli_count, whose extra control is one
    more AND computation; a one- or two-qubit reflection is a Z or a CZ, with no
    Toffoli. The caller converts to T gates at its stated rate.

    Amplitude amplification uses it twice a round: about the zero state of the
    registers the preparation starts from (the reflection of Brassard, Hoyer,
    Mosca and Tapp 2002, arXiv:quant-ph/0005055) and about the zero state of the
    index register that marks success (the good-subspace reflection). For 4He on the
    3**3 lattice those are 128 over the 108 system, 7 index and 15 enumeration qubits,
    and 5 over the index register. A table load's enumeration register counts among the
    registers the preparation starts from, since undoing the load does not always
    return it to zero.

    Args:
        num_qubits: Width n of the register reflected about its zero state, in
            qubits; an integer >= 1.

    Returns:
        The Toffoli count of one uncontrolled reflection; 0 at one or two qubits.

    Raises:
        ValueError: If num_qubits is not an integer >= 1 (a float raises, whole or
            not).
    """
    if type(num_qubits) is not int or num_qubits < 1:
        raise ValueError(
            "reflection_toffoli_count: num_qubits must be an integer >= 1 (a "
            "reflection acts on a whole number of qubits, at least one); got "
            f"{num_qubits!r}."
        )
    return max(num_qubits - 2, 0)


def _is_integer_among(value: object, allowed: tuple[int, ...]) -> bool:
    """Whether value is an int and one of the allowed values.

    Membership alone compares by value, so it would accept True for 1 and 2.0 for 2.
    """
    return type(value) is int and value in allowed


def prepare_rotation_count(
    num_terms: int, *, rotations_per_gate: int, prepares_per_qubiterate: int
) -> int:
    """Number of z-rotations in the prepare stage of one controlled qubiterate.

    Roggero+ 2019 (arXiv:1911.06368) App. C.1 builds the prepare V_P from at most
    2**N_A - N_A - 2 arbitrary one-qubit gates, with
    N_A = ``lcu_ancilla_register_size(num_terms)``. This returns

        prepares_per_qubiterate * (2**N_A - N_A - 2) * rotations_per_gate.

    The two keyword arguments are counting conventions. The library chooses
    neither, so neither has a default:

    - ``rotations_per_gate``. A one-qubit unitary is one to three z-rotations,
      and that range is accepted. 3 is the appendix's stated count ("to be
      conservative, we will count 3 z-rotations per one-qubit unitary"). 1 charges
      each gate as a single R_y, as in a rotation tree on real amplitudes. The
      appendix does not state its gate count for that charge, and the count per
      prepare can then fall below the Gamma - 1 rotations that Gamma generic
      non-negative amplitudes need, by at most N_A + 1 and only when Gamma is
      within N_A of 2**N_A. With 2, the App. C.1 T column plus one prepare's
      rotations comes within 2% of the upper of the two qubitization lines of the
      source's Fig. 5 at its 40Ar point (N_A = 16), which counts each T gate as
      one rotation. That figure plots totals over phase estimation's walk steps,
      and the comparison divides its line by the 2**18 - 1 applications of the
      paper's own register (W_q = 18). What the figure charged per gate is not
      established.
    - ``prepares_per_qubiterate``. 2 is what one controlled qubiterate applies
      (V_P and its inverse). 1 is what the appendix counts, as a lower bound,
      because constructions that need one prepare per step are known.

    Args:
        num_terms: Number of LCU terms Gamma; an integer >= 5.
        rotations_per_gate: z-rotations charged to each one-qubit gate; the
            integer 1, 2 or 3.
        prepares_per_qubiterate: Prepare applications charged to one controlled
            qubiterate; the integer 1 or 2.

    Returns:
        The number of z-rotations per controlled qubiterate, a dimensionless
        count; at least 3.

    Raises:
        ValueError: If rotations_per_gate is not the integer 1, 2 or 3; if
            prepares_per_qubiterate is not the integer 1 or 2 (a float or a
            boolean that equals an accepted value raises as well); if num_terms
            is 3 or 4 (N_A = 2), where the closed form has no one-qubit gates and
            a price built from it would be zero without the caller stating a
            zero; or if num_terms is not an integer or is below 3 (propagated
            from controlled_qubiterate_cost).
    """
    if not _is_integer_among(rotations_per_gate, (1, 2, 3)):
        raise ValueError(
            "prepare_rotation_count: rotations_per_gate must be the integer 1, 2 or "
            "3 (a one-qubit unitary is one to three z-rotations); got "
            f"{rotations_per_gate!r}."
        )
    if not _is_integer_among(prepares_per_qubiterate, (1, 2)):
        raise ValueError(
            "prepare_rotation_count: prepares_per_qubiterate must be the integer 1 "
            "or 2 (a controlled qubiterate applies two prepares and App. C.1 counts "
            f"one); got {prepares_per_qubiterate!r}."
        )
    gate_count = controlled_qubiterate_cost(num_terms).single_qubit_count
    if gate_count == 0:
        raise ValueError(
            "prepare_rotation_count: App. C.1's count of one-qubit gates, "
            "2**N_A - N_A - 2, is 0 at N_A = 2, which would price the prepare of "
            f"num_terms={num_terms} amplitudes at zero; the count needs "
            "num_terms >= 5 (N_A >= 3)."
        )
    return prepares_per_qubiterate * gate_count * rotations_per_gate


def table_lookup_prepare_t_count(num_terms: int, *, t_per_coefficient: float) -> float:
    """T gates of a table-lookup prepare stage of one controlled qubiterate.

    An alternative to the rotation tree of App. C.1: the LCU coefficients are
    loaded by table lookup (QROM with coherent alias sampling; Babbush+ 2018,
    Phys. Rev. X 8, 041015, the reference App. C.1 cites for a single prepare per
    step), whose T count grows linearly with the number of stored coefficients.
    This returns

        t_per_coefficient * num_terms.

    ``t_per_coefficient`` is the caller's rate. It must cover every prepare
    application inside one controlled qubiterate (V_P and its inverse). The
    library asserts no value for it.

    The App. C.1 floor of three terms belongs to the rotation tree's closed forms
    and does not bind a lookup, so a table of one coefficient is accepted.

    The qubits of a lookup are not counted in this module.
    ``QubiterateCost.ancilla_count`` is the index register plus the select's work
    qubits, which is all the rotation tree uses. The lookup circuit of Babbush+
    2018 (its Fig. 11) spans 2 mu + 2 log2(Gamma) + O(1) qubits, index register
    included, for coefficients kept to mu bits. A caller that takes the T price
    from here adds those registers to its qubit count.

    Args:
        num_terms: Number of LCU terms Gamma, which is the number of stored
            coefficients; an integer >= 1.
        t_per_coefficient: T gates per stored coefficient per controlled
            qubiterate; a finite number > 0.

    Returns:
        The prepare's T gates per controlled qubiterate; fractional if the rate
        is.

    Raises:
        ValueError: If t_per_coefficient is not a finite number > 0, or if
            num_terms is not an integer >= 1 (a float raises, whole or not).
    """
    if not 0 < t_per_coefficient < math.inf:
        raise ValueError(
            "table_lookup_prepare_t_count: t_per_coefficient must be a finite "
            f"number > 0; got {t_per_coefficient}."
        )
    if type(num_terms) is not int or num_terms < 1:
        raise ValueError(
            "table_lookup_prepare_t_count: num_terms must be an integer >= 1 (a "
            "table holds a whole number of coefficients, at least one); got "
            f"{num_terms!r}."
        )
    return t_per_coefficient * num_terms


def lookup_keep_probability_bits(
    one_norm: float, resolution: float, *, spectral_norm_bound: float
) -> int:
    """Bits mu of a lookup prepare's keep probabilities (Babbush+ 2018, Eq. 36).

    The lookup prepare of Babbush+ 2018 (arXiv:1805.03662, Phys. Rev. X 8,
    041015, Sec. III D, Fig. 11) stores an alternate index and a mu-bit keep value
    for each of its L entries, and compares the keep value with a mu-bit register
    in equal superposition. The probabilities it prepares are multiples of
    1 / (2**mu L), each within 1 / (2**mu L) of its w_l / lambda (Eq. 35). Eq. (36)
    gives the bits that keep this error within the bound App. A derives for the
    prepare's share of the error budget (Eqs. 25 and A10):

        mu = ceil( log2(2 sqrt(2) lambda / Delta-E)
                   + log2(1 + Delta-E**2 / (8 lambda**2))
                   - log2(1 - ||H||**2 / lambda**2) ).

    The second term is below 0.17 bits while Delta-E < lambda. The third grows
    without bound as ||H|| approaches lambda, so the caller states a bound on
    ||H||, in the frame of the block-encoded Hamiltonian, which drops the
    constants of the Hamiltonian it encodes and so shifts every energy by minus
    their sum (for the pionless Hamiltonian, the kinetic term's 2 D t A and the
    potential's identity M (3/2 C0 + 1/2 D0), Roggero+ 2019 Eq. 1; see
    phase_estimation_energy_bound in the pionless app).

    Which energies ||H|| bounds is this library's choice. App. A bounds the phase
    error of all eigenvalues at once, with the operator norm over the whole space
    the walk acts on (Eqs. A4-A8). Here ||H|| is the largest |E| among the energies
    the phase estimation reads: Eqs. (A4)-(A8) hold for one eigenvalue E with |E|
    in place of ||H||, since rounding moves each eigenvalue by at most lambda / 2**mu
    (Weyl's inequality with Eqs. 35 and A6), and an eigenvalue that is never read
    does not enter the estimate. The two readings need not agree, though for 4He on
    the pionless 3**3 lattice at 1.4 fm and Delta-E = 5 MeV both give mu = 13: the
    4-nucleon energies lie below 0.30 lambda and the whole-space norm below 0.36
    lambda, and mu stays 13 up to 0.61 lambda.

    Eq. (35)'s coefficient count cancels; the three terms are one log2 of a product.

    Units: the three energies share one unit (MeV by project convention); mu is bits.

    Args:
        one_norm: LCU 1-norm lambda; a finite number > 0.
        resolution: Target error Delta-E of Eqs. (35)-(36), the root-mean-square
            error of the energy estimate (Eqs. 21-23), same unit; a finite number
            > 0 and below one_norm. Eqs. (24) and (26) pair 2**m walk steps with
            Delta-E >= sqrt(2) pi lambda / 2**(m + 1); a smaller Delta-E, such as
            the resolution that sizes the energy register, is conservative.
        spectral_norm_bound: Bound on ||H||, the largest |E| among the energies
            the phase estimation reads, in the frame above, same unit; a finite
            number > 0 and below one_norm.

    Returns:
        mu, an integer >= 2.

    Raises:
        ValueError: If any argument is not a finite number > 0, if resolution or
            spectral_norm_bound is not below one_norm, or if one_norm / resolution
            is so large that the logarithm's argument overflows.
    """
    if not 0 < one_norm < math.inf:
        raise ValueError(
            "lookup_keep_probability_bits: one_norm must be a finite number > 0; got "
            f"{one_norm}."
        )
    if not 0 < resolution < math.inf:
        raise ValueError(
            "lookup_keep_probability_bits: resolution must be a finite number > 0; "
            f"got {resolution}."
        )
    if not 0 < spectral_norm_bound < math.inf:
        raise ValueError(
            "lookup_keep_probability_bits: spectral_norm_bound must be a finite number "
            f"> 0; got {spectral_norm_bound}."
        )
    if resolution >= one_norm:
        raise ValueError(
            "lookup_keep_probability_bits: resolution must be below one_norm (the "
            "source does not take Delta-E larger than lambda, and at Delta-E = lambda "
            "an estimate of zero suffices, since every energy lies within lambda of "
            f"zero); got resolution={resolution}, one_norm={one_norm}."
        )
    if spectral_norm_bound >= one_norm:
        raise ValueError(
            "lookup_keep_probability_bits: spectral_norm_bound must be below one_norm "
            "(the term -log2(1 - ||H||**2 / lambda**2) diverges at ||H|| = lambda); "
            f"got spectral_norm_bound={spectral_norm_bound}, one_norm={one_norm}."
        )
    resolution_term = 1 + (resolution / one_norm) ** 2 / 8
    norm_term = 1 - (spectral_norm_bound / one_norm) ** 2
    argument = 2 * math.sqrt(2) * (one_norm / resolution) * resolution_term / norm_term
    if math.isinf(argument):
        raise ValueError(
            "lookup_keep_probability_bits: the logarithm's argument, about "
            "2 sqrt(2) one_norm / resolution, overflows; got "
            f"one_norm={one_norm}, resolution={resolution}."
        )
    # An exact ceiling: a float log2 can round an argument just above 2**j down to j.
    mantissa, exponent = math.frexp(argument)
    return exponent - 1 if mantissa == 0.5 else exponent


@dataclass(frozen=True)
class LookupPrepareCost:
    """Cost of one lookup prepare (Babbush+ 2018, Sec. III D, Figs. 10-12).

    Attributes:
        toffoli_count: AND computations of one prepare, 4 T each in the source;
            the caller converts at its stated rate.
        rotation_count: Single-qubit rotations of one prepare that need
            synthesis; the caller prices them.
        register_qubits: Qubits the prepare leaves holding data until its inverse
            runs: the width of the walk's reflection and the prepare's share of
            the walk's qubits. Transient work qubits are not included.
    """

    toffoli_count: int
    rotation_count: int
    register_qubits: int


@dataclass(frozen=True)
class EqualSuperpositionCost:
    """Cost of an equal superposition over a register's values (Babbush+ Fig. 12).

    Attributes:
        toffoli_count: AND computations of the amplitude amplification over the
            odd part of the size; none for a power of two.
        rotation_count: R_z rotations of that amplification, two or none.
        register_qubits: Qubits holding the superposition, ceil(log2 size).
            Transient work qubits (the inequality test's, log of the odd part
            plus a constant in the source) are not included.
    """

    toffoli_count: int
    rotation_count: int
    register_qubits: int


def equal_superposition_cost(size: int) -> EqualSuperpositionCost:
    """Cost of preparing an equal superposition over size values.

    Babbush+ 2018 (arXiv:1805.03662, Phys. Rev. X 8, 041015) Fig. 12 prepares
    (1/sqrt(2**k L)) sum_l |l> for size = 2**k L with L odd: the k factors of two
    take Hadamards alone, and the odd part takes one round of amplitude
    amplification on ceil(log2 L) qubits, "8 log L" T uncontrolled in the caption,
    2 ceil(log2 L) AND computations at 4 T each, plus two R_z rotations; "If L = 1,
    the Rz rotations are not needed", and the count is zero. This returns

        toffoli_count   = 2 ceil(log2 L)  if L > 1 else 0,
        rotation_count  = 2               if L > 1 else 0,
        register_qubits = ceil(log2 size).

    The walk's one-site prepare uses it twice, for the table's entries and for
    the sites (translation_invariant_lookup_prepare_cost), and a density insertion,
    whose coefficients share one magnitude, uses it once per axis as its whole prepare.

    Args:
        size: Number of values in the superposition; an integer >= 1 (one value
            is the register's zero state, at no cost).

    Returns:
        The AND, rotation and register-qubit counts of one preparation.

    Raises:
        ValueError: If size is not an integer >= 1 (a float raises, whole or
            not).
    """
    if type(size) is not int or size < 1:
        raise ValueError(
            "equal_superposition_cost: size must be an integer >= 1 (a superposition "
            f"spans a whole number of values, at least one); got {size!r}."
        )
    odd_part = size // (size & -size)
    return EqualSuperpositionCost(
        toffoli_count=2 * (odd_part - 1).bit_length(),
        rotation_count=2 if odd_part > 1 else 0,
        register_qubits=(size - 1).bit_length(),
    )


def translation_invariant_lookup_prepare_cost(
    coefficients_per_site: int, site_count: int, *, keep_bits: int
) -> LookupPrepareCost:
    """Cost of a lookup prepare for a Hamiltonian that repeats on every site.

    On a periodic lattice the LCU coefficients of one site repeat at every site.
    The prepare then stores one site's L coefficients, by the coherent alias
    sampling of Babbush+ 2018 (arXiv:1805.03662, Phys. Rev. X 8, 041015,
    Sec. III D, Fig. 11), and reaches the M sites by an equal superposition
    (their Fig. 12). Babbush+ treat their own lattice models this way. Their
    electronic-structure Hamiltonian, Eq. (41), which covers jellium, has "only
    O(N) unique coefficients" (Sec. IV B), and the index over which its
    coefficients repeat is put in equal superposition (p. 20 and the Fig. 16
    caption). Their Hubbard model has "only three unique coefficients" (Sec. V A),
    and their Fig. 20 puts its site registers in equal superposition.

    One prepare costs, in AND computations (4 T each in the source):

    - the table lookup over L entries: L - 1 (Sec. III C, Fig. 10: 4L - 4 T,
      independent of the word length);
    - the comparison of the mu-bit keep and sigma registers: mu, the 4 mu T of
      the Fig. 11 caption;
    - the controlled swap of the index and alternate-index registers: one per
      qubit, ceil(log2 L);
    - the equal superpositions over the L entries and over the M sites: for a
      size 2**k n with n odd, 2 ceil(log2 n) (Fig. 12 caption, uncontrolled:
      8 log n T), and none when n = 1.

    Each equal superposition over a size whose odd part exceeds 1 adds two
    rotations (the R_z gates of Fig. 12). With n_L and n_M the odd parts of L and
    M, this returns

        toffoli_count   = (L - 1) + mu + ceil(log2 L)
                          + 2 ceil(log2 n_L) + 2 ceil(log2 n_M),
        rotation_count  = 2 [n_L > 1] + 2 [n_M > 1],
        register_qubits = 1 + 2 mu + 2 ceil(log2 L) + ceil(log2 M).

    The registers are Eq. (34)'s 1 + 2 mu + 2 log L (index, alternate index,
    keep, sigma and the comparison's result) and the site register. At M = 1 the
    cost is the source's generic 4(L + mu) + O(log L) T (Fig. 11).

    A walk step applies the prepare and its inverse, at this cost each. The
    walk's reflection spans register_qubits, and its AND chain holds
    register_qubits - 1 work qubits at its peak, more than any transient of the
    prepare, so a walk's peak ancillas are register_qubits plus that chain (see
    controlled_reflection_toffoli_count).

    Args:
        coefficients_per_site: Number L of LCU coefficients on one site, all
            stored in the table; an integer >= 2 (one coefficient needs no
            table).
        site_count: Number M of equivalent lattice sites; an integer >= 1.
        keep_bits: Bits mu of each keep probability
            (lookup_keep_probability_bits); an integer >= 1.

    Returns:
        The Toffoli, rotation and register-qubit counts of one prepare.

    Raises:
        ValueError: If coefficients_per_site is not an integer >= 2, or
            site_count or keep_bits is not an integer >= 1 (a float raises, whole
            or not).
    """
    if type(coefficients_per_site) is not int or coefficients_per_site < 2:
        raise ValueError(
            "translation_invariant_lookup_prepare_cost: coefficients_per_site must be "
            "an integer >= 2 (one coefficient needs no table); got "
            f"{coefficients_per_site!r}."
        )
    if type(site_count) is not int or site_count < 1:
        raise ValueError(
            "translation_invariant_lookup_prepare_cost: site_count must be an integer "
            f">= 1; got {site_count!r}."
        )
    if type(keep_bits) is not int or keep_bits < 1:
        raise ValueError(
            "translation_invariant_lookup_prepare_cost: keep_bits must be an integer "
            f">= 1; got {keep_bits!r}."
        )
    entries = equal_superposition_cost(coefficients_per_site)
    sites = equal_superposition_cost(site_count)
    return LookupPrepareCost(
        toffoli_count=(coefficients_per_site - 1)
        + keep_bits
        + entries.register_qubits
        + entries.toffoli_count
        + sites.toffoli_count,
        rotation_count=entries.rotation_count + sites.rotation_count,
        register_qubits=1
        + 2 * keep_bits
        + 2 * entries.register_qubits
        + sites.register_qubits,
    )


@dataclass(frozen=True)
class QubiterateQpeCost:
    """Qubiterate phase-estimation call counts.

    Phase estimation on the qubitization walk operator uses an energy register of
    width ``ancilla_qubits`` (W_q, Roggero+ 2019 Eq. 21) and, in its textbook
    form, ``2**W_q - 1`` controlled-walk applications, bit j of the register
    driving 2**j of them; both follow from the ratio of the LCU 1-norm to the
    target energy resolution. Eq. 21 fixes only the register width: the walk steps
    are the standard phase-estimation structure, not Eq. 21 (App. C.1 costs the
    walk per qubiterate, applied at each step). Phase estimation that runs the
    inverse walk on the 0 branch reaches the same resolution in 2**(W_q - 1)
    applications (two_sided_qpe_application_count).

    Attributes:
        ancilla_qubits: Energy-register width W_q = ceil(log2(one_norm /
            resolution)).
        application_count: Controlled-qubiterate applications 2**W_q - 1, the
            sum_{j=0}^{W_q-1} 2**j of textbook phase estimation (~ one_norm /
            resolution).
    """

    ancilla_qubits: int
    application_count: int


def qubiterate_qpe_ancilla_count(
    one_norm: float, resolution: float
) -> QubiterateQpeCost:
    """Energy-register width and application count for qubiterate-based QPE.

    The walk-operator eigenphases encode the spectrum scaled by the LCU 1-norm,
    so resolving energies to ``resolution`` needs W_q = ceil(log2(one_norm /
    resolution)) register qubits (Roggero+ 2019 Eq. 21). Textbook phase estimation
    then runs 2**W_q - 1 controlled-walk applications, a standard structure, not
    Eq. 21 itself.

    The ceil-log2 is evaluated exactly via ``math.frexp``: writing
    ratio = mantissa * 2**exponent with mantissa in [0.5, 1), the ceiling is
    exponent - 1 when ratio is an exact power of two (mantissa == 0.5) and
    exponent otherwise -- sidestepping any float ``log2`` rounding at the boundary.

    Units: ``one_norm`` and ``resolution`` must share an energy unit (MeV by
    project convention); their ratio -- and hence W_q and the application count --
    is dimensionless.

    Args:
        one_norm: LCU 1-norm lambda (sum of |coefficients|), in any energy unit;
            a finite number > 0.
        resolution: Target energy resolution Delta-omega, same unit as one_norm;
            a finite number > 0.

    Returns:
        The energy-register width and controlled-qubiterate application count.

    Raises:
        ValueError: If resolution or one_norm is not a finite number > 0, if
            one_norm <= resolution (the ratio must exceed 1 so that W_q >= 1), or
            if the ratio one_norm / resolution overflows to infinity. A NaN in
            either argument, an infinite one_norm or an overflowing ratio would
            otherwise yield a register of width 0.
    """
    if not 0 < resolution < math.inf:
        raise ValueError(
            "qubiterate_qpe_ancilla_count: resolution must be a finite number > 0; "
            f"got {resolution}."
        )
    if not 0 < one_norm < math.inf:
        raise ValueError(
            "qubiterate_qpe_ancilla_count: one_norm must be a finite number > 0; "
            f"got {one_norm}."
        )
    if one_norm <= resolution:
        raise ValueError(
            "qubiterate_qpe_ancilla_count: one_norm must exceed resolution so that "
            f"lambda/Delta-omega > 1 (W_q >= 1); got one_norm={one_norm}, "
            f"resolution={resolution}."
        )
    ratio = one_norm / resolution
    if math.isinf(ratio):
        raise ValueError(
            "qubiterate_qpe_ancilla_count: the ratio one_norm / resolution must be "
            f"finite; got one_norm={one_norm}, resolution={resolution}, whose ratio "
            "overflows."
        )
    mantissa, exponent = math.frexp(ratio)
    width = exponent - 1 if mantissa == 0.5 else exponent
    return QubiterateQpeCost(ancilla_qubits=width, application_count=2**width - 1)


def two_sided_qpe_application_count(ancilla_qubits: int) -> int:
    """Walk applications of phase estimation that runs the inverse walk on 0.

    Babbush+ 2018 (arXiv:1805.03662, Phys. Rev. X 8, 041015, Sec. II B, Eq. 16
    and Fig. 2) apply W**n when a control qubit is 1 and (W^dagger)**n when it is
    0, in place of W**n and nothing. The phase difference of the two branches
    doubles, so each application counts twice. The control moves from the walk
    onto reflections, R_L W**n R_L = (W^dagger)**n, which merge with the walk's
    own; the walk and its select run uncontrolled. Doubling reads the phase
    modulo pi, and one walk controlled in the textbook way resolves the ambiguity
    (the first operation of their Fig. 2).

    With a register of W_q qubits, W_q - 1 of them read 2 phi, 4 phi, ...,
    2**(W_q - 1) phi at 1 + 2 + ... + 2**(W_q - 2) applications, and the textbook
    qubit reads phi at one more. This returns

        2**(ancilla_qubits - 1),

    against textbook phase estimation's 2**W_q - 1 (QubiterateQpeCost.application_count)
    at the same register resolution; the source: "the complexity is reduced by
    approximately a half".

    Args:
        ancilla_qubits: Energy-register width W_q
            (QubiterateQpeCost.ancilla_qubits); an integer >= 1.

    Returns:
        The walk applications of one phase estimation.

    Raises:
        ValueError: If ancilla_qubits is not an integer >= 1 (a float raises,
            whole or not).
    """
    if type(ancilla_qubits) is not int or ancilla_qubits < 1:
        raise ValueError(
            "two_sided_qpe_application_count: ancilla_qubits must be an integer >= 1 "
            "(an energy register has a whole number of qubits, at least one); got "
            f"{ancilla_qubits!r}."
        )
    return 2 ** (ancilla_qubits - 1)


def two_sided_qpe_rms_energy_error(one_norm: float, ancilla_qubits: int) -> float:
    """Energy error that phase estimation on a W_q-qubit register delivers.

    Babbush+ 2018 (arXiv:1805.03662, Phys. Rev. X 8, 041015, Eqs. 21-26) measure
    the estimate by its root-mean-square phase error and write it (Eq. 22) as the
    statistical part, the Holevo variance pi**2 / 2**(2 (m + 1)) of a control
    register prepared in their state chi_m (Eq. 17 and Fig. 2, the optimal state
    of their Ref. 56, an O(m) construction by Eqs. 17-20), added in quadrature to
    the circuit's systematic phase error, the prepare's coefficient rounding
    epsilon_prep plus pi times the inverse QFT's synthesis error epsilon_QFT,
    which they set equal to the statistical part. That pairs 2**m walk
    applications with

        Delta-E = sqrt(2) pi lambda / 2**(m + 1)        (their Eq. 24),

    the error at which Eq. 26's 2**m < sqrt(2) pi lambda / Delta-E holds. Phase
    estimation that runs the inverse walk on 0 applies the walk 2**(W_q - 1) times
    (two_sided_qpe_application_count), so m = W_q - 1 and this returns

        sqrt(2) pi one_norm / 2**ancilla_qubits.

    Precondition: the control register is prepared in chi_m. A register in
    uniform superposition has a broader error distribution and does not reach
    this variance, so a caller that prices a window state prices chi_m, or a
    bound that covers it.

    The register's mid-band energy bin, 2 pi lambda / 2**W_q, is sqrt(2) times
    larger. For 4He on 27 sites at a = 1.4 fm (lambda = 9,132.1 MeV, W_q = 11) the
    error is 19.8 MeV and the bin 28.0 MeV; the 5 MeV that sizes the register
    through W_q = ceil(log2(lambda / sigma)) is a sizing parameter, not either. This
    is the Delta-E that lookup_keep_probability_bits takes, the formula's own.

    Units: one_norm and the result share an energy unit (MeV by project
    convention); ancilla_qubits is a number of qubits.

    Args:
        one_norm: LCU 1-norm lambda of the walk; a finite number > 0.
        ancilla_qubits: Energy-register width W_q; an integer >= 1.

    Returns:
        The root-mean-square energy error of one phase estimation.

    Raises:
        ValueError: If one_norm is not a finite number > 0, or ancilla_qubits is
            not an integer >= 1 (a float raises, whole or not).
    """
    if not 0 < one_norm < math.inf:
        raise ValueError(
            "two_sided_qpe_rms_energy_error: one_norm must be a finite number > 0; "
            f"got {one_norm}."
        )
    if type(ancilla_qubits) is not int or ancilla_qubits < 1:
        raise ValueError(
            "two_sided_qpe_rms_energy_error: ancilla_qubits must be an integer >= 1 "
            "(an energy register has a whole number of qubits, at least one); got "
            f"{ancilla_qubits!r}."
        )
    return math.ldexp(math.sqrt(2.0) * math.pi * one_norm, -ancilla_qubits)


@dataclass(frozen=True)
class QsvtDegree:
    """QSVT polynomial degree to realize e^{-iH*tau} on a block-encoding.

    Attributes:
        heuristic: ceil(block_encoding_norm * total_time) -- the d ~ alpha*tau
            scaling, with alpha the block-encoding subnormalization. Dimensionless.
        jacobi_anger: the rigorous constructive Jacobi-Anger truncation degree --
            the smallest d whose Bessel tail 2*sum_{k>d}|J_k(z)| <= epsilon at
            z = block_encoding_norm * total_time. It exceeds the heuristic by only
            a small, sublinear additive overhead, d ~ z + O(z**(1/3) +
            log(1/epsilon)) (~5% above the heuristic at z = 500, the ratio
            approaching 1 as z grows). Dimensionless.
    """

    heuristic: int
    jacobi_anger: int


_JACOBI_ANGER_TAIL_FLOOR = 1e-30


def jacobi_anger_truncation_error(z: float, degree: int) -> float:
    """Jacobi-Anger truncation error 2 * sum_{k>degree} |J_k(z)|.

    Truncating the Jacobi-Anger expansion
    e^{i z sin(theta)} = sum_k J_k(z) e^{i k theta} at |k| <= degree leaves a
    sup-norm tail bounded by 2 * sum_{k>degree} |J_k(z)| (the +k and -k terms
    share |J_k|). This is the error a degree-``degree`` QSVT polynomial of the
    block-encoding must drive below the target epsilon.

    The Bessel terms decay super-exponentially past the turnover k ~ z, so the
    sum is accumulated upward from k = degree + 1 until the terms fall below a
    fixed floor (1e-30) *beyond* the turnover -- yielding the tail to ~1e-28
    absolute accuracy, far below any epsilon a QSVT approximation would target
    (the super-exponential decay makes the few extra terms negligible in cost).

    Args:
        z: Jacobi-Anger argument z = block_encoding_norm * total_time (>= 0,
            dimensionless).
        degree: Truncation degree (>= 0); the retained band is |k| <= degree.

    Returns:
        The truncation error 2 * sum_{k>degree} |J_k(z)| (dimensionless).
    """
    tail = 0.0
    order = degree + 1
    while True:
        term = abs(float(jv(order, z)))
        tail += term
        if order > z and term < _JACOBI_ANGER_TAIL_FLOOR:
            break
        order += 1
    return 2.0 * tail


def _minimal_jacobi_anger_degree(z: float, epsilon: float) -> int:
    """Smallest degree d with jacobi_anger_truncation_error(z, d) <= epsilon.

    The tail is monotone non-increasing in d, so an exponential search brackets
    the crossing and a binary search locates the minimal degree.
    """
    hi = 1
    while jacobi_anger_truncation_error(z, hi) > epsilon:
        hi *= 2
    lo = hi // 2
    while lo < hi:
        mid = (lo + hi) // 2
        if jacobi_anger_truncation_error(z, mid) <= epsilon:
            hi = mid
        else:
            lo = mid + 1
    return lo


def qsvt_evolution_degree(
    block_encoding_norm: float, total_time: float, epsilon: float
) -> QsvtDegree:
    """QSVT degree for e^{-iH*tau}: the heuristic alpha*tau and Jacobi-Anger.

    A degree-d polynomial of the block-encoded operator H/alpha realizes the
    time evolution e^{-iH*tau}, so the degree scales as the dimensionless product
    d ~ alpha*tau, with alpha the block-encoding subnormalization.

    Units convention (load-bearing): this function is unit-agnostic -- it
    multiplies its two arguments directly -- so the caller MUST pass
    ``block_encoding_norm`` in some energy unit and ``total_time`` in the
    *reciprocal* of that SAME unit; the product (hence the returned degree) is
    dimensionless. This project is MeV-native: pass the centered operator norm
    Delta-H/2 in MeV together with the QSVT horizon tau_max = 1/sigma in MeV^-1
    (sigma = 10 MeV -> tau_max = 0.1 MeV^-1). Mixing units -- a MeV norm with a
    GeV^-1 time -- silently inflates the degree by 10**3.

    Choose ``block_encoding_norm`` to match the block-encoding: the centered
    spectral operator norm ||H||_op = Delta-H/2 for an ideal block-encoding (our
    convention), or the 1-norm lambda for an LCU (then d ~ lambda*tau =
    lambda/Delta-omega, the qubiterate-QPE query scale).

    Args:
        block_encoding_norm: Block-encoding subnormalization alpha, in an energy
            unit (MeV by convention); must be > 0.
        total_time: Evolution time tau, in the reciprocal energy unit (MeV^-1 by
            convention); must be > 0.
        epsilon: Target spectral-norm error of the polynomial approximation;
            must satisfy 0 < epsilon < 1.

    Returns:
        The heuristic degree and the rigorous Jacobi-Anger truncation degree.

    Raises:
        ValueError: If block_encoding_norm <= 0, total_time <= 0, or epsilon is
            outside the open interval (0, 1).
    """
    if block_encoding_norm <= 0:
        raise ValueError(
            "qsvt_evolution_degree: block_encoding_norm must be > 0; got "
            f"{block_encoding_norm}."
        )
    if total_time <= 0:
        raise ValueError(
            f"qsvt_evolution_degree: total_time must be > 0; got {total_time}."
        )
    if not 0 < epsilon < 1:
        raise ValueError(
            "qsvt_evolution_degree: epsilon must lie in the open interval (0, 1); "
            f"got {epsilon}."
        )
    z = block_encoding_norm * total_time
    heuristic = math.ceil(z)
    jacobi_anger = _minimal_jacobi_anger_degree(z, epsilon)
    return QsvtDegree(heuristic=heuristic, jacobi_anger=jacobi_anger)


@dataclass(frozen=True)
class PerShotCost:
    """T gates of one qubitization circuit shot, with every part stated.

    A shot applies the walk ``query_count`` times: textbook phase estimation's
    2**W_q - 1 (W_q from lambda/Delta-omega); 2**(W_q - 1) for phase estimation that
    runs the inverse walk on the 0 branch (two_sided_qpe_application_count); or a
    QSVT-on-LCU evolution degree d ~ lambda*tau. All three apply the SAME LCU
    walk, so all pair with the same per-qubiterate parts. In the second kind of
    phase estimation every select but the first, and most reflections, run
    uncontrolled, so a price that counts their control is an upper bound there by
    at most one AND computation each.

    Consistency (load-bearing): ``query_count`` and the three prices must describe
    the SAME LCU block-encoding. A *spectral*-norm QSVT
    degree (d ~ ||H||/sigma, e.g. ~500) does not: it needs an ideal block-encoding
    subnormalized by ||H||, which a plain LCU (subnormalized by lambda >> ||H||)
    does not provide. Pairing a spectral degree with the LCU per-qubiterate cost
    fuses two normalizations.

    One qubiterate, one application of the walk, costs a select, a controlled
    reflection and a prepare, and all three are the caller's statements (see
    per_shot_t_cost). The parts are kept apart, and the total is computed from
    them.

    Attributes:
        query_count: Walk applications (qubiterates) in one shot.
        select_t_per_qubiterate: T gates charged to the select per qubiterate, as
            supplied by the caller.
        reflection_t_per_qubiterate: T gates charged to the controlled reflection
            per qubiterate, as supplied by the caller.
        prepare_t_per_qubiterate: T gates charged to the prepare per qubiterate,
            as supplied by the caller.
    """

    query_count: int
    select_t_per_qubiterate: float
    reflection_t_per_qubiterate: float
    prepare_t_per_qubiterate: float

    @property
    def t_count(self) -> float:
        """Total T gates per shot: query_count times the sum of the three parts.

        An expected count, so it may be fractional.
        """
        return self.query_count * (
            self.select_t_per_qubiterate
            + self.reflection_t_per_qubiterate
            + self.prepare_t_per_qubiterate
        )


def per_shot_t_cost(
    query_count: int,
    *,
    select_t_per_qubiterate: float,
    reflection_t_per_qubiterate: float,
    prepare_t_per_qubiterate: float,
) -> PerShotCost:
    """Per-shot T gates of query_count qubiterates, every part stated.

    This returns

        query_count * (select_t_per_qubiterate
                       + reflection_t_per_qubiterate
                       + prepare_t_per_qubiterate),

    the caller's prices for the select, the controlled reflection and the prepare,
    summed per qubiterate and multiplied by the number of walk applications in one
    circuit shot.

    No part has a default price, and the library chooses none. The caller prices
    each part for the same LCU block-encoding of Gamma terms:

    - the select: App. C.1's, controlled_qubiterate_cost(Gamma).select_t_count, or
      one by unary iteration, unary_iteration_select_toffoli_count(Gamma) times the
      caller's T gates per Toffoli;
    - the controlled reflection, about the zero state of every qubit the prepare
      acts on: App. C.1's, controlled_qubiterate_cost(Gamma).reflection_t_count,
      which covers the appendix's prepare on the N_A index qubits, or an AND
      chain, controlled_reflection_toffoli_count(width) times the T gates per
      Toffoli, at the width of the caller's prepare (for a table lookup,
      LookupPrepareCost.register_qubits);
    - the prepare: by synthesis, prepare_rotation_count(...) times the T gates per
      synthesized rotation (for five terms or more), or by table lookup, at a rate
      per coefficient, table_lookup_prepare_t_count(...), or itemized for a
      Hamiltonian that repeats on every site,
      translation_invariant_lookup_prepare_cost(...), its Toffolis and rotations
      at the caller's rates, twice per step.

    With App. C.1's select and reflection, a prepare of 0 states that the result is
    the App. C.1 T column alone, which is a lower bound on the T gates of a shot
    and can sit far below its full non-Clifford cost.

    One prepare is left out at any price. A shot begins with an initial controlled
    prepare, "whose small cost we neglect in the estimates that follow" (App.
    C.1), and it is not in this total either. It is one prepare against those of
    query_count qubiterates, so it matters only for a shot of a few applications.

    query_count and the three prices must describe the same LCU block-encoding
    (see PerShotCost).

    Args:
        query_count: Walk applications in one shot, the walk steps of a phase
            estimation or a QSVT degree, a dimensionless count; an integer >= 1.
        select_t_per_qubiterate: T gates charged to the select per qubiterate; a
            finite number >= 0.
        reflection_t_per_qubiterate: T gates charged to the controlled reflection
            per qubiterate; a finite number >= 0.
        prepare_t_per_qubiterate: T gates charged to the prepare per qubiterate; a
            finite number >= 0.

    Returns:
        The per-shot T gates, with the select, the controlled reflection and the
        prepare kept apart.

    Raises:
        ValueError: If query_count is not an integer >= 1 (a shot applies a whole
            number of qubiterates, at least one; a float raises, whole or not), or
            if any price is not a finite number >= 0.
    """
    if type(query_count) is not int or query_count < 1:
        raise ValueError(
            "per_shot_t_cost: query_count must be an integer >= 1; got "
            f"{query_count!r}."
        )
    if not 0 <= select_t_per_qubiterate < math.inf:
        raise ValueError(
            "per_shot_t_cost: select_t_per_qubiterate must be a finite number "
            f">= 0; got {select_t_per_qubiterate}."
        )
    if not 0 <= reflection_t_per_qubiterate < math.inf:
        raise ValueError(
            "per_shot_t_cost: reflection_t_per_qubiterate must be a finite number "
            f">= 0; got {reflection_t_per_qubiterate}."
        )
    if not 0 <= prepare_t_per_qubiterate < math.inf:
        raise ValueError(
            "per_shot_t_cost: prepare_t_per_qubiterate must be a finite number "
            f">= 0; got {prepare_t_per_qubiterate}."
        )
    return PerShotCost(
        query_count=query_count,
        select_t_per_qubiterate=select_t_per_qubiterate,
        reflection_t_per_qubiterate=reflection_t_per_qubiterate,
        prepare_t_per_qubiterate=prepare_t_per_qubiterate,
    )
