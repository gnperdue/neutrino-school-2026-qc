"""Cost of loading a classically known sparse state onto the system register.

A ground state computed classically, as amplitudes on a list of occupation
configurations, is written onto the qubits of the system register before it is
used. The count here is the method of Fomichev+ 2023 (arXiv:2310.18410, PRX
Quantum 5, 040339 (2024)), which leaves no junk register behind, so the system
holds the superposition itself and not a mixture of its configurations. A
zero-momentum state on a periodic lattice can be loaded with one nucleon pinned
at the origin, which has fewer amplitudes, and then spread over the lattice
translations; the second function prices that spreading.
"""

import math
from dataclasses import dataclass

from qec_utility_estimation.qubitization import equal_superposition_cost


@dataclass(frozen=True)
class SparseStateLoadCost:
    """Cost of one load of a sparse state, by the method of Fomichev+ 2023.

    The two Toffoli parts come from two places in the source and are kept apart;
    the total is computed from them.

    Attributes:
        leading_toffoli_count: Toffoli gates of Eq. (17), the source's leading
            count: the two table lookups and the erasure of the enumeration
            register.
        rotation_toffoli_count: Toffoli gates of the rotations of the dense state
            on the enumeration register (the source's App. D).
        ancilla_qubits: Ancilla qubits beyond the system register: Eq. (18) and
            step 1's phase-gradient and angle registers.
    """

    leading_toffoli_count: int
    rotation_toffoli_count: int
    ancilla_qubits: int

    @property
    def toffoli_count(self) -> int:
        """Toffoli gates of one load, the two parts summed."""
        return self.leading_toffoli_count + self.rotation_toffoli_count


def sparse_state_load_cost(
    num_configurations: int, *, rotation_bits: int
) -> SparseStateLoadCost:
    """Cost of preparing a superposition of D configurations, with no junk left.

    Fomichev+ 2023 (arXiv:2310.18410, PRX Quantum 5, 040339 (2024)) Sec. III
    prepares sum_i alpha_i |nu_i> on the system register from a classical list of D
    distinct occupation bitstrings nu_i and their amplitudes, with
    L = ceil(log2 D), in six steps:

    1. a dense state on an enumeration register of L qubits;
    2. a table lookup that writes nu_i into the system register;
    3. the first bit of a (2L - 1)-bit identifier, a parity of chosen bits of nu_i,
       by CNOTs alone (their Lemma III.1);
    4. the remaining bits the same way;
    5. the enumeration register returned to zero, conditioned on each identifier;
    6. the identifier uncomputed.

    The erasure of step 5, with the uncomputation of step 6 after it, is what
    leaves the system in the pure superposition. A load without the erasure leaves
    the enumeration register entangled with the system, which then holds a mixture
    of the configurations, not the state. The source calls a junk register "not
    acceptable" here, unlike the prepare of a qubitization walk (their App. D).
    This returns

        leading_toffoli_count  = (2L - 2) D + 2**(L + 1) + D,
        rotation_toffoli_count = (rotation_bits - 3) L,
        ancilla_qubits         = 5 L - 3 + c + (0 if c <= 2L - 1 else c),

    with c = rotation_bits.

    The leading count is Eq. (17), "(2 log D - 2)D + 2^(log D+1) + D <
    (2 log D + 3)D", with log D rounded up as the source directs when D is not a
    power of two; step 5 accounts for the (2L - 2) D. The rotation count is the
    source's cost of the L rotations of step 1, by phase-gradient addition of
    rotation_bits-bit angles (their App. D). It asks for
    rotation_bits > log2(pi log D / eps), with eps the tolerated error of the
    loaded state (chemical accuracy in the source). The library takes the bits as
    stated.

    ancilla_qubits is Eq. (18), the enumeration and identification registers and
    the work qubits of step 5's multi-controlled operations, plus step 1's two
    registers: the phase-gradient state of c qubits, which the source's rotation
    cost assumes (App. D: "assuming access to a gradient state"), and the c-qubit
    angle register (App. E: the qubits "to store the phase"). The angle register
    sits in the identification register when it fits, since that register is
    idle and clean until step 3. Step 1's transient work qubits are not counted,
    as in Eq. (18).

    The caller converts Toffolis to T gates at its stated rate.

    Args:
        num_configurations: Number D of configurations in the superposition, a
            dimensionless count; an integer >= 2.
        rotation_bits: Bits of each rotation angle in step 1; an integer >= 4.

    Returns:
        The Toffoli and ancilla counts of one load.

    Raises:
        ValueError: If num_configurations is not an integer >= 2, or if
            rotation_bits is not an integer >= 4 (a float raises, whole or not).
    """
    if type(num_configurations) is not int or num_configurations < 2:
        raise ValueError(
            "sparse_state_load_cost: num_configurations must be an integer >= 2 (one "
            "configuration is a single bitstring, written without Toffolis); got "
            f"{num_configurations!r}."
        )
    if type(rotation_bits) is not int or rotation_bits < 4:
        raise ValueError(
            "sparse_state_load_cost: rotation_bits must be an integer >= 4 (below "
            "four bits the rotation term (rotation_bits - 3) * L is zero or "
            f"negative); got {rotation_bits!r}."
        )
    register = (num_configurations - 1).bit_length()
    identification_qubits = 2 * register - 1
    angle_qubits = 0 if rotation_bits <= identification_qubits else rotation_bits
    return SparseStateLoadCost(
        leading_toffoli_count=(2 * register - 2) * num_configurations
        + 2 ** (register + 1)
        + num_configurations,
        rotation_toffoli_count=(rotation_bits - 3) * register,
        ancilla_qubits=5 * register - 3 + rotation_bits + angle_qubits,
    )


@dataclass(frozen=True)
class TranslationSymmetrizationCost:
    """Cost of spreading a pinned state over the translations of a cubic lattice.

    The three Toffoli parts come from three places in the construction and are
    kept apart, as SparseStateLoadCost keeps its two; the total is a property.

    Attributes:
        swap_toffoli_count: Controlled swaps of the mode qubits, the cyclic
            shifts as qubit permutations.
        sign_toffoli_count: One controlled-controlled-Z per lattice line and
            shift, the fermionic sign of each shift.
        superposition_toffoli_count: AND computations of the shift register's
            three equal superpositions.
        rotation_count: R_z rotations of those equal superpositions.
        ancilla_qubits: The shift register, three times ceil(log2 n), and the
            two parity qubits of the sign gates; zero when n = 1. The transient
            AND target of each 4-T Fredkin and controlled-controlled-Z is not
            included, as EqualSuperpositionCost and LookupPrepareCost exclude
            theirs.
    """

    swap_toffoli_count: int
    sign_toffoli_count: int
    superposition_toffoli_count: int
    rotation_count: int
    ancilla_qubits: int

    @property
    def toffoli_count(self) -> int:
        """Toffoli gates of one symmetrization, the three parts summed."""
        return (
            self.swap_toffoli_count
            + self.sign_toffoli_count
            + self.superposition_toffoli_count
        )


def translation_symmetrization_cost(
    sites_per_axis: int, modes_per_site: int
) -> TranslationSymmetrizationCost:
    """Cost of making the zero-momentum state from one with a nucleon pinned.

    A zero-momentum state of a translation-invariant Hamiltonian on a periodic
    n x n x n lattice is the sum over the M = n**3 lattice translations of the
    state with one chosen nucleon at the origin: the projections of the state
    on that nucleon's site are orthogonal and sum to it. The pinned state has
    M times fewer amplitudes, so it can be loaded (sparse_state_load_cost) and
    then spread:

    1. a shift register of three axis registers, each in an equal superposition
       over n values (equal_superposition_cost(n) each);
    2. for each axis and each bit j of its register, a cyclic shift of the
       lattice by 2**j sites along that axis, controlled on the bit;
    3. the register erased by CNOTs from the pinned nucleon's occupation
       qubits, which hold its site after the shift. This is Clifford, and it
       needs the pinned nucleon's species to hold exactly one nucleon, as every
       species does in 4He; other nuclei need another pinning.

    The modes are ordered internal index first (Roggero+ 2019, arXiv:1911.06368,
    Eq. 7's qubit 4 i + f) and the sites lexicographically with x fastest, then
    y, then z; the site order is this library's choice, which Roggero+ leave
    open. With it each axis has lines of n contiguous blocks of modes: n**2
    lines of blocks of m along x, n lines of blocks of n m along y, one line of
    blocks of n**2 m along z. A cyclic shift by k of n blocks is n - gcd(n, k)
    block transpositions as a qubit permutation, each a controlled swap per
    qubit, so an axis takes (n - gcd(n, k)) n**2 m Toffolis per shift, whatever
    the ordering; and as a fermionic operator it is one swap of the first n - k
    blocks past the last k, whose sign (-1)**(N_A N_C) is a product of two
    occupation parities, computed by CNOTs into two work qubits and applied as
    one controlled-controlled-Z, one Toffoli per line per shift. The sign count
    needs every line contiguous, which the site order above gives on all three
    axes; the swap count does not depend on the order. This returns

        swap_toffoli_count          = sum_j 3 (n - gcd(n, 2**j)) n**2 m,
        sign_toffoli_count          = sum_j (n**2 + n + 1),
        superposition_toffoli_count = 3 equal_superposition_cost(n).toffoli_count,
        rotation_count              = 3 equal_superposition_cost(n).rotation_count,
        ancilla_qubits              = 3 ceil(log2 n) + 2   (0 at n = 1),

    with j over the ceil(log2 n) bits of an axis register. For 4He on the 3**3
    lattice (n = 3, m = 4): 432 + 26 + 12 = 470 Toffolis, 6 rotations and 8 qubits,
    against about 640,000 Toffolis for the pinned load it completes.

    Args:
        sites_per_axis: Sites n along each axis of the cubic lattice; an
            integer >= 1 (one site per axis has no translations: every count
            is zero).
        modes_per_site: Fermionic modes m per site (4 for the pionless model);
            an integer >= 1.

    Returns:
        The Toffoli, rotation and ancilla counts of one symmetrization.

    Raises:
        ValueError: If either argument is not an integer >= 1 (a float raises,
            whole or not).
    """
    if type(sites_per_axis) is not int or sites_per_axis < 1:
        raise ValueError(
            "translation_symmetrization_cost: sites_per_axis must be an integer >= 1 "
            f"(a lattice has a whole number of sites along each axis); got "
            f"{sites_per_axis!r}."
        )
    if type(modes_per_site) is not int or modes_per_site < 1:
        raise ValueError(
            "translation_symmetrization_cost: modes_per_site must be an integer >= 1 "
            f"(a site holds a whole number of fermionic modes); got {modes_per_site!r}."
        )
    axis_qubits = (sites_per_axis - 1).bit_length()
    superposition = equal_superposition_cost(sites_per_axis)
    lines = sites_per_axis**2 + sites_per_axis + 1
    modes_on_an_axis = sites_per_axis**2 * modes_per_site
    swaps = 0
    for bit in range(axis_qubits):
        transpositions = sites_per_axis - math.gcd(sites_per_axis, 2**bit)
        swaps += 3 * transpositions * modes_on_an_axis
    return TranslationSymmetrizationCost(
        swap_toffoli_count=swaps,
        sign_toffoli_count=axis_qubits * lines,
        superposition_toffoli_count=3 * superposition.toffoli_count,
        rotation_count=3 * superposition.rotation_count,
        ancilla_qubits=3 * axis_qubits + 2 if axis_qubits else 0,
    )
