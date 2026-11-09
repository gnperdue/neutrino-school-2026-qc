"""The cost of one circuit of the neutrino-response calculation, piece by piece.

One circuit prepares a nucleus's ground state, applies the current and measures the
energy by phase estimation of the quantum walk, on a periodic cubic lattice of
leading-order pionless EFT. This module adds the library's counts up into that
circuit, in the order the lecture builds it. Energies are in MeV; counts are of T
gates, Toffolis or qubits.

Sources. The library modules this calls cite their own; these are the ones each
piece here rests on.

- The Hamiltonian, its couplings and lambda: Roggero+ 2019 (arXiv:1911.06368), Eq. 1
  and Tab. I; its 38 Pauli strings a site, App. C.1; the energy register's width,
  Eq. 21.
- The walk and its phase estimation: Babbush+ 2018 (arXiv:1805.03662). The AND at
  4 T, erased free by a measurement, is Gidney's (arXiv:1709.06648), drawn in their
  Fig. 4; the one-site table's prepare, Figs. 10-12 and Eq. 34; its keep bits,
  Eq. 36; the inverse walk on the 0 branch, Sec. II B and Eq. 16, and the energy
  error it delivers, Eq. 24; the window state, Eqs. 17-20.
- The current and its success probability, P_s = eta^2 / Lambda_J^2: Roggero, Gu,
  Baroni and Papenbrock 2020 (arXiv:2009.13485), Eqs. 26-31.
- Amplitude amplification: Brassard, Hoyer, Mosca and Tapp 2002
  (arXiv:quant-ph/0005055).
- The ground state's load: Fomichev+ 2023 (arXiv:2310.18410).
- The rotations: repeat-until-success synthesis, Bocharov, Roetteler and Svore 2015
  (arXiv:1404.5320); the randomized alternative, Campbell 2017 (arXiv:1612.02689),
  Thm. 2; the deterministic method that theorem covers, Ross and Selinger 2016
  (arXiv:1403.2975).
- The budgets: the U.S. Department of Energy's request for information on
  scientifically relevant fault-tolerant quantum computing systems, RFI
  892431-26-RFI-0001 (May 2026). Its scale for the early-to-mid 2030s is what the
  lecture calls the 2033 budget.
- The code distance: the surface code's P_L ~ 0.03 (p / p_th)^((d + 1) / 2)
  (Fowler+ 2012, arXiv:1208.0928, Eq. 11), here without the 0.03.
"""

import math
from dataclasses import dataclass

from qec_utility_estimation.apps.pionless_first_order.couplings import (
    reference_couplings_at_spacing,
)
from qec_utility_estimation.apps.pionless_first_order.norms import (
    hamiltonian_one_norm,
    pauli_coefficients,
    phase_estimation_energy_bound,
)
from qec_utility_estimation.qubitization import (
    controlled_reflection_toffoli_count,
    equal_superposition_cost,
    lookup_keep_probability_bits,
    qubiterate_qpe_ancilla_count,
    reflection_toffoli_count,
    translation_invariant_lookup_prepare_cost,
    two_sided_qpe_application_count,
    two_sided_qpe_rms_energy_error,
    unary_iteration_select_toffoli_count,
)
from qec_utility_estimation.state_preparation import (
    sparse_state_load_cost,
    translation_symmetrization_cost,
)
from qec_utility_estimation.synthesis import (
    coherent_synthesis_precision,
    phase_gradient_angle_bits,
    repeat_until_success_t_count,
)

LATTICE_SPACING_FM = 1.4
AXES = 3  # of the cubic lattice
MODES_PER_SITE = 4  # two spins times two isospins
TERMS_PER_SITE = 38  # 24 hopping, 4 Z, 6 ZZ and 4 ZZZ Pauli strings
T_PER_TOFFOLI = 4  # an AND: 4 T to compute, erased free by a measurement
SYNTHESIS_ERROR_BUDGET = 0.01  # per circuit, shared by every rotation and rounded angle
FAULT_BUDGET = 0.1  # expected logical faults per circuit
# The 2033 budget counts "hard gates"; the same source's 2028 figure names "T or Toffoli
# gates". We charge a Toffoli as 4 T against it, the safe side for a claim that a
# circuit fits. A claim about how large a gap the allowance needs, or a case just over
# the budget, also depends on this count, so OneCircuit.one_per_toffoli gives the other.
HARD_OPERATIONS_2033 = 10**9  # the 2033 budget: hard operations per circuit
LOGICAL_QUBITS_2033 = 1_000  # the 2033 budget: logical qubits
HARD_OPERATIONS_2028 = 10**5  # the 2028 budget: hard operations per circuit
LOGICAL_ERROR_2028 = 1e-8  # the 2028 budget: logical error per hard operation
MIN_REGISTER_BITS = 3
MAX_REGISTER_BITS = 64
LECTURE_COUPLINGS = reference_couplings_at_spacing(LATTICE_SPACING_FM)  # t, C0, D0


def _require_positive(value: float, name: str) -> None:
    """Stop with a message that names the input, not a bare division by zero."""
    if not value > 0.0:
        raise ValueError(f"{name} must be above zero; got {value}.")


def _require_between_zero_and_one(value: float, name: str) -> None:
    """Stop with a message that names the input, for a rate or a fraction."""
    if not 0.0 < value < 1.0:
        raise ValueError(f"{name} must lie strictly between 0 and 1; got {value}.")


def _is_four_counts(species: object) -> bool:
    """A tuple of four whole numbers, each at least zero."""
    if type(species) is not tuple or len(species) != 4:
        return False
    for nucleons in species:
        if type(nucleons) is not int or nucleons < 0:
            return False
    return True


@dataclass(frozen=True, kw_only=True)
class Nucleus:
    """A nucleus by its nucleons in each species (p up, p down, n up, n down).

    Raises:
        ValueError: The species are not four whole numbers of at least zero, there
            are fewer than two nucleons, or there is no proton for the current to see.
    """

    name: str
    nucleons_per_species: tuple[int, int, int, int]

    def __post_init__(self) -> None:
        if not _is_four_counts(self.nucleons_per_species):
            raise ValueError(
                "nucleons_per_species must be a tuple of four whole numbers of at "
                "least 0, for p up, p down, n up and n down; got "
                f"{self.nucleons_per_species!r}."
            )
        if self.nucleon_count < 2:
            raise ValueError(
                f"{self.name} has {self.nucleon_count} nucleon(s); a nucleus needs "
                "at least two nucleons."
            )
        if self.proton_count < 1:
            raise ValueError(
                f"{self.name} has no protons, and the current is the proton density, "
                "so there is no response to price; give it at least one proton."
            )

    @property
    def nucleon_count(self) -> int:
        """A, the nucleons of all four species."""
        return sum(self.nucleons_per_species)

    @property
    def proton_count(self) -> int:
        """Z, the nucleons of the two proton species."""
        proton_up = self.nucleons_per_species[0]
        proton_down = self.nucleons_per_species[1]
        return proton_up + proton_down

    @property
    def strength_band(self) -> tuple[float, float]:
        """[Z/2, Z], the range taken for the strength eta^2: an estimate, not a bound.

        The strength eta^2 = <Psi_0| rho_q^dagger rho_q |Psi_0> is the response's total
        weight at momentum q, and it sets the insertion's success probability,
        P_s = eta^2 / Lambda_J^2. It is Z plus the sum over pairs of protons i != j of
        <exp(i q.(x_i - x_j))>, and its only bound is Z^2, since ||rho_q|| <= Z on
        states of Z protons.

        The band estimates the strength without its elastic line, the ground state
        recoiling at momentum q. Well above the Fermi momentum the pair terms average
        away and that strength approaches Z, the proton sum rule; Pauli blocking holds
        it lower at smaller q. The low end, Z/2, is the fraction of Z that helium's
        band, [1, 2], spans. A floor of 1 for every nucleus would instead put carbon at
        a sixth of its sum rule and argon at an eighteenth, and cut their allowances
        per preparation by about 1.7 and 3, since the preparations grow as
        1 / sqrt(P_s).

        The elastic line adds to eta^2, and below twice the Fermi momentum, where
        221 MeV, the smallest momentum on carbon's 4^3 box, lies, it can carry eta^2
        above Z. A larger eta^2 takes fewer rounds of amplification, so the band's low
        end bounds the circuit's cost from above as long as the strength without the
        elastic line is at least Z/2.
        """
        protons = float(self.proton_count)
        return (protons / 2.0, protons)


HELIUM_4 = Nucleus(name="helium", nucleons_per_species=(1, 1, 1, 1))
CARBON_12 = Nucleus(name="carbon", nucleons_per_species=(3, 3, 3, 3))
ARGON_40 = Nucleus(name="argon", nucleons_per_species=(9, 9, 11, 11))


@dataclass(frozen=True, kw_only=True)
class Lattice:
    """A periodic cubic lattice with box_side sites along each axis.

    Raises:
        ValueError: box_side is not an integer of at least 2.
    """

    box_side: int

    def __post_init__(self) -> None:
        if type(self.box_side) is not int or self.box_side < 2:
            raise ValueError(
                "box_side must be a whole number of at least 2 sites per axis, so "
                f"that a nucleon has somewhere to hop; got {self.box_side!r}."
            )

    @property
    def site_count(self) -> int:
        """M = n^3, the sites."""
        return self.box_side**3

    @property
    def box_length_fm(self) -> float:
        """n a, the length of one side, in fm."""
        return self.box_side * LATTICE_SPACING_FM

    @property
    def system_qubits(self) -> int:
        """4 M, one qubit per mode."""
        return MODES_PER_SITE * self.site_count

    @property
    def term_count(self) -> int:
        """Gamma = 38 M, the Pauli strings of the Hamiltonian as written.

        On 2^3 the hops either way along an axis reach the same site, so the hopping
        strings come in equal pairs, 12 distinct of 24 a site, and a select priced on
        38 M is an upper bound there. lambda is the same either way, since a pair's
        coefficients add.
        """
        return TERMS_PER_SITE * self.site_count

    @property
    def one_norm_mev(self) -> float:
        """lambda, the sum of the sizes of those strings' coefficients, in MeV."""
        return hamiltonian_one_norm(self.site_count, LECTURE_COUPLINGS)


def ground_state_amplitudes(nucleus: Nucleus, lattice: Lattice) -> int:
    """Configurations of the nucleus on the lattice: C(M, n) multiplied over the
    species. Zero when a species has more nucleons than the lattice has sites."""
    amplitudes = 1
    for nucleons in nucleus.nucleons_per_species:
        amplitudes = amplitudes * math.comb(lattice.site_count, nucleons)
    return amplitudes


def pinned_table_amplitudes(nucleus: Nucleus, lattice: Lattice) -> int | None:
    """Amplitudes with one nucleon pinned at the origin: C(M, n) over the other species.

    None when no species holds exactly one nucleon, so nothing can be pinned and erased.
    """
    if 1 not in nucleus.nucleons_per_species:
        return None
    amplitudes = 1
    already_pinned = False
    for nucleons in nucleus.nucleons_per_species:
        if nucleons == 1 and not already_pinned:
            already_pinned = True
        else:
            amplitudes = amplitudes * math.comb(lattice.site_count, nucleons)
    return amplitudes


def amplitudes_after_translations(nucleus: Nucleus, lattice: Lattice) -> float:
    """The configurations over the lattice's M translations: the table left once they
    are used. This is the pinned table where a species of one can be pinned, and an
    estimate otherwise, since a configuration a translation leaves unchanged has fewer
    copies."""
    return ground_state_amplitudes(nucleus, lattice) / lattice.site_count


def gain_at_proton_count(nucleus: Nucleus, lattice: Lattice) -> float:
    """(M / Z)^2, what P_s gains if the current is normalized at Z instead of M."""
    return (lattice.site_count / nucleus.proton_count) ** 2


def _bits_to_number(count: int) -> int:
    """ceil(log2 count): the bits that give each of count things its own number. The
    integer form avoids a floating-point log2 landing just past a power of two."""
    return (count - 1).bit_length()


def position_register_qubits(nucleus: Nucleus, lattice: Lattice) -> int:
    """Qubits to store each nucleon's mode instead: A ceil(log2(4 M))."""
    bits_per_nucleon = _bits_to_number(lattice.system_qubits)
    return nucleus.nucleon_count * bits_per_nucleon


@dataclass(frozen=True, kw_only=True)
class OneSiteOneNorm:
    """One site's share of lambda, by group of Pauli strings, in MeV (Roggero+ 2019
    Eq. 1, through norms.pauli_coefficients, which says how this compares with the
    paper's printed Eqs. 5 and 7)."""

    hopping_mev: float  # 24 strings of |t| / 2: 12 |t|
    single_z_mev: float  # 4 strings: 3 |C0 + D0 / 2|
    pair_zz_mev: float  # 6 strings: 1.5 |C0 + D0|
    triple_zzz_mev: float  # 4 strings: 0.5 |D0|

    @property
    def on_site_mev(self) -> float:
        """The 14 strings that act on one site alone: the Z, ZZ and ZZZ groups."""
        return self.single_z_mev + self.pair_zz_mev + self.triple_zzz_mev

    @property
    def total_mev(self) -> float:
        """The four groups added up: lambda / M."""
        return self.hopping_mev + self.on_site_mev


def one_site_one_norm() -> OneSiteOneNorm:
    """The pieces at the lecture's spacing, which sum to lambda / M."""
    strings = pauli_coefficients(LECTURE_COUPLINGS)
    return OneSiteOneNorm(
        hopping_mev=24.0 * abs(strings.hopping_mev),
        single_z_mev=4.0 * abs(strings.single_z_mev),
        pair_zz_mev=6.0 * abs(strings.pair_zz_mev),
        triple_zzz_mev=4.0 * abs(strings.triple_zzz_mev),
    )


@dataclass(frozen=True, kw_only=True)
class EnergyRegister:
    """What W_q bits of phase estimation buy."""

    bits: int  # W_q
    walk_steps: int  # 2^(W_q - 1)
    bin_width_mev: float  # 2 pi lambda / 2^W_q, at mid-band
    rms_error_mev: float  # sqrt(2) pi lambda / 2^W_q, the delivered error


def energy_register(lattice: Lattice, bits: int) -> EnergyRegister:
    """The steps, the bin and the delivered error of a register of this many bits.

    Raises:
        ValueError: bits is not a whole number from 3 to 64.
    """
    if type(bits) is not int or not MIN_REGISTER_BITS <= bits <= MAX_REGISTER_BITS:
        raise ValueError(
            f"register bits must be a whole number from {MIN_REGISTER_BITS} to "
            f"{MAX_REGISTER_BITS}: below that the delivered error exceeds lambda "
            "itself, and above it the walk alone is more than 2^63 steps; "
            f"got {bits!r}."
        )
    one_norm = lattice.one_norm_mev
    return EnergyRegister(
        bits=bits,
        walk_steps=two_sided_qpe_application_count(bits),
        bin_width_mev=2.0 * math.pi * one_norm / 2**bits,
        rms_error_mev=two_sided_qpe_rms_energy_error(one_norm, bits),
    )


def register_bits_for_sizing(lattice: Lattice, sizing_mev: float) -> int:
    """The smallest register with lambda / 2^W_q <= sizing_mev.

    The sizing is a label, never the resolution delivered.

    Raises:
        ValueError: The register would have more bits than energy_register allows,
            which a smaller box_side or a coarser sizing avoids.
    """
    one_norm = lattice.one_norm_mev
    bits = qubiterate_qpe_ancilla_count(one_norm, sizing_mev).ancilla_qubits
    if bits > MAX_REGISTER_BITS:
        raise ValueError(
            f"the sizing rule at {sizing_mev} MeV asks {bits} register bits on the "
            f"{lattice.box_side}^3 lattice, more than the {MAX_REGISTER_BITS} the walk "
            "can run; choose a smaller box_side or a coarser sizing."
        )
    return bits


@dataclass(frozen=True, kw_only=True)
class WalkStep:
    """The counts of one walk step, before any price."""

    keep_bits: int  # the alias sampling's mu
    prepare_toffolis: int  # one direction
    prepare_rotations: int  # one direction
    select_toffolis: int  # Gamma - 1, by unary iteration
    reflect_toffolis: int  # an AND chain over the ancillas, with the control
    ancilla_qubits: int  # the prepare's register


def walk_step(nucleus: Nucleus, lattice: Lattice, register: EnergyRegister) -> WalkStep:
    """Keep bits from the delivered error and the bound on the energies read, then
    the rest.

    The reflection covers every ancilla the prepare touches, as Babbush+ 2018
    (arXiv:1805.03662, Fig. 1 and Sec. III D) write it. The alternate index, the keep
    register and the comparison bit are set by classical steps from the index and
    sigma, which SELECT only reads, so PREPARE's inverse returns them to zero on every
    input, and a reflection over the index, sigma and the site would do. For carbon on
    4^3 that is 24 qubits instead of 43, 19 ANDs or 76 T fewer a step, 0.7%.

    The select and the reflection are priced controlled on every step. Phase
    estimation that runs the inverse walk on the 0 branch runs every select but the
    first, and most reflections, uncontrolled (qubitization.PerShotCost), one AND
    fewer each: 8 T a step, under 0.1% of carbon's on 4^3.
    """
    one_norm = lattice.one_norm_mev
    energy_bound = phase_estimation_energy_bound(
        nucleus.nucleon_count, lattice.site_count, LECTURE_COUPLINGS
    )
    keep_bits = lookup_keep_probability_bits(
        one_norm, register.rms_error_mev, spectral_norm_bound=energy_bound
    )
    prepare = translation_invariant_lookup_prepare_cost(
        TERMS_PER_SITE, lattice.site_count, keep_bits=keep_bits
    )
    return WalkStep(
        keep_bits=keep_bits,
        prepare_toffolis=prepare.toffoli_count,
        prepare_rotations=prepare.rotation_count,
        select_toffolis=unary_iteration_select_toffoli_count(lattice.term_count),
        reflect_toffolis=controlled_reflection_toffoli_count(prepare.register_qubits),
        ancilla_qubits=prepare.register_qubits,
    )


@dataclass(frozen=True, kw_only=True)
class Insertion:
    """The counts of one application of the current, the proton density."""

    terms: int  # 2 M, a Z on each proton mode
    index_qubits: int  # x, y and z in binary, and one for the spin
    toffolis: int  # a superposition per axis, done and undone, and a select
    rotations: int  # the phases' and the superpositions'
    t_gates: int  # the phases that are exactly a T gate, with nothing to synthesize


@dataclass(frozen=True, kw_only=True)
class PhaseGates:
    """The gates that put the current's phase on one coordinate's bits."""

    rotations: int  # turns that must be synthesized
    t_gates: int  # turns of a whole number of eighths, each exactly one T gate


def _phase_gates(full_turn: int, bits: int) -> PhaseGates:
    """The gates that turn bit b of a register by 2^b / full_turn of a full circle.

    A whole number of quarter turns is a Z or an S, which take no T. A whole number of
    eighths takes one T gate. Any other turn is a rotation to synthesize.
    """
    rotations = 0
    t_gates = 0
    for bit in range(bits):
        whole_quarters = (4 * 2**bit) % full_turn == 0
        whole_eighths = (8 * 2**bit) % full_turn == 0
        if whole_eighths and not whole_quarters:
            t_gates += 1
        if not whole_eighths:
            rotations += 1
    return PhaseGates(rotations=rotations, t_gates=t_gates)


def phase_gates_per_axis(lattice: Lattice) -> PhaseGates:
    """The gates for exp(2 pi i k x / n) on one coordinate x, at the costliest momentum.

    Writing x in binary makes this phase one gate per bit: bit b of x turns by k 2^b / n
    of a full circle. A whole number of quarter turns is a Z or an S, which take no T.
    A whole number of eighths takes one T gate. Any other turn is a rotation to
    synthesize.

    This charges k = 1, the costliest momentum, since the turn at any other k is a whole
    multiple of the turn at k = 1. With n = 4, q.x = (pi / 2) k.x, so every phase is a
    power of i and the phases cost nothing at every momentum.
    """
    box_side = lattice.box_side
    axis_bits = equal_superposition_cost(box_side).register_qubits
    return _phase_gates(box_side, axis_bits)


def gradient_state_gates(bits: int) -> PhaseGates:
    """The gates that make the phase-gradient state a load adds its angles into.

    The state is the sum over k of exp(-2 pi i k / 2^c) |k> on c = bits qubits, one
    qubit per bit: a Hadamard, then a turn of 2^b / 2^c of a full circle on bit b. So
    the top two bits take a Z and an S, the next one T gate, and each of the other
    c - 3 a rotation. The state is never used up, so a circuit makes it once.
    """
    return _phase_gates(2**bits, bits)


def insertion(lattice: Lattice) -> Insertion:
    """A block encoding of the current, whose terms all have one size, so equal
    superpositions are its whole PREPARE.

    The index holds x, y and z in their own registers and one bit for the spin. So
    PREPARE is one superposition per axis, a site's phase is one gate per bit
    (phase_gates_per_axis), and SELECT is still a tree with 2 M leaves, 2 M - 1 ANDs.

    One register counting the terms would take fewer qubits where n is not a power of
    two, 8 instead of 10 on 5^3, but its phases would not factor: 250 rotations instead
    of 9. This takes the fewer T gates, since a low error rate is expected to be much
    harder to come by than qubits. Where n is a power of two the two are the same
    register.
    """
    terms = 2 * lattice.site_count
    axis = equal_superposition_cost(lattice.box_side)
    phases = phase_gates_per_axis(lattice)
    select_toffolis = unary_iteration_select_toffoli_count(terms)
    return Insertion(
        terms=terms,
        index_qubits=AXES * axis.register_qubits + 1,
        toffolis=2 * AXES * axis.toffoli_count + select_toffolis,
        rotations=AXES * phases.rotations + 2 * AXES * axis.rotation_count,
        t_gates=AXES * phases.t_gates,
    )


@dataclass(frozen=True, kw_only=True)
class RoundReflections:
    """The ANDs of one amplification round's two reflections."""

    index_chain_ands: int  # about the index register's zero state
    start_chain_ands: int  # about the zero state the preparation starts from


def round_reflections(lattice: Lattice, *, enumeration_qubits: int) -> RoundReflections:
    """An AND chain over each reflection's qubits, two fewer ANDs than qubits.

    The long reflection is about the state the preparation starts from, so it covers
    every register the undone preparation can leave away from zero where the index is
    zero: the system, the index and a table load's enumeration register. That register
    returns to zero only on the load's own output (Fomichev+ 2023, arXiv:2310.18410,
    Sec. III), and a reflection blind to it amplifies nothing. The spread's shift
    register is left away from zero only where the index is too, so the reflection
    need not cover it, though it is not clean (QubitCount.held_qubits). The load's
    other registers return to zero, so the chain need not cover them.

    A chain holds one ancilla for each AND at its peak. A construction in blocks holds
    fewer ancillas at the price of more ANDs; this takes the chain, the fewer T gates,
    since a low error rate is expected to be much harder to come by than qubits.

    Args:
        lattice: The lattice.
        enumeration_qubits: The table load's enumeration register, 0 with no table.
    """
    index_qubits = insertion(lattice).index_qubits
    reflected_qubits = lattice.system_qubits + index_qubits + enumeration_qubits
    return RoundReflections(
        index_chain_ands=reflection_toffoli_count(index_qubits),
        start_chain_ands=reflection_toffoli_count(reflected_qubits),
    )


@dataclass(frozen=True, kw_only=True)
class Amplification:
    """An amplitude-amplification schedule."""

    success_probability: float  # P_s before amplifying
    rounds: int  # m
    landing_probability: float  # sin^2((2m + 1) theta)

    @property
    def preparations(self) -> int:
        """1 + 2 m: one load to start, and two in each round."""
        return 1 + 2 * self.rounds

    @property
    def preparations_per_success(self) -> float:
        """Preparations for each circuit that lands. A circuit that misses is run again,
        so a success takes 1 / landing circuits on average."""
        return self.preparations / self.landing_probability


def amplification(success_probability: float) -> Amplification:
    """The m that lands (2m + 1) theta nearest a quarter turn, with sin theta =
    sqrt(P_s); the fewer rounds on a tie.

    Amplitude amplification (Brassard, Hoyer, Mosca and Tapp 2002,
    arXiv:quant-ph/0005055) takes 1 + 2m preparations, about (pi / 2) / sqrt(P_s), for
    one success. Post-selection, which applies the current to one preparation,
    measures the index and starts over on a failure, takes 1 / P_s preparations and
    needs neither the preparation's inverse nor P_s in advance: for carbon on 4^3,
    683 to 1,365 preparations a success against 41 to 59.

    The quarter-turn landing is the textbook choice, and the lecture's round counts
    follow it. A miss is found at the index, before the energy measurement, and
    repeats only the amplified preparation (OneCircuit.set_t). So the energy
    measurement runs once per success under either rule, and the preparations per
    success, (1 + 2m) / landing, are fewest at a landing near 0.84, where
    x = (2m + 1) theta solves tan x = 2x. For carbon on 4^3 that is m = 21 and 15 in
    place of 29 and 20, 12 to 14% fewer preparations a success; for helium on 3^3,
    where every part is priced, the circuit per success falls by 9 to 13%.

    Raises:
        ValueError: P_s does not lie above 0 and at most 1.
    """
    if not 0.0 < success_probability <= 1.0:
        raise ValueError(
            "success probability must lie above 0 and at most 1, and it is the "
            "strength divided by the square of the current's one-norm; "
            f"got {success_probability}."
        )
    theta = math.asin(math.sqrt(success_probability))
    lower = math.floor(math.pi / (4.0 * theta) - 0.5)
    upper = lower + 1
    lower_miss = abs((2 * lower + 1) * theta - math.pi / 2.0)
    upper_miss = abs((2 * upper + 1) * theta - math.pi / 2.0)
    rounds = lower
    if upper_miss < lower_miss and not math.isclose(upper_miss, lower_miss):
        rounds = upper
    landing = math.sin((2 * rounds + 1) * theta) ** 2
    return Amplification(
        success_probability=success_probability,
        rounds=rounds,
        landing_probability=landing,
    )


@dataclass(frozen=True, kw_only=True)
class Synthesis:
    """What one circuit's rotation count buys under the shared budget.

    The rotations' errors are coherent and add linearly, so the N rotations share the
    budget evenly (synthesis.coherent_synthesis_precision), and each is priced by
    repeat-until-success synthesis (synthesis.repeat_until_success_t_count), whose T
    count is an expected value. A precision below the machine's logical error per
    operation would buy nothing; carbon's 8.1e-7 a rotation sits far above the 1e-10
    of the 2033 budget.

    N counts every rotation and rounded angle priced. An unpriced preparation's own
    are not in it: one that spent carbon's allowance in walk steps would bring about
    30 times the circuit's rotations, and raise each one's price from 34 T to 40 T,
    a walk step's by 0.2%.
    """

    rotation_count: int  # N, every rotation and rounded angle priced
    precision: float  # the error budget over N, 0.01 / N unless asked otherwise
    t_per_rotation: int  # by repeat-until-success synthesis at that precision
    angle_bits: int  # of the load's rounded angles, at the same precision


def synthesis_budget(
    rotation_count: int, *, error_budget: float = SYNTHESIS_ERROR_BUDGET
) -> Synthesis:
    """Share the circuit's error budget evenly: epsilon = 0.01 / N for each rotation,
    which sets its T gates and the bits of each rounded angle. Another error_budget
    shows how weakly the price follows it."""
    precision = coherent_synthesis_precision(error_budget, rotation_count)
    return Synthesis(
        rotation_count=rotation_count,
        precision=precision,
        t_per_rotation=repeat_until_success_t_count(precision),
        angle_bits=phase_gradient_angle_bits(precision),
    )


@dataclass(frozen=True, kw_only=True)
class GroundStateLoad:
    """The counts of one load of the pinned table with its spreading.

    The spread runs after the load, on ancillas the load has returned to zero, so it
    adds none of its own: it needs 3 ceil(log2 n) + 2, 8 for helium on 3^3, and the
    load frees all but its enumeration and gradient registers, 57 there. Its shift
    register, a coordinate for each axis, is erased by reading the pinned nucleon's
    site, which returns it to zero only on a state of zero momentum; undone at the long
    reflection, after the current, it is left away from zero where the index is not.
    Its two sign qubits are computed from the system and uncomputed, so they are clean.
    """

    amplitudes: int  # D
    load_toffolis: int
    spread_toffolis: int
    spread_rotations: int
    ancilla_qubits: int  # all of them, the next three included
    enumeration_qubits: int  # L, which the long reflection must cover
    gradient_qubits: int  # c, a resource state held throughout
    shift_qubits: int  # 3 ceil(log2 n), not clean at the long reflection
    gradient_state: PhaseGates  # to make that state, once per circuit


def enumeration_register_qubits(amplitudes: int) -> int:
    """The load's enumeration register for a table of D amplitudes, ceil(log2 D)."""
    return _bits_to_number(amplitudes)


def load_coherent_errors(amplitudes: int, lattice: Lattice) -> int:
    """One load's share of the rotation count: a rounded angle per level of the table,
    ceil(log2 D) of them, and the spread's rotations."""
    levels = enumeration_register_qubits(amplitudes)
    spread = translation_symmetrization_cost(lattice.box_side, MODES_PER_SITE)
    return levels + spread.rotation_count


def ground_state_load(
    amplitudes: int, lattice: Lattice, angle_bits: int
) -> GroundStateLoad:
    """The sparse load of the table, then its spread over the lattice's translations."""
    load = sparse_state_load_cost(amplitudes, rotation_bits=angle_bits)
    spread = translation_symmetrization_cost(lattice.box_side, MODES_PER_SITE)
    return GroundStateLoad(
        amplitudes=amplitudes,
        load_toffolis=load.toffoli_count,
        spread_toffolis=spread.toffoli_count,
        spread_rotations=spread.rotation_count,
        ancilla_qubits=load.ancilla_qubits,
        enumeration_qubits=enumeration_register_qubits(amplitudes),
        gradient_qubits=angle_bits,
        shift_qubits=AXES * equal_superposition_cost(lattice.box_side).register_qubits,
        gradient_state=gradient_state_gates(angle_bits),
    )


@dataclass(frozen=True, kw_only=True)
class QubitCount:
    """The register by stage, in logical qubits."""

    system: int
    energy_register: int
    walk_ancillas: int
    walk_chain: int
    insertion_index: int
    amplification_chain: int
    load_ancillas: int  # all of them, zero where the preparation is unpriced
    load_enumeration: int  # of those, the register the long reflection covers
    load_gradient: int  # of those, the resource state that is never clean
    load_shift: int  # of those, the spread's register, not clean at the reflection

    @property
    def walk_stage(self) -> int:
        """The system, the energy register, the ancillas and the AND chain."""
        return self.system + self.energy_register + self.walk_ancillas + self.walk_chain

    @property
    def walk_added(self) -> int:
        """What the walk adds to the system."""
        return self.walk_stage - self.system

    @property
    def load_other_ancillas(self) -> int:
        """The load's ancillas but its enumeration, gradient and shift registers: these
        return to zero after each load and each undone load, so they can take turns
        with the AND chain."""
        held = self.load_enumeration + self.load_gradient + self.load_shift
        return self.load_ancillas - held

    @property
    def reflected_qubits(self) -> int:
        """What the long reflection covers: the system, the index, the enumeration."""
        return self.system + self.insertion_index + self.load_enumeration

    @property
    def held_qubits(self) -> int:
        """What stays held through the long reflection: what it covers, the gradient
        state, and the spread's shift register. The reflection need not cover the shift
        register, which is away from zero only where the index is too, but the chain's
        ANDs need clean qubits, so it cannot take turns with them."""
        return self.reflected_qubits + self.load_gradient + self.load_shift

    @property
    def amplification_stage(self) -> int:
        """What stays held, and the larger of the load's other ancillas and the chain,
        which take turns in every round."""
        taking_turns = max(self.load_other_ancillas, self.amplification_chain)
        return self.held_qubits + taking_turns

    @property
    def amplification_added(self) -> int:
        """What the amplification adds to the system."""
        return self.amplification_stage - self.system

    @property
    def peak(self) -> int:
        """The larger of the two stages, what the machine must hold. An unpriced
        preparation's ancillas come on top where they outnumber the chain's ANDs, or
        where the reflection must cover them, so with no load this is a floor."""
        return max(self.walk_stage, self.amplification_stage)

    @property
    def free_for_preparation(self) -> int:
        """What a preparation's ancillas could use in the 2033 budget, if they are clean
        at every reflection and so take turns with the chain: 1,000 less what stays
        held. None are free where that and the chain beside it already pass 1,000, as
        for argon on 5^3, since ancillas that take turns with the chain leave the stage
        at least that large."""
        if self.held_qubits + self.amplification_chain > LOGICAL_QUBITS_2033:
            return 0
        return LOGICAL_QUBITS_2033 - self.held_qubits

    @property
    def free_if_reflected_over(self) -> int:
        """What a preparation could use if the reflection must cover all its ancillas:
        each then also lengthens the chain by one, so it takes two qubits."""
        return max(0, (LOGICAL_QUBITS_2033 - self.amplification_stage) // 2)


@dataclass(frozen=True, kw_only=True)
class ReadoutRotations:
    """The readout's rotations, both counts bounds."""

    fourier: int  # the inverse Fourier transform's, W_q (W_q - 1)
    window: int  # the window state's, 2^W_q

    @property
    def total(self) -> int:
        """Both parts."""
        return self.fourier + self.window


def readout_rotations(bits: int) -> ReadoutRotations:
    """The rotations of the readout of a register of this many bits.

    The window state is charged 2^W_q rotations, about what a rotation tree over its
    2^W_q amplitudes takes, and the inverse Fourier transform W_q (W_q - 1). Babbush+
    2018 (arXiv:1805.03662, Fig. 2) give both a gate complexity of soft-O(W_q),
    linear in W_q up to logarithmic factors, the window state by their Eqs. 17-20.
    On carbon's 4^3 lattice the window's 4,096 rotations are a third of those sharing
    the budget, so their window would save at most 0.7% of the priced parts.
    """
    return ReadoutRotations(fourier=bits * (bits - 1), window=2**bits)


@dataclass(frozen=True, kw_only=True)
class OnePerToffoli:
    """One circuit's counts if the budget counts a Toffoli as one hard operation."""

    walk_step: int
    energy_measurement: int
    all_but_preparations: int
    whole: int | None  # None where the preparation is unpriced
    allowance_in_walk_steps: float
    rule_of_thumb_mev: float | None  # lambda over those steps; None with nothing left

    @property
    def everything_priced(self) -> int:
        """The whole circuit where the preparation is priced, and otherwise all but the
        preparations."""
        if self.whole is None:
            return self.all_but_preparations
        return self.whole


@dataclass(frozen=True, kw_only=True)
class OneCircuit:
    """One amplified circuit at one strength, with every piece priced."""

    nucleus: Nucleus
    lattice: Lattice
    register: EnergyRegister
    step: WalkStep
    current: Insertion
    reflections: RoundReflections
    schedule: Amplification
    synthesis: Synthesis
    load: GroundStateLoad | None  # None where no classical table exists

    @property
    def prepare_pair_t(self) -> int:
        """PREPARE and its inverse."""
        toffoli_t = T_PER_TOFFOLI * self.step.prepare_toffolis
        rotation_t = self.step.prepare_rotations * self.synthesis.t_per_rotation
        return 2 * (toffoli_t + rotation_t)

    @property
    def select_t(self) -> int:
        """SELECT, 4 T per AND."""
        return T_PER_TOFFOLI * self.step.select_toffolis

    @property
    def reflect_t(self) -> int:
        """REFLECT, 4 T per AND."""
        return T_PER_TOFFOLI * self.step.reflect_toffolis

    @property
    def walk_step_t(self) -> int:
        """One walk step, the three pieces above."""
        return self.prepare_pair_t + self.select_t + self.reflect_t

    @property
    def walk_t(self) -> int:
        """The walk: steps times one step."""
        return self.register.walk_steps * self.walk_step_t

    @property
    def select_share(self) -> float:
        """SELECT's share of one walk step."""
        return self.select_t / self.walk_step_t

    @property
    def readout_t(self) -> int:
        """The inverse QFT and the window state."""
        rotations = readout_rotations(self.register.bits).total
        return rotations * self.synthesis.t_per_rotation

    @property
    def energy_measurement_t(self) -> int:
        """The walk and the readout."""
        return self.walk_t + self.readout_t

    @property
    def insertion_t(self) -> int:
        """One application of the current."""
        toffoli_t = T_PER_TOFFOLI * self.current.toffolis
        rotation_t = self.current.rotations * self.synthesis.t_per_rotation
        return toffoli_t + rotation_t + self.current.t_gates

    @property
    def round_reflections_t(self) -> int:
        """One round's two reflections."""
        ands = self.reflections.index_chain_ands + self.reflections.start_chain_ands
        return T_PER_TOFFOLI * ands

    @property
    def insertions_t(self) -> int:
        """One insertion for each preparation."""
        return self.schedule.preparations * self.insertion_t

    @property
    def reflections_t(self) -> int:
        """Two reflections in each round."""
        return self.schedule.rounds * self.round_reflections_t

    @property
    def all_but_preparations_t(self) -> int:
        """The energy measurement, the insertions and the reflections: everything but
        the preparations, priced or not."""
        return self.energy_measurement_t + self.insertions_t + self.reflections_t

    @property
    def load_t(self) -> int | None:
        """One load with its spreading; None where unpriced."""
        if self.load is None:
            return None
        toffolis = self.load.load_toffolis + self.load.spread_toffolis
        rotation_t = self.load.spread_rotations * self.synthesis.t_per_rotation
        return T_PER_TOFFOLI * toffolis + rotation_t

    @property
    def all_loads_t(self) -> int | None:
        """Every load of the circuit; None where unpriced."""
        one_load_t = self.load_t
        if one_load_t is None:
            return None
        return self.schedule.preparations * one_load_t

    @property
    def gradient_state_t(self) -> int | None:
        """The load's gradient state, made once per circuit; None where unpriced."""
        if self.load is None:
            return None
        gates = self.load.gradient_state
        return gates.rotations * self.synthesis.t_per_rotation + gates.t_gates

    @property
    def circuit_t(self) -> int | None:
        """The whole circuit; None where the preparation is unpriced."""
        loads_t = self.all_loads_t
        gradient_t = self.gradient_state_t
        if loads_t is None or gradient_t is None:
            return None
        return self.all_but_preparations_t + loads_t + gradient_t

    @property
    def everything_priced_t(self) -> int:
        """The whole circuit where the preparation is priced, and otherwise all but the
        preparations, a floor on the circuit."""
        circuit_t = self.circuit_t
        if circuit_t is None:
            return self.all_but_preparations_t
        return circuit_t

    @property
    def needed_code_distance(self) -> int:
        """The code distance at the lower of two logical error rates: the one the
        priced parts need, and the one a circuit that fills the budget needs. The
        second is the lower wherever the priced parts fit, and the first where they
        pass the budget."""
        on_priced = logical_error_rate(self.everything_priced_t)
        on_budget = logical_error_rate(HARD_OPERATIONS_2033)
        return code_distance(min(on_priced, on_budget))

    def set_t(self, t_per_circuit: float, *, successes: int) -> float:
        """The T that so many successful circuits cost, each of t_per_circuit T.

        The index is measured before the energy measurement, so a circuit that misses
        runs only the rest again: its loads, insertions, reflections and gradient
        state. A success then pays the energy measurement once and the rest once per
        attempt, 1 / landing probability attempts on average. t_per_circuit is this
        circuit's priced parts or the whole budget. The gradient state is counted with
        every attempt, though a miss could keep it, under 0.01% of helium's circuit.

        Raises:
            ValueError: successes is not above zero, or t_per_circuit is below the
                energy measurement it holds.
        """
        _require_positive(successes, "successes")
        energy = self.energy_measurement_t
        if t_per_circuit < energy:
            raise ValueError(
                f"t_per_circuit {t_per_circuit:,} is below this circuit's energy "
                f"measurement of {energy:,} T, which it must hold."
            )
        rest = t_per_circuit - energy
        per_success = energy + rest / self.schedule.landing_probability
        return successes * per_success

    @property
    def loads_share(self) -> float | None:
        """Every load's share of the whole circuit; None where unpriced."""
        loads_t = self.all_loads_t
        if loads_t is None:
            return None
        return loads_t / self.everything_priced_t

    @property
    def energy_measurement_share(self) -> float:
        """The energy measurement's share of everything priced."""
        return self.energy_measurement_t / self.everything_priced_t

    @property
    def rotation_share(self) -> float:
        """The rotations' T as a share of everything priced, over the same parts.

        A load's rounded angles count in N but not here: the load adds them into the
        gradient state with Toffolis, which its price already holds.
        """
        rotations = self.synthesis.rotation_count
        if self.load is not None:
            rounded_angles = self.schedule.preparations * self.load.enumeration_qubits
            rotations = rotations - rounded_angles
        rotations_t = rotations * self.synthesis.t_per_rotation
        return rotations_t / self.everything_priced_t

    @property
    def left_over_t(self) -> int:
        """10^9 - all but the preparations: the budget left for the preparations. It is
        negative where the rest of the circuit is already over the budget."""
        return HARD_OPERATIONS_2033 - self.all_but_preparations_t

    @property
    def allowance_t(self) -> float:
        """What is left over, shared by the preparations: what each one may cost."""
        return self.left_over_t / self.schedule.preparations

    @property
    def allowance_in_walk_steps(self) -> float:
        """The allowance, as a number of walk steps."""
        return self.allowance_t / self.walk_step_t

    @property
    def rule_of_thumb_mev(self) -> float | None:
        """lambda over the allowance's walk steps, in MeV. Projecting out the ground
        state takes about lambda / (gamma Delta) steps, with gamma the trial state's
        overlap and Delta the gap, so this is the least gamma Delta the allowance pays
        for. None where nothing is left for the preparations, so no gap is paid for."""
        if self.allowance_t <= 0.0:
            return None
        return self.lattice.one_norm_mev / self.allowance_in_walk_steps

    @property
    def one_per_toffoli(self) -> OnePerToffoli:
        """The same circuit if the budget counts a Toffoli as one hard operation.

        Everything else here charges a Toffoli as 4 T. A rotation's T stay T, so each
        Toffoli simply gives back T_PER_TOFFOLI - 1 of them.
        """
        given_back = T_PER_TOFFOLI - 1
        schedule = self.schedule
        step = self.step
        step_toffolis = (
            2 * step.prepare_toffolis + step.select_toffolis + step.reflect_toffolis
        )
        walk_toffolis = self.register.walk_steps * step_toffolis
        round_ands = (
            self.reflections.index_chain_ands + self.reflections.start_chain_ands
        )
        rest_toffolis = (
            walk_toffolis
            + schedule.preparations * self.current.toffolis
            + schedule.rounds * round_ands
        )
        walk_step = self.walk_step_t - given_back * step_toffolis
        rest = self.all_but_preparations_t - given_back * rest_toffolis
        left_over = HARD_OPERATIONS_2033 - rest
        allowance_in_walk_steps = left_over / schedule.preparations / walk_step
        rule_of_thumb_mev = None
        if left_over > 0:
            rule_of_thumb_mev = self.lattice.one_norm_mev / allowance_in_walk_steps
        whole = None
        circuit_t = self.circuit_t
        if circuit_t is not None and self.load is not None:
            load_toffolis = self.load.load_toffolis + self.load.spread_toffolis
            all_toffolis = rest_toffolis + schedule.preparations * load_toffolis
            whole = circuit_t - given_back * all_toffolis
        return OnePerToffoli(
            walk_step=walk_step,
            energy_measurement=self.energy_measurement_t - given_back * walk_toffolis,
            all_but_preparations=rest,
            whole=whole,
            allowance_in_walk_steps=allowance_in_walk_steps,
            rule_of_thumb_mev=rule_of_thumb_mev,
        )

    @property
    def qubits(self) -> QubitCount:
        """The register at each stage of this circuit."""
        load_ancillas = 0
        load_enumeration = 0
        load_gradient = 0
        load_shift = 0
        if self.load is not None:
            load_ancillas = self.load.ancilla_qubits
            load_enumeration = self.load.enumeration_qubits
            load_gradient = self.load.gradient_qubits
            load_shift = self.load.shift_qubits
        return QubitCount(
            system=self.lattice.system_qubits,
            energy_register=self.register.bits,
            walk_ancillas=self.step.ancilla_qubits,
            walk_chain=self.step.reflect_toffolis,
            insertion_index=self.current.index_qubits,
            amplification_chain=self.reflections.start_chain_ands,
            load_ancillas=load_ancillas,
            load_enumeration=load_enumeration,
            load_gradient=load_gradient,
            load_shift=load_shift,
        )


def _budget_with_gradient_state(rotations: int) -> Synthesis:
    """The budget when a load's gradient state shares it.

    The state's rotations join the count, and the count sets the state's bits, so the
    two are solved together: add the state's rotations until its bits stop changing.
    More rotations never mean fewer bits, so this settles within a pass or two.
    """
    synthesis = synthesis_budget(rotations)
    bits = 0
    while bits != synthesis.angle_bits:
        bits = synthesis.angle_bits
        gradient = gradient_state_gates(bits)
        synthesis = synthesis_budget(rotations + gradient.rotations)
    return synthesis


def price_one_circuit(
    nucleus: Nucleus,
    *,
    box_side: int,
    register_bits: int,
    strength: float,
    current_one_norm: float,
) -> OneCircuit:
    """Price one circuit: the design the lecture explains, at one strength.

    The box side comes as a plain number, as the exercises give it; every piece below
    takes the one Lattice built from it.

    Args:
        nucleus: The nucleus.
        box_side: Sites per axis.
        register_bits: W_q.
        strength: eta^2, the response's total weight at this momentum (dimensionless).
        current_one_norm: Lambda_J, so that P_s = eta^2 / Lambda_J^2: the site count M
            for the current the lecture prices, or the proton count Z for the lever
            nobody has built.

    Raises:
        ValueError: A species holds more nucleons than the lattice has sites (Pauli
            exclusion); strength or current_one_norm is not above zero; P_s exceeds 1;
            or as Lattice and energy_register raise.
    """
    _require_positive(strength, "strength")
    _require_positive(current_one_norm, "current_one_norm")
    lattice = Lattice(box_side=box_side)
    if ground_state_amplitudes(nucleus, lattice) == 0:
        largest_species = max(nucleus.nucleons_per_species)
        raise ValueError(
            f"{nucleus.name} has {largest_species} nucleons of one species, more than "
            f"the {lattice.site_count} sites of this lattice can hold, by Pauli "
            "exclusion; choose a larger box_side."
        )
    register = energy_register(lattice, register_bits)
    step = walk_step(nucleus, lattice, register)
    current = insertion(lattice)
    schedule = amplification(strength / current_one_norm**2)
    amplitudes = pinned_table_amplitudes(nucleus, lattice)
    if amplitudes == 1:
        raise ValueError(
            f"{nucleus.name} has a pinned table of one configuration on this lattice, "
            "since its other species are empty or fill every site; one configuration "
            "needs no load, and this module prices tables of two or more."
        )

    rotations_per_preparation = current.rotations
    if amplitudes is not None:
        rotations_per_preparation += load_coherent_errors(amplitudes, lattice)
    walk_rotations = register.walk_steps * 2 * step.prepare_rotations
    readout = readout_rotations(register_bits).total
    preparation_rotations = schedule.preparations * rotations_per_preparation
    rotations = walk_rotations + readout + preparation_rotations
    synthesis = synthesis_budget(rotations)

    load = None
    enumeration_qubits = 0
    if amplitudes is not None:
        synthesis = _budget_with_gradient_state(rotations)
        load = ground_state_load(amplitudes, lattice, synthesis.angle_bits)
        enumeration_qubits = load.enumeration_qubits

    return OneCircuit(
        nucleus=nucleus,
        lattice=lattice,
        register=register,
        step=step,
        current=current,
        reflections=round_reflections(lattice, enumeration_qubits=enumeration_qubits),
        schedule=schedule,
        synthesis=synthesis,
        load=load,
    )


def logical_error_rate(hard_operations: int) -> float:
    """epsilon_l = 0.1 / N_T, per hard operation.

    This charges the logical errors to the hard operations alone and leaves out the
    volume of the Clifford gates and the idle qubits. Read instead per logical qubit
    per hard operation, a spacetime volume, the budget is 0.1 / (Q N_T): for carbon's
    524 logical qubits at the full 2033 budget, d = 25 and 6.6e5 physical qubits
    against 19 and 3.8e5.

    Raises:
        ValueError: hard_operations is not above zero.
    """
    _require_positive(hard_operations, "hard_operations")
    return FAULT_BUDGET / hard_operations


def expected_faults(hard_operations: int, *, logical_error: float) -> float:
    """N_T epsilon_l, the logical faults a circuit of N_T hard operations expects.

    Raises:
        ValueError: hard_operations is not above zero, or logical_error does not lie
            strictly between 0 and 1.
    """
    _require_positive(hard_operations, "hard_operations")
    _require_between_zero_and_one(logical_error, "logical_error")
    return hard_operations * logical_error


def code_distance(logical_error: float, *, physical_over_threshold: float = 0.1) -> int:
    """The smallest odd d with (p/p_th)^((d + 1) / 2) <= epsilon_l: the lecture's
    schematic. The comparison has a tolerance, since 0.1^10 lands just above 1e-10.

    Raises:
        ValueError: Either rate does not lie strictly between 0 and 1.
    """
    _require_between_zero_and_one(logical_error, "logical_error")
    _require_between_zero_and_one(physical_over_threshold, "physical_over_threshold")
    distance = 1
    error = physical_over_threshold
    while error > logical_error and not math.isclose(error, logical_error):
        distance = distance + 2
        error = physical_over_threshold ** ((distance + 1) // 2)
    return distance


def physical_qubits_per_logical(distance: int) -> int:
    """2 d^2.

    Raises:
        ValueError: distance is not above zero.
    """
    _require_positive(distance, "distance")
    return 2 * distance**2


def physical_qubits(logical_qubits: int, *, distance: int) -> int:
    """2 d^2 for each logical qubit.

    Raises:
        ValueError: logical_qubits or distance is not above zero.
    """
    _require_positive(logical_qubits, "logical_qubits")
    return logical_qubits * physical_qubits_per_logical(distance)


def randomized_synthesis_precision(rotation_count: int) -> float:
    """epsilon = sqrt(0.01 / (5 N)), the randomized alternative.

    Mixing approximations of each rotation at random keeps each within 5 epsilon^2 in
    diamond distance (Campbell 2017, arXiv:1612.02689, Thm. 2 and Eq. 11), so N of
    them stay within 0.01 when 5 N epsilon^2 <= 0.01. The theorem asks for
    epsilon < 0.01, which holds for N above 20, and for sequences from a
    deterministic synthesis such as Ross and Selinger's; repeat-until-success
    synthesis is not covered.

    Raises:
        ValueError: rotation_count is not above zero.
    """
    _require_positive(rotation_count, "rotation_count")
    return math.sqrt(SYNTHESIS_ERROR_BUDGET / (5.0 * rotation_count))


def ross_selinger_t_estimate(precision: float) -> float:
    """3 log2(1 / epsilon), the deterministic method that theorem covers.

    Ross and Selinger 2016 (arXiv:1403.2975) approximate a z-rotation within epsilon
    with 3 log2(1 / epsilon) + O(log(log(1 / epsilon))) T gates in the typical case;
    this leaves out the second term.

    Raises:
        ValueError: precision is not above zero.
    """
    _require_positive(precision, "precision")
    return 3.0 * math.log2(1.0 / precision)


def _round_up(count: float) -> int:
    """A count rounded up to a whole number, unless it is one already but for the
    noise of floating point: 0.16 / 0.0001 lands just above 1,600."""
    nearest = round(count)
    if math.isclose(count, nearest):
        return nearest
    return math.ceil(count)


def bins_in_window(window_mev: float, *, bin_width_mev: float) -> int:
    """Bins a response of this width fills, rounded up.

    Raises:
        ValueError: Either width is not above zero.
    """
    _require_positive(window_mev, "window_mev")
    _require_positive(bin_width_mev, "bin_width_mev")
    return _round_up(window_mev / bin_width_mev)


def runs_to_pin_bin_to_itself(bin_weight: float, *, precision: float) -> int:
    """N = (1 - p) / (p epsilon^2): a bin of weight p to a fraction epsilon of
    itself.

    Raises:
        ValueError: bin_weight does not lie strictly between 0 and 1, or precision is
            not above zero.
    """
    _require_between_zero_and_one(bin_weight, "bin_weight")
    _require_positive(precision, "precision")
    return _round_up((1.0 - bin_weight) / (bin_weight * precision * precision))


def runs_to_pin_bin_to_strength(bin_weight: float, *, precision: float) -> int:
    """N = p (1 - p) / epsilon^2: a bin of weight p to a fraction epsilon of the
    strength.

    Raises:
        ValueError: bin_weight does not lie strictly between 0 and 1, or precision is
            not above zero.
    """
    _require_between_zero_and_one(bin_weight, "bin_weight")
    _require_positive(precision, "precision")
    return _round_up(bin_weight * (1.0 - bin_weight) / (precision * precision))


@dataclass(frozen=True, kw_only=True)
class RunsForTheSet:
    """The circuits a set of histograms takes, one energy sample from each success."""

    histograms: int  # one per response function and momentum transfer
    successes: int  # the runs each histogram needs, over all of them
    circuits: int  # the successes over the landing probability, rounded up


def runs_for_the_set(
    *,
    response_functions: int,
    momentum_transfers: int,
    runs_per_histogram: int,
    landing_probability: float,
) -> RunsForTheSet:
    """A histogram per response function and momentum transfer, each of so many
    successful runs. A circuit that misses is run again, so on average the set takes
    its successes over the landing probability. A miss is found before the energy
    measurement, so it costs only the rest of the circuit (OneCircuit.set_t).

    Raises:
        ValueError: A count is not above zero, or the landing probability does not lie
            above 0 and at most 1.
    """
    _require_positive(response_functions, "response_functions")
    _require_positive(momentum_transfers, "momentum_transfers")
    _require_positive(runs_per_histogram, "runs_per_histogram")
    if not 0.0 < landing_probability <= 1.0:
        raise ValueError(
            "landing_probability must lie above 0 and at most 1; "
            f"got {landing_probability}."
        )
    histograms = response_functions * momentum_transfers
    successes = histograms * runs_per_histogram
    return RunsForTheSet(
        histograms=histograms,
        successes=successes,
        circuits=_round_up(successes / landing_probability),
    )


def wall_time_seconds(t_per_circuit: int, *, runs: int, t_per_second: float) -> float:
    """runs x T per circuit / T rate, on one machine.

    Raises:
        ValueError: An input is not above zero.
    """
    _require_positive(t_per_circuit, "t_per_circuit")
    _require_positive(runs, "runs")
    _require_positive(t_per_second, "t_per_second")
    return runs * t_per_circuit / t_per_second


@dataclass(frozen=True, kw_only=True)
class LectureCase:
    """A nucleus at the lattice and register sizing the lecture quotes it on."""

    nucleus: Nucleus
    box_side: int
    sizing_mev: float


LECTURE_CASES = {
    "carbon": LectureCase(nucleus=CARBON_12, box_side=4, sizing_mev=10.0),
    "helium": LectureCase(nucleus=HELIUM_4, box_side=3, sizing_mev=5.0),
    "argon": LectureCase(nucleus=ARGON_40, box_side=5, sizing_mev=10.0),
}


def price_strength_band(
    nucleus: Nucleus, *, box_side: int, register_bits: int, current_one_norm: float
) -> tuple[OneCircuit, OneCircuit]:
    """One circuit at each end of the strength band [Z/2, Z], the weak end first. The
    weak end needs more rounds, so it never costs less.

    Raises:
        ValueError: As price_one_circuit raises.
    """
    weakest, strongest = nucleus.strength_band
    weak = price_one_circuit(
        nucleus,
        box_side=box_side,
        register_bits=register_bits,
        strength=weakest,
        current_one_norm=current_one_norm,
    )
    strong = price_one_circuit(
        nucleus,
        box_side=box_side,
        register_bits=register_bits,
        strength=strongest,
        current_one_norm=current_one_norm,
    )
    return weak, strong


def price_lecture_case(case: LectureCase) -> tuple[OneCircuit, OneCircuit]:
    """A case's band as the lecture prices it: the register its sizing gives, and the
    current normalized at the site count M.

    Raises:
        ValueError: As register_bits_for_sizing and price_one_circuit raise.
    """
    lattice = Lattice(box_side=case.box_side)
    bits = register_bits_for_sizing(lattice, case.sizing_mev)
    return price_strength_band(
        case.nucleus,
        box_side=case.box_side,
        register_bits=bits,
        current_one_norm=float(lattice.site_count),
    )


@dataclass(frozen=True, kw_only=True)
class Standing:
    """Where a case stands against the 2033 budget, at the two ends of its band.

    Each share is of the budget: of the whole circuit where the preparation is priced,
    and of all but the preparations where it is not. A priced circuit fits while it is
    within the budget. An unpriced one fits, in the sense that it is still open, only
    while something is left for its preparations. The weak end needs more rounds, so it
    never fits where the strong end does not.
    """

    preparation_priced: bool
    weak_end_share: float
    strong_end_share: float

    @property
    def weak_end_fits(self) -> bool:
        """The weak end fits, or is open."""
        return self._fits(self.weak_end_share)

    @property
    def strong_end_fits(self) -> bool:
        """The strong end fits, or is open."""
        return self._fits(self.strong_end_share)

    def _fits(self, share: float) -> bool:
        if self.preparation_priced:
            return share <= 1.0
        return share < 1.0


def standing(weak: OneCircuit, strong: OneCircuit) -> Standing:
    """Where a band stands against the 2033 budget, with a Toffoli charged as 4 T."""
    return Standing(
        preparation_priced=weak.load is not None,
        weak_end_share=weak.everything_priced_t / HARD_OPERATIONS_2033,
        strong_end_share=strong.everything_priced_t / HARD_OPERATIONS_2033,
    )


def standing_if_one_per_toffoli(weak: OneCircuit, strong: OneCircuit) -> Standing:
    """Where the band stands if the budget counts a Toffoli as one hard operation."""
    weak_counted = weak.one_per_toffoli.everything_priced
    strong_counted = strong.one_per_toffoli.everything_priced
    return Standing(
        preparation_priced=weak.load is not None,
        weak_end_share=weak_counted / HARD_OPERATIONS_2033,
        strong_end_share=strong_counted / HARD_OPERATIONS_2033,
    )
