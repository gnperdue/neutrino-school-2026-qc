"""Price one circuit of the neutrino-response calculation, in the order of hour 2.

With no options it works carbon on the 4^3 lattice, one of the lecture's calculations:

    uv run python scripts/price_one_circuit.py

The exercises' inputs are options. Each command gives you the numbers to reason about;
the reasoning is the exercise.

    Exercise 1, a finer register:    --register-bits 13, then --register-bits 14
    Exercise 2, a bigger box:        --box-side 5
    Exercise 3, price a preparation: the allowance, and the uses of H's block
                                     encoding it pays for, are in the default output
    Exercise 4, argon's register:    --nucleus argon
    Exercise 5, the table:           the amplitudes, and what the translations leave
                                     of them, are in the default output
    The bonus challenge, the current at Z:
                                     --nucleus helium --current-normalized-at-protons

Each nucleus starts from the lecture's case: carbon on 4^3 with 12 energy bits, helium
on 3^3 with 11, argon on 5^3 with 13. With a different --box-side and no
--register-bits, the register follows the case's sizing rule: the fewest bits that
bring lambda / 2^W_q down to the case's sizing energy, which the first line prints.

Every sum is in the tested module response_circuit. This script picks the cases and
the assumptions below, prints the module's numbers, and converts units: a count as a
share of a budget, seconds as days.
"""

import argparse
import os
import sys
from dataclasses import dataclass

from qec_utility_estimation.apps.pionless_first_order.response_circuit import (
    FAULT_BUDGET,
    HARD_OPERATIONS_2028,
    HARD_OPERATIONS_2033,
    LATTICE_SPACING_FM,
    LECTURE_CASES,
    LECTURE_COUPLINGS,
    LOGICAL_ERROR_2028,
    MODES_PER_SITE,
    SYNTHESIS_ERROR_BUDGET,
    TERMS_PER_SITE,
    Lattice,
    LectureCase,
    Nucleus,
    OneCircuit,
    RunsForTheSet,
    Standing,
    amplitudes_after_translations,
    bins_in_window,
    expected_faults,
    gain_at_proton_count,
    ground_state_amplitudes,
    logical_error_rate,
    one_site_one_norm,
    physical_qubits,
    physical_qubits_per_logical,
    position_register_qubits,
    price_lecture_case,
    price_strength_band,
    randomized_synthesis_precision,
    readout_rotations,
    register_bits_for_sizing,
    ross_selinger_t_estimate,
    runs_for_the_set,
    runs_to_pin_bin_to_itself,
    runs_to_pin_bin_to_strength,
    standing,
    standing_if_one_per_toffoli,
    synthesis_budget,
    wall_time_seconds,
)

BAD_INPUT_EXIT_STATUS = 2
BROKEN_PIPE_EXIT_STATUS = 1  # what Python returns when a reader closes the pipe early
TOO_LARGE = (
    "these inputs are too large for floating-point arithmetic; choose a smaller "
    "--box-side."
)

# Our assumptions, stated on the slides. Change one, and every line using it follows.
T_PER_SECOND = 10**6
SLOWER_T_PER_SECOND = 10**5
EARLY_LAYOUT_T_PER_SECOND = 7 * 10**3
NARROWEST_RESPONSE_MEV = 200.0
WIDEST_RESPONSE_MEV = 500.0
PEAK_BIN_WEIGHT = 0.15
PRECISION_OF_THE_BIN = 0.05
PRECISION_OF_THE_STRENGTH = 0.005
RESPONSE_FUNCTIONS = 5
MOMENTUM_TRANSFERS = 5
SYNTHESIS_BUDGET_MOVED_BY = 10.0  # how far the synthesis budget is moved either way

SECONDS_PER_MINUTE = 60.0
SECONDS_PER_HOUR = 3_600.0
SECONDS_PER_DAY = 24.0 * SECONDS_PER_HOUR
SECONDS_PER_YEAR = 365.25 * SECONDS_PER_DAY


@dataclass(frozen=True, kw_only=True)
class Choice:
    """The circuits the command line asks for, and how their register was chosen."""

    weak: OneCircuit
    strong: OneCircuit
    sizing_mev: float | None  # None where --register-bits set the register
    current_at_protons: bool


def t_gates(count: int) -> str:
    """A T count in full, and in the short form the slides use."""
    return f"{count:,} T ({count:.2e})"


def share_of_budget(count: float) -> str:
    """A T count as a share of the 2033 budget's hard operations."""
    percent = 100.0 * count / HARD_OPERATIONS_2033
    return f"{percent:.3g}% of the 2033 budget"


def both_ends(weak_text: str, strong_text: str) -> str:
    """One text if the strength band's two ends agree, and both if they do not."""
    if weak_text == strong_text:
        return weak_text
    return f"{weak_text} at the weak end, {strong_text} at the strong end"


def from_weak_to_strong(weak_text: str | None, strong_text: str | None) -> str:
    """'weak to strong', or where nothing is left. The weak end costs more, so it has
    nothing left wherever the strong end has nothing."""
    if strong_text is None:
        return "nothing is left at either end"
    if weak_text is None:
        return f"nothing at the weak end, {strong_text} at the strong end"
    return f"{weak_text} to {strong_text}"


def t_or_nothing(allowance_t: float) -> str | None:
    """An allowance in T, or None where nothing is left."""
    if allowance_t <= 0.0:
        return None
    return f"{allowance_t:.3g} T"


def uses_or_nothing(uses: float) -> str | None:
    """The uses of H's block encoding an allowance pays for, or None where nothing is
    left."""
    if uses <= 0.0:
        return None
    return f"{uses:,.0f}"


def mev_or_nothing(energy_mev: float | None) -> str | None:
    """An energy in MeV, or None where nothing is left to pay for one."""
    if energy_mev is None:
        return None
    return f"{energy_mev:.3g} MeV"


def one_or_two(low: str, high: str) -> str:
    """'low to high', or one number where the two agree."""
    if low == high:
        return low
    return f"{low} to {high}"


def print_title(title: str) -> None:
    """Open a section of the output."""
    print()
    print(title)


def print_header(choice: Choice) -> None:
    """The case, the register, and the current's normalization."""
    weak = choice.weak
    register = f"{weak.register.bits} energy bits"
    if choice.sizing_mev is not None:
        register = f"{register} (the sizing rule at {choice.sizing_mev:g} MeV)"
    print(
        f"One circuit for {weak.nucleus.name} on the {weak.lattice.box_side}^3 "
        f"lattice, with {register}"
    )
    if choice.current_at_protons:
        print(
            f"The current is normalized at Z = {weak.nucleus.proton_count}, not at the "
            f"{weak.lattice.site_count} sites"
        )


def print_model(lattice: Lattice) -> None:
    """The model: nucleons on a lattice."""
    print_title("The model: leading-order pionless EFT on a periodic cubic lattice")
    print(f"  spacing a: {LATTICE_SPACING_FM} fm")
    print(f"  sites: {lattice.box_side}^3 = {lattice.site_count}")
    print(f"  box: {lattice.box_length_fm:.1f} fm on a side")
    print(f"  hopping t: {LECTURE_COUPLINGS.hopping_scale_mev} MeV")
    print(f"  two-body contact C0: {LECTURE_COUPLINGS.two_body_mev} MeV")
    print(f"  three-body contact D0: {LECTURE_COUPLINGS.three_body_mev} MeV")


def print_register_and_terms(nucleus: Nucleus, lattice: Lattice) -> None:
    """The register, one qubit per mode, and the Hamiltonian's terms."""
    amplitudes = ground_state_amplitudes(nucleus, lattice)
    positions = position_register_qubits(nucleus, lattice)
    print_title("The register: one qubit per mode")
    print(f"  modes per site: {MODES_PER_SITE}")
    print(f"  system qubits: {lattice.system_qubits}")
    print(f"  nucleons: {nucleus.nucleon_count}")
    print(
        "  nucleons per species (p up, p down, n up, n down): "
        f"{nucleus.nucleons_per_species}"
    )
    print(f"  configurations of the nucleus: {amplitudes:.2e} amplitudes")
    print(f"  storing positions instead: {positions} qubits")
    print_title("The terms of H, in Pauli strings")
    print(f"  per site: 24 hopping + 4 Z + 6 ZZ + 4 ZZZ = {TERMS_PER_SITE}")
    print(f"  Gamma, over the lattice: {lattice.term_count:,} terms")


def print_one_norm(lattice: Lattice) -> None:
    """Lambda, site by site."""
    pieces = one_site_one_norm()
    print_title("lambda: the size of the Hamiltonian")
    print(f"  24 hopping strings: {pieces.hopping_mev:.1f} MeV")
    print(f"  4 single Z strings: {pieces.single_z_mev:.1f} MeV")
    print(f"  6 ZZ strings: {pieces.pair_zz_mev:.1f} MeV")
    print(f"  4 ZZZ strings: {pieces.triple_zzz_mev:.1f} MeV")
    print(f"  the 14 on-site strings together: {pieces.on_site_mev:.1f} MeV")
    print(f"  one site: {pieces.total_mev:.1f} MeV")
    print(f"  lambda, over {lattice.site_count} sites: {lattice.one_norm_mev:,.0f} MeV")


def print_walk_step(circuit: OneCircuit) -> None:
    """PREPARE, SELECT and REFLECT, priced for one step."""
    step = circuit.step
    print_title(f"One walk step, at {circuit.synthesis.t_per_rotation} T per rotation")
    print(f"  keep number: {step.keep_bits} bits")
    print(
        f"  PREPARE and its inverse: 2 x {step.prepare_toffolis} Toffolis and "
        f"2 x {step.prepare_rotations} rotations = {circuit.prepare_pair_t:,} T"
    )
    print(
        f"  SELECT: {step.select_toffolis:,} Toffolis = {circuit.select_t:,} T, "
        f"{100.0 * circuit.select_share:.0f}% of the step"
    )
    print(
        f"  REFLECT: {step.ancilla_qubits} ancilla qubits, "
        f"{step.reflect_toffolis:,} ANDs = {circuit.reflect_t:,} T"
    )
    print(f"  one step: {circuit.walk_step_t:,} T")


def print_phase_estimation(circuit: OneCircuit) -> None:
    """What the energy register buys."""
    register = circuit.register
    print_title("Phase estimation")
    print(f"  energy register: {register.bits} bits")
    print(f"  walk steps: {register.walk_steps:,}")
    print(f"  bin width at mid-band: {register.bin_width_mev:.1f} MeV")
    print(f"  delivered error: {register.rms_error_mev:.1f} MeV RMS")


def print_energy_measurement(weak: OneCircuit, strong: OneCircuit) -> None:
    """The walk and the readout, and their shares of the budget."""
    readout = readout_rotations(weak.register.bits)
    print_title("The energy measurement")
    print(f"  the walk: {t_gates(weak.walk_t)}, {share_of_budget(weak.walk_t)}")
    print(
        f"  the readout: at most {readout.fourier} rotations in the inverse Fourier "
        f"transform and {readout.window:,} in the window state"
    )
    print(f"    = {t_gates(weak.readout_t)}, {share_of_budget(weak.readout_t)}")
    weak_total = (
        f"{t_gates(weak.energy_measurement_t)}, "
        f"{share_of_budget(weak.energy_measurement_t)}"
    )
    strong_total = (
        f"{t_gates(strong.energy_measurement_t)}, "
        f"{share_of_budget(strong.energy_measurement_t)}"
    )
    print(f"  the energy measurement: {both_ends(weak_total, strong_total)}")


def print_rotations(weak: OneCircuit, strong: OneCircuit) -> None:
    """The rotations, the shared budget, and the randomized alternative."""
    count = weak.synthesis.rotation_count
    precision = weak.synthesis.precision
    moved = SYNTHESIS_BUDGET_MOVED_BY
    looser = synthesis_budget(count, error_budget=SYNTHESIS_ERROR_BUDGET * moved)
    tighter = synthesis_budget(count, error_budget=SYNTHESIS_ERROR_BUDGET / moved)
    randomized = randomized_synthesis_precision(count)
    budget = SYNTHESIS_ERROR_BUDGET
    print_title(f"Rotations: one error budget of {budget:g} per circuit")
    weak_count = f"{count:,}"
    strong_count = f"{strong.synthesis.rotation_count:,}"
    print(
        "  rotations and rounded angles in what is priced: "
        f"{both_ends(weak_count, strong_count)}"
    )
    if weak.load is None:
        print("    an unpriced preparation's own rotations would come on top")
    weak_precision = f"{precision:.2e}"
    strong_precision = f"{strong.synthesis.precision:.2e}"
    print(f"  precision per rotation: {both_ends(weak_precision, strong_precision)}")
    weak_price = f"{weak.synthesis.t_per_rotation} T"
    strong_price = f"{strong.synthesis.t_per_rotation} T"
    print(f"  price per rotation: {both_ends(weak_price, strong_price)}")
    print(
        f"  a budget {moved:g} times looser: {looser.t_per_rotation} T; "
        f"{moved:g} times tighter: {tighter.t_per_rotation} T"
    )
    share = 100.0 * weak.rotation_share
    print(f"  rotations' share of everything priced: {share:.2g}%")
    print(
        "  randomized compiling (Campbell 2017, Thm. 2): "
        f"precision {randomized:.2e} per rotation"
    )
    print(
        "  Ross-Selinger, the deterministic method that theorem covers: "
        f"{ross_selinger_t_estimate(randomized):.0f} T there, "
        f"{ross_selinger_t_estimate(precision):.0f} T at the coherent precision"
    )


def print_helium_load_for_scale() -> None:
    """Helium's load, the one priced in the lecture."""
    case = LECTURE_CASES["helium"]
    helium, _ = price_lecture_case(case)
    load = helium.load
    print(
        f"  for scale, helium on {case.box_side}^3: {load.amplitudes:,} amplitudes, "
        f"{load.load_toffolis:,} Toffolis to load, and {load.spread_toffolis} "
        f"Toffolis and {load.spread_rotations} rotations to spread"
    )
    print(f"    = {t_gates(helium.load_t)} per load")


def print_preparation(weak: OneCircuit, strong: OneCircuit) -> None:
    """The ground state: priced where a table exists, and otherwise a blank."""
    print_title("The ground state")
    load = weak.load
    if load is None:
        amplitudes = ground_state_amplitudes(weak.nucleus, weak.lattice)
        after = amplitudes_after_translations(weak.nucleus, weak.lattice)
        print(f"  {weak.nucleus.name} has no species of one nucleon: no table to load")
        print(
            f"  amplitudes: {amplitudes:.2e}; the translations would leave about "
            f"{after:.2e}"
        )
        print("  the preparation is unpriced")
        print_helium_load_for_scale()
        return
    gradient = load.gradient_state
    print(f"  table: {load.amplitudes:,} amplitudes")
    print(
        f"  load: {load.load_toffolis:,} Toffolis, {load.ancilla_qubits} ancilla qubits"
    )
    print(
        f"  spread: {load.spread_toffolis:,} Toffolis, "
        f"{load.spread_rotations} rotations"
    )
    print(f"  one load: {t_gates(weak.load_t)}")
    print(
        f"  the gradient state, made once: {gradient.rotations} rotations and "
        f"{gradient.t_gates} T gate = {weak.gradient_state_t:,} T"
    )
    weak_loads = t_gates(weak.all_loads_t)
    strong_loads = t_gates(strong.all_loads_t)
    print(f"  all the loads: {both_ends(weak_loads, strong_loads)}")
    print(
        f"  their share of the circuit: {100.0 * weak.loads_share:.0f}% to "
        f"{100.0 * strong.loads_share:.0f}%"
    )
    print(
        f"  the energy measurement's: {100.0 * weak.energy_measurement_share:.0f}% to "
        f"{100.0 * strong.energy_measurement_share:.0f}%"
    )


def print_amplification(weak: OneCircuit, strong: OneCircuit) -> None:
    """The insertion, the strength band, the rounds and the preparations."""
    print_title("The current and its amplification")
    print(
        f"  the insertion: {weak.current.terms:,} terms, "
        f"{weak.current.index_qubits} index qubits, "
        f"{weak.current.toffolis:,} Toffolis, {weak.current.rotations:,} rotations, "
        f"{weak.current.t_gates} T gates"
    )
    weak_insertion = f"{weak.insertion_t:,} T"
    strong_insertion = f"{strong.insertion_t:,} T"
    print(f"  one insertion: {both_ends(weak_insertion, strong_insertion)}")
    weakest, strongest = weak.nucleus.strength_band
    print(f"  strength band: {weakest} to {strongest}")
    print(
        f"  P_s: {weak.schedule.success_probability:.2e} to "
        f"{strong.schedule.success_probability:.2e}"
    )
    print(f"  rounds: {weak.schedule.rounds} to {strong.schedule.rounds}")
    print(
        f"  preparations: {weak.schedule.preparations} to "
        f"{strong.schedule.preparations}"
    )
    print(
        f"  landing probability: {weak.schedule.landing_probability:.6f} to "
        f"{strong.schedule.landing_probability:.6f}"
    )
    print(
        "  preparations per success, as a circuit that misses runs again: "
        f"{weak.schedule.preparations_per_success:.2f} to "
        f"{strong.schedule.preparations_per_success:.2f}"
    )
    print(
        f"  one round's reflections: {weak.reflections.index_chain_ands:,} + "
        f"{weak.reflections.start_chain_ands:,} ANDs = {weak.round_reflections_t:,} T"
    )


def print_everything_priced(circuit: OneCircuit) -> None:
    """Every priced piece of the costliest circuit, added up."""
    preparations = circuit.schedule.preparations
    rounds = circuit.schedule.rounds
    print_title("Everything priced, at the weak end of the strength band")
    print(
        "  the energy measurement: "
        f"{t_gates(circuit.energy_measurement_t)}, "
        f"{share_of_budget(circuit.energy_measurement_t)}"
    )
    print(
        f"  insertions, {circuit.insertion_t:,} T x {preparations}: "
        f"{t_gates(circuit.insertions_t)}, {share_of_budget(circuit.insertions_t)}"
    )
    print(
        f"  reflections, {circuit.round_reflections_t:,} T x {rounds}: "
        f"{t_gates(circuit.reflections_t)}, {share_of_budget(circuit.reflections_t)}"
    )
    if circuit.load is not None:
        print(
            f"  the loads, {circuit.load_t:,} T x {preparations}: "
            f"{t_gates(circuit.all_loads_t)}, {share_of_budget(circuit.all_loads_t)}"
        )
        print(
            f"  the gradient state, made once: {t_gates(circuit.gradient_state_t)}, "
            f"{share_of_budget(circuit.gradient_state_t)}"
        )
    print(
        f"  everything priced: {t_gates(circuit.everything_priced_t)}, "
        f"{share_of_budget(circuit.everything_priced_t)}"
    )
    if circuit.load is None:
        print("  the preparations: unpriced")


def print_allowance(weak: OneCircuit, strong: OneCircuit) -> None:
    """What each preparation may cost for the circuit to fit the 2033 budget."""
    weak_counted = weak.one_per_toffoli
    strong_counted = strong.one_per_toffoli
    print_title("The allowance: what each preparation may cost")
    if weak.left_over_t > 0:
        print(f"  the budget left over: {weak.left_over_t:.3g} T, at the weak end")
    else:
        times = weak.all_but_preparations_t / HARD_OPERATIONS_2033
        print(
            "  the budget left over: nothing at the weak end, where all but the "
            f"preparations is {times:.3g} times the budget"
        )
    allowance = from_weak_to_strong(
        t_or_nothing(weak.allowance_t), t_or_nothing(strong.allowance_t)
    )
    print(f"  per preparation: {allowance}")
    uses = from_weak_to_strong(
        uses_or_nothing(weak.allowance_in_walk_steps),
        uses_or_nothing(strong.allowance_in_walk_steps),
    )
    print(
        f"  uses of H's block encoding it pays for, {weak.walk_step_t:,} T each: {uses}"
    )
    energies = from_weak_to_strong(
        mev_or_nothing(weak.rule_of_thumb_mev), mev_or_nothing(strong.rule_of_thumb_mev)
    )
    print(f"  lambda over those uses: {energies}")
    print("  if the budget counts a Toffoli as one hard operation instead of 4 T:")
    counted_uses = from_weak_to_strong(
        uses_or_nothing(weak_counted.allowance_in_walk_steps),
        uses_or_nothing(strong_counted.allowance_in_walk_steps),
    )
    print(f"    uses of H's block encoding it pays for: {counted_uses}")
    counted_energies = from_weak_to_strong(
        mev_or_nothing(weak_counted.rule_of_thumb_mev),
        mev_or_nothing(strong_counted.rule_of_thumb_mev),
    )
    print(f"    lambda over those uses: {counted_energies}")


def print_lever(choice: Choice) -> None:
    """The same circuit if the current could be normalized at Z."""
    weak = choice.weak
    nucleus = weak.nucleus
    print_title("The lever nobody has priced: the current normalized at Z")
    if choice.current_at_protons:
        print("  this run already normalizes the current at Z")
        return
    weak_at_z, strong_at_z = price_strength_band(
        nucleus,
        box_side=weak.lattice.box_side,
        register_bits=weak.register.bits,
        current_one_norm=float(nucleus.proton_count),
    )
    gain = gain_at_proton_count(nucleus, weak.lattice)
    print(f"  P_s rises by (M / Z)^2 = {gain:.3g}")
    print(f"  rounds: {weak_at_z.schedule.rounds} to {strong_at_z.schedule.rounds}")
    print(
        f"  preparations: {weak_at_z.schedule.preparations} to "
        f"{strong_at_z.schedule.preparations}"
    )
    allowance = from_weak_to_strong(
        t_or_nothing(weak_at_z.allowance_t), t_or_nothing(strong_at_z.allowance_t)
    )
    print(f"  allowance per preparation: {allowance}")


def print_error_budget(circuit: OneCircuit) -> None:
    """The logical error rate, the code distance and the physical qubits."""
    priced = circuit.everything_priced_t
    on_priced = logical_error_rate(priced)
    on_budget = logical_error_rate(HARD_OPERATIONS_2033)
    distance = circuit.needed_code_distance
    peak = circuit.qubits.peak
    rate_2028 = LOGICAL_ERROR_2028
    faults_priced = expected_faults(priced, logical_error=rate_2028)
    faults_full = expected_faults(HARD_OPERATIONS_2033, logical_error=rate_2028)
    print_title(
        f"The error budget: {FAULT_BUDGET:g} expected logical faults per circuit"
    )
    print(f"  logical error rate, on the priced parts: {on_priced:.2e} per operation")
    print(f"  if a circuit uses the whole budget: {on_budget:.2e} per operation")
    print(
        "  code distance at the lower of the two, ten times below threshold: "
        f"{distance}"
    )
    print(
        "  physical qubits per logical one, 2 d^2: "
        f"{physical_qubits_per_logical(distance)}"
    )
    print(
        f"  for the {peak:,}-qubit stage: "
        f"{physical_qubits(peak, distance=distance):.2e} physical qubits"
    )
    print(
        f"  at the 2028 figure of {LOGICAL_ERROR_2028:.0e} per operation: "
        f"{faults_priced:.2g} expected faults on the priced parts, "
        f"{faults_full:.2g} if a circuit uses the whole budget"
    )


def print_qubits(circuit: OneCircuit) -> None:
    """The register by stage."""
    qubits = circuit.qubits
    print_title("The qubit count, by stage")
    print(f"  system: {qubits.system:,}")
    print("  during the walk:")
    print(
        f"    energy register {qubits.energy_register}, "
        f"ancillas {qubits.walk_ancillas}, "
        f"AND chain {qubits.walk_chain}: {qubits.walk_added} added"
    )
    print(f"    stage total: {qubits.walk_stage:,}")
    print("  during the amplification:")
    print(f"    index: {qubits.insertion_index}")
    if circuit.load is None:
        print(
            f"    AND chain over the {qubits.reflected_qubits:,} system and index "
            f"qubits, one of them the reflection's target: "
            f"{qubits.amplification_chain:,}"
        )
    else:
        print(f"    the load's enumeration register: {qubits.load_enumeration}")
        print(f"    the load's gradient state: {qubits.load_gradient}")
        print(
            f"    AND chain over the {qubits.reflected_qubits:,} system, index and "
            f"enumeration qubits, one of them the reflection's target: "
            f"{qubits.amplification_chain:,}"
        )
        print(
            f"    the spread's shift register, held, not clean there: "
            f"{qubits.load_shift}"
        )
        print(
            f"    the load's other {qubits.load_other_ancillas} ancillas take turns "
            "with that chain"
        )
    print(f"    {qubits.amplification_added:,} added")
    print(f"    stage total: {qubits.amplification_stage:,}")
    if circuit.load is not None:
        print(f"  peak: {qubits.peak:,}")
        return
    print(
        f"  peak: {qubits.peak:,}, plus any of the preparation's ancillas beyond the "
        f"chain's {qubits.amplification_chain:,}"
    )
    if qubits.free_for_preparation == 0:
        print(
            "  free for a preparation in the budget: none, since what the reflection "
            "holds and its AND chain already pass its 1,000 qubits"
        )
        return
    print(
        "  free for a preparation in the budget: "
        f"{qubits.free_for_preparation} if its ancillas are clean at every "
        f"reflection, {qubits.free_if_reflected_over} if the reflection must "
        "cover them"
    )


def the_set(circuit: OneCircuit, runs_per_histogram: int) -> RunsForTheSet:
    """The lecture's set of histograms, at this circuit's landing probability."""
    return runs_for_the_set(
        response_functions=RESPONSE_FUNCTIONS,
        momentum_transfers=MOMENTUM_TRANSFERS,
        runs_per_histogram=runs_per_histogram,
        landing_probability=circuit.schedule.landing_probability,
    )


def runs_per_peak_bin() -> tuple[int, int]:
    """The runs to pin a peak bin to a fraction of itself, and of the strength."""
    to_itself = runs_to_pin_bin_to_itself(
        PEAK_BIN_WEIGHT, precision=PRECISION_OF_THE_BIN
    )
    to_strength = runs_to_pin_bin_to_strength(
        PEAK_BIN_WEIGHT, precision=PRECISION_OF_THE_STRENGTH
    )
    return to_itself, to_strength


def print_runs(weak: OneCircuit, strong: OneCircuit) -> None:
    """One run, one bin: the runs to pin a peak bin, and for the whole set."""
    bin_width = weak.register.bin_width_mev
    fewest_bins = bins_in_window(NARROWEST_RESPONSE_MEV, bin_width_mev=bin_width)
    most_bins = bins_in_window(WIDEST_RESPONSE_MEV, bin_width_mev=bin_width)
    to_itself, to_strength = runs_per_peak_bin()
    fewest = the_set(weak, to_itself)
    most = the_set(weak, to_strength)
    weak_circuits = f"{fewest.circuits:,} to {most.circuits:,}"
    strong_circuits = (
        f"{the_set(strong, to_itself).circuits:,} to "
        f"{the_set(strong, to_strength).circuits:,}"
    )
    print_title("How many runs: one run, one bin")
    print(
        f"  a response {NARROWEST_RESPONSE_MEV:g} to {WIDEST_RESPONSE_MEV:g} MeV wide "
        f"fills {fewest_bins} to {most_bins} bins"
    )
    print(
        f"  a peak bin of weight {PEAK_BIN_WEIGHT:g}, as on the default register's "
        "bins, to "
        f"{100.0 * PRECISION_OF_THE_BIN:g}% of itself: {to_itself:,} runs"
    )
    print(
        f"  the same bin, to {100.0 * PRECISION_OF_THE_STRENGTH:g}% of the strength: "
        f"{to_strength:,} runs"
    )
    print(
        f"  {fewest.histograms} histograms: {fewest.successes:,} to "
        f"{most.successes:,} successful circuits"
    )
    print(
        "  circuits run, as one that misses runs again: "
        f"{both_ends(weak_circuits, strong_circuits)}"
    )


def set_seconds(circuit: OneCircuit, t_per_circuit: int) -> tuple[float, float]:
    """The whole set's wall time, for the fewest runs and for the most. A circuit
    that misses runs only the rest again, not its energy measurement."""
    to_itself, to_strength = runs_per_peak_bin()
    fewest = circuit.set_t(
        t_per_circuit, successes=the_set(circuit, to_itself).successes
    )
    most = circuit.set_t(
        t_per_circuit, successes=the_set(circuit, to_strength).successes
    )
    return fewest / T_PER_SECOND, most / T_PER_SECOND


def set_years_in_the_budget(circuit: OneCircuit, fits: bool) -> str:
    """The whole set's wall time if each circuit uses the whole budget, in years,
    where this end of the band fits in it."""
    if not fits:
        return "does not apply, since the priced parts pass the budget"
    return in_years(set_seconds(circuit, HARD_OPERATIONS_2033))


def in_days(seconds: tuple[float, float]) -> str:
    """A range of seconds, in days."""
    fewest, most = seconds
    return f"{fewest / SECONDS_PER_DAY:,.0f} to {most / SECONDS_PER_DAY:,.0f} days"


def in_years(seconds: tuple[float, float]) -> str:
    """A range of seconds, in years."""
    fewest, most = seconds
    return f"{fewest / SECONDS_PER_YEAR:.1f} to {most / SECONDS_PER_YEAR:.1f} years"


def print_wall_times(weak: OneCircuit, strong: OneCircuit) -> None:
    """T gates over the T rate, on one machine."""
    budget = HARD_OPERATIONS_2033
    priced_s = wall_time_seconds(
        weak.everything_priced_t, runs=1, t_per_second=T_PER_SECOND
    )
    budget_s = wall_time_seconds(budget, runs=1, t_per_second=T_PER_SECOND)
    slower_s = wall_time_seconds(budget, runs=1, t_per_second=SLOWER_T_PER_SECOND)
    early_s = wall_time_seconds(budget, runs=1, t_per_second=EARLY_LAYOUT_T_PER_SECOND)
    weak_priced = in_days(set_seconds(weak, weak.everything_priced_t))
    strong_priced = in_days(set_seconds(strong, strong.everything_priced_t))
    band = standing(weak, strong)
    weak_budget = set_years_in_the_budget(weak, band.weak_end_fits)
    strong_budget = set_years_in_the_budget(strong, band.strong_end_fits)
    print_title(f"Wall time, at {T_PER_SECOND:,} T per second on one machine")
    print(f"  one circuit, the priced parts: {priced_s:,.0f} s")
    print(
        f"  one circuit, the whole budget: {budget_s / SECONDS_PER_MINUTE:.0f} minutes"
    )
    print(f"  the whole set, the priced parts: {both_ends(weak_priced, strong_priced)}")
    print(f"  the whole set, the whole budget: {both_ends(weak_budget, strong_budget)}")
    print(
        "  one circuit, the whole budget, at slower rates: "
        f"{slower_s / SECONDS_PER_HOUR:.1f} hours at {SLOWER_T_PER_SECOND:,} T per "
        f"second, {early_s / SECONDS_PER_HOUR:.0f} hours at "
        f"{EARLY_LAYOUT_T_PER_SECOND:,}"
    )


def standing_words(where: Standing) -> str:
    """A standing in words that say no more than it checks."""
    weak = where.weak_end_share
    strong = where.strong_end_share
    times = one_or_two(f"{strong:.1f}", f"{weak:.1f}")
    if where.preparation_priced:
        if where.weak_end_fits:
            return f"fits: {100.0 * strong:.1f}% to {100.0 * weak:.1f}% of the budget"
        if where.strong_end_fits:
            return (
                f"fits at the strong end only, at {100.0 * strong:.1f}% of the "
                f"budget; {100.0 * weak:.1f}% of it at the weak end"
            )
        return f"over: {times} times the budget"
    if where.weak_end_fits:
        return "open: fits if a preparation meets its allowance"
    if where.strong_end_fits:
        return (
            "open at the strong end only; the weak end is over before its preparations"
        )
    return f"over before any preparation: {times} times the budget"


def print_verdict_row(name: str, weak: OneCircuit, strong: OneCircuit) -> None:
    """One row of the verdict, for one case."""
    weak_rate = f"{logical_error_rate(weak.everything_priced_t):.2e}"
    strong_rate = f"{logical_error_rate(strong.everything_priced_t):.2e}"
    counted = standing_words(standing_if_one_per_toffoli(weak, strong))
    counted_measurement = share_of_budget(weak.one_per_toffoli.energy_measurement)
    print(f"  {name}, {weak.register.rms_error_mev:.1f} MeV RMS:")
    print(
        f"    energy measurement: {weak.energy_measurement_t:.2e} T, "
        f"{share_of_budget(weak.energy_measurement_t)}"
    )
    print(
        f"    preparations: {strong.schedule.preparations} to "
        f"{weak.schedule.preparations}"
    )
    if weak.load_t is None:
        allowance = from_weak_to_strong(
            t_or_nothing(weak.allowance_t), t_or_nothing(strong.allowance_t)
        )
        print(f"    unpriced; allowance per preparation: {allowance}")
        print(f"    peak qubits: {weak.qubits.peak:,}, plus the preparation's")
    else:
        print(
            f"    loads of {weak.load_t:.2e} T: {strong.all_loads_t:.2e} to "
            f"{weak.all_loads_t:.2e} T"
        )
        print(f"    peak qubits: {weak.qubits.peak:,}")
    print(
        "    logical error rate per operation, on the priced parts: "
        f"{both_ends(weak_rate, strong_rate)}"
    )
    print(f"    standing: {standing_words(standing(weak, strong))}")
    print(
        f"    if a Toffoli counts as one hard operation: {counted}; the energy "
        f"measurement {counted_measurement}"
    )


def print_verdict(choice: Choice) -> None:
    """The lecture's three rows, then the case asked for if it is not one of them."""
    weak = choice.weak
    carbon = LECTURE_CASES["carbon"]
    helium = LECTURE_CASES["helium"]
    next_helium = LectureCase(
        nucleus=helium.nucleus, box_side=4, sizing_mev=helium.sizing_mev
    )
    carbon_weak, carbon_strong = price_lecture_case(carbon)
    helium_weak, helium_strong = price_lecture_case(helium)
    next_weak, next_strong = price_lecture_case(next_helium)
    print_title("The verdict against the 2033 budget")
    for case, case_weak, case_strong in (
        (carbon, carbon_weak, carbon_strong),
        (helium, helium_weak, helium_strong),
        (next_helium, next_weak, next_strong),
    ):
        name = f"{case.nucleus.name}, {case.box_side}^3"
        print_verdict_row(name, case_weak, case_strong)
    if weak not in (carbon_weak, helium_weak, next_weak):
        name = f"the case you asked for, {weak.nucleus.name}, {weak.lattice.box_side}^3"
        print_verdict_row(name, weak, choice.strong)


def print_2028_comparison() -> None:
    """Nothing in this calculation fits the 2028 budget."""
    carbon, _ = price_lecture_case(LECTURE_CASES["carbon"])
    helium, _ = price_lecture_case(LECTURE_CASES["helium"])
    carbon_times = carbon.energy_measurement_t / HARD_OPERATIONS_2028
    helium_times = helium.energy_measurement_t / HARD_OPERATIONS_2028
    print_title(f"Against the 2028 budget of {HARD_OPERATIONS_2028:,} hard operations")
    print(f"  carbon's energy measurement alone: {carbon_times:.0f} times the budget")
    print(f"  helium's energy measurement alone: {helium_times:.0f} times the budget")


def parse_options() -> argparse.Namespace:
    """Read the command line; argparse stops on an unknown nucleus or a non-integer."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--nucleus",
        choices=("carbon", "helium", "argon"),
        default="carbon",
        help="the nucleus (default: carbon)",
    )
    parser.add_argument(
        "--box-side",
        type=int,
        default=None,
        help="sites per axis of the lattice (default: the nucleus's lecture case)",
    )
    parser.add_argument(
        "--register-bits",
        type=int,
        default=None,
        help="bits of the energy register (default: the case's sizing rule)",
    )
    parser.add_argument(
        "--current-normalized-at-protons",
        action="store_true",
        help="normalize the current at Z rather than at the site count M",
    )
    return parser.parse_args()


def choose_circuits(options: argparse.Namespace) -> Choice:
    """Fill in the lecture's case for anything the command line left out."""
    case = LECTURE_CASES[options.nucleus]
    box_side = options.box_side
    if box_side is None:
        box_side = case.box_side
    lattice = Lattice(box_side=box_side)
    register_bits = options.register_bits
    sizing_mev = None
    if register_bits is None:
        sizing_mev = case.sizing_mev
        register_bits = register_bits_for_sizing(lattice, sizing_mev)
    current_one_norm = float(lattice.site_count)
    if options.current_normalized_at_protons:
        current_one_norm = float(case.nucleus.proton_count)
    weak, strong = price_strength_band(
        case.nucleus,
        box_side=box_side,
        register_bits=register_bits,
        current_one_norm=current_one_norm,
    )
    return Choice(
        weak=weak,
        strong=strong,
        sizing_mev=sizing_mev,
        current_at_protons=options.current_normalized_at_protons,
    )


def print_the_calculation(choice: Choice) -> None:
    """Every section, in the order of hour 2's slides."""
    weak = choice.weak
    strong = choice.strong
    print_header(choice)
    print_model(weak.lattice)
    print_register_and_terms(weak.nucleus, weak.lattice)
    print_one_norm(weak.lattice)
    print_walk_step(weak)
    print_phase_estimation(weak)
    print_energy_measurement(weak, strong)
    print_rotations(weak, strong)
    print_preparation(weak, strong)
    print_amplification(weak, strong)
    print_everything_priced(weak)
    print_allowance(weak, strong)
    print_lever(choice)
    print_error_budget(weak)
    print_qubits(weak)
    print_runs(weak, strong)
    print_wall_times(weak, strong)
    print_verdict(choice)
    print_2028_comparison()


def refuse(message: str) -> int:
    """Say what is wrong with the inputs, and return the status for bad input."""
    print(f"price_one_circuit: {message}", file=sys.stderr)
    return BAD_INPUT_EXIT_STATUS


def main() -> int:
    """Price the circuit the options name, or say what is wrong with them.

    Only the pricing of the circuit asked for can refuse its inputs, so a ValueError
    anywhere else is a fault in this code, and shows as one. Large inputs can overflow
    while the output prints, so that check covers both. A reader that closes the pipe
    early, as head does, ends the run quietly, as Python's documentation advises.
    """
    options = parse_options()
    try:
        choice = choose_circuits(options)
    except ValueError as error:
        return refuse(str(error))
    except OverflowError:
        return refuse(TOO_LARGE)
    try:
        print_the_calculation(choice)
        sys.stdout.flush()
    except OverflowError:
        return refuse(TOO_LARGE)
    except BrokenPipeError:
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, sys.stdout.fileno())
        return BROKEN_PIPE_EXIT_STATUS
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
