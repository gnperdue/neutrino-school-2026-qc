"""Tests for the cost of one circuit of the neutrino-response calculation.

They pin the numbers the lecture quotes, so each one reads as a worked example:
take a nucleus, a lattice and a register, and check that the pieces add up. The last
few run the script scripts/price_one_circuit.py itself, the way you would, and check
only that it finishes and names its case, or names the input it refuses.
"""

import cmath
import math
import subprocess
import sys
from pathlib import Path

import pytest

from qec_utility_estimation.apps.pionless_first_order.response_circuit import (
    ARGON_40,
    CARBON_12,
    HARD_OPERATIONS_2028,
    HARD_OPERATIONS_2033,
    HELIUM_4,
    LECTURE_CASES,
    LECTURE_COUPLINGS,
    LOGICAL_ERROR_2028,
    Lattice,
    Nucleus,
    PhaseGates,
    ReadoutRotations,
    Standing,
    amplification,
    amplitudes_after_translations,
    bins_in_window,
    code_distance,
    energy_register,
    expected_faults,
    gain_at_proton_count,
    gradient_state_gates,
    ground_state_amplitudes,
    ground_state_load,
    insertion,
    load_coherent_errors,
    logical_error_rate,
    one_site_one_norm,
    phase_gates_per_axis,
    physical_qubits,
    physical_qubits_per_logical,
    pinned_table_amplitudes,
    position_register_qubits,
    price_lecture_case,
    price_one_circuit,
    price_strength_band,
    randomized_synthesis_precision,
    readout_rotations,
    register_bits_for_sizing,
    ross_selinger_t_estimate,
    round_reflections,
    runs_for_the_set,
    runs_to_pin_bin_to_itself,
    runs_to_pin_bin_to_strength,
    standing,
    standing_if_one_per_toffoli,
    synthesis_budget,
    walk_step,
    wall_time_seconds,
)


def test_carbon_circuit_prices_every_piece_the_lecture_quotes() -> None:
    """Carbon on the 4^3 lattice, at 12 register bits and strength 3.

    One walk step is a PREPARE pair, a SELECT and a REFLECT, and the SELECT is 92% of
    it. The walk repeats the step 2^11 times, and the readout's inverse Fourier
    transform and window state complete the energy measurement. Then 59 insertions and
    29 rounds of reflections complete what can be priced. Every rotation priced shares
    one error budget, which sets its 34 T.
    """
    circuit = price_one_circuit(
        CARBON_12, box_side=4, register_bits=12, strength=3.0, current_one_norm=64.0
    )

    assert circuit.register.walk_steps == 2_048

    assert circuit.prepare_pair_t == 656
    assert circuit.select_t == 9_724
    assert circuit.reflect_t == 168
    assert circuit.walk_step_t == 656 + 9_724 + 168
    assert circuit.walk_step_t == 10_548
    assert circuit.select_share == pytest.approx(9_724 / 10_548)
    assert circuit.select_share == pytest.approx(0.92, abs=0.005)

    assert circuit.walk_t == 21_602_304
    assert readout_rotations(12) == ReadoutRotations(fourier=12 * 11, window=2**12)
    assert circuit.readout_t == (132 + 4_096) * 34
    assert circuit.readout_t == 143_752
    assert circuit.energy_measurement_t == 21_746_056

    assert circuit.schedule.rounds == 29
    assert circuit.schedule.preparations == 59
    assert circuit.insertion_t == 508
    assert circuit.round_reflections_t == 1_064

    assert circuit.synthesis.rotation_count == 2_048 * 2 * 2 + 132 + 4_096
    assert circuit.synthesis.rotation_count == 12_420
    assert circuit.synthesis.t_per_rotation == 34

    assert circuit.all_but_preparations_t == 21_806_884


def test_carbon_has_no_table_so_its_preparation_is_unpriced() -> None:
    """Carbon has three nucleons of every species, so none can be pinned and erased.

    With no table to load, its preparation has no price. That must read as unknown,
    never as a cost of zero, so the circuit's total is unknown too.
    """
    circuit = price_one_circuit(
        CARBON_12, box_side=4, register_bits=12, strength=3.0, current_one_norm=64.0
    )

    assert circuit.load is None
    assert circuit.load_t is None
    assert circuit.all_loads_t is None
    assert circuit.circuit_t is None
    assert circuit.loads_share is None


def test_carbon_strength_band_gives_the_allowance_and_the_steps_it_buys() -> None:
    """What each preparation may cost for the circuit to fit the 2033 budget.

    Take the budget of 10^9, subtract everything priced, and share what is left over
    the preparations. A stronger response needs fewer preparations, so each may cost
    more. That is 17 to 24 million T each, or 1,572 to 2,262 walk steps. The insertion
    has no rotations on this lattice, so the strength does not move the rotation count.
    """
    weak = price_one_circuit(
        CARBON_12, box_side=4, register_bits=12, strength=3.0, current_one_norm=64.0
    )
    assert weak.schedule.rounds == 29
    assert weak.schedule.preparations == 59
    assert weak.all_but_preparations_t == 21_806_884
    assert weak.left_over_t == 10**9 - 21_806_884
    assert weak.allowance_t == pytest.approx(weak.left_over_t / 59)
    assert weak.allowance_t == pytest.approx(16_579_544, abs=0.5)
    assert weak.allowance_in_walk_steps == pytest.approx(1_572, abs=0.5)

    strong = price_one_circuit(
        CARBON_12, box_side=4, register_bits=12, strength=6.0, current_one_norm=64.0
    )
    assert strong.schedule.rounds == 20
    assert strong.schedule.preparations == 41
    assert strong.all_but_preparations_t == 21_788_164
    assert strong.allowance_t == pytest.approx(23_858_825, abs=0.5)
    assert strong.allowance_in_walk_steps == pytest.approx(2_262, abs=0.5)
    assert strong.synthesis.rotation_count == weak.synthesis.rotation_count


def test_helium_fits_the_budget_with_its_load_the_biggest_cost() -> None:
    """Helium on the 3^3 lattice is the one nucleus priced in full.

    Each of its 43 or 29 loads costs about 2.5 million T, so the loads dwarf the 5.3
    million of the energy measurement, and the circuit still fits under 10^9. Each
    preparation adds its load's 21 rounded angles and rotations and its insertion's 18
    rotations to the shared budget, and the gradient state adds its 19 once.
    """
    weak = price_one_circuit(
        HELIUM_4, box_side=3, register_bits=11, strength=1.0, current_one_norm=27.0
    )
    assert weak.synthesis.rotation_count == 10_350 + 43 * (21 + 18) + 19
    assert weak.synthesis.rotation_count == 12_046
    assert weak.synthesis.t_per_rotation == 34
    assert weak.synthesis.angle_bits == 22
    assert weak.energy_measurement_t == 5_316_252
    assert weak.load_t == 2_548_596
    assert weak.schedule.preparations == 43
    assert weak.all_loads_t == 43 * 2_548_596
    assert weak.circuit_t == 114_956_612 + 647
    assert weak.loads_share == pytest.approx(43 * 2_548_596 / weak.circuit_t)
    assert weak.loads_share == pytest.approx(0.95, abs=0.005)
    assert weak.energy_measurement_share == pytest.approx(5_316_252 / weak.circuit_t)

    strong = price_one_circuit(
        HELIUM_4, box_side=3, register_bits=11, strength=2.0, current_one_norm=27.0
    )
    assert strong.synthesis.rotation_count == 10_350 + 29 * (21 + 18) + 19
    assert strong.synthesis.rotation_count == 11_500
    assert strong.synthesis.t_per_rotation == 34
    assert strong.synthesis.angle_bits == 22
    assert strong.energy_measurement_t == 5_316_252
    assert strong.load_t == 2_548_596
    assert strong.schedule.preparations == 29
    assert strong.all_loads_t == 29 * 2_548_596
    assert strong.circuit_t == 79_259_664 + 647


def test_everything_priced_takes_in_the_loads_where_they_have_a_price() -> None:
    """Everything priced is helium's whole circuit, and all but carbon's preparations.

    Helium's loads have a price, so everything priced is its whole circuit, 1.15e8 T,
    of which the energy measurement, 43 insertions and 21 rounds of reflections are only
    5.4 million. Carbon's preparation has none, so its everything priced is the 2.18e7 T
    of those three parts alone, a floor on its circuit.
    """
    helium = price_one_circuit(
        HELIUM_4, box_side=3, register_bits=11, strength=1.0, current_one_norm=27.0
    )
    assert helium.insertions_t == 43 * 920
    assert helium.reflections_t == 21 * 532
    assert helium.all_but_preparations_t == 5_316_252 + 43 * 920 + 21 * 532
    assert helium.everything_priced_t == helium.circuit_t
    assert helium.everything_priced_t == 114_957_259

    carbon = price_one_circuit(
        CARBON_12, box_side=4, register_bits=12, strength=3.0, current_one_norm=64.0
    )
    assert carbon.insertions_t == 59 * 508
    assert carbon.reflections_t == 29 * 1_064
    assert carbon.everything_priced_t == carbon.all_but_preparations_t
    assert carbon.everything_priced_t == 21_806_884


def test_the_rotations_share_divides_like_by_like() -> None:
    """The rotations' T as a share of everything priced, counted over the same parts.

    Carbon's 12,420 rotations at 34 T are 1.9% of its 2.18e7 T. Helium's share is of its
    whole circuit, so its spreads' and its gradient state's rotations count, but not the
    15 rounded angles of each load: the load adds those into the gradient state with
    Toffolis, which its price already holds.
    """
    carbon = price_one_circuit(
        CARBON_12, box_side=4, register_bits=12, strength=3.0, current_one_norm=64.0
    )
    assert carbon.rotation_share == pytest.approx(12_420 * 34 / 21_806_884)
    assert carbon.rotation_share == pytest.approx(0.019, abs=0.0005)

    helium = price_one_circuit(
        HELIUM_4, box_side=3, register_bits=11, strength=1.0, current_one_norm=27.0
    )
    synthesized = 12_046 - 43 * 15
    assert helium.rotation_share == pytest.approx(synthesized * 34 / 114_957_259)


def test_helium_on_the_next_lattice_goes_over_through_the_loads() -> None:
    """One size up, the table grows to 64^3 amplitudes and takes the circuit past 10^9.

    The larger circuit has more rotations, so each one gets less of the budget: 35 T
    a rotation, and 23 bits for the load's rounded angles. The gradient state is then
    20 rotations and a T gate, 701 T.
    """
    weak = price_one_circuit(
        HELIUM_4, box_side=4, register_bits=13, strength=1.0, current_one_norm=64.0
    )
    assert weak.synthesis.t_per_rotation == 35
    assert weak.synthesis.angle_bits == 23
    assert weak.energy_measurement_t == 43_578_708
    assert weak.load_t == 38_802_760
    assert weak.schedule.preparations == 101
    assert weak.gradient_state_t == 20 * 35 + 1
    assert weak.circuit_t == 3_962_765_576 + 701

    strong = price_one_circuit(
        HELIUM_4, box_side=4, register_bits=13, strength=2.0, current_one_norm=64.0
    )
    assert strong.synthesis.t_per_rotation == 35
    assert strong.synthesis.angle_bits == 23
    assert strong.load_t == 38_802_760
    assert strong.schedule.preparations == 71
    assert strong.circuit_t == 2_798_650_496 + 701


def test_argon_passes_the_qubit_budget_through_the_and_chain() -> None:
    """Argon on the 5^3 lattice: an odd lattice, and no table to load.

    Its index holds three coordinates of three bits and the spin, so an insertion takes
    21 rotations, and each rotation's precision costs 36 T and 24 bits of rounded angle
    at both ends of the band. The 1,018 qubits come from the amplification's chain of
    508.
    """
    weak = price_one_circuit(
        ARGON_40, box_side=5, register_bits=13, strength=9.0, current_one_norm=125.0
    )
    assert weak.synthesis.t_per_rotation == 36
    assert weak.synthesis.angle_bits == 24
    assert weak.energy_measurement_t == 82_646_512
    assert weak.insertion_t == 1_896
    assert weak.schedule.preparations == 65
    assert weak.allowance_t == pytest.approx(14_110_218, abs=0.5)
    assert weak.qubits.peak == 1_018
    assert weak.qubits.amplification_chain == 508

    strong = price_one_circuit(
        ARGON_40, box_side=5, register_bits=13, strength=18.0, current_one_norm=125.0
    )
    assert strong.synthesis.t_per_rotation == 36
    assert strong.synthesis.angle_bits == 24
    assert strong.energy_measurement_t == 82_646_512
    assert strong.schedule.preparations == 47
    assert strong.allowance_t == pytest.approx(19_515_253, abs=0.5)
    assert strong.qubits.peak == 1_018


def test_argon_on_the_larger_box_is_over_on_the_measurement_alone() -> None:
    """Argon on the 8^3 lattice: nothing is left to spend, and the code must say so.

    The energy measurement alone is 1.3 times the budget, so the allowance is negative,
    never clamped to zero, which would read as a preparation that must simply be free.
    Its walk steps are negative too, and lambda over them would be a negative energy,
    so there is no rule of thumb. The system and the index alone pass 1,000 qubits. An
    insertion here is its 1,023 ANDs and one T gate per axis, with no rotation.

    Counted one per Toffoli the energy measurement is a third of the budget, and the
    rule of thumb returns, until a 17-bit register takes it over the budget again.
    """
    circuit = price_one_circuit(
        ARGON_40, box_side=8, register_bits=15, strength=9.0, current_one_norm=512.0
    )
    assert circuit.energy_measurement_t == 1_291_001_644
    assert circuit.insertion_t == 1_023 * 4 + 3
    assert circuit.qubits.walk_stage == 2_166
    assert circuit.allowance_t == pytest.approx(-1_089_998, abs=0.5)
    assert circuit.allowance_in_walk_steps < 0.0
    assert circuit.rule_of_thumb_mev is None
    assert circuit.qubits.free_for_preparation == 0
    assert circuit.one_per_toffoli.rule_of_thumb_mev is not None

    finer = price_one_circuit(
        ARGON_40, box_side=8, register_bits=17, strength=9.0, current_one_norm=512.0
    )
    assert finer.one_per_toffoli.rule_of_thumb_mev is None


def test_the_current_normalized_at_the_proton_count_needs_few_preparations() -> None:
    """If the current could be normalized at Z = 6 instead of M = 64.

    P_s rises by (64 / 6)^2, so carbon needs only 5 or 3 preparations instead of 59 or
    41. The insertion has no rotations here, so a rotation stays at 34 T. The allowance
    grows past 190 million T. Nobody has built this current, and it is the slide's
    lever.
    """
    gain = gain_at_proton_count(CARBON_12, Lattice(box_side=4))
    assert gain == pytest.approx((64.0 / 6.0) ** 2)
    assert gain == pytest.approx(114.0, abs=0.5)

    weak = price_one_circuit(
        CARBON_12, box_side=4, register_bits=12, strength=3.0, current_one_norm=6.0
    )
    assert weak.schedule.rounds == 2
    assert weak.schedule.preparations == 5
    assert weak.synthesis.t_per_rotation == 34
    assert weak.allowance_t == pytest.approx(195_649_855, abs=0.5)

    strong = price_one_circuit(
        CARBON_12, box_side=4, register_bits=12, strength=6.0, current_one_norm=6.0
    )
    assert strong.schedule.rounds == 1
    assert strong.schedule.preparations == 3
    assert strong.synthesis.t_per_rotation == 34
    assert strong.allowance_t == pytest.approx(326_083_785, abs=0.5)


def test_a_toffoli_counted_as_one_hard_operation_moves_the_rule_of_thumb() -> None:
    """The budget's source names "T or Toffoli gates", and we charge a Toffoli as 4 T.

    Charged as one operation instead, a rotation's T stay T and the rest falls by four.
    Carbon's step is 2,603 Toffolis and 4 rotations at 34 T, and the allowance buys
    about four times the walk steps. So the rule of thumb, lambda over those steps,
    falls from 10 to 14 MeV to 2.4 to 3.5 MeV: how hard the open problem looks depends
    on this count.
    """
    weak = price_one_circuit(
        CARBON_12, box_side=4, register_bits=12, strength=3.0, current_one_norm=64.0
    )
    assert weak.rule_of_thumb_mev == pytest.approx(13.77, abs=0.005)
    counted = weak.one_per_toffoli
    assert counted.walk_step == 2_603 + 4 * 34
    assert counted.energy_measurement == 2_048 * 2_739 + (132 + 4_096) * 34
    assert counted.energy_measurement == 5_753_224
    assert counted.all_but_preparations == 5_753_224 + 59 * 127 + 29 * 266
    assert counted.all_but_preparations == 5_768_431
    assert counted.whole is None
    assert counted.everything_priced == 5_768_431
    assert counted.allowance_in_walk_steps == pytest.approx(6_152, abs=0.5)
    assert counted.rule_of_thumb_mev == pytest.approx(3.52, abs=0.005)

    strong = price_one_circuit(
        CARBON_12, box_side=4, register_bits=12, strength=6.0, current_one_norm=64.0
    )
    assert strong.rule_of_thumb_mev == pytest.approx(9.57, abs=0.005)
    counted_strong = strong.one_per_toffoli
    assert counted_strong.allowance_in_walk_steps == pytest.approx(8_853, abs=0.5)
    assert counted_strong.rule_of_thumb_mev == pytest.approx(2.44, abs=0.005)


def test_two_standings_change_if_a_toffoli_counts_as_one_hard_operation() -> None:
    """At 4 T helium on 4^3 is over the budget, and argon on 8^3 over on its energies.

    Charged as one operation, helium's loads are 9,700,690 Toffolis each, so the circuit
    fits at 70% to 99% of the budget. Its gradient state is rotations and a T gate, so
    its 701 T stay T. Argon's energy measurement is about a third of the budget.
    """
    weak = price_one_circuit(
        HELIUM_4, box_side=4, register_bits=13, strength=1.0, current_one_norm=64.0
    )
    assert weak.circuit_t is not None
    assert weak.circuit_t > 10**9
    assert weak.one_per_toffoli.energy_measurement == 4_096 * 2_747 + 292_180
    assert weak.one_per_toffoli.all_but_preparations == (
        11_543_892 + 101 * 127 + 50 * 284
    )
    assert weak.one_per_toffoli.whole == 11_570_919 + 101 * 9_700_690 + 701
    assert weak.one_per_toffoli.whole == 991_341_310

    strong = price_one_circuit(
        HELIUM_4, box_side=4, register_bits=13, strength=2.0, current_one_norm=64.0
    )
    assert strong.one_per_toffoli.whole == 700_312_540

    argon = price_one_circuit(
        ARGON_40, box_side=8, register_bits=15, strength=9.0, current_one_norm=512.0
    )
    assert argon.energy_measurement_t > 10**9
    assert argon.one_per_toffoli.energy_measurement == 16_384 * 19_794 + 1_253_164
    assert argon.one_per_toffoli.energy_measurement == 325_558_060
    assert argon.one_per_toffoli.whole is None


def test_a_standing_at_the_budget_fits_but_leaves_nothing_open() -> None:
    """At exactly the whole budget, a priced circuit fits.

    An unpriced one at the whole budget has nothing left for its preparations, so it is
    not open: a preparation that must cost nothing is not one anybody has.
    """
    full = Standing(preparation_priced=True, weak_end_share=1.0, strong_end_share=0.9)
    assert full.weak_end_fits
    assert full.strong_end_fits

    nothing_left = Standing(
        preparation_priced=False, weak_end_share=1.0, strong_end_share=0.9
    )
    assert not nothing_left.weak_end_fits
    assert nothing_left.strong_end_fits


def test_the_standing_checks_each_end_of_the_band() -> None:
    """A case fits, stays open or goes over at each end of its band, checked apart.

    Carbon's preparation has no price, and something is left for it at both ends, so
    it is open. Helium on 3^3 fits, at 7.9% to 11.5% of the budget, and on 4^3 it is
    over at both ends. Argon on 8^3 is over before any preparation. Counted one per
    Toffoli, helium on 4^3 with a 14-bit register fits at its strong end only.
    """
    carbon_weak, carbon_strong = price_lecture_case(LECTURE_CASES["carbon"])
    carbon = standing(carbon_weak, carbon_strong)
    assert not carbon.preparation_priced
    assert carbon.weak_end_fits
    assert carbon.strong_end_fits

    helium_weak, helium_strong = price_lecture_case(LECTURE_CASES["helium"])
    helium = standing(helium_weak, helium_strong)
    assert helium.preparation_priced
    assert helium.weak_end_fits
    assert helium.strong_end_share == pytest.approx(0.079, abs=0.0005)
    assert helium.weak_end_share == pytest.approx(0.115, abs=0.0005)

    next_weak, next_strong = price_strength_band(
        HELIUM_4, box_side=4, register_bits=13, current_one_norm=64.0
    )
    next_helium = standing(next_weak, next_strong)
    assert not next_helium.weak_end_fits
    assert not next_helium.strong_end_fits

    argon_weak, argon_strong = price_strength_band(
        ARGON_40, box_side=8, register_bits=15, current_one_norm=512.0
    )
    argon = standing(argon_weak, argon_strong)
    assert not argon.preparation_priced
    assert not argon.weak_end_fits
    assert not argon.strong_end_fits

    finer_weak, finer_strong = price_strength_band(
        HELIUM_4, box_side=4, register_bits=14, current_one_norm=64.0
    )
    finer = standing_if_one_per_toffoli(finer_weak, finer_strong)
    assert finer.weak_end_share == pytest.approx(1.003, abs=0.0005)
    assert not finer.weak_end_fits
    assert finer.strong_end_fits


def test_one_more_register_bit_doubles_the_walk_and_halves_the_bin() -> None:
    """The law of phase estimation: each bit doubles the cost and halves the error."""
    lattice = Lattice(box_side=3)
    twelve = energy_register(lattice, bits=12)
    thirteen = energy_register(lattice, bits=13)

    assert thirteen.walk_steps == 2 * twelve.walk_steps
    assert thirteen.bin_width_mev == pytest.approx(twelve.bin_width_mev / 2.0)
    assert thirteen.rms_error_mev == pytest.approx(twelve.rms_error_mev / 2.0)


def test_the_lecture_cases_size_the_registers_the_lecture_quotes() -> None:
    """Each nucleus's case gives the register and the delivered error on the slides.

    The sizing is only a label: 10 MeV on carbon's lattice delivers 23.5 MeV RMS. The
    same rule gives the slides' other rows their registers.
    """
    carbon = LECTURE_CASES["carbon"]
    assert carbon.nucleus == CARBON_12
    carbon_lattice = Lattice(box_side=carbon.box_side)
    carbon_bits = register_bits_for_sizing(carbon_lattice, carbon.sizing_mev)
    assert carbon_bits == 12
    carbon_register = energy_register(carbon_lattice, carbon_bits)
    assert carbon_register.rms_error_mev == pytest.approx(23.5, abs=0.05)

    helium = LECTURE_CASES["helium"]
    assert helium.nucleus == HELIUM_4
    helium_lattice = Lattice(box_side=helium.box_side)
    helium_bits = register_bits_for_sizing(helium_lattice, helium.sizing_mev)
    assert helium_bits == 11
    helium_register = energy_register(helium_lattice, helium_bits)
    assert helium_register.rms_error_mev == pytest.approx(19.8, abs=0.05)

    argon = LECTURE_CASES["argon"]
    assert argon.nucleus == ARGON_40
    argon_lattice = Lattice(box_side=argon.box_side)
    argon_bits = register_bits_for_sizing(argon_lattice, argon.sizing_mev)
    assert argon_bits == 13
    argon_register = energy_register(argon_lattice, argon_bits)
    assert argon_register.rms_error_mev == pytest.approx(22.9, abs=0.05)

    assert register_bits_for_sizing(Lattice(box_side=4), helium.sizing_mev) == 13
    assert register_bits_for_sizing(Lattice(box_side=8), argon.sizing_mev) == 15


def test_a_band_is_priced_at_both_its_ends() -> None:
    """The weak end of the band, Z/2, needs more rounds than the strong end, Z.

    Carbon's lecture case sizes its register at 10 MeV, so 12 bits, and normalizes the
    current at the 64 sites: 59 preparations at the weak end and 41 at the strong one.
    """
    weak, strong = price_lecture_case(LECTURE_CASES["carbon"])
    assert weak.register.bits == 12
    assert weak.schedule.success_probability == pytest.approx(3.0 / 64.0**2)
    assert strong.schedule.success_probability == pytest.approx(6.0 / 64.0**2)
    assert weak.schedule.preparations == 59
    assert strong.schedule.preparations == 41

    same_weak, same_strong = price_strength_band(
        CARBON_12, box_side=4, register_bits=12, current_one_norm=64.0
    )
    assert same_weak == weak
    assert same_strong == strong


def test_the_sizing_rule_names_the_box_when_it_asks_too_many_bits() -> None:
    """A box of ten million sites a side needs more register bits than the walk can run.

    Whoever asked for that box chose no register, so the message must name the box
    and the remedy: a smaller box_side, or a coarser sizing.
    """
    with pytest.raises(ValueError) as refused:
        register_bits_for_sizing(Lattice(box_side=10**7), 10.0)
    message = str(refused.value)
    assert "box_side" in message
    assert "coarser sizing" in message


def test_the_smallest_lattices_each_nucleus_fits_on_still_price() -> None:
    """Inputs a student can reach must price, not stop on an error from the library.

    Helium and carbon fit on 2^3 and argon first fits on 3^3. Three bits is the smallest
    register that delivers an error below lambda.
    """
    helium = price_one_circuit(
        HELIUM_4, box_side=2, register_bits=11, strength=1.0, current_one_norm=8.0
    )
    assert helium.circuit_t is not None

    carbon = price_one_circuit(
        CARBON_12, box_side=2, register_bits=13, strength=3.0, current_one_norm=8.0
    )
    assert carbon.all_but_preparations_t > 0

    argon = price_one_circuit(
        ARGON_40, box_side=3, register_bits=12, strength=9.0, current_one_norm=27.0
    )
    assert argon.all_but_preparations_t > 0

    coarse = price_one_circuit(
        HELIUM_4, box_side=3, register_bits=3, strength=1.0, current_one_norm=27.0
    )
    assert coarse.register.walk_steps == 4


def test_nuclei_know_their_nucleons_protons_and_strength_band() -> None:
    """Helium, carbon and argon, counted from their four species.

    The strength band [Z/2, Z] sets every success probability in the lecture.
    """
    assert HELIUM_4.nucleon_count == 4
    assert HELIUM_4.proton_count == 2
    assert HELIUM_4.strength_band == (1.0, 2.0)

    assert CARBON_12.nucleon_count == 12
    assert CARBON_12.proton_count == 6
    assert CARBON_12.strength_band == (3.0, 6.0)

    assert ARGON_40.nucleon_count == 40
    assert ARGON_40.proton_count == 18
    assert ARGON_40.strength_band == (9.0, 18.0)


@pytest.mark.parametrize(
    ("species", "named"),
    [
        ((1, 0, 0, 0), "at least two nucleons"),
        ((0, 0, 2, 2), "no protons"),
        ((1, 1, -1, 0), "four whole numbers"),
        ((1, 1, 1), "four whole numbers"),
        ((1.0, 1, 1, 1), "four whole numbers"),
    ],
)
def test_a_nucleus_refuses_what_it_cannot_be(species: object, named: str) -> None:
    """A nucleus you make yourself must be four counts of at least zero, in the order
    p up, p down, n up, n down, with two nucleons or more.

    It also needs a proton, since the current is the proton density: with none, there
    is no response to measure.
    """
    with pytest.raises(ValueError, match=named):
        Nucleus(name="custom", nucleons_per_species=species)


def test_a_table_of_one_configuration_is_refused_by_name() -> None:
    """A species that fills every site leaves the pinned table one configuration.

    One configuration is a bitstring, set without a load, so there is no table to
    price; the message says so rather than stopping in the library.
    """
    filled = Nucleus(name="filled", nucleons_per_species=(1, 8, 0, 0))
    with pytest.raises(ValueError, match="filled has a pinned table of one"):
        price_one_circuit(
            filled, box_side=2, register_bits=8, strength=4.5, current_one_norm=8.0
        )


def test_lattice_counts_its_sites_modes_and_terms() -> None:
    """The 4^3 lattice: 64 sites, 5.6 fm across, 256 modes and 2,432 Pauli strings.

    Its one-norm, lambda, is the sum of the sizes of those strings' coefficients.
    """
    lattice = Lattice(box_side=4)

    assert lattice.site_count == 64
    assert lattice.box_length_fm == pytest.approx(5.6)
    assert lattice.system_qubits == 256
    assert lattice.term_count == 2_432
    assert lattice.one_norm_mev == pytest.approx(21_646.4)


@pytest.mark.parametrize("box_side", [1, 0, -2, 4.0, True])
def test_lattice_refuses_fewer_than_two_sites_per_axis(box_side: object) -> None:
    """A lattice needs at least two sites per axis, or a nucleon has nowhere to hop.

    A float or a boolean is refused too, even one that looks like a whole number.
    """
    with pytest.raises(ValueError, match="box_side"):
        Lattice(box_side=box_side)


def test_ground_state_amplitudes_multiply_each_species_choices() -> None:
    """Each species chooses its sites, and the choices multiply: the nuclear wall.

    Carbon on 64 sites has C(64, 3) = 41,664 choices per species, so 3.0e18 amplitudes.
    Argon on 8 sites cannot fit: 11 nucleons of one species have no way to sit on 8
    sites, by Pauli exclusion, and the count says so with a zero.
    """
    carbon_on_64_sites = ground_state_amplitudes(CARBON_12, Lattice(box_side=4))
    assert carbon_on_64_sites == 41_664**4
    assert carbon_on_64_sites == pytest.approx(3.0e18, rel=0.01)

    assert ground_state_amplitudes(HELIUM_4, Lattice(box_side=3)) == 27**4

    argon_on_125_sites = ground_state_amplitudes(ARGON_40, Lattice(box_side=5))
    assert argon_on_125_sites == pytest.approx(8.04e56, rel=0.001)

    assert ground_state_amplitudes(ARGON_40, Lattice(box_side=2)) == 0


def test_a_pinned_table_needs_a_species_of_one_nucleon() -> None:
    """Holding one nucleon at the origin takes its species out of the count.

    Helium has one nucleon of each species: on 27 sites the other three species give
    27^3 = 19,683 amplitudes, and on 64 sites 64^3 = 262,144. Carbon has no species of
    one, so it gets no table. Only one nucleon is pinned, even where two species have
    one each: with species of 1, 1, 1 and 0 nucleons, 27^2 amplitudes remain.
    """
    assert pinned_table_amplitudes(HELIUM_4, Lattice(box_side=3)) == 19_683
    assert pinned_table_amplitudes(HELIUM_4, Lattice(box_side=4)) == 262_144
    assert pinned_table_amplitudes(CARBON_12, Lattice(box_side=4)) is None

    three_nucleons = Nucleus(name="three", nucleons_per_species=(1, 1, 1, 0))
    assert pinned_table_amplitudes(three_nucleons, Lattice(box_side=3)) == 27**2


def test_the_translations_divide_the_configurations_by_the_sites() -> None:
    """Each of the M translations of the box carries a configuration to another one.

    For helium, pinning its one proton up at the origin uses exactly these: its 27^4
    configurations over the 27 translations are its pinned table of 27^3.
    """
    lattice = Lattice(box_side=3)
    after = amplitudes_after_translations(HELIUM_4, lattice)
    assert after == ground_state_amplitudes(HELIUM_4, lattice) / 27
    assert after == pinned_table_amplitudes(HELIUM_4, lattice)


def test_the_position_register_counts_nucleons_times_mode_bits() -> None:
    """Storing where each nucleon sits instead takes far fewer qubits than 256.

    Each of carbon's 12 nucleons needs 8 bits to name one of 256 modes, so 96 qubits.
    """
    assert position_register_qubits(CARBON_12, Lattice(box_side=4)) == 96


def test_one_site_one_norm_pieces_sum_to_the_library_lambda() -> None:
    """Lambda, built site by site from t, C0 and D0, and checked against the library.

    The hopping gives 126.96 MeV, and the Z, ZZ and ZZZ groups 102.93, 44.415 and
    63.92, the 14 on-site strings 211.3 between them. That is 338.2 MeV per site and
    21,646 MeV on 64 sites.
    """
    assert LECTURE_COUPLINGS.hopping_scale_mev == 10.58
    assert LECTURE_COUPLINGS.two_body_mev == -98.23
    assert LECTURE_COUPLINGS.three_body_mev == 127.84

    pieces = one_site_one_norm()
    assert pieces.hopping_mev == pytest.approx(126.96)
    assert pieces.single_z_mev == pytest.approx(102.93)
    assert pieces.pair_zz_mev == pytest.approx(44.415)
    assert pieces.triple_zzz_mev == pytest.approx(63.92)
    assert pieces.on_site_mev == pytest.approx(102.93 + 44.415 + 63.92)
    assert pieces.on_site_mev + pieces.hopping_mev == pytest.approx(pieces.total_mev)

    lattice = Lattice(box_side=4)
    assert pieces.total_mev * lattice.site_count == pytest.approx(lattice.one_norm_mev)


def test_energy_register_gives_steps_bins_and_the_delivered_error() -> None:
    """Twelve bits on the 4^3 lattice: 2,048 walk steps and 33.2 MeV bins.

    The bin is not the error: what a run delivers is 23.5 MeV RMS, smaller than the
    bin by sqrt(2).
    """
    register = energy_register(Lattice(box_side=4), bits=12)

    assert register.bits == 12
    assert register.walk_steps == 2_048
    assert register.bin_width_mev == pytest.approx(33.21, abs=0.005)
    assert register.rms_error_mev == pytest.approx(23.48, abs=0.005)


@pytest.mark.parametrize("bits", [2, 0, 13.0, True, 65, 5_000])
def test_energy_register_refuses_a_size_outside_its_range(bits: object) -> None:
    """Below 3 bits the delivered error exceeds lambda itself, and above 64 the walk
    alone is more than 2^63 steps, so a student is told why instead of getting a
    number with no meaning or an error from deep inside the library.
    """
    with pytest.raises(ValueError, match="register bits"):
        energy_register(Lattice(box_side=4), bits=bits)


def test_energy_register_accepts_the_ends_of_its_range() -> None:
    """Three bits and sixty-four bits both give a register."""
    assert energy_register(Lattice(box_side=4), bits=3).walk_steps == 4
    assert energy_register(Lattice(box_side=4), bits=64).walk_steps == 2**63


def test_walk_step_counts_on_both_kinds_of_lattice() -> None:
    """One walk step's counts, on an even lattice and an odd one.

    On 64 sites the superposition over sites takes Hadamards alone. On 27 sites it
    needs amplification, which adds Toffolis and two more rotations.
    """
    carbon_lattice = Lattice(box_side=4)
    carbon_register = energy_register(carbon_lattice, bits=12)
    carbon_step = walk_step(CARBON_12, carbon_lattice, carbon_register)
    assert carbon_step.keep_bits == 12
    assert carbon_step.prepare_toffolis == 65
    assert carbon_step.prepare_rotations == 2
    assert carbon_step.select_toffolis == 2_431
    assert carbon_step.reflect_toffolis == 42
    assert carbon_step.ancilla_qubits == 43

    helium_lattice = Lattice(box_side=3)
    helium_register = energy_register(helium_lattice, bits=11)
    helium_step = walk_step(HELIUM_4, helium_lattice, helium_register)
    assert helium_step.keep_bits == 11
    assert helium_step.prepare_toffolis == 74
    assert helium_step.prepare_rotations == 4
    assert helium_step.select_toffolis == 1_025
    assert helium_step.reflect_toffolis == 39
    assert helium_step.ancilla_qubits == 40


def test_insertion_counts_its_registers_and_gates() -> None:
    """The current is a Z on each proton mode, so it has 2 M terms of one size.

    The index holds x, y and z in binary and one bit for the spin. With 4 sites per
    axis, Hadamards make the superposition and every phase is a Z or an S, so the
    select's 127 ANDs are the whole price. With 3 sites per axis, each axis's
    superposition takes amplification, 4 ANDs and 2 rotations, done and undone, and each
    of its 2 bits takes a phase rotation. With 5, each axis has 3 bits. With 8,
    Hadamards again, and one T gate per axis is all that the phases cost.
    """
    on_64_sites = insertion(Lattice(box_side=4))
    assert on_64_sites.terms == 128
    assert on_64_sites.index_qubits == 7
    assert on_64_sites.toffolis == 127
    assert on_64_sites.rotations == 0
    assert on_64_sites.t_gates == 0

    on_27_sites = insertion(Lattice(box_side=3))
    assert on_27_sites.terms == 54
    assert on_27_sites.index_qubits == 3 * 2 + 1
    assert on_27_sites.toffolis == 2 * 3 * 4 + 53
    assert on_27_sites.rotations == 2 * 3 * 2 + 3 * 2
    assert on_27_sites.t_gates == 0

    on_125_sites = insertion(Lattice(box_side=5))
    assert on_125_sites.terms == 250
    assert on_125_sites.index_qubits == 3 * 3 + 1
    assert on_125_sites.toffolis == 2 * 3 * 6 + 249
    assert on_125_sites.rotations == 2 * 3 * 2 + 3 * 3
    assert on_125_sites.t_gates == 0

    on_512_sites = insertion(Lattice(box_side=8))
    assert on_512_sites.terms == 1_024
    assert on_512_sites.index_qubits == 10
    assert on_512_sites.toffolis == 1_023
    assert on_512_sites.rotations == 0
    assert on_512_sites.t_gates == 3


def _phase_on_bits(coordinate: int, momentum: int, box_side: int, bits: int) -> complex:
    """exp(2 pi i k x / n) as the gates build it: one phase for each set bit of x."""
    phase = complex(1.0)
    for bit in range(bits):
        if (coordinate // 2**bit) % 2 == 1:
            phase *= cmath.exp(2j * math.pi * momentum * 2**bit / box_side)
    return phase


TOY_PROTONS = (
    (0, 0, 0, 0),
    (1, 0, 0, 1),
    (2, 1, 0, 0),
    (0, 2, 1, 1),
    (1, 1, 2, 0),
    (1, 1, 2, 1),
)  # x, y, z and spin of six protons, a fixed basis state
TOY_MOMENTA = ((1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 1, 1), (2, 1, 1))


def _average_of_the_terms(box_side: int, momentum: tuple[int, int, int]) -> complex:
    """The index's amplitude on zero after PREPARE, phases, SELECT, PREPARE undone."""
    k_x, k_y, k_z = momentum
    current = insertion(Lattice(box_side=box_side))
    bits_per_axis = (current.index_qubits - 1) // 3
    total = complex(0.0)
    for x in range(box_side):
        for y in range(box_side):
            for z in range(box_side):
                phase = _phase_on_bits(x, k_x, box_side, bits_per_axis)
                phase *= _phase_on_bits(y, k_y, box_side, bits_per_axis)
                phase *= _phase_on_bits(z, k_z, box_side, bits_per_axis)
                for spin in (0, 1):
                    z_eigenvalue = -1.0 if (x, y, z, spin) in TOY_PROTONS else 1.0
                    total += phase * z_eigenvalue
    return total / current.terms


def _density(box_side: int, momentum: tuple[int, int, int]) -> complex:
    """The proton density at this momentum on the toy's six protons."""
    k_x, k_y, k_z = momentum
    density = complex(0.0)
    for x, y, z, _spin in TOY_PROTONS:
        density += cmath.exp(2j * math.pi * (k_x * x + k_y * y + k_z * z) / box_side)
    return density


def test_the_insertion_applies_the_proton_density() -> None:
    """The index's coordinates and their phase gates really do apply the current.

    Take six protons in a fixed basis state. The equal superposition spreads over the
    terms, each term takes its site's phase, one gate per coordinate bit, and a Z on its
    own mode, and undoing the superposition returns the average of the terms. Since
    n = (1 - Z) / 2 and the phases sum to zero at nonzero momentum, that average is the
    density over the site count, with a minus sign. That is why the current's one-norm
    is M, and P_s = eta^2 / M^2.
    """
    for box_side in (3, 4, 5):
        for momentum in TOY_MOMENTA:
            density = _density(box_side, momentum)
            assert abs(density) > 0.1
            expected = -density / Lattice(box_side=box_side).site_count
            assert _average_of_the_terms(box_side, momentum) == pytest.approx(expected)


def test_a_coordinate_register_holds_every_coordinate() -> None:
    """The index must have room for the largest coordinate on each axis and the spin."""
    for box_side in range(2, 17):
        index_qubits = insertion(Lattice(box_side=box_side)).index_qubits
        bits_per_axis = (index_qubits - 1) // 3
        assert 3 * bits_per_axis + 1 == index_qubits
        assert 2**bits_per_axis >= box_side
        assert 2 ** (bits_per_axis - 1) < box_side


def _what_the_gate_costs(momentum: int, bit: int, box_side: int) -> str:
    """Name a coordinate bit's phase gate from the gate itself.

    A power of i is a Z or an S, which take no T. An eighth root of one takes one T
    gate. Anything else must be synthesized as a rotation.
    """
    phase = cmath.exp(2j * math.pi * momentum * 2**bit / box_side)
    if abs(phase**4 - 1.0) < 1e-9:
        return "free"
    if abs(phase**8 - 1.0) < 1e-9:
        return "T"
    return "rotation"


def _costs_of_the_gates(momentum: int, box_side: int, bits: int) -> list[str]:
    """What each bit's phase gate costs on one coordinate at this momentum."""
    costs = []
    for bit in range(bits):
        costs.append(_what_the_gate_costs(momentum, bit, box_side))
    return costs


def test_phase_gates_charge_the_costliest_momentum_by_what_each_gate_is() -> None:
    """Each bit of a coordinate takes one phase gate, priced by what that gate is.

    The module takes no momentum, so it must charge the costliest one. Name each gate
    from its phase, for every momentum on every side up to 16: the charge matches
    k = 1, and no other momentum costs more.
    """
    charges = {2: (0, 0), 3: (2, 0), 4: (0, 0), 5: (3, 0), 8: (0, 1), 16: (1, 1)}
    for box_side, (rotations, t_gates) in charges.items():
        charged = phase_gates_per_axis(Lattice(box_side=box_side))
        assert charged == PhaseGates(rotations=rotations, t_gates=t_gates)

    for box_side in range(2, 17):
        lattice = Lattice(box_side=box_side)
        bits_per_axis = (insertion(lattice).index_qubits - 1) // 3
        charged = phase_gates_per_axis(lattice)

        at_one = _costs_of_the_gates(1, box_side, bits_per_axis)
        assert at_one.count("rotation") == charged.rotations
        assert at_one.count("T") == charged.t_gates

        for momentum in range(2, box_side):
            costs = _costs_of_the_gates(momentum, box_side, bits_per_axis)
            assert costs.count("rotation") <= charged.rotations
            assert costs.count("free") >= at_one.count("free")


def test_round_reflections_count_two_and_chains() -> None:
    """One amplification round reflects about two zero states.

    The chain over the index register is short. The other reflects about the state the
    preparation starts from, so it covers the system, the index and a table load's
    enumeration register. With no table it holds 261 ancillas on 64 sites, and sets the
    largest register the circuit needs.
    """
    on_64_sites = round_reflections(Lattice(box_side=4), enumeration_qubits=0)
    assert on_64_sites.index_chain_ands == 5
    assert on_64_sites.start_chain_ands == 261

    on_27_sites = round_reflections(Lattice(box_side=3), enumeration_qubits=15)
    assert on_27_sites.index_chain_ands == 5
    assert on_27_sites.start_chain_ands == 108 + 7 + 15 - 2


def _toy_basis_state(system: int, enumeration: int, index: int) -> int:
    """Where a basis state sits among the toy's 32: 2 + 2 + 1 qubits."""
    return 8 * system + 2 * enumeration + index


def _toy_dense_entry(row: int, column: int) -> float:
    """A Householder reflection whose first column is the toy's table of amplitudes."""
    direction = [0.9 - 1.0, 0.3, 0.3, 0.1]
    squared_length = 0.0
    for size in direction:
        squared_length += size * size
    entry = -2.0 * direction[row] * direction[column] / squared_length
    if row == column:
        entry += 1.0
    return entry


def _toy_load_then_current() -> list[list[float]]:
    """A toy preparation and current, as the matrix of a real orthogonal circuit.

    The load writes a four-amplitude table on the enumeration register, adds it into
    the system, and clears the enumeration register with what the system held before:
    |s, e> goes to the sum over k of dense[k][e] |s xor k, s>. From |0, 0> that is the
    table on the system with the enumeration register back at zero, and from elsewhere
    the enumeration register does not return to zero. The current then leaves the index
    at 0, which marks success, only when the system holds configuration 1.
    """
    circuit = []
    for _ in range(32):
        circuit.append([0.0] * 32)
    for system in range(4):
        for enumeration in range(4):
            for index in range(2):
                column = _toy_basis_state(system, enumeration, index)
                for k in range(4):
                    index_after = index
                    if system ^ k != 1:
                        index_after = 1 - index
                    row = _toy_basis_state(system ^ k, system, index_after)
                    circuit[row][column] = _toy_dense_entry(k, enumeration)
    return circuit


def _toy_apply(
    circuit: list[list[float]], state: list[float], *, undo: bool
) -> list[float]:
    """The circuit on a state, or undone, by its transpose: the matrix is orthogonal."""
    result = [0.0] * 32
    for row in range(32):
        for column in range(32):
            if undo:
                result[column] += circuit[row][column] * state[row]
            else:
                result[row] += circuit[row][column] * state[column]
    return result


def _toy_flip_signs(state: list[float], positions: list[int]) -> None:
    """A reflection: flip the sign of the named basis states."""
    for position in positions:
        state[position] = -state[position]


def _toy_success_after_rounds(
    circuit: list[list[float]], flipped_at_the_start: list[int], rounds: int
) -> float:
    """The probability of success after some rounds of amplitude amplification.

    Each round flips the sign of the successful states, the ones with index 0, undoes
    the circuit, flips the sign of the basis states the start reflection names, and
    redoes the circuit.
    """
    successful = list(range(0, 32, 2))
    zero_state = [0.0] * 32
    zero_state[_toy_basis_state(0, 0, 0)] = 1.0
    state = _toy_apply(circuit, zero_state, undo=False)
    for _ in range(rounds):
        _toy_flip_signs(state, successful)
        state = _toy_apply(circuit, state, undo=True)
        _toy_flip_signs(state, flipped_at_the_start)
        state = _toy_apply(circuit, state, undo=False)
    success = 0.0
    for position in successful:
        success += state[position] * state[position]
    return success


def test_amplification_needs_the_enumeration_register_in_its_reflection() -> None:
    """A reflection blind to the load's enumeration register amplifies nothing.

    The inverse load sends a state the table can reach to system zero with the
    enumeration register away from zero. A reflection about the zero state of every
    register lands where the schedule says. One that looks only at the system and the
    index flips those other components too, so the state leaves the plane amplitude
    amplification turns in, and success stays where it started.
    """
    circuit = _toy_load_then_current()
    success = 0.3 * 0.3
    schedule = amplification(success)
    assert schedule.rounds == 2

    every_register = [_toy_basis_state(0, 0, 0)]
    landing = _toy_success_after_rounds(circuit, every_register, schedule.rounds)
    assert landing == pytest.approx(schedule.landing_probability)
    assert landing > 0.99

    blind_to_enumeration = []
    for enumeration in range(4):
        blind_to_enumeration.append(_toy_basis_state(0, enumeration, 0))
    blind = _toy_success_after_rounds(circuit, blind_to_enumeration, schedule.rounds)
    assert blind == pytest.approx(success)


def test_amplification_lands_nearest_a_quarter_turn() -> None:
    """Each round turns the state by 2 theta, so m rounds land at (2 m + 1) theta.

    Carbon's two strengths at the norm 64 take 29 and 20 rounds. At theta a twentieth
    of a turn, (2 m + 1) theta is a quarter turn at m = 2, so two rounds land exactly.
    """
    weak = amplification(3 / 4_096)
    assert weak.success_probability == 3 / 4_096
    assert weak.rounds == 29
    assert weak.preparations == 59
    assert weak.landing_probability == pytest.approx(0.999317, abs=5e-7)

    strong = amplification(6 / 4_096)
    assert strong.rounds == 20
    assert strong.preparations == 41

    exact = amplification(math.sin(math.pi / 10) ** 2)
    assert exact.rounds == 2
    assert exact.landing_probability == pytest.approx(1.0)


def test_amplification_takes_fewer_rounds_on_a_tie() -> None:
    """Two round counts can miss the quarter turn by exactly the same angle.

    Then the cheaper one wins, whichever way rounding noise leans. At theta a twelfth
    of a half turn, two rounds land a twelfth short of the quarter turn and three land
    a twelfth past, so two are taken, though the small-angle formula pi / (4 sqrt(P_s))
    would say three. In floating point the second miss comes out a hair the smaller,
    which is why the comparison needs a tolerance.
    """
    twelfth = amplification(math.sin(math.pi / 12) ** 2)
    assert twelfth.rounds == 2
    assert twelfth.landing_probability == pytest.approx(math.sin(5 * math.pi / 12) ** 2)


def test_amplification_of_a_certain_success_needs_no_rounds() -> None:
    """With P_s = 1 the state already sits at the quarter turn: no rounds."""
    certain = amplification(1.0)
    assert certain.rounds == 0
    assert certain.preparations == 1
    assert certain.landing_probability == pytest.approx(1.0)


@pytest.mark.parametrize("success_probability", [0.0, -0.1, 1.5, math.nan])
def test_amplification_refuses_a_probability_outside_zero_to_one(
    success_probability: float,
) -> None:
    """A success probability must lie above zero and at most one."""
    with pytest.raises(ValueError, match="success probability"):
        amplification(success_probability)


def test_synthesis_budget_shares_one_budget_over_every_rotation() -> None:
    """A budget of 0.01 per circuit, over all its rotations and rounded angles.

    Carbon's 12,420 rotations each get about 8.1e-7, which costs 34 T. Argon's 42,481
    each get less room, which costs 36 T. The budget is a weak lever: ten times looser
    or tighter moves carbon's price only to 30 or 38 T.
    """
    carbon = synthesis_budget(12_420)
    assert carbon.rotation_count == 12_420
    assert carbon.precision == pytest.approx(8.05e-7, abs=5e-9)
    assert carbon.t_per_rotation == 34
    assert carbon.angle_bits == 22
    assert synthesis_budget(12_420, error_budget=0.1).t_per_rotation == 30
    assert synthesis_budget(12_420, error_budget=0.001).t_per_rotation == 38

    argon = synthesis_budget(42_481)
    assert argon.t_per_rotation == 36
    assert argon.angle_bits == 24


def test_ground_state_load_counts_the_table_and_the_spread() -> None:
    """Helium's table of 19,683 amplitudes, loaded with 22-bit angles and spread out.

    The table takes 636,628 Toffolis and the spreading only 470, with six rotations.
    The 15 levels of the table and those six rotations are 21 places where one load
    spends the error budget. The spread's shift register holds a coordinate of
    ceil(log2 3) = 2 bits for each of the three axes.
    """
    lattice = Lattice(box_side=3)

    load = ground_state_load(19_683, lattice, angle_bits=22)
    assert load.amplitudes == 19_683
    assert load.load_toffolis == 636_628
    assert load.spread_toffolis == 470
    assert load.spread_rotations == 6
    assert load.ancilla_qubits == 94
    assert load.enumeration_qubits == 15
    assert load.gradient_qubits == 22
    assert load.shift_qubits == 3 * 2
    assert load.gradient_state == PhaseGates(rotations=19, t_gates=1)

    assert load_coherent_errors(19_683, lattice) == 21


def test_the_gradient_state_is_prepared_once_per_circuit() -> None:
    """The load adds its rounded angles into a phase-gradient state, which must be made.

    Helium's angles take 22 bits, so the state is 22 qubits, each a Hadamard and then
    a turn of 2^b / 2^22 of a full circle on bit b. The top two turns are a Z and an S,
    the next is one T gate, and the other 19 are rotations, which share the error
    budget with the rest of the circuit. The state is never used up, so it is made once
    per circuit, not once per load. Carbon has no load, so it has no gradient state.
    """
    assert gradient_state_gates(22) == PhaseGates(rotations=19, t_gates=1)
    assert gradient_state_gates(4) == PhaseGates(rotations=1, t_gates=1)

    helium = price_one_circuit(
        HELIUM_4, box_side=3, register_bits=11, strength=1.0, current_one_norm=27.0
    )
    assert helium.synthesis.angle_bits == 22
    assert helium.gradient_state_t == 19 * 34 + 1
    assert helium.all_loads_t is not None
    assert helium.circuit_t == (
        helium.all_but_preparations_t + helium.all_loads_t + helium.gradient_state_t
    )

    carbon = price_one_circuit(
        CARBON_12, box_side=4, register_bits=12, strength=3.0, current_one_norm=64.0
    )
    assert carbon.gradient_state_t is None


def test_the_gradient_state_and_the_budget_settle_together() -> None:
    """The gradient's rotations share the budget, and the budget sets its size.

    Helium on 2^3 with 5 bits sits on a threshold. Without the gradient state it has
    16 x 2 x 2 + 5 x 4 + 2^5 + 9 x 9 = 197 rotations, which take 16-bit angles. A
    16-bit state's 13 rotations push the count past what 16 bits cover, so the two
    settle at 17 bits, whose state has 14 rotations. Priced in one pass, the circuit
    would make a state one qubit short of its own angles.
    """
    assert synthesis_budget(197).angle_bits == 16
    circuit = price_one_circuit(
        HELIUM_4, box_side=2, register_bits=5, strength=2.0, current_one_norm=8.0
    )
    assert circuit.synthesis.angle_bits == 17
    assert circuit.synthesis.rotation_count == 197 + 14
    assert circuit.load is not None
    assert circuit.load.gradient_qubits == 17
    assert circuit.load.gradient_state == PhaseGates(rotations=14, t_gates=1)


def test_qubit_count_by_stage() -> None:
    """The register at each stage of the circuit, and which stage sets the peak.

    Carbon's peak comes from the amplification's long AND chain: 256 + 7 + 261 = 524.
    A preparation's ancillas that are clean at every reflection take turns with that
    chain, so 737 fit in the budget. Ancillas the reflection must cover also lengthen
    the chain, so each takes two qubits, and only 238 fit.

    Helium on 3^3 has a load. Its 15 enumeration and 22 gradient qubits stay held
    through the reflection, and so does the spread's 6-qubit shift register, which the
    undone spread leaves away from zero where the index is not; its other 51 ancillas
    take turns with the chain of 128: 108 + 7 + 15 + 22 + 6 + 128 = 286. The next
    lattice is 256 + 7 + 18 + 23 + 6 + 279 = 589. On
    the 2^3 lattice the chain is so short that helium's other ancillas outnumber it, and
    carbon's walk stage is the larger one. Argon on 5^3 holds 510 qubits through the
    reflection and a chain of 508 beside them, 1,018 before a preparation adds any
    ancillas, so none are free.
    """
    carbon = price_one_circuit(
        CARBON_12, box_side=4, register_bits=12, strength=3.0, current_one_norm=64.0
    ).qubits
    assert carbon.walk_stage == 256 + 12 + 43 + 42
    assert carbon.walk_stage == 353
    assert carbon.walk_added == 12 + 43 + 42
    assert carbon.amplification_added == 7 + 261
    assert carbon.amplification_stage == 524
    assert carbon.peak == 524
    assert carbon.free_for_preparation == 737
    assert carbon.free_if_reflected_over == 238

    helium = price_one_circuit(
        HELIUM_4, box_side=3, register_bits=11, strength=1.0, current_one_norm=27.0
    ).qubits
    assert helium.load_ancillas == 94
    assert helium.load_enumeration == 15
    assert helium.load_gradient == 22
    assert helium.load_shift == 6
    assert helium.load_other_ancillas == 94 - 15 - 22 - 6
    assert helium.amplification_chain == 128
    assert helium.walk_stage == 198
    assert helium.amplification_stage == 108 + 7 + 15 + 22 + 6 + 128
    assert helium.peak == 286

    next_helium = price_one_circuit(
        HELIUM_4, box_side=4, register_bits=13, strength=1.0, current_one_norm=64.0
    ).qubits
    assert next_helium.amplification_stage == 256 + 7 + 18 + 23 + 6 + 279
    assert next_helium.peak == 589

    small_helium = price_one_circuit(
        HELIUM_4, box_side=2, register_bits=11, strength=1.0, current_one_norm=8.0
    ).qubits
    held = small_helium.load_enumeration + small_helium.load_gradient
    held += small_helium.load_shift
    assert small_helium.load_ancillas - held > small_helium.amplification_chain
    assert small_helium.amplification_stage == 32 + 4 + small_helium.load_ancillas

    small_carbon = price_one_circuit(
        CARBON_12, box_side=2, register_bits=13, strength=3.0, current_one_norm=8.0
    ).qubits
    assert small_carbon.amplification_stage == 32 + 4 + 34
    assert small_carbon.walk_stage > small_carbon.amplification_stage
    assert small_carbon.peak == small_carbon.walk_stage

    argon = price_one_circuit(
        ARGON_40, box_side=5, register_bits=13, strength=9.0, current_one_norm=125.0
    ).qubits
    assert argon.amplification_stage == 500 + 10 + 508
    assert argon.free_for_preparation == 0
    assert argon.free_if_reflected_over == 0


def test_price_one_circuit_refuses_a_nucleus_the_lattice_cannot_hold() -> None:
    """Argon has 11 nucleons of one species, and the 2^3 lattice has only 8 sites.

    Zero configurations must not price as a circuit, so the message says what is
    wrong in the nucleus's own numbers.
    """
    with pytest.raises(ValueError) as refused:
        price_one_circuit(
            ARGON_40, box_side=2, register_bits=13, strength=9.0, current_one_norm=8.0
        )
    message = str(refused.value)
    assert "argon" in message
    assert "11 nucleons" in message
    assert "8 sites" in message


@pytest.mark.parametrize(
    ("strength", "current_one_norm", "named"),
    [
        (0.0, 27.0, "strength must"),
        (-1.0, 27.0, "strength must"),
        (1.0, 0.0, "current_one_norm must"),
        (1.0, -6.0, "current_one_norm must"),
        (2.0, 1.0, "success probability must"),
    ],
)
def test_price_one_circuit_refuses_a_strength_or_norm_out_of_range(
    strength: float, current_one_norm: float, named: str
) -> None:
    """A strength or a norm must lie above zero, and their P_s must not exceed 1.

    A negative norm is the quiet case: its square would give a sensible-looking P_s.
    """
    with pytest.raises(ValueError, match=named):
        price_one_circuit(
            HELIUM_4,
            box_side=3,
            register_bits=12,
            strength=strength,
            current_one_norm=current_one_norm,
        )


def test_numbers_a_call_could_swap_must_be_passed_by_name() -> None:
    """Two numbers of one kind, swapped, would price another circuit with no error.

    So each such pair is passed by name: the box side and the register bits, the
    strength and the current's norm, the logical qubits and the distance, the two
    error rates, the runs and the T rate, a bin's weight and its precision, and a
    window and its bins' width. A call that passes even the first named number by
    position is refused, whatever it names after it.
    """
    with pytest.raises(TypeError, match="positional argument"):
        price_one_circuit(
            CARBON_12, 4, register_bits=12, strength=3.0, current_one_norm=64.0
        )
    with pytest.raises(TypeError, match="positional argument"):
        price_strength_band(CARBON_12, 4, register_bits=12, current_one_norm=64.0)
    with pytest.raises(TypeError, match="positional argument"):
        physical_qubits(524, 19)
    with pytest.raises(TypeError, match="positional argument"):
        code_distance(4.6e-9, 0.1)
    with pytest.raises(TypeError, match="positional argument"):
        wall_time_seconds(21_806_884, 1, t_per_second=10**6)
    with pytest.raises(TypeError, match="positional argument"):
        runs_to_pin_bin_to_itself(0.15, 0.05)
    with pytest.raises(TypeError, match="positional argument"):
        runs_to_pin_bin_to_strength(0.15, 0.005)
    with pytest.raises(TypeError, match="positional argument"):
        bins_in_window(200.0, 33.21)


def test_logical_error_rate_divides_the_fault_budget() -> None:
    """A budget of 0.1 expected faults per circuit, shared over its hard operations:
    4.6e-9 for each of the 21,806,884 T of carbon's priced parts."""
    assert logical_error_rate(21_806_884) == pytest.approx(4.6e-9, abs=5e-11)
    assert logical_error_rate(10**9) == pytest.approx(1e-10)


def test_faults_add_up_over_the_hard_operations() -> None:
    """Expected faults are the hard operations times the logical error rate.

    The 2028 configuration runs 10^5 hard operations at 1e-8 each, 1e-3 faults a shot.
    At that rate a circuit that uses the whole 2033 budget carries 10 faults, so almost
    every run holds one; carbon's priced parts alone carry 0.22, about twice the 0.1 a
    circuit is allowed.
    """
    in_2028 = expected_faults(HARD_OPERATIONS_2028, logical_error=LOGICAL_ERROR_2028)
    assert in_2028 == pytest.approx(1e-3)
    full = expected_faults(HARD_OPERATIONS_2033, logical_error=LOGICAL_ERROR_2028)
    assert full == pytest.approx(10.0)
    carbon = expected_faults(21_806_884, logical_error=LOGICAL_ERROR_2028)
    assert carbon == pytest.approx(0.218, abs=0.0005)


def test_code_distance_meets_the_logical_error() -> None:
    """The lecture's schematic: (p / p_th)^((d + 1) / 2) with p / p_th = 0.1.

    An error of 1e-10 takes (d + 1) / 2 = 10 factors of 0.1, so d = 19, at 2 x 19^2 =
    722 physical qubits per logical one, and about 4e5 for carbon's 524-qubit stage. In
    floating point 0.1^10 lands just above 1e-10, so a bare comparison would wrongly
    ask for 21.
    """
    assert code_distance(1e-10) == 19
    assert code_distance(1e-8) == 15
    assert code_distance(4.6e-9) == 17
    assert physical_qubits_per_logical(19) == 722
    assert physical_qubits(524, distance=19) == 524 * 722


def test_a_circuit_needs_the_distance_of_the_lower_error_rate() -> None:
    """A circuit within the budget needs the rate of one that fills it, 1e-10, so
    d = 19. Helium on 4^3 prices past the budget, so its priced parts alone need a rate
    near 2.5e-11, and a longer code, d = 21.
    """
    carbon = price_one_circuit(
        CARBON_12, box_side=4, register_bits=12, strength=3.0, current_one_norm=64.0
    )
    assert carbon.needed_code_distance == 19

    helium = price_one_circuit(
        HELIUM_4, box_side=4, register_bits=13, strength=1.0, current_one_norm=64.0
    )
    on_priced = logical_error_rate(helium.everything_priced_t)
    assert on_priced < logical_error_rate(HARD_OPERATIONS_2033)
    assert on_priced == pytest.approx(2.5e-11, rel=0.02)
    assert helium.needed_code_distance == 21


@pytest.mark.parametrize("rate", [0.0, 1.0, 1.5])
def test_code_distance_refuses_rates_outside_zero_to_one(rate: float) -> None:
    """Both rates must lie strictly between 0 and 1.

    At a ratio of 1 or more, a larger code never helps, and the search would not end.
    """
    with pytest.raises(ValueError, match="logical_error"):
        code_distance(rate)
    with pytest.raises(ValueError, match="physical_over_threshold"):
        code_distance(1e-10, physical_over_threshold=rate)
    with pytest.raises(ValueError, match="logical_error"):
        expected_faults(10**9, logical_error=rate)


def test_randomized_alternative_costs_what_the_coherent_budget_does() -> None:
    """Randomizing loosens each rotation's precision, but the savings do not combine.

    Carbon's budget under randomized compiling asks only 4.0e-4 per rotation, yet the
    theorem covers a deterministic method, whose 3 log2(1 / epsilon) comes to 34 T: the
    same as repeat-until-success at the coherent 8.1e-7. That same deterministic method
    at 8.1e-7 would cost 61 T.
    """
    randomized = randomized_synthesis_precision(12_420)
    assert randomized == pytest.approx(4.0e-4, abs=5e-6)
    assert ross_selinger_t_estimate(randomized) == pytest.approx(34, abs=0.5)

    coherent = synthesis_budget(12_420).precision
    assert ross_selinger_t_estimate(coherent) == pytest.approx(61, abs=0.5)


def test_runs_to_pin_a_bin() -> None:
    """One run returns one bin, so a bin's weight is pinned by counting.

    A bin holding 15% of the strength, pinned to 5% of itself, takes 2,267 runs. Pinned
    to 0.5% of the total strength it takes 5,100, the other way to state a target.
    """
    assert runs_to_pin_bin_to_itself(0.15, precision=0.05) == 2_267
    assert runs_to_pin_bin_to_strength(0.15, precision=0.005) == 5_100


@pytest.mark.parametrize("bin_weight", [0.0, 1.0, 1.5])
def test_runs_refuse_a_weight_outside_zero_to_one(bin_weight: float) -> None:
    """A bin's weight is a fraction: with none or all of it, counting has no error."""
    with pytest.raises(ValueError, match="bin_weight"):
        runs_to_pin_bin_to_itself(bin_weight, precision=0.05)
    with pytest.raises(ValueError, match="bin_weight"):
        runs_to_pin_bin_to_strength(bin_weight, precision=0.05)


def test_a_circuit_that_misses_is_run_again() -> None:
    """A circuit lands with probability sin^2((2m + 1) theta), and runs again if not.

    At P_s = 0.6 no round helps, so a circuit lands 60% of the time: a success takes
    1 / 0.6 circuits on average, and as many preparations. A set of 25 histograms of
    1,000 successes each then takes 41,667 circuits, and at a landing of 0.8 exactly
    31,250, not one more.
    """
    schedule = amplification(0.6)
    assert schedule.rounds == 0
    assert schedule.landing_probability == pytest.approx(0.6)
    assert schedule.preparations_per_success == pytest.approx(1.0 / 0.6)

    the_set = runs_for_the_set(
        response_functions=5,
        momentum_transfers=5,
        runs_per_histogram=1_000,
        landing_probability=0.6,
    )
    assert the_set.histograms == 25
    assert the_set.successes == 25_000
    assert the_set.circuits == 41_667

    exact = runs_for_the_set(
        response_functions=5,
        momentum_transfers=5,
        runs_per_histogram=1_000,
        landing_probability=0.8,
    )
    assert exact.circuits == 31_250

    with pytest.raises(ValueError, match="landing_probability"):
        runs_for_the_set(
            response_functions=5,
            momentum_transfers=5,
            runs_per_histogram=1_000,
            landing_probability=0.0,
        )


def test_a_success_pays_the_energy_measurement_once() -> None:
    """A miss is found when the index is measured, before the energy measurement, so it
    runs only the rest of the circuit again.

    With the current's norm at 1.25 a circuit lands with P_s = 0.64. A success takes k
    attempts with probability 0.36^(k - 1) 0.64; each pays the rest of the circuit, and
    the last the energy measurement too. Summed over k, that is what one success costs.
    A circuit priced below its own energy measurement is refused.
    """
    circuit = price_one_circuit(
        HELIUM_4, box_side=3, register_bits=11, strength=1.0, current_one_norm=1.25
    )
    landing = circuit.schedule.landing_probability
    assert landing == pytest.approx(0.64)
    per_circuit = circuit.everything_priced_t
    energy = circuit.energy_measurement_t
    expected = 0.0
    for attempts in range(1, 200):
        chance = (1.0 - landing) ** (attempts - 1) * landing
        expected += chance * (attempts * (per_circuit - energy) + energy)

    assert circuit.set_t(per_circuit, successes=1) == pytest.approx(expected)
    assert circuit.set_t(per_circuit, successes=2_000) == pytest.approx(
        2_000 * expected
    )
    with pytest.raises(ValueError, match="energy measurement"):
        circuit.set_t(energy - 1, successes=1)
    with pytest.raises(ValueError, match="successes"):
        circuit.set_t(per_circuit, successes=0)


def test_bins_in_a_window() -> None:
    """A response 200 to 500 MeV wide, at carbon's 33.21 MeV bins, fills 7 to 16."""
    assert bins_in_window(200.0, bin_width_mev=33.21) == 7
    assert bins_in_window(500.0, bin_width_mev=33.21) == 16


def test_counts_round_up_only_past_a_whole_number() -> None:
    """A count that is exactly a whole number must not gain one from rounding noise.

    A bin of weight 0.2 pinned to 1% of the strength takes 0.2 x 0.8 / 0.01^2 = 1,600
    runs, and a window of 2.1 MeV holds exactly seven bins of 0.3 MeV. In floating point
    both ratios land just above the whole number, so a bare rounding up would say 1,601
    and 8. A window of 2.2 MeV does need an eighth bin.
    """
    assert runs_to_pin_bin_to_strength(0.2, precision=0.01) == 1_600
    assert bins_in_window(2.1, bin_width_mev=0.3) == 7
    assert bins_in_window(2.2, bin_width_mev=0.3) == 8


def test_wall_time_divides_t_gates_by_the_rate() -> None:
    """Wall time is T gates over the T rate, at one million T per second.

    The priced parts of one carbon circuit take 22 s, and a circuit that uses the whole
    budget 1,000 s. At 56,675 runs of that, the campaign takes 1.8 years.
    """
    priced = wall_time_seconds(21_806_884, runs=1, t_per_second=10**6)
    assert priced == pytest.approx(22.0, abs=0.5)

    whole_budget = wall_time_seconds(10**9, runs=1, t_per_second=10**6)
    assert whole_budget == pytest.approx(1_000.0)

    campaign = wall_time_seconds(10**9, runs=56_675, t_per_second=10**6)
    seconds_per_year = 365.25 * 24.0 * 3_600.0
    assert campaign / seconds_per_year == pytest.approx(1.8, abs=0.05)


@pytest.mark.parametrize("value", [0, -1])
def test_arithmetic_refuses_inputs_at_or_below_zero(value: int) -> None:
    """Each of these would divide by zero, or square a negative into a sensible number.

    The message names the input instead.
    """
    with pytest.raises(ValueError, match="hard_operations"):
        logical_error_rate(value)
    with pytest.raises(ValueError, match="hard_operations"):
        expected_faults(value, logical_error=1e-8)
    with pytest.raises(ValueError, match="distance"):
        physical_qubits_per_logical(value)
    with pytest.raises(ValueError, match="logical_qubits"):
        physical_qubits(value, distance=19)
    with pytest.raises(ValueError, match="rotation_count"):
        randomized_synthesis_precision(value)
    with pytest.raises(ValueError, match="precision"):
        ross_selinger_t_estimate(value)
    with pytest.raises(ValueError, match="window_mev"):
        bins_in_window(value, bin_width_mev=33.21)
    with pytest.raises(ValueError, match="bin_width_mev"):
        bins_in_window(200.0, bin_width_mev=value)
    with pytest.raises(ValueError, match="precision"):
        runs_to_pin_bin_to_itself(0.15, precision=value)
    with pytest.raises(ValueError, match="precision"):
        runs_to_pin_bin_to_strength(0.15, precision=value)
    with pytest.raises(ValueError, match="t_per_second"):
        wall_time_seconds(10**9, runs=1, t_per_second=value)
    with pytest.raises(ValueError, match="t_per_circuit"):
        wall_time_seconds(value, runs=1, t_per_second=10**6)
    with pytest.raises(ValueError, match="runs must"):
        wall_time_seconds(10**9, runs=value, t_per_second=10**6)
    with pytest.raises(ValueError, match="response_functions"):
        runs_for_the_set(
            response_functions=value,
            momentum_transfers=5,
            runs_per_histogram=1_000,
            landing_probability=0.8,
        )
    with pytest.raises(ValueError, match="momentum_transfers"):
        runs_for_the_set(
            response_functions=5,
            momentum_transfers=value,
            runs_per_histogram=1_000,
            landing_probability=0.8,
        )
    with pytest.raises(ValueError, match="runs_per_histogram"):
        runs_for_the_set(
            response_functions=5,
            momentum_transfers=5,
            runs_per_histogram=value,
            landing_probability=0.8,
        )


THE_SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "price_one_circuit.py"


def _run_the_script(options: list[str]) -> subprocess.CompletedProcess[str]:
    """Run the script with these options, in the Python running the tests."""
    return subprocess.run(
        [sys.executable, str(THE_SCRIPT)] + options,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )


@pytest.mark.parametrize(
    ("options", "first_line"),
    [
        ([], "One circuit for carbon on the 4^3 lattice"),
        (
            ["--register-bits", "13"],
            "One circuit for carbon on the 4^3 lattice, with 13",
        ),
        (
            ["--register-bits", "14"],
            "One circuit for carbon on the 4^3 lattice, with 14",
        ),
        (["--box-side", "5"], "One circuit for carbon on the 5^3 lattice"),
        (
            ["--nucleus", "helium", "--current-normalized-at-protons"],
            "One circuit for helium on the 3^3 lattice",
        ),
        (["--nucleus", "argon"], "One circuit for argon on the 5^3 lattice"),
        (["--nucleus", "argon", "--box-side", "8"], "One circuit for argon on the 8^3"),
        (["--nucleus", "helium", "--box-side", "4"], "One circuit for helium on the 4"),
    ],
)
def test_the_script_runs_every_exercise_to_the_end(
    options: list[str], first_line: str
) -> None:
    """Every exercise's options, and argon on 8^3 and helium on 4^3, run to the end.

    Only the status and the case are checked, never a number: the numbers are the
    module's, tested above, and reasoning about them is the exercise.
    """
    finished = _run_the_script(options)
    assert finished.returncode == 0
    assert finished.stderr == ""
    assert finished.stdout.startswith(first_line)
    assert "Against the 2028 budget" in finished.stdout


@pytest.mark.parametrize(
    ("options", "named"),
    [
        (["--box-side", "1"], "box_side must be"),
        (["--register-bits", "2"], "register bits must be"),
        (["--nucleus", "argon", "--box-side", "2"], "Pauli exclusion"),
        (["--box-side", "10000000"], "coarser sizing"),
        (["--box-side", "2.5"], "--box-side"),
        (["--nucleus", "oxygen"], "--nucleus"),
    ],
)
def test_the_script_names_the_input_it_refuses(options: list[str], named: str) -> None:
    """A bad input stops before any output, with status 2 and a message naming it."""
    finished = _run_the_script(options)
    assert finished.returncode == 2
    assert named in finished.stderr
    assert finished.stdout == ""


def test_the_script_says_when_a_box_is_too_large_to_compute() -> None:
    """A box of a thousand sites a side overflows floating point while it prints.

    The output stops there, and the message asks for a smaller box.
    """
    finished = _run_the_script(["--nucleus", "argon", "--box-side", "1000"])
    assert finished.returncode == 2
    assert "choose a smaller --box-side" in finished.stderr
    assert "Traceback" not in finished.stderr


def test_the_script_ends_quietly_when_its_reader_stops_early() -> None:
    """A reader such as head can close the pipe before the output ends.

    The script then stops without a traceback, with Python's status for a closed pipe.
    """
    process = subprocess.Popen(
        [sys.executable, str(THE_SCRIPT)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert process.stdout is not None
    assert process.stderr is not None
    process.stdout.close()
    errors = process.stderr.read()
    process.wait(timeout=60)
    assert process.returncode == 1
    assert errors == ""
