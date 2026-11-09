"""Tests for rotation synthesis: the precision per rotation, and its T cost."""

import math

import pytest

from qec_utility_estimation.state_preparation import sparse_state_load_cost
from qec_utility_estimation.synthesis import (
    coherent_synthesis_precision,
    phase_gradient_angle_bits,
    repeat_until_success_t_count,
)


@pytest.mark.parametrize(
    ("epsilon", "expected_t_count"),
    [(1e-3, 22), (1e-4, 26), (1e-5, 30), (1e-12, 57), (5.124e-13, 58)],
)
def test_repeat_until_success_t_count_at_known_precision_points(
    epsilon: float, expected_t_count: int
) -> None:
    """T-count matches hand-computed ceilings of Eq. 80 at five precisions.

    The formula is ceil(1.15 log2(2/eps) + 9.2): Watson+ Eq. 80 read in base 2
    and rounded up per rotation. At eps=1e-3 this is ceil(21.81) = 22; at
    eps=1e-4, ceil(25.63) = 26; at eps=1e-5, ceil(29.45) = 30; at eps=1e-12,
    ceil(56.19) = 57; at eps=5.124e-13 (the per-rotation precision at the
    pionless Tab. 9 headline point), ceil(57.30) = 58. Every pre-ceiling value
    sits clear of an integer boundary, so the ceiling is robust. The 1e-4 and
    1e-5 points locate the common "~30 T per rotation" figure: eps = 1e-5 costs
    30 T (as does any eps in [7.2e-6, 1.3e-5)), while eps = 1e-4 costs only 26.
    """
    assert repeat_until_success_t_count(epsilon) == expected_t_count


def test_repeat_until_success_t_count_is_monotone_in_precision() -> None:
    """Tightening the precision never decreases the T-count.

    Eq. 80 decreases in epsilon, so smaller (more precise) epsilon requires
    at least as many T-gates. Precisions from 1e-2 down to 1e-15 must yield a
    non-decreasing count.
    """
    epsilons = [1e-2, 1e-4, 1e-6, 1e-9, 1e-12, 1e-15]
    counts = [repeat_until_success_t_count(eps) for eps in epsilons]

    assert counts == sorted(counts)


@pytest.mark.parametrize("epsilon", [0.0, -1e-3, 1.0, 2.0])
def test_repeat_until_success_t_count_rejects_epsilon_out_of_unit_interval(
    epsilon: float,
) -> None:
    """Precision outside the open interval (0, 1) is rejected.

    A non-positive epsilon makes log2(2/eps) undefined or divides by zero,
    and eps >= 1 corresponds to no meaningful synthesis precision; both must
    fail fast with a ValueError rather than a raw math/arithmetic error.
    """
    with pytest.raises(ValueError):
        repeat_until_success_t_count(epsilon)


def test_coherent_precision_reproduces_the_linear_budget_arithmetic() -> None:
    """Twenty-five million rotations within 2.5e-3 need 1e-10 each, 49 T apiece.

    Errors that add linearly keep N rotations at epsilon each within N epsilon,
    so a budget B allows epsilon = B / N. This is the arithmetic of a synthesized
    prepare for 4He on 27 sites at a = 1.4 fm with 11 energy bits, three rotations
    per gate in both prepares: 12,210 rotations a walk step over textbook phase
    estimation's 2,047 steps, 2.5e7 in the walk. Within 2.5e-3 each must reach
    1e-10, and Eq. 80 then charges ceil(1.15 log2(2e10) + 9.2) = ceil(48.55) = 49 T
    per rotation.
    """
    precision = coherent_synthesis_precision(2.5e-3, 25_000_000)

    assert precision == pytest.approx(1e-10, rel=1e-12)
    assert repeat_until_success_t_count(precision) == 49


@pytest.mark.parametrize(
    ("error_budget", "rotation_count", "message"),
    [
        (0.0, 16_384, "error_budget must be a finite number in"),
        (1.0, 16_384, "error_budget must be a finite number in"),
        (-0.1, 16_384, "error_budget must be a finite number in"),
        (math.nan, 16_384, "error_budget must be a finite number in"),
        (math.inf, 16_384, "error_budget must be a finite number in"),
        (0.01, 0, "rotation_count must be an integer >= 1"),
        (0.01, 16384.0, "rotation_count must be an integer >= 1"),
        (0.01, True, "rotation_count must be an integer >= 1"),
    ],
)
def test_coherent_precision_rejects_budgets_and_counts_out_of_range(
    error_budget: float, rotation_count: int, message: str
) -> None:
    """A budget lies strictly between 0 and 1, and a rotation count is whole.

    A zero budget asks for exact rotations, which no finite synthesis gives. The
    budget is an operator-norm distance, which bounds the diamond distance of the
    two channels (Campbell 2017, Eq. 3), and that never exceeds 1. So a budget of
    1 or more bounds nothing a measurement can see, and one rotation at it is past
    the precisions repeat_until_success_t_count accepts. Both raise, and so do a
    negative, NaN or infinite budget. A circuit synthesizes a whole number of
    rotations, at least one: zero raises, and so do a whole-valued float and True,
    a bool that equals 1. Each error names the argument.
    """
    with pytest.raises(ValueError, match=message):
        coherent_synthesis_precision(error_budget, rotation_count)


def test_phase_gradient_angle_bits_follow_the_shared_precision() -> None:
    """A rounded angle takes ceil(log2(pi / epsilon)) bits at the shared precision.

    An angle rounded to c bits and added into a phase-gradient register is within
    pi / 2**c of the exact rotation, so a per-rotation precision epsilon needs
    c = ceil(log2(pi / epsilon)) bits. The precision is the one every coherent
    error in the circuit shares, budget over count. The lecture's circuits share
    their budget among some ten thousand coherent errors, and a budget of 0.01 over
    10,000 is 1e-6 per error and takes 22 bits; a budget of 0.1 takes 19 and one
    of 0.001 takes 25, so a decade of budget moves the bits by about three.

    The ceiling is exact at its boundary: a precision of exactly pi / 2**16 is
    met by 16 bits, since the quotient is exactly 2**16 and log2 of a power of
    two is exact, and a precision one part in 10**9 below it needs 17.

    The bits are the rotation_bits of sparse_state_load_cost: 22 of them over
    the L = 15 levels of helium's 19,683-configuration load on 27 sites cost
    (22 - 3) * 15 = 285 rotation Toffolis, so the return is a count the load
    accepts.
    """
    precision = coherent_synthesis_precision(0.01, 10_000)
    assert phase_gradient_angle_bits(precision) == 22
    assert phase_gradient_angle_bits(coherent_synthesis_precision(0.1, 10_000)) == 19
    assert phase_gradient_angle_bits(coherent_synthesis_precision(0.001, 10_000)) == 25

    assert phase_gradient_angle_bits(math.pi / 2**16) == 16
    assert phase_gradient_angle_bits(math.pi / 2**16 * (1 - 1e-9)) == 17

    load = sparse_state_load_cost(
        19_683, rotation_bits=phase_gradient_angle_bits(precision)
    )
    assert load.rotation_toffoli_count == 285


@pytest.mark.parametrize("precision", [0.0, -1e-7, math.pi, 4.0, math.inf, math.nan])
def test_phase_gradient_angle_bits_reject_unusable_precisions(precision: float) -> None:
    """A precision is a finite operator-norm error in the open interval (0, pi).

    A zero precision asks for an exact angle, which no finite register holds, and
    a negative one bounds nothing. A precision of pi or more constrains no angle,
    since a rounded angle is already within pi of any rotation, and would give
    zero or negative bits. An infinite or NaN precision is no bound. Each raises
    with a message that names the argument, in place of a division error or
    math's own domain error.
    """
    with pytest.raises(ValueError, match="precision must be a finite number in"):
        phase_gradient_angle_bits(precision)
