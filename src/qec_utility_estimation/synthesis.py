"""Rotation synthesis: the precision per rotation, and its T cost.

The precision each rotation needs follows from the circuit's error budget and
rotation count (coherent errors). The T cost at that precision is the
repeat-until-success R_z synthesis of Watson+ 2023 Eqs. 79-80, and the bits of
an angle a phase-gradient rotation rounds to follow from the same precision.
"""

from math import ceil, log2, pi


def repeat_until_success_t_count(epsilon_synthesis: float) -> int:
    """Expected T-gate count to synthesize an R_z rotation to precision epsilon.

    Repeat-until-success (RUS) synthesis: a Clifford+T circuit with one ancilla
    qubit and a measurement, retried until it succeeds, so the count is an
    expected value. This is Watson+ 2023 Eqs. 79-80 as applied in their Sec. 5
    T-gate lemmas (Lemmas 12, 16, 23) and hence their Tab. 9 T-counts, where
    epsilon bounds the operator-norm error ||R_z - R~_z|| of the rotation:

        T_count = ceil(1.15 * log2(2 / epsilon) + 9.2).

    Watson+ print "1.15 log(2/eps) + 9.2", with no log base and no ceiling. The
    1.15 / 9.2 coefficients are the Bocharov, Roetteler & Svore fit, PRL 114,
    080502 (2015), arXiv:1404.5320: "1.149 log2(1/eps) + 9.2" for random-angle
    z-rotations, which fixes the base as 2. The 2 inside the log is Watson+'s
    own (+1.15 T per rotation over the fit; not derived there). The ceiling is
    ours; their lemmas carry the unrounded value.

    Watson+ cite Paetznick & Svore, QIC 14, 1277 (2014), arXiv:1311.1074, whose
    own fit is 1.26 log2(1/eps) - 3.53. Ross & Selinger, QIC 16, 901 (2016),
    arXiv:1403.2975, is a different method (deterministic and ancilla-free,
    ~3 log2(1/eps) T-gates, more T than RUS at the precisions used here) and is
    not what this function computes.

    Args:
        epsilon_synthesis: Per-rotation synthesis precision epsilon, the
            dimensionless operator-norm bound of Eq. 79; strictly in (0, 1).

    Returns:
        Integer T-count (ceiling of the analytic expression).

    Raises:
        ValueError: If epsilon_synthesis is not in the open interval (0, 1).
    """
    if not 0.0 < epsilon_synthesis < 1.0:
        raise ValueError(
            f"epsilon_synthesis must be in (0, 1); got {epsilon_synthesis}."
        )
    return ceil(1.15 * log2(2.0 / epsilon_synthesis) + 9.2)


def coherent_synthesis_precision(error_budget: float, rotation_count: int) -> float:
    """Precision per rotation that keeps N synthesized rotations within a budget.

    Each synthesized rotation differs from its target by at most epsilon in
    operator norm. Unitary errors add at worst linearly: ||prod_i U_i -
    prod_i V_i|| <= sum_i ||U_i - V_i||, by the triangle inequality and the
    unitary invariance of the norm. N rotations at epsilon each keep the circuit
    within N epsilon of its target, which meets the budget at

        epsilon = error_budget / rotation_count.

    Watson+ 2023 (arXiv:2312.05344) budget synthesis this way: their Eq. 86
    charges each of a Trotter step's 42 L**3 rotations delta / (42 L**3). The
    bound holds for the diamond distance too, half the diamond norm of the two
    channels' difference (Campbell 2017, arXiv:1612.02689, Phys. Rev. A 95,
    042306, Eq. 2), which between two unitary channels is at most the
    operator-norm distance (his Eq. 3). The result is the epsilon of
    repeat_until_success_t_count.

    Randomly mixing approximations of each rotation does better: Campbell 2017
    (Thm. 2, Eq. 11) bounds a mixed axial rotation by 5 epsilon**2 in diamond
    distance, for epsilon < 0.01 and gate sequences from a deterministic
    synthesis algorithm. Repeat-until-success synthesis is not covered by that
    theorem, and this module offers no randomized model.

    Args:
        error_budget: Total synthesis error allowed for the circuit, an
            operator-norm distance (dimensionless); a finite number in (0, 1).
        rotation_count: Number N of synthesized rotations in the circuit; an
            integer >= 1.

    Returns:
        The operator-norm precision epsilon each rotation must reach.

    Raises:
        ValueError: If error_budget is not a finite number in (0, 1), or if
            rotation_count is not an integer >= 1 (a float raises, whole or not).
    """
    if not 0.0 < error_budget < 1.0:
        raise ValueError(
            "coherent_synthesis_precision: error_budget must be a finite number in "
            f"(0, 1); got {error_budget}."
        )
    if type(rotation_count) is not int or rotation_count < 1:
        raise ValueError(
            "coherent_synthesis_precision: rotation_count must be an integer >= 1 (a "
            f"circuit synthesizes a whole number of rotations); got {rotation_count!r}."
        )
    return error_budget / rotation_count


def phase_gradient_angle_bits(precision: float) -> int:
    """Bits of a rotation angle rounded for a phase-gradient rotation to precision.

    A rotation applied by adding a c-bit angle into a phase-gradient register
    (Fomichev+ 2023, arXiv:2310.18410, PRX Quantum 5, 040339 (2024), App. D, the
    rotations of the sparse-state load) rounds the angle to a multiple of
    2 pi / 2**c, so it is within pi / 2**c of the exact rotation in operator
    norm. At a per-rotation precision epsilon the bits that keep it inside are

        c = ceil(log2(pi / epsilon)).

    Fomichev+'s own criterion, c > log2(pi log2(D) / epsilon_state) for a state
    error epsilon_state over the log2(D) levels of the load, is this rule with
    the state error split evenly over the levels. The result is the
    rotation_bits of sparse_state_load_cost, which accepts four bits and up: a
    precision coarser than pi / 8 gives bits the load rejects, by the load's
    contract. A circuit that shares one coherent budget over its synthesized
    rotations and its rounded angles passes the same
    coherent_synthesis_precision to this function and to
    repeat_until_success_t_count. For 4He on 27 sites at a = 1.4 fm with 11 energy
    bits (a budget of 0.01 over 11,500 to 12,046 coherent errors, epsilon about
    8.5e-7) c = 22.

    Args:
        precision: Operator-norm error allowed per rounded angle
            (dimensionless); a finite number in (0, pi).

    Returns:
        The angle bits c, an integer >= 1.

    Raises:
        ValueError: If precision is not a finite number in (0, pi).
    """
    if not 0.0 < precision < pi:
        raise ValueError(
            "phase_gradient_angle_bits: precision must be a finite number in (0, pi) "
            "(a rounded angle is within pi of any rotation, so pi or more constrains "
            f"nothing); got {precision}."
        )
    return ceil(log2(pi / precision))
