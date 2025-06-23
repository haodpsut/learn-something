import jax.numpy as jnp


def shors_algorithm(N):
    p, q = 0, 0

    while p * q != N:
        a = jnp.random.choice(jnp.arange(2, N - 1))

        if jnp.gcd(N, a) != 1:
            p = jnp.gcd(N, a)
            return p, N // p

        guess_r = guess_order(N, a)

        if guess_r % 2 == 0:
            guess_square_root = (a ** (guess_r // 2)) % N

            if guess_square_root not in [1, N - 1]:
                p = jnp.gcd(N, guess_square_root - 1)
                q = jnp.gcd(N, guess_square_root + 1)

    return p, q

import pennylane as qml


@qml.qjit
def shors_algorithm(N):
    # Implementation goes here
    return p, q

@qml.qjit(autograph=True, static_argnums=(1))
def shors_algorithm(N, n_bits):
    # Implementation goes here
    return p, q


from jax import numpy as jnp


def repeated_squaring(a, exp, N):
    """QJIT-compatible function to determine (a ** exp) % N.

    Source: https://en.wikipedia.org/wiki/Modular_exponentiation#Left-to-right_binary_method
    """
    bits = jnp.array(jnp.unpackbits(jnp.array([exp]).view("uint8"), bitorder="little"))
    total_bits_one = jnp.sum(bits)

    result = jnp.array(1, dtype=jnp.int32)
    x = jnp.array(a, dtype=jnp.int32)

    idx, num_bits_added = 0, 0

    while num_bits_added < total_bits_one:
        if bits[idx] == 1:
            result = (result * x) % N
            num_bits_added += 1
        x = (x**2) % N
        idx += 1

    return result


def modular_inverse(a, N):
    """QJIT-compatible modular multiplicative inverse routine.

    Source: https://en.wikipedia.org/wiki/Extended_Euclidean_algorithm#Modular_integers
    """
    t = jnp.array(0, dtype=jnp.int32)
    newt = jnp.array(1, dtype=jnp.int32)
    r = jnp.array(N, dtype=jnp.int32)
    newr = jnp.array(a, dtype=jnp.int32)

    while newr != 0:
        quotient = r // newr
        t, newt = newt, t - quotient * newt
        r, newr = newr, r - quotient * newr

    if t < 0:
        t = t + N

    return t


def fractional_binary_to_float(sample):
    """Convert an n-bit sample [k1, k2, ..., kn] to a floating point
    value using fractional binary representation,

        k = (k1 / 2) + (k2 / 2 ** 2) + ... + (kn / 2 ** n)
    """
    powers_of_two = 2 ** (jnp.arange(len(sample)) + 1)
    return jnp.sum(sample / powers_of_two)


def as_integer_ratio(f):
    """QJIT compatible conversion of a floating point number to two 64-bit
    integers such that their quotient equals the input to available precision.
    """
    mantissa, exponent = jnp.frexp(f)

    i = 0
    while jnp.logical_and(i < 300, mantissa != jnp.floor(mantissa)):
        mantissa = mantissa * 2.0
        exponent = exponent - 1
        i += 1

    numerator = jnp.asarray(mantissa, dtype=jnp.int64)
    denominator = jnp.asarray(1, dtype=jnp.int64)
    abs_exponent = jnp.abs(exponent)

    if exponent > 0:
        num_to_return, denom_to_return = numerator << abs_exponent, denominator
    else:
        num_to_return, denom_to_return = numerator, denominator << abs_exponent

    return num_to_return, denom_to_return


def phase_to_order(phase, max_denominator):
    """Given some floating-point phase, estimate integers s, r such that s / r =
    phase.  Uses a JIT-compatible re-implementation of Fraction.limit_denominator.
    """
    numerator, denominator = as_integer_ratio(phase)

    order = 0

    if denominator <= max_denominator:
        order = denominator

    else:
        p0, q0, p1, q1 = 0, 1, 1, 0

        a = numerator // denominator
        q2 = q0 + a * q1

        while q2 < max_denominator:
            p0, q0, p1, q1 = p1, q1, p0 + a * p1, q2
            numerator, denominator = denominator, numerator - a * denominator

            a = numerator // denominator
            q2 = q0 + a * q1

        k = (max_denominator - q0) // q1
        bound1 = p0 + k * p1 / q0 + k * q1
        bound2 = p1 / q1

        loop_res = 0

        if jnp.abs(bound2 - phase) <= jnp.abs(bound1 - phase):
            loop_res = q1
        else:
            loop_res = q0 + k * q1

        order = loop_res

    return order


import pennylane as qml
import catalyst
from catalyst import measure

catalyst.autograph_strict_conversion = True


def QFT(wires):
    """The standard QFT, redefined because the PennyLane one uses terminal SWAPs."""
    shifts = jnp.array([2 * jnp.pi * 2**-i for i in range(2, len(wires) + 1)])

    for i in range(len(wires)):
        qml.Hadamard(wires[i])

        for j in range(len(shifts) - i):
            qml.ControlledPhaseShift(shifts[j], wires=[wires[(i + 1) + j], wires[i]])


def fourier_adder_phase_shift(a, wires):
    """Sends QFT(|b>) -> QFT(|b + a>)."""
    n = len(wires)
    a_bits = jnp.unpackbits(jnp.array([a]).view("uint8"), bitorder="little")[:n][::-1]
    powers_of_two = jnp.array([1 / (2**k) for k in range(1, n + 1)])
    phases = jnp.array([jnp.dot(a_bits[k:], powers_of_two[: n - k]) for k in range(n)])

    for i in range(len(wires)):
        if phases[i] != 0:
            qml.PhaseShift(2 * jnp.pi * phases[i], wires=wires[i])


def doubly_controlled_adder(N, a, control_wires, wires, aux_wire):
    """Sends |c>|x>QFT(|b>)|0> -> |c>|x>QFT(|b + c x a) mod N>)|0>."""
    qml.ctrl(fourier_adder_phase_shift, control=control_wires)(a, wires)

    qml.adjoint(fourier_adder_phase_shift)(N, wires)

    qml.adjoint(QFT)(wires)
    qml.CNOT(wires=[wires[0], aux_wire])
    QFT(wires)

    qml.ctrl(fourier_adder_phase_shift, control=aux_wire)(N, wires)

    qml.adjoint(qml.ctrl(fourier_adder_phase_shift, control=control_wires))(a, wires)

    qml.adjoint(QFT)(wires)
    qml.PauliX(wires=wires[0])
    qml.CNOT(wires=[wires[0], aux_wire])
    qml.PauliX(wires=wires[0])
    QFT(wires)

    qml.ctrl(fourier_adder_phase_shift, control=control_wires)(a, wires)


def controlled_ua(N, a, control_wire, target_wires, aux_wires, mult_a_mask, mult_a_inv_mask):
    """Sends |c>|x>|0> to |c>|ax mod N>|0> if c = 1.

    The mask arguments allow for the removal of unnecessary double-controlled additions.
    """
    n = len(target_wires)

    # Apply double-controlled additions where bits of a can be 1.
    for i in range(n):
        if mult_a_mask[n - i - 1] > 0:
            pow_a = (a * (2**i)) % N
            doubly_controlled_adder(
                N, pow_a, [control_wire, target_wires[n - i - 1]], aux_wires[:-1], aux_wires[-1]
            )

    qml.adjoint(QFT)(wires=aux_wires[:-1])

    # Controlled SWAP the target and aux wires; note that the top-most aux wire
    # is only to catch overflow, so we ignore it here.
    for i in range(n):
        qml.CNOT(wires=[aux_wires[i + 1], target_wires[i]])
        qml.Toffoli(wires=[control_wire, target_wires[i], aux_wires[i + 1]])
        qml.CNOT(wires=[aux_wires[i + 1], target_wires[i]])

    # Adjoint of controlled multiplication with the modular inverse of a
    a_mod_inv = modular_inverse(a, N)

    QFT(wires=aux_wires[:-1])

    for i in range(n):
        if mult_a_inv_mask[i] > 0:
            pow_a_inv = (a_mod_inv * (2 ** (n - i - 1))) % N
            qml.adjoint(doubly_controlled_adder)(
                N,
                pow_a_inv,
                [control_wire, target_wires[i]],
                aux_wires[:-1],
                aux_wires[-1],
            )


from jax import random


# ``n_bits`` is a static argument because ``jnp.arange`` does not currently
# support dynamically-shaped arrays when jitted.
@qml.qjit(autograph=True, static_argnums=(3))
def shors_algorithm(N, key, a, n_bits, n_trials):
    # If no explicit a is passed (denoted by a = 0), randomly choose a
    # non-trivial value of a that does not have a common factor with N.
    if a == 0:
        while jnp.gcd(a, N) != 1:
            key, subkey = random.split(key)
            a = random.randint(subkey, (1,), 2, N - 1)[0]

    est_wire = 0
    target_wires = jnp.arange(n_bits) + 1
    aux_wires = jnp.arange(n_bits + 2) + n_bits + 1

    dev = qml.device("lightning.qubit", wires=2 * n_bits + 3, shots=1)

    @qml.qnode(dev)
    def run_qpe():
        meas_results = jnp.zeros((n_bits,), dtype=jnp.int32)
        cumulative_phase = jnp.array(0.0)
        phase_divisors = 2.0 ** jnp.arange(n_bits + 1, 1, -1)

        a_mask = jnp.zeros(n_bits, dtype=jnp.int64)
        a_mask = a_mask.at[0].set(1) + jnp.array(
            jnp.unpackbits(jnp.array([a]).view("uint8"), bitorder="little")[:n_bits]
        )
        a_inv_mask = a_mask

        # Initialize the target register of QPE in |1>
        qml.PauliX(wires=target_wires[-1])

        # The "first" QFT on the auxiliary register; required here because
        # QFT are largely omitted in the modular arithmetic routines due to
        # cancellation between adjacent blocks of the algorithm.
        QFT(wires=aux_wires[:-1])

        # First iteration: add a - 1 using the Fourier adder.
        qml.Hadamard(wires=est_wire)

        QFT(wires=target_wires)
        qml.ctrl(fourier_adder_phase_shift, control=est_wire)(a - 1, target_wires)
        qml.adjoint(QFT)(wires=target_wires)

        # Measure the estimation wire and store the phase
        qml.Hadamard(wires=est_wire)
        meas_results[0] = measure(est_wire, reset=True)
        cumulative_phase = -2 * jnp.pi * jnp.sum(meas_results / jnp.roll(phase_divisors, 1))

        # For subsequent iterations, determine powers of a, and apply controlled
        # U_a when the power is not 1. Unnecessary double-controlled operations
        # are removed, based on values stored in the two "mask" variables.
        powers_cua = jnp.array([repeated_squaring(a, 2**p, N) for p in range(n_bits)])

        loop_bound = n_bits
        if jnp.min(powers_cua) == 1:
            loop_bound = jnp.argmin(powers_cua)

        # The core of the QPE routine
        for pow_a_idx in range(1, loop_bound):
            pow_cua = powers_cua[pow_a_idx]

            if not jnp.all(a_inv_mask):
                for power in range(2**pow_a_idx, 2 ** (pow_a_idx + 1)):
                    next_pow_a = jnp.array([repeated_squaring(a, power, N)])
                    a_inv_mask = a_inv_mask + jnp.array(
                        jnp.unpackbits(next_pow_a.view("uint8"), bitorder="little")[:n_bits]
                    )

            qml.Hadamard(wires=est_wire)

            controlled_ua(N, pow_cua, est_wire, target_wires, aux_wires, a_mask, a_inv_mask)

            a_mask = a_mask + a_inv_mask
            a_inv_mask = jnp.zeros_like(a_inv_mask)

            # Rotate the estimation wire by the accumulated phase, then measure and reset it
            qml.PhaseShift(cumulative_phase, wires=est_wire)
            qml.Hadamard(wires=est_wire)
            meas_results[pow_a_idx] = measure(est_wire, reset=True)
            cumulative_phase = (
                -2 * jnp.pi * jnp.sum(meas_results / jnp.roll(phase_divisors, pow_a_idx + 1))
            )

        # The adjoint partner of the QFT from the beginning, to reset the auxiliary register
        qml.adjoint(QFT)(wires=aux_wires[:-1])

        return meas_results

    # The "classical" part of Shor's algorithm: run QPE, extract a candidate
    # order, then check whether a valid solution is found. We run multiple trials,
    # and return a success probability.
    p, q = jnp.array(0, dtype=jnp.int32), jnp.array(0, dtype=jnp.int32)
    successful_trials = jnp.array(0, dtype=jnp.int32)

    for _ in range(n_trials):
        sample = run_qpe()
        phase = fractional_binary_to_float(sample)
        guess_r = phase_to_order(phase, N)

        # If the guess order is even, we may have a non-trivial square root.
        # If so, try to compute p and q.
        if guess_r % 2 == 0:
            guess_square_root = repeated_squaring(a, guess_r // 2, N)

            if guess_square_root != 1 and guess_square_root != N - 1:
                candidate_p = jnp.gcd(guess_square_root - 1, N).astype(jnp.int32)

                if candidate_p != 1:
                    candidate_q = N // candidate_p
                else:
                    candidate_q = jnp.gcd(guess_square_root + 1, N).astype(jnp.int32)

                    if candidate_q != 1:
                        candidate_p = N // candidate_q

                if candidate_p * candidate_q == N:
                    p, q = candidate_p, candidate_q
                    successful_trials += 1

    return p, q, key, a, successful_trials / n_trials


key = random.PRNGKey(123456789)

N = 15
n_bits = int(jnp.ceil(jnp.log2(N)))

p, q, _, a, success_prob = shors_algorithm(N, key.astype(jnp.uint32), 0, n_bits, 100)

print(f"Found {N} = {p} x {q} (using random a = {a}) with probability {success_prob:.2f}")