import numpy as np
from qiskit_aer.noise.errors.quantum_error import QuantumError
from qiskit_aer.noise.noiseerror import NoiseError
from qiskit.circuit import Reset
from qiskit.circuit.library.standard_gates import IGate, ZGate


# pylint: disable=invalid-name
def dephasing_circuit(t2, time):
    r"""
    Return a dephasing quantum error channel.

    Args:
        t2 (double): the :math:`T_2` relaxation time constant.
        time (double): the gate time for relaxation error.

    Returns:
        QuantumError: a quantum error object for a noise model.

    Raises:
        NoiseError: If noise parameters are invalid.
    """

    if time < 0:
        raise NoiseError("Invalid gate_time ({} < 0)".format(time))
    if t2 <= 0:
        raise NoiseError("Invalid T_2 relaxation time parameter: T_2 <= 0.")

    rate2 = 1 / t2

    circuits = [
        [(IGate(), [0])],
        [(ZGate(), [0])],
    ]

    # circuits = [
    #     [(IGate(), [0])],
    #     [(Reset(), [0])],
    # ]

    p_z = (1 - np.exp(-time * rate2)) / 2
    p_identity = 1 - p_z
    probabilities = [p_identity, p_z]

    return QuantumError(zip(circuits, probabilities))
