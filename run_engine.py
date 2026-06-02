import os
import time
import pickle
import configparser
from pprint import pprint
import tempfile
import glob
import numpy as np
import matplotlib.pyplot as plt
from qiskit_nature.second_q.operators import FermionicOp, PolynomialTensor
from qiskit_nature.second_q.mappers import JordanWignerMapper, QubitMapper
# from qiskit_nature.second_q.transformers import
from qiskit_nature.second_q.hamiltonians import QuadraticHamiltonian
from qiskit.synthesis import LieTrotter, SuzukiTrotter
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import PauliEvolutionGate
import constants as const
from qiskit_aer.primitives import Estimator, EstimatorV2
from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError, phase_damping_error, thermal_relaxation_error
from qiskit_algorithms.utils import algorithm_globals
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit_aer import noise
from dephasing import dephasing_circuit
try:
    from mpi4py import MPI
    mpi_available = True
except ImportError:
    mpi_available = False

print("MPI available: ", mpi_available)

if mpi_available:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
else:
    # Fallback for non-MPI run
    comm = None
    rank = 0
    size = 1


algorithm_globals.random_seed = 111

if rank == 0:
    config = configparser.ConfigParser()
    config.read('config.ini')
else:
    config = None

if mpi_available:
    config = comm.bcast(config, root=0)

class System:
    def __init__(self, **kwargs):

        with_temp = config.getboolean("system", "temp")
        self.with_temp = kwargs.get('with_temp', with_temp)
        self.N_sites = kwargs.get('N_sites', config.getint("system", 'N_sites'))
        self.num_spin_orbitals = 2 * self.N_sites
        self.k = np.linspace(0, config.getfloat("system", 'k_max'), self.N_sites)

        self.band_gap = kwargs.get('band_gap', config.getfloat("band_structure", 'band_gap'))  # in eV
        m_e = config.getfloat("band_structure", 'm_e')
        m_h = config.getfloat("band_structure", 'm_h')
        self.cond_band = 1.0 / m_e * self.k ** 2 + self.band_gap
        self.val_band = -1.0 / m_h * self.k ** 2
        self.label = None
        self.tmp_dir = config["system"]['tmp_dir']

    def set_system(self, val_band, cond_band):

        self.with_temp = False
        self.cond_band = cond_band
        self.val_band = val_band
        self.N_sites = len(cond_band)
        self.num_spin_orbitals = 2 * self.N_sites
        self.band_gap = cond_band[0] - val_band[0]

    def backup(self, time, signal_func, pop_data_cond, pop_data_val, pol_data1, pol_data2):

        if self.with_temp:
            os.makedirs(self.tmp_dir, exist_ok=True)
            directory = self.tmp_dir
            for item in os.listdir(directory):
                if item.endswith(".npy") and item.startswith("pop_data_cond_" + self.label):
                    os.remove(os.path.join(directory, item))
                if item.endswith(".npy") and item.startswith("pop_data_val_" + self.label):
                    os.remove(os.path.join(directory, item))
                if item.endswith(".npy") and item.startswith("pol_data1_" + self.label):
                    os.remove(os.path.join(directory, item))
                if item.endswith(".npy") and item.startswith("pol_data2_" + self.label):
                    os.remove(os.path.join(directory, item))
                if item.endswith(".npy") and item.startswith("time_" + self.label):
                    os.remove(os.path.join(directory, item))
                if item.endswith(".npy") and item.startswith("field_" + self.label):
                    os.remove(os.path.join(directory, item))

            self._backup(pop_data_cond, "pop_data_cond_" + self.label, directory)
            self._backup(pop_data_val, "pop_data_val_" + self.label, directory)
            self._backup(pol_data1, "pol_data1_" + self.label, directory)
            self._backup(pol_data2, "pol_data2_" + self.label, directory)
            self._backup(time, "time_" + self.label, directory)
            self._backup(signal_func(time), "field_" + self.label, directory)

    def _backup(self, data, name, directory):

        timestamp = time.strftime("%Y%m%d_%H%M%S")

        with tempfile.NamedTemporaryFile(mode='w+b',
                                         prefix=name + f"{timestamp}_",
                                         suffix=".npy",
                                         dir=directory,
                                         delete=False) as tmp:
            temp_filename = tmp.name
            print(f"Temporary file created at: {temp_filename}")

            # Save the NumPy array to the temporary file
            np.save(tmp, data)


class Dynamics(System):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # ----------- set up energy bands/levels -----------

        self.with_temp = kwargs.get('with_temp', True)

        # ------------------ set up time -------------------

        self.num_steps = kwargs.get('num_steps', config.getint("dynamics", 'num_steps'))
        self.evolution_time = kwargs.get('evolution_time', config.getfloat("dynamics", 'evolution_time'))


        # ----------------- EM pulse ------------------------

        def probe(tt):
            # t0 = 5
            # sigma = 1.5
            # amp = 1.5
            t0 = config.getfloat("pulse", "t0")
            sigma = config.getfloat("pulse", "sigma")
            amp = config.getfloat("pulse", "amp")

            return amp * np.exp(-((tt - t0) ** 2) / (2 * sigma ** 2))  # * np.cos(1.1 * self.band_gap * tt)

        self.field = kwargs.get('field', probe)
        self.time = np.linspace(0, self.evolution_time, self.num_steps)
        self.dt = (self.time[3] - self.time[2])
        self.dephasing_energy = config.getfloat("dynamics", 'dephasing_energy', fallback=0.025)

        self.init_cb = config.getfloat("dynamics", 'init_cb', fallback=0.0)
        self.init_vb = config.getfloat("dynamics", 'init_vb', fallback=1.0)

    def visualise(self):

        plt.plot(self.time, self.field(self.time))
        plt.show()

        pol_data1 = np.load(glob.glob(os.path.join(self.tmp_dir, "pol_data1_" + self.label + '*'))[0])
        pol_data2 = np.load(glob.glob(os.path.join(self.tmp_dir, "pol_data2_" + self.label + '*'))[0])
        pop_data_cond = np.load(glob.glob(os.path.join(self.tmp_dir, "pop_data_cond_" + self.label + '*'))[0])
        pop_data_val = np.load(glob.glob(os.path.join(self.tmp_dir, "pop_data_val_" + self.label + '*'))[0])

        # compute macroscopic polarization and visualize microscopic polarizations
        if pol_data1.shape[0] == 1:

            pp = pol_data1[0, :] + 1j * pol_data2[0, :]

            plt.plot(np.real(pp))
            plt.plot(np.imag(pp))
            plt.show()

            plt.plot(pol_data1.T)
            plt.plot(pol_data2.T)
            plt.show()

            plt.plot(pop_data_cond.T)
            plt.show()

            plt.plot(pop_data_val.T)
            plt.show()

        else:

            pp = (np.sum(pol_data1, axis=0) + 1j * np.sum(pol_data2, axis=0)) * (self.k[2] - self.k[1]) / (2 * np.pi)

            plt.plot(np.real(pp))
            plt.plot(np.imag(pp))
            plt.show()

            plt.contourf(pol_data1, 50)
            plt.show()
            plt.contourf(pol_data2, 50)
            plt.show()

            plt.contourf(pop_data_cond, 50)
            plt.show()
            plt.contourf(pop_data_val, 50)
            plt.show()

        freqs, pol = self.make_spectra(self.time, pol_data1, pol_data2, self.field(self.time), k=self.k)

        plt.plot(freqs, np.real(pol))
        plt.plot(freqs, np.imag(pol))
        plt.xlim([-0.05, 2 * self.band_gap])
        plt.ylim([-0.5*np.max(np.abs(pol[freqs < 2])), 1.1*np.max(np.abs(pol[freqs < 2]))])
        plt.show()

    @staticmethod
    def make_spectra(time, pol_data1, pol_data2, field, k=None):

        if pol_data1.shape[0] == 1:
            pp = pol_data1[0, :] + 1j * pol_data2[0, :]
        else:
            pp = (np.sum(pol_data1, axis=0) + 1j * np.sum(pol_data2, axis=0)) * (k[2] - k[1]) / (2 * np.pi)

        # use padding for better frequency resolution
        pad = np.zeros(len(time) + 50000, dtype=complex)
        pad[:len(time)] = field

        # Fourier transform of optical signal
        E_f = np.fft.ifft(pad)
        # E_f = np.fft.fft(self.field(self.time))

        # use padding for better frequency resolution
        pad = np.zeros(len(pp) + 50000, dtype=complex)
        pad[:len(time)] = pp
        # Fourier transform of the polarization function
        pp_f = -np.fft.ifft(pad)

        # array of frequencies
        dt = time[2] - time[1]
        freqs = np.fft.fftfreq(len(pp_f), dt) * 2 * np.pi

        # tackle the causality (get rid of negative frequencies in a proper way)
        pp_ff = pp_f[freqs >= 0]
        pp_ff[1:] = pp_ff[1:] + np.conj(pp_f[freqs < 0])[::-1][:-1]
        E_ff = E_f[freqs >= 0]
        freqs = freqs[freqs >= 0]

        pol = pp_ff / E_ff

        return freqs, pol


class Dynamics_quantum(Dynamics):

    def __init__(self, **kwargs):

        super().__init__(**kwargs)

        self.t = Parameter("t")
        self.label = 'quant_'

        # --------- set up time propagation method -----------

        # self.product_formula = LieTrotter()
        self.product_formula = SuzukiTrotter(order=1)
        self.mapper = JordanWignerMapper()
        self.backend = kwargs.get('backend_mode', 'local')

        # ------------------ dephasing -----------------------

        dephasing_time = 1.0 / self.dephasing_energy
        self.dephasing_duration = self.dt
        self.dephasing = dephasing_time * self.dt / self.dephasing_duration

    def _setup_hamiltonian(self):

        # --------------- set up Hamiltonian ---------------

        hermitian_part = np.diag(np.append(np.flip(self.val_band), self.cond_band))
        antisymmetric_part = np.zeros(hermitian_part.shape)
        hamiltonian = QuadraticHamiltonian(hermitian_part=hermitian_part, antisymmetric_part=antisymmetric_part)

        # convert it to a FermionicOp and print it
        hamiltonian_ferm = hamiltonian.second_q_op()

        # --------------- add semi-classical EM interaction -----------

        for j in range(self.N_sites):
            hamiltonian_ferm += FermionicOp({"+_{} -_{}".format(j, 2 * self.N_sites - j - 1): -self.field(self.t),
                                             "+_{} -_{}".format(2 * self.N_sites - j - 1, j): -self.field(self.t)},
                                            num_spin_orbitals=self.num_spin_orbitals)

        # --------------- add semi-classical EM interaction -----------

        hamiltonian_converted = self.mapper.map(hamiltonian_ferm, register_length=self.num_spin_orbitals)
        print(hamiltonian_converted)

        # --------------- set up initial conditions ---------------

        initial_circuit = QuantumCircuit(self.num_spin_orbitals)
        # initial_circuit.prepare_state(self.N_sites * '0' + self.N_sites * '0')
        # initial_circuit.prepare_state('00')

        # --------------- exponentiate Hamiltonian ----------------

        evol_gate = PauliEvolutionGate(hamiltonian_converted.assign_parameters({self.t: self.dt}),
                                       time=self.dt,
                                       synthesis=self.product_formula)
        initial_circuit.append(evol_gate, initial_circuit.qubits)

        if rank == 0:
            print(
                f"""
            Trotter step with Lie-Trotter
            -----------------------------
            Depth: {initial_circuit.decompose(reps=3).depth()}
            Gate count: {len(initial_circuit.decompose(reps=3))}
            Nonlocal gate count: {initial_circuit.decompose(reps=3).num_nonlocal_gates()}
            Gate breakdown: {", ".join([f"{k.upper()}: {v}" for k, v in initial_circuit.decompose(reps=3).count_ops().items()])}
            """
            )

        return hamiltonian_converted

    def _polarization(self):

        # --------------- prepare observables --------------------

        pols1 = []
        pols2 = []

        for j in range(self.N_sites):
            # polarization1 = FermionicOp({"+_{} -_{}".format(j, 2 * N_sites - j - 1): 1,
            #                              "+_{} -_{}".format(2 * N_sites - j - 1, j): 1},
            #                             num_spin_orbitals=num_spin_orbitals)

            polarization1 = FermionicOp({"+_{} -_{}".format(j, 2 * self.N_sites - j - 1): 1,
                                         "+_{} -_{}".format(2 * self.N_sites - j - 1, j): 1},
                                        num_spin_orbitals=self.num_spin_orbitals)

            pols1.append(self.mapper.map(polarization1, register_length=self.num_spin_orbitals))

            polarization1 = FermionicOp({"+_{} -_{}".format(j, 2 * self.N_sites - j - 1): 1j,
                                         "+_{} -_{}".format(2 * self.N_sites - j - 1, j): -1j},
                                        num_spin_orbitals=self.num_spin_orbitals)

            pols2.append(self.mapper.map(polarization1, register_length=self.num_spin_orbitals))

        return pols1, pols2

    def _population(self):

        pops = []

        for j in range(2 * self.N_sites):
            pop1 = FermionicOp({"+_{} -_{}".format(j, j): 1},
                               num_spin_orbitals=self.num_spin_orbitals)
            pops.append(self.mapper.map(pop1, register_length=self.num_spin_orbitals))

        return pops

    def _generate_steps(self, hamiltonian_converted):

        initial_circuit = QuantumCircuit(self.num_spin_orbitals)
        initial_circuit.prepare_state(self.N_sites * str(int(self.init_vb)) + self.N_sites * str(int(self.init_cb)) )
        # Initiate the circuit
        evolved_state = QuantumCircuit(initial_circuit.num_qubits)
        # Start from the initial spin configuration
        evolved_state.append(initial_circuit, evolved_state.qubits)
        evolved_state_list = [evolved_state]

        time = 0

        # Start time evolution
        for j in range(self.num_steps):
            print(j)
            time += self.dt
            # Expand the circuit to describe delta-t
            single_step_evolution_gates_lt = PauliEvolutionGate(hamiltonian_converted.assign_parameters({self.t: time}),
                                                                time=self.dt,
                                                                synthesis=self.product_formula)

            evolved_state.append(single_step_evolution_gates_lt, evolved_state.qubits)

            # evolved_state.global_phase = np.pi

            for qubit in range(evolved_state.num_qubits):
                evolved_state.delay(100, qubit)

            evolved_state_list.append(evolved_state.copy())

        return evolved_state_list

    def _prepare_estimator(self, shots=10000):

        noise_model = noise.NoiseModel()

        # device noise model
        if config.getboolean("noise_model", "noise") and (config["noise_model"]["model"] == 'frombackend'):
            from qiskit_ibm_runtime.fake_provider import FakeSydneyV2, FakeManilaV2, FakeMarrakesh, FakeSherbrooke, FakeKyoto, FakeBrisbane
            #backend = FakeSydneyV2()
            #backend = FakeManilaV2()
            #backend = FakeMarrakesh()
            backend = FakeSherbrooke()
            backend = FakeKyoto()
            # backend = FakeBrisbane()
            noise_model = NoiseModel.from_backend(backend)

            #from qiskit_ibm_runtime import QiskitRuntimeService

            #service = QiskitRuntimeService()
            #backend = service.backend("ibm_fez")
            #noise_model = NoiseModel.from_backend(backend)

        if config.getboolean("noise_model", "noise") and config["noise_model"]["model"] == 'bitflip':
            p = 0.003
            my_bitflip = noise.pauli_error([('X', p), ('I', 1 - p)])
            noise_model.add_all_qubit_quantum_error(my_bitflip, ['u1', 'u2', 'u3', 'uc', 'UCPauliRotGate'])

        if config.getboolean("noise_model", "noise") and config["noise_model"]["model"] == 'depol':
            p = 0.05
            error_1q = depolarizing_error(p, 1)
            error_2q = depolarizing_error(p, 2)
            noise_model.add_all_qubit_quantum_error(error_1q, ["u1", "u2", "u3"])
            noise_model.add_all_qubit_quantum_error(error_2q, ["cx"])

        # dephasing model
        dephasing = dephasing_circuit(self.dephasing, self.dephasing_duration)
        noise_model.add_all_qubit_quantum_error(dephasing, ['delay'])

        estimator = EstimatorV2(options=dict(run_options={"optimization_level": config.getint("run_options", "optimization_level"),
                                                          "shots": config.getint("run_options", "shots"),
                                                          "blocking_enable": config.getboolean("run_options", "blocking_enable"),
                                                          "blocking_qubits": config.getint("run_options", "blocking_qubits"),
                                                          "cuStateVec_enable": config.getboolean("run_options", "cuStateVec_enable")
                                                          },
                                             backend_options={"noise_model": noise_model,
                                                              "method": config["backend_options"]['method'],
                                                              "device": config["backend_options"]['device'],
                                                              "batched_shots_gpu": config["backend_options"]["batched_shots_gpu"]
                                                              }
                                             )
                                )

        if rank == 0:
            print(
                f"""
            Backend setup
            -----------------------------
            optimization_level: {config.getint("run_options", "optimization_level")}
            shots: {config.getint("run_options", "shots")}
            blocking_enable: {config.getboolean("run_options", "blocking_enable")}
            blocking_qubits: {config.getint("run_options", "blocking_qubits")}
            cuStateVec_enable: {config.getboolean("run_options", "cuStateVec_enable")}
            method: {config["backend_options"]['method']}
            device: {config["backend_options"]['device']}
            batched_shots_gpu: {config["backend_options"]["batched_shots_gpu"]}
            """
            )


        return estimator

    def prerun(self, save=True):

        hamiltonian_converted = self._setup_hamiltonian()
        evolved_state_list = self._generate_steps(hamiltonian_converted)
        pols1, pols2 = self._polarization()
        pops = self._population()

        time = 0
        pubs = []

        for j in range(self.num_steps):
            print(j)
            time += self.dt

            obs = [hamiltonian_converted.assign_parameters({self.t: time}), *pops, *pols1, *pols2]

            pass_manager = generate_preset_pass_manager(3, AerSimulator())
            isa_circuit = pass_manager.run(evolved_state_list[j + 1])

            pubs.append([[isa_circuit, obs[jj]] for jj in range(len(obs))])

        if save:
            with open("pubs.pkl", "wb") as f:  # Open in binary write mode
                pickle.dump(pubs, f)

        return pubs

    def run_with_prerun(self):

        pubs = self.prerun()

        energy_list = []
        pop_data_cond = np.empty((self.N_sites, self.num_steps))
        pop_data_val = np.empty((self.N_sites, self.num_steps))
        pol_data1 = np.empty((self.N_sites, self.num_steps))
        pol_data2 = np.empty((self.N_sites, self.num_steps))
        estimator = self._prepare_estimator()

        for j in range(self.num_steps):
            print(j)
            job = estimator.run(pubs=pubs[j])

            energy_list.append(job.result()[0].data.evs)
            pop_data_cond[:, j] = np.array(
                [float(item.data.evs) for item in job.result()[1:self.num_spin_orbitals + 1]])[:self.N_sites]
            pop_data_val[:, j] = np.array(
                [float(item.data.evs) for item in job.result()[1:self.num_spin_orbitals + 1]])[self.N_sites:]
            pol_data1[:, j] = np.array([float(item.data.evs) for item in job.result()[
                                                                         self.num_spin_orbitals + 1:self.num_spin_orbitals + 1 + self.N_sites]])
            pol_data2[:, j] = np.array([float(item.data.evs) for item in job.result()[
                                                                         self.num_spin_orbitals + 1 + self.N_sites:self.num_spin_orbitals + 1 + 2 * self.N_sites]])

        self.backup(self.time, self.field, pop_data_cond, pop_data_val, pol_data1, pol_data2)

        return pop_data_cond, pop_data_val, pol_data1, pol_data2


    def run(self):

        hamiltonian_converted = self._setup_hamiltonian()
        evolved_state_list = self._generate_steps(hamiltonian_converted)
        pols1, pols2 = self._polarization()
        pops = self._population()

        estimator = self._prepare_estimator()

        energy_list = []
        pop_data_cond = np.empty((self.N_sites, self.num_steps))
        pop_data_val = np.empty((self.N_sites, self.num_steps))
        pol_data1 = np.empty((len(pols1), self.num_steps))
        pol_data2 = np.empty((len(pols2), self.num_steps))

        time = 0

        #for j in range(self.num_steps):
        for j in range(self.num_steps-1, -1, -1):
            print(j)
            time += self.dt

            obs = [hamiltonian_converted.assign_parameters({self.t: time}), *pops, *pols1, *pols2]

            pass_manager = generate_preset_pass_manager(3, AerSimulator())
            isa_circuit = pass_manager.run(evolved_state_list[j + 1])

            pubs = [[isa_circuit, obs[jj]] for jj in range(len(obs))]
            job = estimator.run(pubs=pubs)

            # evs = job.result().values
            # energy_list.append(evs[0])
            # pop_data[:, j] = evs[1:self.num_spin_orbitals + 1]
            # pol_data1[:, j] = evs[self.num_spin_orbitals + 1:self.num_spin_orbitals + 1 + self.N_sites]
            # pol_data2[:, j] = evs[self.num_spin_orbitals + 1 + self.N_sites:self.num_spin_orbitals + 1 + 2 * self.N_sites]
            
            if rank == 0:
                pprint(job.result()[0].metadata)

            energy_list.append(job.result()[0].data.evs)
            pop_data_cond[:, j] = np.array(
                [float(item.data.evs) for item in job.result()[1:self.num_spin_orbitals + 1]])[:self.N_sites]
            pop_data_val[:, j] = np.array(
                [float(item.data.evs) for item in job.result()[1:self.num_spin_orbitals + 1]])[self.N_sites:]
            pol_data1[:, j] = np.array([float(item.data.evs) for item in job.result()[
                                                                         self.num_spin_orbitals + 1:self.num_spin_orbitals + 1 + self.N_sites]])
            pol_data2[:, j] = np.array([float(item.data.evs) for item in job.result()[
                                                                         self.num_spin_orbitals + 1 + self.N_sites:self.num_spin_orbitals + 1 + 2 * self.N_sites]])

        self.backup(self.time, self.field, pop_data_cond, pop_data_val, pol_data1, pol_data2)

        return pop_data_cond, pop_data_val, pol_data1, pol_data2


class Dynamics_quantum_decoupled(Dynamics):

    def __init__(self, **kwargs):

        super().__init__(**kwargs)
        self.label = 'quant_dec_'
        self.N_systems = self.N_sites
        self.dyns = []

        for j in range(self.N_systems):
            dyn = Dynamics_quantum(**kwargs)
            dyn.set_system(np.array([self.val_band[j]]), np.array([self.cond_band[j]]))
            self.dyns.append(dyn)

    def run(self):


        pop_data_cond = np.zeros((self.N_systems, self.num_steps))
        pop_data_val = np.zeros((self.N_systems, self.num_steps))
        pol_data1 = np.zeros((self.N_systems, self.num_steps))
        pol_data2 = np.zeros((self.N_systems, self.num_steps))

        # Determine which indices this rank will process
        indices = list(range(rank, self.N_systems, size))

        for j in indices:

            pop_data_cond_tmp, pop_data_val_tmp, pol_data1_tmp, pol_data2_tmp = self.dyns[j].run_with_prerun()

            pop_data_cond[j, :] = pop_data_cond_tmp
            pop_data_val[j, :] = pop_data_val_tmp
            pol_data1[j, :] = pol_data1_tmp
            pol_data2[j, :] = pol_data2_tmp

        # collect the partial results and add to the total sum
        if mpi_available:
            comm.Allreduce(pop_data_cond, pop_data_cond)
            comm.Allreduce(pop_data_val, pop_data_val)
            comm.Allreduce(pol_data1, pol_data1)
            comm.Allreduce(pol_data2, pol_data2)

        if rank == 0:
            self.backup(self.time, self.field, pop_data_cond, pop_data_val, pol_data1, pol_data2)

        return pop_data_cond, pop_data_val, pol_data1, pol_data2


class Dynamics_classical(Dynamics):

    def __init__(self, **kwargs):

        # ----------- set up energy bands/levels -----------

        super().__init__(**kwargs)
        self.dephasing = self.dephasing_energy * 2
        self.label = 'class_'

    def _rhs(self, t, y):

        ans = np.zeros((3 * self.N_sites,), dtype=complex)

        for j in range(self.N_sites):
            ans[j * 3] = (1j * (self.cond_band[j] - self.val_band[j]) * y[j * 3] +
                          1j * 2 * self.field(t) * (y[j * 3 + 2] - y[j * 3 + 1]) -
                          self.dephasing * y[j * 3])  # polarization
            ans[j * 3 + 1] = -np.imag(np.conj(y[j * 3]) * self.field(t))  # population cond. band
            ans[j * 3 + 2] = np.imag(np.conj(y[j * 3]) * self.field(t))  # population val. band

        return ans

    def _initial_conditions(self):

        ans = np.zeros(3 * self.N_sites)

        for j in range(self.N_sites):
            ans[j * 3 + 1] = self.init_cb  # population cond. band
            ans[j * 3 + 2] = self.init_vb  # population val. band

        return ans

    def _rk4(self, x, y, dx):
        # 4th-order explicit Runge-Kutta

        dy1 = dx * self._rhs(x, y)
        dy2 = dx * self._rhs(x + 0.5 * dx, y + 0.5 * dy1)
        dy3 = dx * self._rhs(x + 0.5 * dx, y + 0.5 * dy2)
        dy4 = dx * self._rhs(x + dx, y + dy3)

        return x + dx, y + (dy1 + 2.0 * dy2 + 2.0 * dy3 + dy4) / 6.0

    def run(self):

        pop_data_cond = np.empty((self.N_sites, self.num_steps))
        pop_data_val = np.empty((self.N_sites, self.num_steps))
        pol_data1 = np.empty((self.N_sites, self.num_steps))
        pol_data2 = np.empty((self.N_sites, self.num_steps))

        time = 0
        y = self._initial_conditions()

        for j in range(self.num_steps):
            print(j)
            time, y = self._rk4(time, y, self.dt)
            data = np.reshape(y, (self.N_sites, -1))

            pop_data_cond[:, j] = data[:, 2]
            pop_data_val[:, j] = data[:, 1]
            pol_data1[:, j] = np.real(data[:, 0])
            pol_data2[:, j] = np.imag(data[:, 0])

        if rank == 0:
            self.backup(self.time, self.field, pop_data_cond, pop_data_val, pol_data1, pol_data2)

        return pop_data_cond, pop_data_val, pol_data1, pol_data2


if __name__ == "__main__":
    def field(tt):
        t0 = 10
        sigma = 1.0
        return 0.0001 * np.exp(-((tt - t0) ** 2) / (2 * sigma ** 2))# * np.cos(1.4*tt)


    #if rank == 0:
        #dyn = Dynamics_quantum()
        #dyn.run()
        #dyn.visualise()

    dyn = Dynamics_quantum_decoupled()
    dyn.run()
    # dyn.visualise()

    if mpi_available:
        comm.Barrier()
        MPI.Finalize()
    
    if rank == 0:
        dyn = Dynamics_classical()
        dyn.run()
#        dyn.visualise()

