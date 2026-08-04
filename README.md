# **Quantum Simulations of Semiconductor Spectroscopy**

This repository provides a framework for simulating spectroscopic experiments of semiconductor materials using digital quantum computers.
The framework combines Qiskit quantum simulations with MPI-enabled parallel execution and efficient classical solvers, making it well suited for high-performance computing (HPC) environments.

## **Features**

### **1. Classical Dynamics**
### **2. Fully Quantum Dynamics**
### **3. Quantum Dynamics with Decoupled Subsystems**
### **4. Data Output & Visualization**

## **Dependencies**

* Python 3.9+
* Qiskit
* All libraries listed in `requirements.txt`


## **Installation**

Before running the application, ensure that all required Python packages are installed.

1. Clone the repository:

```bash
git clone https://github.com/freude/sbe_qc.git
cd sbe_qc
````

2. Create and activate a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
````

3. Install the dependencies:

```bash
pip install -r requirements.txt
```

4. Optional: MPI Support to use parallel execution

```bash
sudo apt install openmpi-bin openmpi-common libopenmpi-dev
pip install mpi4py
```

## **Running the Application**

To start the demo:

```bash
python run_engine.py
```

This launches the main application.

## **Settings, configuration file: `config.ini`**

This file defines all runtime parameters for the simulation, including system size, band-structure properties, dynamical evolution, external pulse characteristics, backend execution options, and noise modeling. The configuration is organized into logical sections to simplify experimentation and reproducibility.

---

### `[system]`
General system-level parameters.

- **`N_sites`**  
  Number of lattice sites (or discretization points) used to represent the system.

- **`k_max`**  
  Maximum wave vector value defining the extent of reciprocal-space sampling.

- **`temp`**  
  Enables the use of a temporary working directory during execution (`True` or `False`).

- **`tmp_dir`**  
  Path to the temporary directory used for intermediate files and data.

---

### `[band_structure]`
Parameters defining the electronic band structure of the material.

- **`band_gap`**  
  Energy gap between the valence and conduction bands.

- **`m_e`**  
  Effective mass of electrons in the conduction band.

- **`m_h`**  
  Effective mass of holes in the valence band.

---

### `[dynamics]`
Time-evolution and initial-state parameters.

- **`num_steps`**  
  Number of discrete time steps used in the dynamical simulation.

- **`evolution_time`**  
  Total duration of the time evolution.

- **`dephasing_energy`**  
  Energy scale associated with dephasing processes.

- **`init_cb`**  
  Initial population of the conduction band.

- **`init_vb`**  
  Initial population of the valence band.

---

### `[pulse]`
External driving pulse parameters.

- **`t0`**  
  Center time of the pulse.

- **`sigma`**  
  Temporal width of the pulse.

- **`amp`**  
  Pulse amplitude.

---

### `[run_options]`
Execution and performance-related settings.

- **`shots`**  
  Number of measurement shots used for sampling-based simulations.

- **`blocking_enable`**  
  Enables qubit blocking to reduce circuit width.

- **`blocking_qubits`**  
  Number of qubits per block when blocking is enabled.

- **`cuStateVec_enable`**  
  Enables NVIDIA cuStateVec acceleration when available.

- **`optimization_level`**  
  Circuit optimization level applied by the compiler/backend.

---

### `[backend_options]`
Quantum backend configuration.

- **`method`**  
  Simulation method used by the backend.  
  Available options:
  - `statevector`
  - `matrix_product_state`

- **`device`**  
  Execution device.  
  Available options:
  - `CPU`
  - `GPU`

- **`batched_shots_gpu`**  
  Enables batched execution of shots on GPU backends.

---

### `[noise_model]`
Noise modeling configuration.

- **`noise`**  
  Enables or disables noise in the simulation.

- **`model`**  
  Noise model selection.  
  Available options:
  - `bitflip`
  - `frombackend` (uses the backend’s native noise model)

---

### Notes

- All parameters are read at runtime and can be modified to explore different physical regimes or backend configurations.
- For reproducibility, keep a copy of the exact `config.ini` file used for each simulation run.
- Backend- and noise-related options may depend on the availability of specific simulators or hardware.



## **License**

This project is licensed under the MIT License.
See the LICENSE file for details.

