import run_engine
import argparse


if run_engine.rank == 0:
    parser = argparse.ArgumentParser(description="A simple script to process text.")
    parser.add_argument("input_file", help="Path to the input text file", default='config.ini')
    #parser.add_argument("-o", "--output", help="Path to the output file", default="output.txt")
    args = parser.parse_args()
    run_engine.config.read(args.input_file)

if run_engine.comm is not None:
    run_engine.config = comm.bcast(run_engine.config, root=0) 

# dyn = run_engine.Dynamics_quantum()
# dyn.run()
# dyn.visualise()

dyn = run_engine.Dynamics_quantum_decoupled()
dyn.run()
# dyn.visualise()

# dyn = run_engine.Dynamics_classical()
# dyn.run()
# dyn.visualise()


