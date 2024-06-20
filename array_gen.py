from mpi4py import MPI
import meep as mp
import numpy as np
import os

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Define function to run a Meep simulation
def run_meep_simulation(wavelength, column_length):
    # Simulation parameters
    resolution = 40
    cell_size = mp.Vector3(3, 3, 3.25)
    pml_layers = [mp.PML(1)]

    # Geometry: Single SiO2 column
    geometry = [mp.Block(mp.Vector3(2, 2, column_length), center=mp.Vector3(0, 0, 0), material=mp.Medium(index=1.4576))]

    # Source
    sources = [mp.Source(mp.ContinuousSource(frequency=1/wavelength), component=mp.Ez, center=mp.Vector3(0, 0, -1.5), size=mp.Vector3(2, 2, 0))]

    # Simulation
    sim = mp.Simulation(cell_size=cell_size,
                        boundary_layers=pml_layers,
                        geometry=geometry,
                        sources=sources,
                        resolution=resolution)

    # Run simulation
    sim.run(until=100)

    # Extract field data
    z_slice = column_length / 2 + 0.5
    field_data = sim.get_array(center=mp.Vector3(0, 0, z_slice), size=mp.Vector3(2, 2, 0), component=mp.Ez)

    # Save data to file
    filename = f'arrays/{wavelength}_{column_length}.npz'
    np.savez_compressed(filename, field_data)

# Main program
if __name__ == "__main__":
    try:
        # Ensure arrays folder exists
        os.makedirs('arrays', exist_ok=True)

        # Run simulations
        wavelengths = [605, 625]
        for wavelength in wavelengths:
            for i in range(rank, 51, size):  # Run 51 simulations, each process handles one subset
                column_length = 0.5 + (i / 20)
                run_meep_simulation(wavelength, column_length)

        # Synchronize MPI processes
        comm.Barrier()

    finally:
        # Finalize MPI
        MPI.Finalize()

csv_filename = 'differ_count.csv'
with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(differ_count)
