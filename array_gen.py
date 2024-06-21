from mpi4py import MPI
import meep as mp
import numpy as np
import os
    # Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
mp.divide_parallel_processes(4)
# Define function to run a Meep simulation
def run_meep_simulation(wavelength, column_length):
    # Simulation parameters
    resolution = 40
    cell_size_z = 2.75 + column_length
    cell_size = mp.Vector3(3.1, 3.1, 3.25)
    pml_layers = [mp.PML(1)]
    frequency = 1 / wavelength
    z_slice = column_length / 2 + 0.5
    # Geometry: Single SiO2 column
    geometry = [mp.Block(mp.Vector3(2, 2, column_length), center=mp.Vector3(0, 0, 0), material=mp.Medium(index=1.4576))]
    # Source
    sources = [mp.Source(mp.ContinuousSource(frequency=frequency), component=mp.Ez, center=mp.Vector3(0, 0, (-(cell_size_z/2) + 1.01)), size=mp.Vector3(2, 2, 0))]
    # Define C4 symmetry
    symmetries = [mp.Mirror(mp.X, phase=+1), mp.Mirror(mp.Y, phase=+1)]
    # Simulation
    sim = mp.Simulation(cell_size=cell_size,
                        boundary_layers=pml_layers,
                        geometry=geometry,
                        sources=sources,
                        resolution=resolution,
                        symmetries=symmetries)

    incident_region = mp.FluxRegion(center=mp.Vector3(0, 0, -(column_length / 2) ), size=mp.Vector3(2, 2, 0))
    transmitted_region = mp.FluxRegion(center=mp.Vector3(0, 0, z_slice), size=mp.Vector3(2, 2, 0))

    incident_flux = sim.add_flux(frequency, 0, 1, incident_region)
    transmitted_flux = sim.add_flux(frequency, 0, 1, transmitted_region)
    # Run simulation
    sim.run(until=100)

    # Calculate intensity profile
    intensity_profile = sim.get_array(center=mp.Vector3(0, 0, z_slice), size=mp.Vector3(2, 2, 0), component=mp.Ez)**2

    # Calculate flux spectra
    incident_flux_spectra = mp.get_fluxes(incident_flux)
    transmitted_flux_spectra = mp.get_fluxes(transmitted_flux)
    # Transmission coefficient
    if incident_flux_spectra[0] != 0:
        transmission_coefficient = transmitted_flux_spectra[0] / incident_flux_spectra[0]  # Normalize to incident power
    else:
        transmission_coefficient = 0  # To avoid division by zero

    # Print or save data (you can adjust how you want to save this data)
    filename_intensity = f'output/intensity_{wavelength}_{column_length}.txt'
    filename_flux = f'output/transflux_{wavelength}_{column_length}.txt'
    filename_transmission = f'output/transmission_{wavelength}_{column_length}.txt'

    np.savetxt(filename_intensity, intensity_profile)
    np.savetxt(filename_flux, transmitted_flux_spectra)
    np.savetxt(filename_transmission, [transmission_coefficient])

# Main program
if __name__ == "__main__":


    try:
        # Ensure output folder exists
        os.makedirs('output', exist_ok=True)
        sim_count = 51
        # Run simulations
        wavelengths = [605, 625] 
        for wavelength in wavelengths:
            # Inside your main simulation loop
            for i in range(rank, sim_count, size):
                print(f"Process {rank} handling simulation {i}")
                column_length = 0.5 + (i / 20)
                run_meep_simulation(wavelength, column_length)

                
        comm.Barrier()
        # Synchronize MPI processes
        

    finally:
        # Finalize MPI
        MPI.Finalize()
