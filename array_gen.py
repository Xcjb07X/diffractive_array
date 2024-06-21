import meep as mp
import numpy as np
import os

# Initialize MPI if not already initialized
mp.init()
# Define function to run a Meep simulation
def run_meep_simulation(wavelength, column_length):
    # Simulation parameters
    resolution = 20
    cell_size_z = 2.75 + column_length
    cell_size = mp.Vector3(3.1, 3.1, cell_size_z)
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

    incident_region = mp.FluxRegion(center=mp.Vector3(0, 0, -(column_length / 2)), size=mp.Vector3(2, 2, 0))
    transmitted_region = mp.FluxRegion(center=mp.Vector3(0, 0, z_slice), size=mp.Vector3(2, 2, 0))

    incident_flux = sim.add_flux(frequency, 0, 1, incident_region)
    transmitted_flux = sim.add_flux(frequency, 0, 1, transmitted_region)

    # Run simulation
    sim.run(until=100)

    # Calculate flux spectra
    incident_flux_spectra = mp.get_fluxes(incident_flux)
    transmitted_flux_spectra = mp.get_fluxes(transmitted_flux)

    # Transmission coefficient
    if incident_flux_spectra[0] != 0:
        transmission_coefficient = transmitted_flux_spectra[0] / incident_flux_spectra[0]
    else:
        transmission_coefficient = 0

    # Save data
    filename_flux = f'output/transflux_{wavelength}_{column_length:.2f}.txt'
    filename_transmission = f'output/transmission_{wavelength}_{column_length:.2f}.txt'

    np.savetxt(filename_flux, transmitted_flux_spectra)
    np.savetxt(filename_transmission, [transmission_coefficient])

    print(f"Finished simulation: wavelength={wavelength}, column_length={column_length}")

# Main program
if __name__ == "__main__":
    mp.verbosity(1)  # Increase verbosity to see more Meep output

    # Ensure output folder exists
    if mp.am_master():
        os.makedirs('output', exist_ok=True)

    # Generate all parameter combinations
    wavelengths = [605, 625]
    column_lengths = [0.5 + (i / 20) for i in range(51)]
    all_params = [(w, cl) for w in wavelengths for cl in column_lengths]

    # Divide work across processes
    local_params = mp.divide_parallel_processes(all_params)

    if mp.am_master():
        print(f"Total simulations: {len(all_params)}")
        print(f"Number of Meep processes: {mp.count_processors()}")

    # Run simulations
    for wavelength, column_length in local_params:
        print(f"Meep process {mp.my_rank()} starting simulation: wavelength={wavelength}, column_length={column_length}")
        run_meep_simulation(wavelength, column_length)

    # Synchronize processes
    mp.all_wait()

    if mp.am_master():
        print("All simulations completed")

    # Finalize MPI
    mp.finalize()
