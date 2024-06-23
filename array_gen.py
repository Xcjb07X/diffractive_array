import meep as mp
import numpy as np
import os

def run_meep_simulation(wavelength, column_length):
    # Simulation parameters
    resolution = 50
    cell_size_z = 2.75 + column_length
    cell_size = mp.Vector3(3.1, 3.1, cell_size_z)
    pml_layers = [mp.PML(1.0)]
    frequency = 1 / wavelength
    z_slice = column_length / 2 + 0.5

    # Add a small amount of loss to the material
    material = mp.Medium(index=1.4576, D_conductivity=1e-6)

    # Geometry: Single SiO2 column
    geometry = [mp.Block(mp.Vector3(2, 2, column_length), center=mp.Vector3(0, 0, 0), material=material)]

    # Source
    sources = [mp.Source(mp.ContinuousSource(frequency=frequency), component=mp.Ez, center=mp.Vector3(0, 0, (-(cell_size_z/2) + 1.01)), size=mp.Vector3(2, 2, 0))]

    # C4 symmetry
    symmetries = [mp.Mirror(mp.X, phase=+1),
                  mp.Mirror(mp.Y, phase=+1)]

    # Simulation
    sim = mp.Simulation(cell_size=cell_size,
                        boundary_layers=pml_layers,
                        geometry=geometry,
                        sources=sources,
                        resolution=resolution,
                        
                        symmetries=symmetries)

    # Only simulate 1/4 of the flux regions due to symmetry
    flux_size = mp.Vector3(1, 1, 0)
    incident_region = mp.FluxRegion(center=mp.Vector3(0.5, 0.5, -(column_length / 2)), size=flux_size)
    transmitted_region = mp.FluxRegion(center=mp.Vector3(0.5, 0.5, z_slice), size=flux_size)

    incident_flux = sim.add_flux(frequency, 0, 1, incident_region)
    transmitted_flux = sim.add_flux(frequency, 0, 1, transmitted_region)

    # Run simulation
    sim.run(until=100)

    intensity_profile = sim.get_array(center=mp.Vector3(0.5, 0.5, z_slice), size=flux_size, component=mp.Ez)**2

    # Calculate flux spectra
    incident_flux_spectra = mp.get_fluxes(incident_flux)
    transmitted_flux_spectra = mp.get_fluxes(transmitted_flux)

    # Transmission coefficient (multiply by 4 due to symmetry)
    if incident_flux_spectra[0] != 0:
        transmission_coefficient = 4 * transmitted_flux_spectra[0] / incident_flux_spectra[0]
    else:
        transmission_coefficient = 0

    # Save data
    if mp.am_master():
        filename_intensity = f'output/intensity_{wavelength}_{column_length:.2f}'
        filename_flux = f'output/transflux_{wavelength}_{column_length:.2f}.txt'
        filename_transmission = f'output/transmission_{wavelength}_{column_length:.2f}.txt'
        np.savez_compressed(filename_intensity, intensity_profile)
        np.savetxt(filename_flux, transmitted_flux_spectra)
        np.savetxt(filename_transmission, [transmission_coefficient])

    print(f"Finished simulation: wavelength={wavelength}, column_length={column_length}")

if __name__ == "__main__":
    mp.verbosity(1)  # Increase verbosity to see more Meep output

    # Generate all parameter combinations
    wavelengths = [605, 625]
    column_lengths = [0.5 + (i / 20) for i in range(51)]
    all_params = [(w, cl) for w in wavelengths for cl in column_lengths]

    # Divide work across processes
    num_processes = mp.count_processors()
    process_rank = mp.divide_parallel_processes(num_processes)
    
    # Ensure output folder exists (only on master process)
    if mp.am_master():
        os.makedirs('output', exist_ok=True)

    # Run simulations
    for i, params in enumerate(all_params):
        if i % num_processes == process_rank:
            wavelength, column_length = params
            print(f"Meep process {process_rank} starting simulation: wavelength={wavelength}, column_length={column_length}")
            try:
                run_meep_simulation(wavelength, column_length)
                print(f"Meep process {process_rank} finished simulation: wavelength={wavelength}, column_length={column_length}")
            except Exception as e:
                print(f"Meep process {process_rank} encountered an error: {str(e)}")

    # Make sure all processes are done before exiting
    mp.all_wait()

    if mp.am_master():
        print("All simulations completed.")
