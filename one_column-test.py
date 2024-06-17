import meep as mp
import numpy as np
import matplotlib.pyplot as plt

# Define the simulation parameters
resolution = 15  # Reduced resolution for quicker simulation
wavelengths = [605e-9, 625e-9]  # wavelengths in meters
frequencies = [1 / wl for wl in wavelengths]  # frequencies corresponding to the wavelengths
block_length = 2

# Simulation domain size
sx = 5  # size of cell in x direction
sy = 5  # size of cell in y direction
sz = block_length * 2.5  # size of cell in z direction

# Material properties
n_siO2 = 1.45  # refractive index of SiO2
siO2 = mp.Medium(index=n_siO2)

# Geometry: single SiO2 column
column_size = 1.0  # size of the SiO2 column
geometry = [
    mp.Block(
        size=mp.Vector3(column_size, column_size, block_length),  # size of the SiO2 column
        center=mp.Vector3(0, 0, 0),  # position of the column in the center
        material=siO2
    )
]

# Simulation domain and boundary layers
cell = mp.Vector3(sx, sy, sz)
pml_layers = [mp.PML(1)]  # Reduced PML thickness

def run_simulation(frequency):
    # Define source
    sources = [
        mp.Source(
            mp.ContinuousSource(frequency=frequency),
            component=mp.Ez,
            center=mp.Vector3(0, 0, -1.5),  # position the source in front of the column
            size=mp.Vector3(1, 1, 0)
        )
    ]

    # Initialize and run the simulation
    sim = mp.Simulation(
        cell_size=cell,
        boundary_layers=pml_layers,
        geometry=geometry,
        sources=sources,
        resolution=resolution
    )

    # Run the simulation until steady state is reached
    sim.run(until=100)  # Reduced run time

    # Extract the field data behind the column (e.g., at z = 2.5)
    z_slice = block_length / 2 + 0.5
    field_data = sim.get_array(center=mp.Vector3(0, 0, z_slice), size=mp.Vector3(sx, sy, 0), component=mp.Ez)
    return field_data

# Run simulations for both wavelengths
fields_605nm = run_simulation(frequencies[0])
fields_625nm = run_simulation(frequencies[1])

# Plotting the results for 605 nm
plt.figure()
plt.imshow(np.abs(fields_605nm.transpose()), interpolation='spline36', cmap='RdBu')
plt.colorbar()
plt.title('Field Intensity at z = {:.2f} for 605 nm (Behind the Column)'.format(block_length / 2 + 0.5))
plt.show()

# Plotting the results for 625 nm
plt.figure()
plt.imshow(np.abs(fields_625nm.transpose()), interpolation='spline36', cmap='RdBu')
plt.colorbar()
plt.title('Field Intensity at z = {:.2f} for 625 nm (Behind the Column)'.format(block_length / 2 + 0.5))
plt.show()

