import meep as mp
import numpy as np
import matplotlib.pyplot as plt

# Define the simulation parameters
resolution = 25  # pixels per unit length
wavelengths = [605e-9, 625e-9]  # wavelengths in meters
frequencies = [1 / wl for wl in wavelengths]  # frequency corresponding to the wavelengths

# Simulation domain size
sx = 4  # size of cell in x direction
sy = 4   # size of cell in y direction

# Material properties
n_siO2 = 1.45  # refractive index of SiO2
siO2 = mp.Medium(index=n_siO2)

# Geometry: array of SiO2 columns
geometry = [
    mp.Block(
        size=mp.Vector3(1.5, 1, mp.inf),  # size of the SiO2 columns
        center=mp.Vector3(1.5/2+.25, .5, 0),  # x = 37-62
        material=siO2),
    mp.Block(
        size=mp.Vector3(.75, 1, mp.inf),  # size of the SiO2 columns
        center=mp.Vector3(.75/2+.25, -.5, 0),  # x = 37-62
        material=siO2) 
    
]

# Simulation domain and boundary layers
cell = mp.Vector3(sx, sy, 0)
pml_layers = [mp.PML(1.0)]

# Function to create and run the simulation for a given frequency
def run_simulation(frequency):
    sources = [
        mp.Source(
            mp.ContinuousSource(frequency=frequency),
            component=mp.Ez,
            center=mp.Vector3(-1, 0, 0), #at x=25
            size=mp.Vector3(0, sy, 0)
        )
    ]

    sim = mp.Simulation(
        cell_size=cell,
        boundary_layers=pml_layers,
        geometry=geometry,
        sources=sources,
        resolution=resolution
    )

    # Run the simulation until steady state is reached
    sim.run(until=200)
    
    # Extract the field data
    field_data = sim.get_array(center=mp.Vector3(), size=cell, component=mp.Ez)
    return field_data

# Run simulations for both wavelengths
fields_605nm = run_simulation(frequencies[0])
fields_625nm = run_simulation(frequencies[1])

# Plotting the results
plt.figure()
plt.imshow(np.abs(fields_605nm.transpose()), interpolation='spline36', cmap='RdBu')
plt.colorbar()
plt.title('605 nm Field Intensity')
plt.show()

plt.figure()
plt.imshow(np.abs(fields_625nm.transpose()), interpolation='spline36', cmap='RdBu')
plt.colorbar()
plt.title('625 nm Field Intensity')
plt.show()
