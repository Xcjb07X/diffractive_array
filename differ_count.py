import meep as mp
import numpy as np
import matplotlib.pyplot as plt
import csv

def run_simulation(frequency, cell, pml_layers, geometry, resolution, column_length, run_time):
    # Define source
    sources = [
        mp.Source(
            mp.ContinuousSource(frequency=frequency),
            component=mp.Ez,
            center=mp.Vector3(0, 0, -1.5),  # position the source in front of the column
            size=mp.Vector3(1, 1, 0))]
    
    # Initialize and run the simulation with C4 symmetry
    symmetries = [mp.Mirror(mp.X, phase=1), mp.Mirror(mp.Y, phase=1)]
    sim = mp.Simulation(
        cell_size=cell,
        boundary_layers=pml_layers,
        geometry=geometry,
        sources=sources,
        resolution=resolution,
        symmetries=symmetries)

    # Run the simulation until steady state is reached
    sim.run(until=run_time)

    # Extract the field data behind the column (e.g., at z = 2.5)
    z_slice = column_length / 2 + 0.5
    field_data = sim.get_array(center=mp.Vector3(0, 0, z_slice), size=mp.Vector3(2.5, 2.5, 0), component=mp.Ez)
    return np.array(field_data)

def normalize_array(array):
    min_val = np.min(array)
    max_val = np.max(array)
    normalized_array = (array - min_val) / (max_val - min_val)
    return normalized_array

def convert_arrays(fields_605, fields_625):
    fields_605_array = np.array(fields_605)
    fields_625_array = np.array(fields_625)
    convert_605 = normalize_array(fields_605_array)
    convert_625 = normalize_array(fields_625_array)
    return convert_605, convert_625

def compare_arrays(convert_605, convert_625, threshold):
    differ_count = 0
    array_x, array_y = np.shape(convert_605)
    
    for i in range(array_x):
        for j in range(array_y):
            if convert_625[i, j] != 0 and (convert_605[i, j] / convert_625[i, j]) < threshold:
                differ_count += 1
            elif convert_605[i, j] != 0 and (convert_625[i, j] / convert_605[i, j]) < threshold:
                differ_count += 1
                
    return differ_count

def run(resolution, column_length, wavelength_1, wavelength_2, run_time):
    # Parameters
    resolution = resolution
    wavelength_1 = wavelength_1
    wavelength_2 = wavelength_2
    frequencies = [1 / wavelength_1, 1 / wavelength_2]
    column_length = column_length
    run_time = run_time
    threshold = 0.25
    
    # Simulation domain size
    sx = 3.5  # reduce by half for symmetry
    sy = 3.5  # reduce by half for symmetry
    sz = column_length * 2.5 + 2
    
    # Material properties
    n_siO2 = 1.4576
    siO2 = mp.Medium(index=n_siO2)
    
    # Geometry: single SiO2 column
    column_size = 2.5
    geometry = [
        mp.Block(
            size=mp.Vector3(column_size, column_size, column_length),
            center=mp.Vector3(0, 0, 0),
            material=siO2)]
    
    # Simulation domain and boundary layers
    cell = mp.Vector3(sx, sy, sz)
    pml_layers = [mp.PML(1)]
    
    fields_605 = run_simulation(frequencies[0], cell, pml_layers, geometry, resolution, column_length, run_time)
    fields_625 = run_simulation(frequencies[1], cell, pml_layers, geometry, resolution, column_length, run_time)
    convert_605, convert_625 = convert_arrays(fields_605, fields_625)
    differ_count = compare_arrays(convert_605, convert_625, threshold)
    return differ_count

differ_count = []
for i in range(52):
    column_length = 0.5 + (i / 20)
    differ_count_indv = run(50, column_length, 605e-9, 625e-9, 100)
    differ_count.append((column_length, differ_count_indv))
    if i % 5 == 0:
        print(f'Finished epoch: {i}')

csv_filename = 'differ_count.csv'
with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(differ_count)
