# Author: Dr. Zahra Tabatabaei
# Contact: elec.tabatabaei@gmail.com

import gc
import numpy as np
import gudhi as gd
import matplotlib.pyplot as plt
import pandas as pd
from tifffile import imread
from joblib import Parallel, delayed

#%%

file_path = r"D:\Delifood\Milk\Images\NA-data_for_CV_paper\NA_1.8GDL\20251029_NaCas_2.6_1.8GDL_30C_1.tif"
images = imread(file_path)

#%% Betti curve function

def betti_curve(persistence_intervals, resolution=100):

    if len(persistence_intervals) == 0:
        x_vals = np.linspace(0, 1, resolution)
        y_vals = np.zeros(resolution)
        return x_vals, y_vals

    birth, death = persistence_intervals[:, 0], persistence_intervals[:, 1]
    death[death == np.inf] = max(birth) + 1

    x_vals = np.linspace(min(birth), max(death), resolution)
    y_vals = np.zeros_like(x_vals)

    for b, d in zip(birth, death):
        y_vals += (x_vals >= b) & (x_vals < d)

    return x_vals, y_vals


#%% Function for one image

def process_one_image(i):

    print(f"Processing image {i+1}/{len(images)}")

    min_val = np.min(images[i])
    max_val = np.max(images[i])

    image_uint8 = ((images[i] - min_val) / (max_val - min_val) * 255).astype(np.uint8)

    cubical_complex = gd.CubicalComplex(
        dimensions=image_uint8.shape,
        top_dimensional_cells=image_uint8.flatten()
    )

    persistence = cubical_complex.persistence()
    H1_train = cubical_complex.persistence_intervals_in_dimension(1)

    x_betti, y_betti = betti_curve(H1_train, resolution=50)

    x_vals, y_vals = betti_curve(H1_train, resolution=100)
    max_betti1 = np.max(y_vals)

    result = {
        "y_betti": y_betti,
        "max_betti1": max_betti1,
        "x_vals": x_vals
    }

    del image_uint8
    del cubical_complex
    del persistence
    del H1_train
    del x_betti
    del y_vals
    gc.collect()

    return result


#%% Parallel processing

num_cores = 3   # You have 12 cores, but 12 may crash because of memory

results = Parallel(n_jobs=num_cores)(
    delayed(process_one_image)(i) for i in range(len(images))
)


#%% Collect results

betti_curves_train = [r["y_betti"] for r in results]
betti1_max_per_frame = [r["max_betti1"] for r in results]
x_betti_all = [r["x_vals"] for r in results]

betti_matrix = np.array(betti_curves_train)


#%% Plot Betti-1 max per frame

plt.figure(figsize=(8, 4))
plt.plot(betti1_max_per_frame, marker='o', color='green', label='Max Betti-1 per Frame')
plt.xlabel("Time (Frame Index)")
plt.ylabel("Max Number of Loops (Betti-1)")
plt.title(f"Temporal Tracking of Betti-1 During Aggregation {file_path[-30:-4]}")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


#%% Save result

df_betti1_max_per_frame = pd.DataFrame(betti1_max_per_frame, columns=["Max_Betti1"])

df_betti1_max_per_frame.to_csv(
    f"{file_path[:-4]}_max_betti1.csv",
    index=False
)