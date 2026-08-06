# Author: Dr. Zahra Tabatabaei
# Contact: elec.tabatabaei@gmail.com

import numpy as np
import matplotlib.pyplot as plt
import tifffile as tiff
from joblib import Parallel, delayed
import multiprocessing
from skimage.filters import threshold_otsu
import os,re,cv2


def box_probabilities(binary_image, box_size):
    N = binary_image.shape[0]
    trimmed = binary_image[:N//box_size*box_size, :N//box_size*box_size]
    reshaped = trimmed.reshape(N//box_size, box_size, N//box_size, box_size)
    sums = reshaped.sum(axis=(1, 3))
    return sums.flatten() / binary_image.sum()

def multifractal_spectrum(image, q_vals=np.linspace(-5, 5, 11), box_sizes=None):
    binary_image = image > threshold_otsu(image)
    N = binary_image.shape[0]
    if box_sizes is None:
        box_sizes = np.unique(np.logspace(1, np.log2(N), num=6, base=2, dtype=int))
    epsilons = box_sizes / N
    Dq = []
    for q in q_vals:
        Z_q = []
        for box_size in box_sizes:
            probs = box_probabilities(binary_image, box_size)
            probs = probs[probs > 0]
            if q == 1:
                H = -np.sum(probs * np.log(probs + 1e-12))
                Z_q.append(H)
            else:
                Z_q.append(np.sum(probs ** q))
        Z_q = np.array(Z_q)
        if q == 1:
            slope, _ = np.polyfit(np.log(1/epsilons), Z_q, 1)
            Dq.append(slope)
        else:
            slope, _ = np.polyfit(np.log(epsilons), np.log(Z_q), 1)
            Dq.append(slope / (q - 1))
    return Dq

file_path = 'D:/Delifood/Milk/codes/fractals'  # your image folder
image_data = []

# Regex to extract roughness from filename like: fractal_r0.85.png
pattern = re.compile(r"fractal_r([\d.]+)\.png")
roughness_all = []
for filename in os.listdir(file_path):
    if filename.endswith('.png'):
        match = pattern.match(filename)
        if match:
            roughness = float(match.group(1))
            img_path = os.path.join(file_path, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                image_data.append(img)
                roughness_all.append(roughness)

# Sort by roughness if needed
# image_data.sort(key=lambda x: x[0])

print(f"Loaded {len(image_data)} images.")
images = np.array(image_data)
N_time, H, W = images.shape
#%%
q_vals = np.linspace(-5, 5, 11)
num_cores = multiprocessing.cpu_count()

results = Parallel(n_jobs=num_cores)(
    delayed(multifractal_spectrum)(
        (images[t]),
        q_vals
    )
    for t in range(N_time)
    
)

results = np.array(results).reshape(N_time,  len(q_vals))
#%%
for t in range(N_time):
    plt.figure(figsize=(7,5))
    plt.plot(q_vals, results[t, :], marker='o' )
    plt.xlabel("q")
    plt.ylabel("Generalized Dimension D(q)")
    plt.title(f"Multifractal Spectrum at Time {t+1}")
    plt.legend()
    plt.grid(True)
    plt.show()
    
#%%
 

plt.figure(figsize=(8,6))
for t in range(int(np.round(N_time-15)),N_time):
    plt.plot(q_vals, results[t, :], marker='o', label=f"Time {t+1}")

plt.xlabel("q values",fontsize = 20)
plt.ylabel("Generalized Dimension D(q)",fontsize = 20)
plt.title(f"Multifractal Spectrum Over Time ",fontsize = 20)
plt.legend()
plt.grid(True)
plt.show()



# 
q_min_idx = np.where(q_vals == -5)[0][0]
q_max_idx = np.where(q_vals == 5)[0][0]

# Compute spectrum width ΔD for each time
delta_D = results[:, q_min_idx] - results[:,q_max_idx]

# Plot ΔD over time
plt.figure(figsize=(8,5))
plt.plot(range(1, N_time+1), delta_D, marker='o')
plt.xlabel("Frame",fontsize = 20)
plt.ylabel("Spectrum Width ΔD",fontsize = 20)
plt.title(f"Evolution of Multifractality Width",fontsize = 20)
plt.grid(True)
plt.show()
    
