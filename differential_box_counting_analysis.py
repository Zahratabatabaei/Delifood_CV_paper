# Author: Dr. Zahra Tabatabaei
# Contact: elec.tabatabaei@gmail.com

import os
import re
from pathlib import Path

import numpy as np
import cv2
import matplotlib.pyplot as plt

# -------- Differential Box Counting --------
def differential_box_counting(image, num_scales=10):
    H, W = image.shape
    G = 256
    scales = np.floor(np.logspace(1, np.log10(min(H, W)), num=num_scales)).astype(int)

    counts = []
    used = []
    for s in scales:
        if s < 2:
            continue
        count = 0
        for i in range(0, H, s):
            for j in range(0, W, s):
                block = image[i:i + s, j:j + s]
                if block.size == 0:
                    continue
                g_min, g_max = block.min(), block.max()
                n_boxes = int(np.floor((g_max - g_min) / (G / s) + 1))
                n_boxes = max(1, n_boxes)
                count += n_boxes
        if count > 0:
            counts.append(count)
            used.append(s)

    counts = np.array(counts, dtype=float)
    used = np.array(used, dtype=float)

    log_counts = np.log(counts)
    log_scales = np.log(1.0 / used)

    coeffs = np.polyfit(log_scales, log_counts, 1)
    return coeffs[0]  # fractal dimension


def normalize_to_uint8(img):
    img = img.astype(np.float32)
    mn, mx = img.min(), img.max()
    if mx <= mn:
        return np.zeros_like(img, dtype=np.uint8)
    return ((img - mn) / (mx - mn) * 255).astype(np.uint8)


#%% -------- Load PNGs and compute FD --------
folder = r'D:/Delifood/Milk/codes/fractals'
pattern = re.compile(r"fractal_r([\d.]+)\.png")

images = []
roughness = []

for filename in os.listdir(folder):
    if not filename.lower().endswith('.png'):
        continue
    m = pattern.match(filename)
    if not m:
        continue

    r_val = float(m.group(1))
    img_path = os.path.join(folder, filename)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        continue

    images.append(img)
    roughness.append(r_val)

images = np.array(images)
roughness = np.array(roughness)

# sort by roughness
idx = np.argsort(roughness)
images = images[idx]
roughness = roughness[idx]

fd_values = []
for img in images:
    img_u8 = normalize_to_uint8(img)
    fd = differential_box_counting(img_u8)
    fd_values.append(fd)

fd_values = np.array(fd_values)

# -------- Save CSV --------
csv_path = Path(folder) / "fractal_DBC.csv"
with open(csv_path, "w", newline="") as f:
    f.write("roughness,fractal_dimension_dbc\n")
    for r, fd in zip(roughness, fd_values):
        f.write(f"{r},{fd}\n")
print("Saved:", csv_path)

# -------- Plot FD vs roughness --------
plt.figure(figsize=(8, 5))
plt.plot(roughness, fd_values, marker='o')
plt.xlabel("Roughness r")
plt.ylabel("DBC")
plt.title("DBC vs Roughness")
plt.grid(True)
plt.tight_layout()
plt.show()
