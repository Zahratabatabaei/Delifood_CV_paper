# Author: Dr. Zahra Tabatabaei
# Contact: elec.tabatabaei@gmail.com

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import cv2
cv2.setNumThreads(0)

import matplotlib.pyplot as plt
from scipy.stats import entropy
from skimage.feature import local_binary_pattern
from skimage.filters import gabor
import tifffile as tiff
import pandas as pd
from joblib import Parallel, delayed

radius = 3
n_points = 8 * radius
lbp_method = 'uniform'
gabor_frequencies = [0.1, 0.2, 0.3]
gabor_angles = [0, 45, 90, 135]

file_paths = [
    r"D:\Delifood\Milk\Images\NaCas_2.6_GDL1.8\20251029_NaCas_2.6_1.8GDL_30C_1.tif",
    r"D:\Delifood\Milk\Images\NaCas_2.6_GDL1.8\20251029_NaCas_2.6_1.8GDL_30C_2.tif"
]

def normalize_to_uint8(img):
    img = img.astype(np.float32)
    mn, mx = img.min(), img.max()
    if mx <= mn:
        return np.zeros_like(img, dtype=np.uint8)
    return ((img - mn) / (mx - mn) * 255).astype(np.uint8)

def enhanced_gabor(image_u8):
    acc = np.zeros_like(image_u8, dtype=np.float32)
    for f in gabor_frequencies:
        for ang in gabor_angles:
            real, _ = gabor(image_u8, frequency=f, theta=np.deg2rad(ang))
            acc += real.astype(np.float32)
    acc = cv2.normalize(acc, None, 0, 255, cv2.NORM_MINMAX)
    return acc.astype(np.uint8)

def raw_histogram(img_u8):
    h, _ = np.histogram(img_u8.ravel(), bins=256, range=(0, 255), density=True)
    return h

def lbp_histogram(img_u8):
    lbp = local_binary_pattern(img_u8, n_points, radius, method=lbp_method)
    lbp = np.nan_to_num(lbp, nan=0.0)
    n_bins = int(n_points + 2) if lbp_method == 'uniform' else int(lbp.max() + 1)
    h, _ = np.histogram(lbp, bins=n_bins, range=(0, n_bins), density=True)
    return h

def stats_from_hist(h):
    return float(np.std(h)), float(entropy(h + 1e-12))

def process_frame(frame_2d):
    frame_u8 = normalize_to_uint8(frame_2d)
    gab_u8 = enhanced_gabor(frame_u8)
    h_raw = raw_histogram(frame_u8)
    h_lbp = lbp_histogram(gab_u8)
    s_raw, e_raw = stats_from_hist(h_raw)
    s_lbp, e_lbp = stats_from_hist(h_lbp)
    return s_raw, e_raw, s_lbp, e_lbp

all_series = {"std_raw": [], "ent_raw": [], "std_lbp": [], "ent_lbp": [], "label": []}

for fp in file_paths:
    if not os.path.exists(fp):
        print("Missing:", fp)
        continue

    stack = tiff.imread(fp)
    if stack.ndim == 2:
        stack = stack[None, ...]
    if stack.ndim != 3:
        print("Skipping (not dynamic 2D):", fp, "shape", stack.shape)
        continue

    T, H, W = stack.shape

    results = Parallel(n_jobs=-1, backend="loky", batch_size="auto")(
        delayed(process_frame)(stack[t]) for t in range(T)
    )
    results = np.array(results)  

    std_raw_ts = results[:, 0]
    ent_raw_ts = results[:, 1]
    std_lbp_ts = results[:, 2]
    ent_lbp_ts = results[:, 3]

    out_csv = os.path.splitext(fp)[0] + "_stats2D_timeseries.csv"
    df = pd.DataFrame({
        "time": np.arange(1, T+1),
        "std_raw": std_raw_ts,
        "ent_raw": ent_raw_ts,
        "std_lbp": std_lbp_ts,
        "ent_lbp": ent_lbp_ts
    })
    df.to_csv(out_csv, index=False)
    print("Saved:", out_csv)

    all_series["std_raw"].append(std_raw_ts)
    all_series["ent_raw"].append(ent_raw_ts)
    all_series["std_lbp"].append(std_lbp_ts)
    all_series["ent_lbp"].append(ent_lbp_ts)
    all_series["label"].append(os.path.basename(fp))

if len(all_series["label"]) == 0:
    raise SystemExit("No valid files processed.")

save_dir = os.path.dirname(file_paths[0]) if len(file_paths) > 0 else os.getcwd()

def overlay_plot(metric_key, ylabel, title, fname):
    plt.figure(figsize=(10,5))
    for y, lab in zip(all_series[metric_key], all_series["label"]):
        x = np.arange(1, len(y)+1)
        plt.plot(x, y, marker='o', linewidth=1.2, label=lab)
    plt.xlabel("Time")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, fname), dpi=150)
    plt.show()

overlay_plot("std_raw", "Std Dev (Histogram)", "Std over Time (raw, 2D)", "overlay_std_raw_2D.png")
overlay_plot("ent_raw", "Entropy (Histogram)", "Entropy over Time (raw, 2D)", "overlay_ent_raw_2D.png")
overlay_plot("std_lbp", "Std Dev (LBP)", "Std over Time (LBP, 2D)", "overlay_std_lbp_2D.png")
overlay_plot("ent_lbp", "Entropy (LBP)", "Entropy over Time (LBP, 2D)", "overlay_ent_lbp_2D.png")
