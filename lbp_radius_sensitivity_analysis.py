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

# ---------------------------------------------------------------------
# Original settings kept the same
# ---------------------------------------------------------------------
radius = 3
n_points = 8 * radius
lbp_method = 'uniform'
gabor_frequencies = [0.1, 0.2, 0.3]
gabor_angles = [0, 45, 90, 135]

# ---------------------------------------------------------------------
# LBP sensitivity settings
# Baseline remains radius = 3 and n_points = 24.
# Each radius uses n_points = 8 * radius, exactly as in the original code.
# ---------------------------------------------------------------------
sensitivity_radii = [1, 2, 3, 4, 5]

# -1 uses all available logical CPUs.
# Change to a fixed number, for example 8, if needed.
n_jobs = -1

# ---------------------------------------------------------------------
# Only one dataset
# ---------------------------------------------------------------------
file_path = r"D:\Delifood\Milk\Images\NA-data_for_CV_paper\NA_3.5GDL\20251017_NaCas_2.6_40C_3.5GDL_sted_3.tif"


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
            real, _ = gabor(
                image_u8,
                frequency=f,
                theta=np.deg2rad(ang)
            )
            acc += real.astype(np.float32)

    acc = cv2.normalize(
        acc,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )
    return acc.astype(np.uint8)


def raw_histogram(img_u8):
    h, _ = np.histogram(
        img_u8.ravel(),
        bins=256,
        range=(0, 255),
        density=True
    )
    return h


def lbp_histogram(img_u8, current_radius, current_n_points):
    lbp = local_binary_pattern(
        img_u8,
        current_n_points,
        current_radius,
        method=lbp_method
    )

    lbp = np.nan_to_num(lbp, nan=0.0)

    n_bins = (
        int(current_n_points + 2)
        if lbp_method == 'uniform'
        else int(lbp.max() + 1)
    )

    h, _ = np.histogram(
        lbp,
        bins=n_bins,
        range=(0, n_bins),
        density=True
    )

    return h


def stats_from_hist(h):
    return float(np.std(h)), float(entropy(h + 1e-12))


def process_frame(frame_2d, current_radius, current_n_points):
    frame_u8 = normalize_to_uint8(frame_2d)
    gab_u8 = enhanced_gabor(frame_u8)

    h_raw = raw_histogram(frame_u8)
    h_lbp = lbp_histogram(
        gab_u8,
        current_radius,
        current_n_points
    )

    s_raw, e_raw = stats_from_hist(h_raw)
    s_lbp, e_lbp = stats_from_hist(h_lbp)

    return s_raw, e_raw, s_lbp, e_lbp


def process_radius(stack, current_radius):
    """
    Process the complete TIFF stack for one LBP radius.

    Parallel processing is performed across radius settings, so each
    radius is assigned to a separate CPU worker. Frames are processed
    sequentially inside that worker to avoid nested multiprocessing.
    """
    current_n_points = 8 * current_radius
    T = stack.shape[0]

    print(
        "Processing radius:",
        current_radius,
        "n_points:",
        current_n_points
    )

    results = [
        process_frame(
            stack[t],
            current_radius,
            current_n_points
        )
        for t in range(T)
    ]

    results = np.asarray(results, dtype=np.float64)

    return {
        "radius": current_radius,
        "n_points": current_n_points,
        "std_raw": results[:, 0],
        "ent_raw": results[:, 1],
        "std_lbp": results[:, 2],
        "ent_lbp": results[:, 3]
    }


def overlay_plot(
    sensitivity_results,
    metric_key,
    ylabel,
    title,
    fname,
    save_dir
):
    plt.figure(figsize=(10, 5))

    for result in sensitivity_results:
        y = result[metric_key]
        x = np.arange(1, len(y) + 1)

        current_radius = result["radius"]
        current_n_points = result["n_points"]

        label = (
            f"R={current_radius}, "
            f"P={current_n_points}"
        )

        # Make the original manuscript setting more visible.
        if current_radius == radius:
            plt.plot(
                x,
                y,
                marker='o',
                markersize=3,
                linewidth=2.2,
                label=label + " (baseline)"
            )
        else:
            plt.plot(
                x,
                y,
                marker='o',
                markersize=2,
                linewidth=1.2,
                alpha=0.8,
                label=label
            )

    plt.xlabel("Time")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    output_path = os.path.join(save_dir, fname)
    plt.savefig(output_path, dpi=300)
    print("Saved:", output_path)
    plt.show()


def calculate_sensitivity_summary(sensitivity_results):
    """
    Compare every LBP sensitivity curve with the original setting R=3.
    Pearson correlation and normalized RMSE are calculated separately
    for LBP standard deviation and entropy.
    """
    baseline = None

    for result in sensitivity_results:
        if result["radius"] == radius:
            baseline = result
            break

    if baseline is None:
        raise RuntimeError(
            "The baseline radius must be included in sensitivity_radii."
        )

    summary_rows = []

    for result in sensitivity_results:
        for metric_key in ["std_lbp", "ent_lbp"]:
            baseline_values = np.asarray(
                baseline[metric_key],
                dtype=np.float64
            )
            current_values = np.asarray(
                result[metric_key],
                dtype=np.float64
            )

            if (
                np.std(baseline_values) == 0
                or np.std(current_values) == 0
            ):
                correlation = np.nan
            else:
                correlation = np.corrcoef(
                    baseline_values,
                    current_values
                )[0, 1]

            rmse = np.sqrt(
                np.mean(
                    (current_values - baseline_values) ** 2
                )
            )

            baseline_range = (
                baseline_values.max()
                - baseline_values.min()
            )

            if baseline_range > 0:
                normalized_rmse = rmse / baseline_range
            else:
                normalized_rmse = np.nan

            summary_rows.append({
                "radius": result["radius"],
                "n_points": result["n_points"],
                "metric": metric_key,
                "baseline_radius": radius,
                "baseline_n_points": n_points,
                "pearson_correlation": correlation,
                "rmse": rmse,
                "normalized_rmse": normalized_rmse
            })

    return pd.DataFrame(summary_rows)


def main():
    if not os.path.exists(file_path):
        raise SystemExit("Missing: " + file_path)

    stack = tiff.imread(file_path)

    if stack.ndim == 2:
        stack = stack[None, ...]

    if stack.ndim != 3:
        raise SystemExit(
            "Skipping (not dynamic 2D): "
            + file_path
            + " shape "
            + str(stack.shape)
        )

    T, H, W = stack.shape
    print("Stack shape:", stack.shape)
    print("Using CPUs:", n_jobs)
    print("Sensitivity radii:", sensitivity_radii)

    # -------------------------------------------------------------
    # Parallel CPU processing across the different LBP radii
    # -------------------------------------------------------------
    sensitivity_results = Parallel(
        n_jobs=n_jobs,
        backend="loky",
        batch_size=1,
        verbose=10
    )(
        delayed(process_radius)(
            stack,
            current_radius
        )
        for current_radius in sensitivity_radii
    )

    # Sort results so plots and CSVs follow increasing radius.
    sensitivity_results = sorted(
        sensitivity_results,
        key=lambda result: result["radius"]
    )

    save_dir = os.path.dirname(file_path)

    # -------------------------------------------------------------
    # Save one CSV containing all sensitivity time series
    # -------------------------------------------------------------
    all_rows = []

    for result in sensitivity_results:
        for t in range(T):
            all_rows.append({
                "time": t + 1,
                "radius": result["radius"],
                "n_points": result["n_points"],
                "std_raw": result["std_raw"][t],
                "ent_raw": result["ent_raw"][t],
                "std_lbp": result["std_lbp"][t],
                "ent_lbp": result["ent_lbp"][t]
            })

    sensitivity_df = pd.DataFrame(all_rows)

    out_csv = (
        os.path.splitext(file_path)[0]
        + "_LBP_radius_sensitivity_timeseries.csv"
    )

    sensitivity_df.to_csv(out_csv, index=False)
    print("Saved:", out_csv)

    # -------------------------------------------------------------
    # Save numerical comparison with the original R=3 setting
    # -------------------------------------------------------------
    summary_df = calculate_sensitivity_summary(
        sensitivity_results
    )

    summary_csv = (
        os.path.splitext(file_path)[0]
        + "_LBP_radius_sensitivity_summary.csv"
    )

    summary_df.to_csv(summary_csv, index=False)
    print("Saved:", summary_csv)

    # -------------------------------------------------------------
    # Figures for the paper
    # -------------------------------------------------------------
    overlay_plot(
        sensitivity_results,
        "std_lbp",
        "Std Dev (LBP)",
        "LBP Standard Deviation Sensitivity to Radius",
        "LBP_radius_sensitivity_std.png",
        save_dir
    )

    overlay_plot(
        sensitivity_results,
        "ent_lbp",
        "Entropy (LBP)",
        "LBP Entropy Sensitivity to Radius",
        "LBP_radius_sensitivity_entropy.png",
        save_dir
    )

    print("\nSensitivity summary:")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    # Required for joblib multiprocessing on Windows.
    main()
