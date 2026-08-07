# Image-Based Texture, Fractal, and Topological Analysis Code

**Author:** Dr. Zahra Tabatabaei  
**Contact:** [elec.tabatabaei@gmail.com](mailto:elec.tabatabaei@gmail.com)

## Overview

This repository contains Python scripts used for image-based texture, fractal, and topological analyses. The scripts process either dynamic 2D TIFF image stacks or grayscale PNG images and generate numerical results and plots.

The analysis logic, parameters, input-path variables, and output naming used in the supplied scripts have been retained. Before running a script, update its input path to match the location of your own data.
## Dataset

The complete microscopy dataset used in this study is archived separately on
Zenodo:

[https://doi.org/10.5281/zenodo.12345678](https://doi.org/10.5281/zenodo.21833632)

After downloading and extracting the dataset, update the input paths in the
analysis scripts to point to the corresponding TIFF files.
## Repository contents

### `lbp_histogram_overlay_analysis.py`

Processes multiple dynamic 2D TIFF stacks and compares histogram- and Local Binary Pattern-based measurements over time.

**Input**

- One or more TIFF files listed in the `file_paths` variable.
- Each file must contain either a single 2D image or a 3D stack with shape `(frames, height, width)`.

**Processing**

- Normalizes each frame to 8-bit intensity.
- Applies a Gabor filter bank.
- Calculates raw-intensity and LBP histograms.
- Calculates histogram standard deviation and entropy for every frame.
- Uses parallel processing across frames.

**Output**

- One `<input_name>_stats2D_timeseries.csv` file for each input stack.
- `overlay_std_raw_2D.png`
- `overlay_ent_raw_2D.png`
- `overlay_std_lbp_2D.png`
- `overlay_ent_lbp_2D.png`

The CSV columns are `time`, `std_raw`, `ent_raw`, `std_lbp`, and `ent_lbp`.

### `lbp_radius_sensitivity_analysis.py`

Performs an LBP radius sensitivity analysis on one dynamic 2D TIFF stack. The baseline setting is radius 3 with 24 sampling points. The tested radii are defined in `sensitivity_radii`, and each radius uses `n_points = 8 * radius`.

**Input**

- One TIFF file specified by `file_path`.
- The TIFF must contain either a single 2D image or a 3D stack with shape `(frames, height, width)`.

**Processing**

- Normalizes each frame to 8-bit intensity.
- Applies the same Gabor filter bank used in the overlay analysis.
- Calculates raw and LBP histogram statistics for each radius.
- Compares each LBP curve with the baseline using Pearson correlation, RMSE, and normalized RMSE.
- Uses parallel processing across radius settings.

**Output**

- `<input_name>_LBP_radius_sensitivity_timeseries.csv`
- `<input_name>_LBP_radius_sensitivity_summary.csv`
- `LBP_radius_sensitivity_std.png`
- `LBP_radius_sensitivity_entropy.png`

### `tda_betti1_analysis.py`

Applies cubical persistent homology to every frame of a dynamic 2D TIFF stack and tracks the maximum Betti-1 value over time.

**Input**

- One TIFF stack specified by `file_path`.
- Expected shape: `(frames, height, width)`.

**Processing**

- Normalizes each frame to 8-bit intensity.
- Constructs a GUDHI cubical complex.
- Computes persistence intervals in homology dimension 1.
- Generates Betti curves and extracts the maximum Betti-1 value per frame.
- Uses the number of parallel workers specified by `num_cores`.

**Output**

- A temporal Betti-1 plot displayed by Matplotlib.
- `<input_name>_max_betti1.csv` with the column `Max_Betti1`.

### `differential_box_counting_analysis.py`

Calculates the differential box-counting fractal dimension of grayscale fractal PNG images.

**Input**

- A folder specified by `folder`.
- PNG filenames must follow the pattern `fractal_r<roughness>.png`, for example `fractal_r0.85.png`.

**Processing**

- Reads each matching PNG as a grayscale image.
- Extracts the roughness value from the filename.
- Normalizes the image to 8-bit intensity.
- Calculates the differential box-counting fractal dimension.
- Sorts results by roughness.

**Output**

- `fractal_DBC.csv` with the columns `roughness` and `fractal_dimension_dbc`.
- A plot of differential box-counting dimension versus roughness.

### `multifractal_spectrum_analysis.py`

Calculates generalized multifractal dimensions for grayscale fractal PNG images.

**Input**

- A folder specified by `file_path`.
- PNG filenames must follow the pattern `fractal_r<roughness>.png`.

**Processing**

- Reads matching files as grayscale images.
- Binarizes each image using Otsu thresholding.
- Calculates generalized dimensions over `q` values from -5 to 5.
- Calculates the spectrum width as `D(-5) - D(5)`.
- Uses all available CPU cores through Joblib.

**Output**

- One multifractal-spectrum plot for each image.
- An overlay plot for the final set of images.
- A plot of multifractality width across image index.

This script displays plots but does not save numerical results to a file.

## Requirements

The scripts were prepared from a Python 3.12 Windows environment. The minimal direct dependencies used by the code are listed in `requirements.txt`:

- NumPy
- OpenCV
- Matplotlib
- SciPy
- scikit-image
- tifffile
- pandas
- Joblib
- GUDHI

Install them with:

```bash
python -m pip install -r requirements.txt
```

## Running the scripts

1. Open the script you want to run.
2. Replace the existing Windows input path in `file_path`, `file_paths`, or `folder` with the path to your data.
3. Keep the expected input structure and filename pattern described above.
4. Run the script from the repository directory, for example:

```bash
python lbp_histogram_overlay_analysis.py
```

The scripts write CSV and image outputs either next to the input data or in the configured image folder. Matplotlib figures are also displayed interactively.

## Notes

- The LBP scripts restrict internal OpenMP, MKL, NumExpr, and OpenCV thread use to reduce conflicts with Joblib parallel processing.
- `n_jobs = -1` and `multiprocessing.cpu_count()` use all available logical CPU cores. Reduce these values if memory use is too high.
- The TDA script uses three workers by default to limit memory consumption.
- The scripts retain the supplied analysis settings. Users should document any parameter changes when reproducing or extending the analysis.

## Contact

For questions regarding the paper or the code, please contact [elec.tabatabaei@gmail.com](mailto:elec.tabatabaei@gmail.com).
