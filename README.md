# README Draft

## Overview
This repository provides a pipeline that builds a Proper Orthogonal Decomposition (POD) Galerkin reduced-order model from 2D flow snapshots exported from Ansys Fluent. Given a target Reynolds number, the reduced model is solved to reconstruct the full flow field.

## Requirements
- Python 3.9+
- Scientific Python stack (NumPy, Pandas, SciPy, Matplotlib, etc.)

## Key Components
- `PreProcessing/preprocessing.py`: Sorts Fluent CSV snapshots by coordinates and generates `_sorted` files in batch.
- `Galerkin_offline.py`: Extracts the interior grid from the sorted snapshots, computes POD modes, pre-computes Galerkin tensors, and stores them in `rom_offline_data.npz`.
- `Galerkin_online.py`: Loads the offline data, solves the nonlinear equations for a specified Reynolds number, and exports the reconstructed pressure/velocity fields as CSV files and interpolated plots.
- `PostProcessing/CalculateL2.py`: Calculates the relative L2 error between the reference solution and ROM solution, and visualizes the error trend versus Reynolds number.

## Usage
1. **Data sorting**  
   Place Fluent `case*.csv` files in a working folder such as `offlineDATA/` and run `python PreProcessing/preprocessing.py` to generate the `_sorted` files.
2. **Offline stage**  
   Run `python Galerkin_offline.py` to read `offlineDATA/case*_sorted.csv`, compute POD modes and Galerkin tensors, and store the results in `rom_offline_data.npz`.
3. **Online stage**  
   Run `python Galerkin_online.py`, enter the desired Reynolds number, and the reduced model will reconstruct the flow field, saving the output to `FinalResult/rom_solution_Re_<Re>.csv` along with PNG plots.
4. **(Optional) Post-processing**  
   Run `python PostProcessing/CalculateL2.py` to compare the reference and ROM solutions and generate the Reynolds-number-dependent relative L2 error plot.

## Outputs
- `rom_offline_data.npz`: POD modes, boundary conditions, and precomputed Galerkin coefficient tensors from the offline stage.
- `FinalResult/rom_solution_Re_*.csv` and plots: Reconstructed pressure/velocity fields from the online stage.
- `rom_error_vs_re.png`: Relative ROM error versus Reynolds number produced by the post-processing script.
