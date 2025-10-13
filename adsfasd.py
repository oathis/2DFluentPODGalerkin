"""Utility script to inspect the mean-field separation used by the ROM pipeline.

This script rebuilds the snapshot matrices exactly as the offline routine does, prints the
analytical formulas that are used for the mean/fluctuation split, and verifies that the
online recombination step reproduces the original snapshots when the Galerkin coefficients
obtained from projection are used.  It is meant for debugging and documentation purposes.
"""
from __future__ import annotations

import os
import sys
import glob
import numpy as np

_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = r"C:\Users\spearlab05\Desktop\Galerkin ROM"
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from Galerkin_offline_pressureFullmode_normed_average import (
    NX,
    NY,
    N_NODES,
    load_and_preprocess_data,
)

# Direct path to the training snapshots used by the offline stage
DEFAULT_DATA_DIRECTORY = os.path.join(_REPO_ROOT, "offlineDATA")
OFFLINE_DATA_FILE = os.path.join(_REPO_ROOT, "rom_offline_data.npz")


def _sorted_case_files(data_directory: str) -> list[str]:
    """Return the list of snapshot files sorted by their case number."""
    pattern = os.path.join(data_directory, "case*_sorted.csv")
    filepaths = glob.glob(pattern)

    def extract_case_id(path: str) -> int:
        basename = os.path.basename(path)
        digits = "".join(ch for ch in basename if ch.isdigit())
        return int(digits) if digits else -1

    filepaths.sort(key=extract_case_id)
    return filepaths


def _build_snapshot_matrix(data_directory: str) -> tuple[np.ndarray, np.ndarray]:
    """Load all snapshots and build the full Q matrix used offline."""
    filepaths = _sorted_case_files(data_directory)
    if not filepaths:
        raise FileNotFoundError(
            f"No snapshot files found in '{data_directory}'."
        )

    Q = np.zeros((3 * N_NODES, len(filepaths)))
    coords = None

    for idx, fpath in enumerate(filepaths):
        df = load_and_preprocess_data(fpath)
        snapshot = np.concatenate([
            df["pressure"].to_numpy(),
            df["x-velocity"].to_numpy(),
            df["y-velocity"].to_numpy(),
        ])
        Q[:, idx] = snapshot
        if coords is None:
            coords = df[["x-coordinate", "y-coordinate"]].to_numpy()

    return Q, coords


def _restrict_velocity_to_interior(field: np.ndarray) -> np.ndarray:
    """Restrict a velocity field defined on the full grid to the interior nodes."""
    return field.reshape(NY, NX)[1:-1, 1:-1].ravel()


def expand_interior_to_full(field_int: np.ndarray) -> np.ndarray:
    """Embed an interior-only field back into the full grid by zero-padding the boundary."""
    nx_int, ny_int = NX - 2, NY - 2
    field_full = np.zeros((NY, NX))
    field_full[1:-1, 1:-1] = field_int.reshape(ny_int, nx_int)
    return field_full.ravel()


def build_Q_interior(Q_full: np.ndarray) -> np.ndarray:
    """Construct Q_interior = [p_full; u_int; v_int] for each snapshot."""
    nx_int, ny_int = NX - 2, NY - 2
    n_nodes_int = nx_int * ny_int
    Q_int = np.zeros((N_NODES + 2 * n_nodes_int, Q_full.shape[1]))

    for i in range(Q_full.shape[1]):
        p = Q_full[0:N_NODES, i]
        u = Q_full[N_NODES:2 * N_NODES, i]
        v = Q_full[2 * N_NODES:3 * N_NODES, i]

        Q_int[:, i] = np.concatenate([
            p,
            _restrict_velocity_to_interior(u),
            _restrict_velocity_to_interior(v),
        ])

    return Q_int


def compute_mean_and_fluctuations(Q_int: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute the ensemble mean and the fluctuation matrix."""
    q_mean = Q_int.mean(axis=1)
    fluctuations = Q_int - q_mean[:, None]
    return q_mean, fluctuations


def project_onto_pod(fluctuations: np.ndarray, modes: np.ndarray) -> np.ndarray:
    """Project fluctuations onto the POD basis to get the modal coefficients."""
    return modes.T @ fluctuations


def rebuild_modes_from_offline_data(data: dict[str, np.ndarray]) -> np.ndarray:
    """Reassemble Phi_interior using the stored modal blocks."""
    nx_int, ny_int = NX - 2, NY - 2
    n_nodes_int = nx_int * ny_int

    u_modes_int = np.zeros((n_nodes_int, data["u_modes"].shape[1]))
    v_modes_int = np.zeros_like(u_modes_int)

    for k in range(data["u_modes"].shape[1]):
        u_modes_int[:, k] = _restrict_velocity_to_interior(data["u_modes"][:, k])
        v_modes_int[:, k] = _restrict_velocity_to_interior(data["v_modes"][:, k])

    Phi = np.vstack([
        data["p_modes"],
        u_modes_int,
        v_modes_int,
    ])
    return Phi


def verify_mean_field_coupling(data_directory: str = DEFAULT_DATA_DIRECTORY, offline_file: str = OFFLINE_DATA_FILE) -> None:
    print("[1] Building snapshot matrices and mean-field split...")
    Q_full, coords = _build_snapshot_matrix(data_directory)
    Q_int = build_Q_interior(Q_full)
    q_mean, fluctuations = compute_mean_and_fluctuations(Q_int)

    print("    Mean-field formula: q̄ = (1/M) Σ qᵢ")
    print("    Fluctuation definition: q'_i = qᵢ - q̄")

    reconstruction_error = np.linalg.norm(Q_int - (q_mean[:, None] + fluctuations))
    print(f"    Reconstruction error after split (should be ~0): {reconstruction_error:.3e}")

    if not os.path.exists(offline_file):
        raise FileNotFoundError(
            f"Offline data '{offline_file}' not found. Run the offline stage first."
        )

    with np.load(offline_file) as data:
        offline_data = {key: data[key] for key in data}

    U, _, _ = np.linalg.svd(fluctuations, full_matrices=False)
    K = offline_data["p_modes"].shape[1]
    Phi_svd = U[:, :K]
    coeffs_svd = project_onto_pod(fluctuations, Phi_svd)

    coeff_mean = coeffs_svd.mean(axis=1)
    coeff_std = coeffs_svd.std(axis=1)
    print("\n[2] Statistics of projected coefficients (POD/SVD basis):")
    for i, (mval, sval) in enumerate(zip(coeff_mean, coeff_std), start=1):
        print(f"    Mode {i:02d}: mean = {mval:+.3e}, std = {sval:.3e}")

    print("\n[3] Verifying mean-field reconstruction with the POD basis...")
    pod_residual = Q_int - (q_mean[:, None] + Phi_svd @ coeffs_svd)
    max_pod_error = np.max(np.linalg.norm(pod_residual, axis=0))
    print(f"    Maximum reconstruction error (interior state, POD basis): {max_pod_error:.3e}")

    Phi_offline = rebuild_modes_from_offline_data(offline_data)
    alignment = np.sign(np.sum(Phi_svd * Phi_offline, axis=0))
    Phi_offline_aligned = Phi_offline * alignment
    mode_mismatch = np.linalg.norm(Phi_svd - Phi_offline_aligned, ord=np.inf)
    print(f"    Max absolute difference between POD basis and stored modes: {mode_mismatch:.3e}")

    coeffs, _, _, _ = np.linalg.lstsq(Phi_offline, fluctuations, rcond=None)
    offline_residual = fluctuations - Phi_offline @ coeffs
    max_offline_error = np.max(np.linalg.norm(offline_residual, axis=0))
    print(f"    Maximum reconstruction error (interior state, stored modes): {max_offline_error:.3e}")

    print("\n[4] Cross-check with stored mean fields and boundary coupling...")
    nx_int, ny_int = NX - 2, NY - 2
    n_nodes_int = nx_int * ny_int

    p_mean = offline_data.get("p_mean", q_mean[0:N_NODES])
    u_mean_full = offline_data.get(
        "u_mean",
        expand_interior_to_full(q_mean[N_NODES:N_NODES + n_nodes_int]),
    )
    v_mean_full = offline_data.get(
        "v_mean",
        expand_interior_to_full(
            q_mean[N_NODES + n_nodes_int:N_NODES + 2 * n_nodes_int]
        ),
    )

    u_bc = offline_data["u_bc"]
    boundary_mask = np.abs(u_bc) > 1e-12

    for sample_idx in range(min(3, Q_full.shape[1])):
        alpha = coeffs[:, sample_idx]
        p_full = p_mean + offline_data["p_modes"] @ alpha
        u_full = u_bc + u_mean_full + offline_data["u_modes"] @ alpha
        v_full = v_mean_full + offline_data["v_modes"] @ alpha

        original = Q_full[:, sample_idx]
        reconstructed = np.concatenate([p_full, u_full, v_full])
        diff_norm = np.linalg.norm(original - reconstructed)
        u_error = original[N_NODES:2 * N_NODES] - reconstructed[N_NODES:2 * N_NODES]
        boundary_err = np.linalg.norm(u_error[boundary_mask])
        interior_err = np.linalg.norm(u_error[~boundary_mask])
        max_boundary_diff = np.max(np.abs(u_error[boundary_mask])) if boundary_mask.any() else 0.0

        print(f"    Sample {sample_idx + 1}: full-field reconstruction error = {diff_norm:.3e}")
        print(
            f"        velocity L2 error (boundary/interior) = {boundary_err:.3e} / {interior_err:.3e}"
        )
        print(f"        max boundary velocity mismatch = {max_boundary_diff:.3e}")

    if boundary_mask.any():
        unique_bc = np.unique(u_bc[boundary_mask])
        print(
            "\n    Stored boundary condition enforces the following velocity values "
            f"on the active nodes: {unique_bc}"
        )
        sample_velocities = original[N_NODES:2 * N_NODES][boundary_mask]
        print(
            "    Training snapshots contain boundary velocities with unique values: "
            f"{np.unique(np.round(sample_velocities, decimals=6))}"
        )
        print(
            "    If these sets differ, the residual boundary mismatch explains the non-zero"
            " full-field error while the interior reconstruction remains exact."
        )

    print("\nConclusion: if all reported errors are near machine precision, the mean-field separation\n            and recombination steps are internally consistent.")


if __name__ == "__main__":
    verify_mean_field_coupling()