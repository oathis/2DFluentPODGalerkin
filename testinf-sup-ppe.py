import os
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------
# 1) 동일 정의의 DerivativeHelper (offline.py와 맞춤)
# ---------------------------
class DerivativeHelper:
    def __init__(self, nx:int, ny:int):
        self.nx, self.ny = nx, ny
        x = np.linspace(0.0, 1.0, nx)
        y = np.linspace(-1.0, 0.0, ny)
        self.dx = x[1]-x[0]
        self.dy = y[1]-y[0]
    def _to_2d(self,f): return f.reshape((self.ny,self.nx))
    def _to_1d(self,F): return F.reshape(self.ny*self.nx)
    def dfx(self,f):
        F=self._to_2d(f); return self._to_1d(np.gradient(F,self.dx,axis=1))
    def dfy(self,f):
        F=self._to_2d(f); return self._to_1d(np.gradient(F,self.dy,axis=0))

# ---------------------------
# 2) 가중 내적 / L2 노름
# ---------------------------
def weighted_inner(a, b, dx, dy):
    return float(np.dot(a, b) * dx * dy)

def weighted_norm(a, dx, dy):
    return np.sqrt(weighted_inner(a, a, dx, dy))

# ---------------------------
# 3) 체커보드 지표
# ---------------------------
def checkerboard_index(mode_1d, nx, ny):
    m2d = mode_1d.reshape(ny, nx)
    JJ, II = np.meshgrid(np.arange(nx), np.arange(ny))
    pattern = ((-1.0) ** (II + JJ))
    num = np.sum(m2d * pattern)
    denom = np.linalg.norm(m2d.ravel()) * np.linalg.norm(pattern.ravel())
    return abs(num / denom) if denom > 0 else 0.0

# ---------------------------
# 4) 시각화
# ---------------------------
def plot_pressure_modes(p_modes, nx, ny, outdir, first_k=6):
    os.makedirs(outdir, exist_ok=True)
    kshow = min(first_k, p_modes.shape[1])
    for k in range(kshow):
        mode = p_modes[:, k]
        m2d = mode.reshape(ny, nx)
        cidx = checkerboard_index(mode, nx, ny)
        plt.figure(figsize=(5.2, 4.6))
        im = plt.imshow(m2d, origin='lower', aspect='equal')
        plt.colorbar(im)
        plt.title(f"Pressure mode #{k+1}\ncheckerboard index = {cidx:.3f}")
        plt.tight_layout()
        fname = os.path.join(outdir, f"pressure_mode_{k+1}.png")
        plt.savefig(fname, dpi=150)
        plt.close()
        print(f"[Saved] {fname}  (checkerboard index={cidx:.3f})")

def plot_velocity_modes(u_modes, v_modes, nx, ny, outdir, first_k=6, quiver_stride=None):
    os.makedirs(outdir, exist_ok=True)
    kshow = min(first_k, u_modes.shape[1])
    if quiver_stride is None:
        quiver_stride = max(nx // 25, 1)
    X = np.linspace(0.0, 1.0, nx)
    Y = np.linspace(-1.0, 0.0, ny)
    XX, YY = np.meshgrid(X, Y)
    for k in range(kshow):
        u = u_modes[:, k].reshape(ny, nx)
        v = v_modes[:, k].reshape(ny, nx)
        mag = np.sqrt(u**2 + v**2)
        cidx_u = checkerboard_index(u.ravel(), nx, ny)
        cidx_v = checkerboard_index(v.ravel(), nx, ny)
        # u
        plt.figure(figsize=(5.2, 4.6))
        im = plt.imshow(u, origin='lower', aspect='equal')
        plt.colorbar(im)
        plt.title(f"u-mode #{k+1}\ncheckerboard index(u) = {cidx_u:.3f}")
        plt.tight_layout()
        fname_u = os.path.join(outdir, f"u_mode_{k+1}.png")
        plt.savefig(fname_u, dpi=150); plt.close()
        print(f"[Saved] {fname_u}  (checkerboard index={cidx_u:.3f})")
        # v
        plt.figure(figsize=(5.2, 4.6))
        im = plt.imshow(v, origin='lower', aspect='equal')
        plt.colorbar(im)
        plt.title(f"v-mode #{k+1}\ncheckerboard index(v) = {cidx_v:.3f}")
        plt.tight_layout()
        fname_v = os.path.join(outdir, f"v_mode_{k+1}.png")
        plt.savefig(fname_v, dpi=150); plt.close()
        print(f"[Saved] {fname_v}  (checkerboard index={cidx_v:.3f})")
        # |u|
        plt.figure(figsize=(5.2, 4.6))
        im = plt.imshow(mag, origin='lower', aspect='equal')
        plt.colorbar(im)
        plt.title(f"|velocity| mode #{k+1}")
        plt.tight_layout()
        fname_mag = os.path.join(outdir, f"velmag_mode_{k+1}.png")
        plt.savefig(fname_mag, dpi=150); plt.close()
        print(f"[Saved] {fname_mag}")
        # quiver
        plt.figure(figsize=(5.6, 5.0))
        step = quiver_stride
        plt.quiver(XX[::step, ::step], YY[::step, ::step],
                   u[::step, ::step], v[::step, ::step], scale=None)
        plt.title(f"Velocity vector (quiver) mode #{k+1}")
        plt.gca().set_aspect('equal', 'box')
        plt.tight_layout()
        fname_q = os.path.join(outdir, f"velvec_mode_{k+1}.png")
        plt.savefig(fname_q, dpi=150); plt.close()
        print(f"[Saved] {fname_q}")

# ---------------------------
# 5) offline.npz 포맷에 맞춘 진단
# ---------------------------
def rom_stability_diagnostics_offline(npz_path='ppe_rom_offline_data.npz',
                                      outdir='Diagnostics',
                                      n_show_pressure_modes=None,
                                      n_show_velocity_modes=None):
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"offline data not found: {npz_path}")

    data = np.load(npz_path, allow_pickle=True)

    # ---- 필수 키 로드 (offline.py 포맷) ----
    chi     = data['chi']        # (N, Kp)
    u_modes = data['u_modes']    # (N, Ku)
    v_modes = data['v_modes']    # (N, Ku)
    NX = int(np.array(data['NX']).item())
    NY = int(np.array(data['NY']).item())
    Ku = int(np.array(data['Ku']).item())
    Kp = int(np.array(data['Kp']).item())

    deriv = DerivativeHelper(NX, NY)
    w = deriv.dx * deriv.dy

    print("=== ROM offline diagnostics (matching offline.py) ===")
    print(f"Grid: NX={NX}, NY={NY}, Ku={Ku}, Kp={Kp}  (dx={deriv.dx:.3e}, dy={deriv.dy:.3e})\n")

    # ---- 시각화 ----
    if n_show_pressure_modes is None: n_show_pressure_modes = Kp
    if n_show_velocity_modes is None: n_show_velocity_modes = Ku
    if Kp > 0:
        plot_pressure_modes(chi, NX, NY, outdir, first_k=n_show_pressure_modes)
    if Ku > 0:
        plot_velocity_modes(u_modes, v_modes, NX, NY, outdir, first_k=n_show_velocity_modes)

    # ---- 속도 모드 발산 L2 ----
    print("Divergence L2 norms of velocity modes (weighted):")
    div_list = []
    div_norms = []
    for i in range(Ku):
        div_ui = deriv.dfx(u_modes[:, i]) + deriv.dfy(v_modes[:, i])
        nrm = weighted_norm(div_ui, deriv.dx, deriv.dy)
        div_list.append(div_ui)
        div_norms.append(nrm)
        print(f"  i={i+1:2d}  ||div(phi_{i+1})||_2 = {nrm:.4e}")
    print("")
    div_list = np.column_stack(div_list) if Ku>0 else np.zeros((u_modes.shape[0], 0))

    # ---- B 재구성(offline 정의와 동일: <phi_u, dchi/dx> + <phi_v, dchi/dy>) ----
    if Kp > 0 and Ku > 0:
        dchi_dx = np.column_stack([deriv.dfx(chi[:, i]) for i in range(Kp)])
        dchi_dy = np.column_stack([deriv.dfy(chi[:, i]) for i in range(Kp)])
        B_recon = np.zeros((Ku, Kp))
        for m in range(Ku):
            for i in range(Kp):
                B_recon[m, i] = (weighted_inner(u_modes[:, m], dchi_dx[:, i], deriv.dx, deriv.dy) +
                                 weighted_inner(v_modes[:, m], dchi_dy[:, i], deriv.dx, deriv.dy))
    else:
        B_recon = np.zeros((Ku, Kp))

    # ---- 저장된 B와 비교 ----
    B_saved = data['B']  # (Ku, Kp)
    diff = np.linalg.norm(B_saved - B_recon)
    print(f"‖B_saved - B_recon‖_F = {diff:.3e}\n")

    # ---- inf-sup 스케일된 B~ SVD ----
    if Ku>0 and Kp>0:
        Mu = w * (u_modes.T @ u_modes + v_modes.T @ v_modes)     # (Ku,Ku)
        Mp = w * (chi.T @ chi)                                    # (Kp,Kp)
        # 수치안정용 조그마한 jitter
        Mu += 1e-15 * np.eye(Ku)
        Mp += 1e-15 * np.eye(Kp)
        Lu = np.linalg.cholesky(Mu)
        Lp = np.linalg.cholesky(Mp)
        Btilde = np.linalg.solve(Lu, B_saved) @ np.linalg.inv(Lp).T
        svals = np.linalg.svd(Btilde, compute_uv=False)
        beta = float(np.min(svals)) if svals.size else 0.0
        condB = float(np.max(svals)/max(beta,1e-16)) if svals.size else np.inf
        print("Singular values of B~ (descending):")
        print("  " + ", ".join([f"{sv:.3e}" for sv in sorted(svals, reverse=True)]))
        print(f"\nβ (inf-sup proxy) ≈ {beta:.3e}")
        print(f"cond(B~) ≈ {condB:.3e}\n")
    else:
        svals = np.array([]); beta = 0.0; condB = np.inf
        print("B~ SVD skipped (Ku==0 or Kp==0)\n")

    # ---- D 고유값/조건수 ----
    D = data['D']  # (Kp,Kp)
    Dsym = 0.5*(D + D.T)
    evals = np.linalg.eigvalsh(Dsym) if Dsym.size else np.array([0.0])
    Dcond = float(np.max(evals)/max(np.min(evals),1e-16)) if evals.size else np.inf
    print(f"D eig(min,max)=({np.min(evals):.3e}, {np.max(evals):.3e})   cond≈{Dcond:.3e}\n")

    # ---- 속도 모드별 결합 세기(행 노름) ----
    if Ku>0 and Kp>0:
        row_norms = np.linalg.norm(B_saved, axis=1)  # each velocity mode i: ||B[i,:]||_2
        print("Coupling strength per velocity mode  ||B[i,:]||_2:")
        for i in range(Ku):
            print(f"  i={i+1:2d}  ||B[{i+1},:]||_2 = {row_norms[i]:.4e}")
        print("")

    return {
        "B_saved": B_saved,
        "B_recon": B_recon,
        "B_diff_F": diff,
        "singular_values_Btilde": svals,
        "beta": beta,
        "cond_Btilde": condB,
        "D_eigs": evals,
        "D_cond": Dcond,
        "divergence_norms": np.array(div_norms),
    }

# ---------------------------
# 6) 예시 실행
# ---------------------------
if __name__ == "__main__":
    rom_stability_diagnostics_offline(
        npz_path="ppe_rom_offline_data.npz",
        outdir="Diagnostics",
        n_show_pressure_modes=36,   # Kp=1이면 1로 둬도 좋음
        n_show_velocity_modes=36
    )
