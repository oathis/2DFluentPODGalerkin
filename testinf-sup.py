import os
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------
# 1) 도우미: 수치 미분/라플라시안 (당신 코드와 동일한 정의)
# ---------------------------
class DerivativeHelper:
    def __init__(self, nx, ny):
        self.nx, self.ny = nx, ny
        x = np.linspace(0.0, 1.0, nx)
        y = np.linspace(-1.0, 0.0, ny)
        self.delta_x = x[1] - x[0]
        self.delta_y = y[1] - y[0]

    def _to_2d(self, field_1d):
        return field_1d.reshape((self.ny, self.nx))

    def _to_1d(self, field_2d):
        return field_2d.flatten()

    def dx(self, field_1d):
        f2d = self._to_2d(field_1d)
        return self._to_1d(np.gradient(f2d, self.delta_x, axis=1))

    def dy(self, field_1d):
        f2d = self._to_2d(field_1d)
        return self._to_1d(np.gradient(f2d, self.delta_y, axis=0))

# ---------------------------
# 2) 가중 내적 / L2 노름
# ---------------------------
def weighted_inner(a, b, dx, dy):
    return float(np.dot(a, b) * dx * dy)

def weighted_norm(a, dx, dy):
    return np.sqrt(weighted_inner(a, a, dx, dy))

# ---------------------------
# 3) 체커보드 지표 (checkerboard index)
#    - (-1)^(i+j) 패턴과의 정규화된 상관값(절대값)
#    - 0~1 범위; 1에 가까울수록 "격자 한 칸 교번" 성향이 강함
# ---------------------------
def checkerboard_index(mode_1d, nx, ny):
    m2d = mode_1d.reshape(ny, nx)
    JJ, II = np.meshgrid(np.arange(nx), np.arange(ny))
    pattern = ((-1.0) ** (II + JJ))
    num = np.sum(m2d * pattern)
    denom = np.linalg.norm(m2d.ravel()) * np.linalg.norm(pattern.ravel())
    return abs(num / denom) if denom > 0 else 0.0

# ---------------------------
# 4) 압력 모드 시각화
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
        plt.title(f"Pressure POD mode #{k+1}\ncheckerboard index = {cidx:.3f}")
        plt.tight_layout()
        fname = os.path.join(outdir, f"pressure_mode_{k+1}.png")
        plt.savefig(fname, dpi=150)
        plt.close()
        print(f"[Saved] {fname}  (checkerboard index={cidx:.3f})")

# ---------------------------
# 4-추가) 속도 모드 시각화 (u, v, |u|, quiver)
# ---------------------------
def plot_velocity_modes(u_modes, v_modes, nx, ny, outdir, first_k=6, quiver_stride=None):
    os.makedirs(outdir, exist_ok=True)
    kshow = min(first_k, u_modes.shape[1])
    # quiver 화살표 간격 자동 설정(너무 빽빽하지 않게)
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

        # u-mode
        plt.figure(figsize=(5.2, 4.6))
        im = plt.imshow(u, origin='lower', aspect='equal')
        plt.colorbar(im)
        plt.title(f"u-mode #{k+1}\ncheckerboard index(u) = {cidx_u:.3f}")
        plt.tight_layout()
        fname_u = os.path.join(outdir, f"u_mode_{k+1}.png")
        plt.savefig(fname_u, dpi=150)
        plt.close()
        print(f"[Saved] {fname_u}  (checkerboard index={cidx_u:.3f})")

        # v-mode
        plt.figure(figsize=(5.2, 4.6))
        im = plt.imshow(v, origin='lower', aspect='equal')
        plt.colorbar(im)
        plt.title(f"v-mode #{k+1}\ncheckerboard index(v) = {cidx_v:.3f}")
        plt.tight_layout()
        fname_v = os.path.join(outdir, f"v_mode_{k+1}.png")
        plt.savefig(fname_v, dpi=150)
        plt.close()
        print(f"[Saved] {fname_v}  (checkerboard index={cidx_v:.3f})")

        # |u|-magnitude
        plt.figure(figsize=(5.2, 4.6))
        im = plt.imshow(mag, origin='lower', aspect='equal')
        plt.colorbar(im)
        plt.title(f"|velocity| mode #{k+1}")
        plt.tight_layout()
        fname_mag = os.path.join(outdir, f"velmag_mode_{k+1}.png")
        plt.savefig(fname_mag, dpi=150)
        plt.close()
        print(f"[Saved] {fname_mag}")

        # quiver (벡터 시각화)
        plt.figure(figsize=(5.6, 5.0))
        step = quiver_stride
        Q = plt.quiver(XX[::step, ::step], YY[::step, ::step],
                       u[::step, ::step], v[::step, ::step], scale=None)
        plt.title(f"Velocity vector (quiver) mode #{k+1}")
        plt.gca().set_aspect('equal', 'box')
        plt.tight_layout()
        fname_q = os.path.join(outdir, f"velvec_mode_{k+1}.png")
        plt.savefig(fname_q, dpi=150)
        plt.close()
        print(f"[Saved] {fname_q}")

# ---------------------------
# 5) B 행렬(SVD) & 발산 노름 진단 + 모드 성분 에너지/결합 세기
# ---------------------------
def rom_stability_diagnostics(npz_path='rom_offline_data.npz',
                              outdir='Diagnostics',
                              n_show_pressure_modes=6,
                              n_show_velocity_modes=6):
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"offline data not found: {npz_path}")

    data = np.load(npz_path)
    # 저장된 것들: p_modes (N,K), u_modes (N,K), v_modes (N,K), NX, NY, K, ...
    p_modes = data['p_modes']      # (N, K)
    u_modes = data['u_modes']      # (N, K)  // 이미 full-domain로 확장 저장됨
    v_modes = data['v_modes']      # (N, K)
    NX = int(data['NX'])
    NY = int(data['NY'])
    K  = int(data['K'])

    dx = 1.0 / (NX - 1)   # x: [0,1]
    dy = 1.0 / (NY - 1)   # y: [-1,0] 길이 1
    deriv = DerivativeHelper(NX, NY)

    print("=== ROM inf-sup / spurious-pressure diagnostics ===")
    print(f"Grid: NX={NX}, NY={NY}, K={K}  (dx={dx:.3e}, dy={dy:.3e})\n")

    # 5-0) 모드 성분별 에너지 비율(가중 L2) 출력
    print("Mode-wise component energy fractions (weighted):")
    for k in range(K):
        ep = weighted_norm(p_modes[:, k], dx, dy)**2
        eu = weighted_norm(u_modes[:, k], dx, dy)**2
        ev = weighted_norm(v_modes[:, k], dx, dy)**2
        etot = ep + eu + ev + 1e-30
        print(f"  k={k+1:2d}:  p={ep/etot:6.2%},  u={eu/etot:6.2%},  v={ev/etot:6.2%}")
    print("")

    # 5-1) 압력 모드 시각화(+체커보드 지표)
    plot_pressure_modes(p_modes, NX, NY, outdir, first_k=n_show_pressure_modes)

    # 5-1b) 속도 모드 시각화(u, v, |u|, quiver)
    plot_velocity_modes(u_modes, v_modes, NX, NY, outdir, first_k=n_show_velocity_modes)

    # 5-2) div(u_i) 사전 계산, 발산 노름 출력
    print("Divergence L2 norms of velocity modes (weighted)")
    div_list = []
    div_norms = []
    for i in range(K):
        div_ui = deriv.dx(u_modes[:, i]) + deriv.dy(v_modes[:, i])
        nrm = weighted_norm(div_ui, dx, dy)
        div_list.append(div_ui)
        div_norms.append(nrm)
        print(f"  i={i+1:2d}  ||div(phi_u,{i+1})||_2 = {nrm:.4e}")
    print("")

    # 5-3) 축소 inf-sup 커플링 행렬 B[m,i] = <div u_i, p_m>_M (가중 내적)
    B = np.zeros((K, K))
    for m in range(K):
        p_m = p_modes[:, m]
        for i in range(K):
            B[m, i] = weighted_inner(div_list[i], p_m, dx, dy)

    # (추가) 각 속도 모드의 결합 세기: ||B[:,i]||_2
    col_norms = np.linalg.norm(B, axis=0)
    print("Coupling strength per velocity mode  ||B[:, i]||_2:")
    for i in range(K):
        print(f"  i={i+1:2d}  ||B[:,{i+1}]||_2 = {col_norms[i]:.4e}")
    print("")

    # 5-4) SVD → sigma_min(B), cond(B)
    svals = np.linalg.svd(B, compute_uv=False)
    sigma_min = float(np.min(svals))
    sigma_max = float(np.max(svals))
    cond = (sigma_max / sigma_min) if sigma_min > 0 else np.inf

    print("Singular values of B (descending):")
    print("  " + ", ".join([f"{sv:.3e}" for sv in sorted(svals, reverse=True)]))
    print(f"\nσ_min(B) = {sigma_min:.3e}")
    print(f"cond(B)  = {cond:.3e}")
    if sigma_min < 1e-10:
        print(">> WARNING: σ_min(B) is ~0. Reduced inf-sup likely broken → spurious pressure risk HIGH.")
    elif sigma_min < 1e-6:
        print(">> CAUTION: σ_min(B) is small. Potential coupling weakness.")
    else:
        print(">> OK: σ_min(B) seems reasonably away from 0.")

    # 5-5) 결과 요약 반환
    return {
        "B": B,
        "singular_values": svals,
        "sigma_min": sigma_min,
        "cond": cond,
        "divergence_norms": np.array(div_norms),
        "coupling_col_norms": col_norms
    }

# ---------------------------
# 6) 예시 실행
# ---------------------------
if __name__ == "__main__":
    rom_stability_diagnostics(
        npz_path="rom_offline_data.npz",
        outdir="Diagnostics",
        n_show_pressure_modes=19,
        n_show_velocity_modes=19
    )
