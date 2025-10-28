# offline.py — Steady PPE-ROM (Lifting + POD + Galerkin, with constant/linear/quadratic split)

import glob, os, re, time
from typing import List, Tuple
import numpy as np
import pandas as pd
from scipy.linalg import svd

# --- Config ---
NUM_CASES = 36
NX, NY = 101, 101
N_NODES = NX * NY
K_VEL = 36
K_PRS = 36
DEFAULT_DATA_DIRECTORY = os.path.join(os.path.dirname(__file__), "TrainData")
OUTPUT_FILENAME = "ppe_rom_offline_data.npz"

# --- I/O ---
def load_and_preprocess_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()
    df['x-coordinate'] = df['x-coordinate'].round(5)
    df['y-coordinate'] = df['y-coordinate'].round(5)
    df = df.sort_values(['y-coordinate','x-coordinate'])

    req = ['x-coordinate','y-coordinate','pressure','x-velocity','y-velocity']
    missing = set(req)-set(df.columns)
    if missing:
        raise ValueError(f"{os.path.basename(filepath)} missing {sorted(missing)}")

    # non-dim
    rho, u0, l0 = 998.2, 0.1, 0.01
    df['x-coordinate'] /= l0
    df['y-coordinate'] /= l0
    df['x-velocity']   /= u0
    df['y-velocity']   /= u0
    df['pressure']     /= (rho*u0**2)
    return df[req]

def collect_snapshot_files(directory: str) -> List[str]:
    files = glob.glob(os.path.join(directory, "*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV in '{directory}'")
    def key(p): 
        m = re.search(r"(\d+)", os.path.basename(p)); 
        return (int(m.group(1)) if m else -1, os.path.basename(p))
    files.sort(key=key)
    return files[:NUM_CASES]

# --- Derivatives on uniform grid ---
class DerivativeHelper:
    def __init__(self, nx:int, ny:int):
        self.nx, self.ny = nx, ny
        x = np.linspace(0.0, 1.0, nx)
        y = np.linspace(-1.0, 0.0, ny)
        self.dx = x[1]-x[0]; self.dy = y[1]-y[0]
    def _to_2d(self,f): return f.reshape((self.ny,self.nx))
    def _to_1d(self,F): return F.reshape(self.ny*self.nx)
    def dfx(self,f):
        F=self._to_2d(f); return self._to_1d(np.gradient(F,self.dx,axis=1))
    def dfy(self,f):
        F=self._to_2d(f); return self._to_1d(np.gradient(F,self.dy,axis=0))
    def lap(self,f):
        F=self._to_2d(f)
        gx=np.gradient(F,self.dx,axis=1); gy=np.gradient(F,self.dy,axis=0)
        lx=np.gradient(gx,self.dx,axis=1); ly=np.gradient(gy,self.dy,axis=0)
        return self._to_1d(lx+ly)

def ip(f,g,deriv:DerivativeHelper)->float:
    return float(np.dot(f,g)*(deriv.dx*deriv.dy))

def curl2d(u,v,deriv:DerivativeHelper):
    return deriv.dfx(v)-deriv.dfy(u)

def div_outer(ui,vi, uj,vj, deriv:DerivativeHelper):
    ui_uj = ui*uj; vi_uj = vi*uj; ui_vj = ui*vj; vi_vj = vi*vj
    div_x = deriv.dfx(ui_uj) + deriv.dfy(vi_uj)
    div_y = deriv.dfx(ui_vj) + deriv.dfy(vi_vj)
    return div_x, div_y


def boundary_indices(nx, ny):
    top    = np.arange((ny-1)*nx, ny*nx)[1:-1]
    bottom = np.arange(0, nx)[1:-1]
    left   = np.arange(0, ny*nx, nx)[1:-1]
    right  = left + (nx-1)
    return [(top,(0,1)), (bottom,(0,-1)), (left,(-1,0)), (right,(1,0))]
# --- Lifting field for lid-driven cavity ---
def build_lifting(coords: np.ndarray, nx:int, ny:int):
    # lid(top) u=1, others 0; v_bc=0 everywhere
    y = coords[:,1].reshape((ny,nx))
    u_bc = np.zeros(nx*ny)
    v_bc = np.zeros(nx*ny)
    top_mask = (y == y.max())
    u_bc[top_mask.flatten()] = 1.0
    return u_bc, v_bc

# --- Operator assembly with lifting split ---
def assemble_operators_lifting(phi_u, phi_v, chi, u_bc, v_bc, p_bar, deriv:DerivativeHelper):
    Ku = phi_u.shape[1]; Kp = chi.shape[1]
    # Pre-derivatives
    print("Precomputing derivatives...")
    lap_phi_u = np.column_stack([deriv.lap(phi_u[:,j]) for j in range(Ku)])
    lap_phi_v = np.column_stack([deriv.lap(phi_v[:,j]) for j in range(Ku)])
    dchi_dx = np.column_stack([deriv.dfx(chi[:,i]) for i in range(Kp)]) if Kp>0 else np.zeros((phi_u.shape[0],0))
    dchi_dy = np.column_stack([deriv.dfy(chi[:,i]) for i in range(Kp)]) if Kp>0 else np.zeros((phi_u.shape[0],0))

    # Viscous block: A and fA (lifting contribution)
    print("Assembling A and fA ...")
    A = np.zeros((Ku,Ku))
    for m in range(Ku):
        for j in range(Ku):
            A[m,j] = ip(phi_u[:,m], -lap_phi_u[:,j], deriv) + ip(phi_v[:,m], -lap_phi_v[:,j], deriv)
    fA_u = deriv.lap(u_bc); fA_v = deriv.lap(v_bc)  # note: fA enters as -nu * fA
    fA = np.array([ ip(phi_u[:,m], -fA_u, deriv) + ip(phi_v[:,m], -fA_v, deriv) for m in range(Ku) ])

    # Pressure coupling B
    print("Assembling B ...")
    B = np.zeros((Ku,Kp))
    for m in range(Ku):
        for i in range(Kp):
            B[m,i] = ip(phi_u[:,m], dchi_dx[:,i], deriv) + ip(phi_v[:,m], dchi_dy[:,i], deriv)

    # PPE Laplacian D
    print("Assembling D ...")
    D = np.zeros((Kp,Kp))
    for i in range(Kp):
        for k in range(Kp):
            D[i,k] = ip(dchi_dx[:,i], dchi_dx[:,k], deriv) + ip(dchi_dy[:,i], dchi_dy[:,k], deriv)

    # Convective split: c0, L, Q  (using divergence form)
    print("Assembling convective split (c0, L, Q) ...")
    # Q: modal-modal
    Q = np.zeros((Ku,Ku,Ku))
    for j in range(Ku):
        for k in range(Ku):
            div_x, div_y = div_outer(phi_u[:,j], phi_v[:,j], phi_u[:,k], phi_v[:,k], deriv)
            for m in range(Ku):
                Q[m,j,k] = ip(phi_u[:,m], div_x, deriv) + ip(phi_v[:,m], div_y, deriv)
    # L: lifting–modal
    L = np.zeros((Ku,Ku))
    for j in range(Ku):
        div_x, div_y = div_outer(u_bc, v_bc, phi_u[:,j], phi_v[:,j], deriv)
        div_x2, div_y2 = div_outer(phi_u[:,j], phi_v[:,j], u_bc, v_bc, deriv)
        div_x += div_x2; div_y += div_y2
        for m in range(Ku):
            L[m,j] = ip(phi_u[:,m], div_x, deriv) + ip(phi_v[:,m], div_y, deriv)
    # c0: lifting–lifting
    div_x, div_y = div_outer(u_bc, v_bc, u_bc, v_bc, deriv)
    c0 = np.array([ ip(phi_u[:,m], div_x, deriv) + ip(phi_v[:,m], div_y, deriv) for m in range(Ku) ])



    # PPE source split: g0, G1, G2
    print("Assembling PPE source split (g0, G1, G2) ...")
    # G2: modal-modal
    G2 = np.zeros((Kp,Ku,Ku))
    for j in range(Ku):
        for k in range(Ku):
            div_x, div_y = div_outer(phi_u[:,j], phi_v[:,j], phi_u[:,k], phi_v[:,k], deriv)
            for i in range(Kp):
                G2[i,j,k] = ip(dchi_dx[:,i], div_x, deriv) + ip(dchi_dy[:,i], div_y, deriv)
    # G1: lifting–modal
    G1 = np.zeros((Kp,Ku))
    for j in range(Ku):
        div_x, div_y = div_outer(u_bc, v_bc, phi_u[:,j], phi_v[:,j], deriv)
        div_x2, div_y2 = div_outer(phi_u[:,j], phi_v[:,j], u_bc, v_bc, deriv)
        div_x += div_x2; div_y += div_y2
        for i in range(Kp):
            G1[i,j] = ip(dchi_dx[:,i], div_x, deriv) + ip(dchi_dy[:,i], div_y, deriv)
    # g0: lifting–lifting
    div_x, div_y = div_outer(u_bc, v_bc, u_bc, v_bc, deriv)
    g0 = np.array([ ip(dchi_dx[:,i], div_x, deriv) + ip(dchi_dy[:,i], div_y, deriv) for i in range(Kp) ])

    # PPE boundary split: N0, N1  ( ∮ (t·∇χ) ω ds )
    print("Assembling PPE boundary split (N0, N1) ...")
    omega_modes = np.column_stack([curl2d(phi_u[:,j], phi_v[:,j], deriv) for j in range(Ku)])
    omega_bc = curl2d(u_bc, v_bc, deriv)
    N1 = np.zeros((Kp,Ku))
    N0 = np.zeros(Kp)
    for i in range(Kp):
        dchix_i = dchi_dx[:,i]; dchiy_i = dchi_dy[:,i]
        for indices, normal in boundary_indices(deriv.nx, deriv.ny):
            nx, ny = normal
            tgrad = nx*dchiy_i[indices] - ny*dchix_i[indices]   # t·∇χ
            ds = (deriv.dx if abs(ny)==1 else deriv.dy)
            N1[i,:] += ds*(tgrad @ omega_modes[indices,:])
            N0[i]   += ds*np.dot(tgrad, omega_bc[indices])

    # with this
    print("Assembling F (mean pressure term for PPE) ...")
    d_pbar_dx = deriv.dfx(p_bar)
    d_pbar_dy = deriv.dfy(p_bar)
    F = np.array([
        ip(dchi_dx[:,i], d_pbar_dx, deriv) + ip(dchi_dy[:,i], d_pbar_dy, deriv)
        for i in range(Kp)
    ])
    c_p = np.array([
        ip(phi_u[:,m], deriv.dfx(p_bar), deriv)
    + ip(phi_v[:,m], deriv.dfy(p_bar), deriv)
    for m in range(Ku)
    ])    
    return dict(
        A=A, fA=fA, B=B, Q=Q, L=L, c0=c0,
        D=D, G2=G2, G1=G1, g0=g0, N1=N1, N0=N0, F=F, c_p=c_p
    )

def main(data_directory: str = DEFAULT_DATA_DIRECTORY):
    t0 = time.time()
    print("OFFLINE: Steady PPE-ROM with lifting")
    files = collect_snapshot_files(data_directory)
    print(f"Using {len(files)} snapshots")

    Qp = np.zeros((N_NODES, len(files)))
    Qu = np.zeros((2*N_NODES, len(files)))
    coords = None
    for k,fp in enumerate(files):
        df = load_and_preprocess_data(fp)
        if coords is None:
            coords = df[['x-coordinate','y-coordinate']].to_numpy()
        Qp[:,k]          = df['pressure'].to_numpy()
        Qu[:N_NODES,k]   = df['x-velocity'].to_numpy()
        Qu[N_NODES:,k]   = df['y-velocity'].to_numpy()
        print(f"  loaded {k+1:02d}/{len(files)}: {os.path.basename(fp)}")

    # Lifting
    u_bc, v_bc = build_lifting(coords, NX, NY)

    # Build fluctuation snapshots for velocity
    Qu_fluc = Qu.copy()
    Qu_fluc[:N_NODES,:]   -= u_bc[:,None]
    Qu_fluc[N_NODES:,:]   -= v_bc[:,None]   # v_bc=0

    # Pressure mean-removed
    # 1) 스냅샷별 공간 평균 먼저 제거(게이지 고정)
    Qp_gauge = Qp - Qp.mean(axis=0, keepdims=True)

    # 2) 그 '게이지 압력'들에 대해 평균장 p̄ 계산
    p_bar = Qp_gauge.mean(axis=1)

    # 3) 변동장은 게이지에서 p̄를 뺀 것
    Qp_fluc = Qp_gauge - p_bar[:, None]

    # POD
    print("POD (SVD) ...")
    Up, sp, _ = svd(Qp_fluc, full_matrices=False)
    Uu, su, _ = svd(Qu_fluc, full_matrices=False)
    Kp = min(K_PRS, Up.shape[1]); Ku = min(K_VEL, Uu.shape[1])
    chi = Up[:, :Kp]
    N = chi.shape[0]
    const = np.ones((N, 1))
    const /= np.linalg.norm(const)  # L2 정규화

    # 1) chi에서 상수 성분을 정투영 제거
    #    chi <- chi - (const const^T) chi
    chi = chi - const @ (const.T @ chi)

    # 2) 수치적으로 깔끔하게 재직교(QR)
    chi, _ = np.linalg.qr(chi)

    # 3) 모드 개수 갱신(혹시 수치적으로 축소되면 반영)
    Kp = chi.shape[1]    
    phi_uv = Uu[:, :Ku]
    phi_u = phi_uv[:N_NODES,:]
    phi_v = phi_uv[N_NODES:,:]

    print(f"Truncation: Ku={Ku}, Kp={Kp}")

    # Operators
    deriv = DerivativeHelper(NX, NY)
    ops = assemble_operators_lifting(phi_u, phi_v, chi, u_bc, v_bc,p_bar, deriv)

    # Scales for block-balancing (optional)
    Ru_scale = max(np.linalg.norm(ops['A']), 1.0)
    Rp_scale = max(np.linalg.norm(ops['D']), 1.0)

    out = os.path.join(os.path.dirname(__file__), OUTPUT_FILENAME)
    np.savez(
        out,
        # bases
        u_modes=phi_u, v_modes=phi_v, chi=chi,
        u_bc=u_bc, v_bc=v_bc, p_bar=p_bar,
        # operators
        **ops,
        # sizes & grid
        Ku=Ku, Kp=Kp, NX=NX, NY=NY, coords=coords,
        # scales
        Ru_scale=Ru_scale, Rp_scale=Rp_scale
    )
    print(f"OFFLINE complete. Saved: {out}  (time {time.time()-t0:.2f}s)")

if __name__ == "__main__":
    main()
