# online.py — Steady PPE-ROM (Lifting + POD + Galerkin, split form)

import os
from typing import Iterable, Optional, Tuple, Sequence
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.optimize import fsolve
from scipy.interpolate import griddata

# physical scales (match offline nondim)
RHO=998.2; U0=0.1; L0=0.01
OFFLINE_WAS_NONDIM = True
DEFAULT_OFFLINE_DATA = os.path.join(os.path.dirname(__file__), "ppe_rom_offline_data.npz")
DEFAULT_OUTPUT_DIRECTORY = os.path.join(os.path.dirname(__file__), "FinalResult")
#debug
# === DEBUG: 잔차 블록/항목별 분해 ===
def residual_blocks_with_parts(z, Re, offline, use_boundary_term=True, scaled=True):
    Ku = int(np.array(offline["Ku"]).item()); Kp = int(np.array(offline["Kp"]).item())
    a = z[:Ku]; b = z[Ku:]
    nu = 1.0/float(Re)

    A=offline["A"]; fA=offline["fA"]; B=offline["B"]
    Q=offline["Q"]; L=offline["L"]; c0=offline["c0"]
    D=offline["D"]; G2=offline["G2"]; G1=offline["G1"]; g0=offline["g0"]
    N1=offline["N1"]; N0=offline["N0"]; F=offline["F"]
    Ru_scale = float(np.array(offline.get("Ru_scale",1.0)).item())
    Rp_scale = float(np.array(offline.get("Rp_scale",1.0)).item())
    c_p = offline.get("c_p", np.zeros_like(c0))

    Qa  = np.einsum('mij,i,j->m', Q, a, a, optimize=True)
    aGa = np.einsum('ijk,j,k->i', G2, a, a, optimize=True)

    Ru_terms = {
        "nu(Aa+fA)" : nu*(A@a + fA),
        "c0"        : c0,
        "c_p"       : c_p,
        "L a"       : L@a,
        "Q(a,a)"    : Qa,
        "B b"       : B@b,
    }
    Rp_terms = {
        "D b"       : D@b,
        "g0"        : g0,
        "G1 a"      : G1@a,
        "G2(a,a)"   : aGa,
        "-nu*N"     : (-nu)*(N0 + (N1@a)) if use_boundary_term else 0.0,
        "F"         : F,
    }

    Ru = sum(Ru_terms.values())
    Rp = sum(Rp_terms.values())

    if scaled:
        Ru = Ru/max(Ru_scale,1e-12)
        Rp = Rp/max(Rp_scale,1e-12)

    # 각 항의 (스케일된) L2 노름도 함께 리턴
    term_norms_Ru = {k: np.linalg.norm(v/max(Ru_scale,1e-12)) for k,v in Ru_terms.items()}
    term_norms_Rp = {k: (np.linalg.norm(v/max(Rp_scale,1e-12)) if not np.isscalar(v) else np.abs(v)/max(Rp_scale,1e-12))
                     for k,v in Rp_terms.items()}

    return Ru, Rp, term_norms_Ru, term_norms_Rp

# === DEBUG: 야코비안 조건수/최소특이값 ===
def jacobian_with_cond(z, Re, offline, use_boundary_term=True):
    J = jacobian_split(z, Re, offline, use_boundary_term)
    s = np.linalg.svd(J, compute_uv=False)
    smin = s.min()
    cond = s.max()/max(smin,1e-16)
    return J, smin, cond







# I/O
def jacobian_split(z, Re, offline, use_boundary_term=True):
    """
    J = dR/dz for residual_steady_ppe_split.
    z = [a(0:Ku), b(0:Kp)]
    블록:
      [ dRu/da   dRu/db ]
      [ dRp/da   dRp/db ]
    주의: residual에서 Ru, Rp를 각각 Ru_scale, Rp_scale로 나눴으므로
         야코비안도 같은 스케일로 나눠준다.
    """
    Ku = int(np.array(offline["Ku"]).item())
    Kp = int(np.array(offline["Kp"]).item())
    a  = z[:Ku]
    nu = 1.0/float(Re)

    # 언팩
    A=offline["A"]; fA=offline["fA"]; B=offline["B"]
    Q=offline["Q"]; L=offline["L"]
    D=offline["D"]; G2=offline["G2"]; G1=offline["G1"]
    N1=offline["N1"]
    Ru_scale = float(np.array(offline.get("Ru_scale",1.0)).item())
    Rp_scale = float(np.array(offline.get("Rp_scale",1.0)).item())


    # term1[m,ℓ] = ∑_k Q[m,ℓ,k] a_k
    term1 = np.tensordot(Q, a, axes=(2, 0))      # (Ku, Ku)
    # term2[m,ℓ] = ∑_k Q[m,k,ℓ] a_k
    term2 = np.tensordot(Q, a, axes=(1, 0))      # (Ku, Ku)
    dRu_da = + nu * A + L + (term1 + term2)

    # ---- dRu/db = B ----
    dRu_db = B

    # ---- dRp/da = G1 + (∑_k (G2[:,ℓ,k] + G2[:,k,ℓ]) a_k) - nu*N1 ----
    H1 = np.tensordot(G2, a, axes=(2, 0))        # (Kp, Ku)  -> ∑_k G2[i,ℓ,k] a_k
    H2 = np.tensordot(G2, a, axes=(1, 0))        # (Kp, Ku)  -> ∑_k G2[i,k,ℓ] a_k
    dRp_da = G1 + (H1 + H2)
    if use_boundary_term:
        dRp_da = dRp_da - nu * N1

    # ---- dRp/db = D ----
    dRp_db = D

    # 스케일 반영
    dRu_da /= max(Ru_scale, 1e-12)
    dRu_db /= max(Ru_scale, 1e-12)
    dRp_da /= max(Rp_scale, 1e-12)
    dRp_db /= max(Rp_scale, 1e-12)

    # 큰 야코비안 조립
    J = np.zeros((Ku + Kp, Ku + Kp), dtype=float)
    J[:Ku, :Ku]   = dRu_da
    J[:Ku, Ku:]   = dRu_db
    J[Ku:, :Ku]   = dRp_da
    J[Ku:, Ku:]   = dRp_db
    return J



def load_offline_data(path: str = DEFAULT_OFFLINE_DATA)->dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing offline data: {path}")
    print(f"Loading offline data from '{path}' ...")
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}

def save_solution_to_csv(coords, p,u,v, Re, outdir):
    os.makedirs(outdir, exist_ok=True)
    fn = os.path.join(outdir, f"rom_solution_Re_{int(Re)}.csv")
    df = pd.DataFrame({"x-coordinate":coords[:,0], "y-coordinate":coords[:,1],
                       "pressure":p, "x-velocity":u, "y-velocity":v})
    df.to_csv(fn, index=False)
    print(f"Saved: {fn}")
    return fn

def to_physical_units(coords, p,u,v, nondim=OFFLINE_WAS_NONDIM):
    if not nondim: return coords,p,u,v
    c = coords.astype(float).copy(); c[:,0]*=L0; c[:,1]*=L0
    return c, p*(RHO*U0**2), u*U0, v*U0

def plot_solution_interpolated(coords,p,u,v, Re, outdir):
    os.makedirs(outdir, exist_ok=True)
    fn = os.path.join(outdir, f"rom_solution_Re_{int(Re)}_interpolated.png")
    x,y = coords[:,0], coords[:,1]
    gx,gy = np.mgrid[x.min():x.max():200j, y.min():y.max():200j]
    gp = griddata(coords, p, (gx,gy), method="linear")
    gu = griddata(coords, u, (gx,gy), method="linear")
    gv = griddata(coords, v, (gx,gy), method="linear")
    fig,ax = plt.subplots(1,3,figsize=(18,5))
    fig.suptitle(f"ROM Solution Re={Re}")
    c0=ax[0].contourf(gx,gy,gp,50,cmap="viridis"); fig.colorbar(c0,ax=ax[0]); ax[0].set_title("p")
    c1=ax[1].contourf(gx,gy,gu,50,cmap="viridis"); fig.colorbar(c1,ax=ax[1]); ax[1].set_title("u")
    c2=ax[2].contourf(gx,gy,gv,50,cmap="viridis"); fig.colorbar(c2,ax=ax[2]); ax[2].set_title("v")
    for a in ax: a.set_aspect("equal","box"); a.set_xlabel("x"); a.set_ylabel("y")
    plt.tight_layout(); plt.savefig(fn); plt.close(fig); print(f"Saved: {fn}")
    return fn

# Residual (split form)
def residual_steady_ppe_split(z, Re, offline, use_boundary_term=True):
    Ku = int(np.array(offline["Ku"]).item()); Kp = int(np.array(offline["Kp"]).item())
    a = z[:Ku]; b = z[Ku:]
    nu = 1.0/float(Re)

    # unpack
    A=offline["A"]; fA=offline["fA"]; B=offline["B"]
    Q=offline["Q"]; L=offline["L"]; c0=offline["c0"]
    D=offline["D"]; G2=offline["G2"]; G1=offline["G1"]; g0=offline["g0"]
    N1=offline["N1"]; N0=offline["N0"]; F=offline["F"]
    Ru_scale = float(np.array(offline.get("Ru_scale",1.0)).item())
    Rp_scale = float(np.array(offline.get("Rp_scale",1.0)).item())

    Qa = np.einsum('mij,i,j->m', Q, a, a, optimize=True)
    c_p = offline.get("c_p", np.zeros_like(c0))
    Ru  = + nu*(A@a + fA) + (c0 + c_p + L@a + Qa) + (B@b)

    aGa = np.einsum('ijk,j,k->i', G2, a, a, optimize=True)
    Rp = (D@b) + (g0 + G1@a + aGa) - (nu*((N0 + (N1@a)) if use_boundary_term else 0.0)) + F

    # block scaling (optional)
    return np.concatenate([Ru/max(Ru_scale,1e-12), Rp/max(Rp_scale,1e-12)])

def solve_steady_rom(Re, offline, z0=None, tol=1e-10, max_iter=10000, use_boundary_term=True):
    Ku = int(np.array(offline["Ku"]).item())
    Kp = int(np.array(offline["Kp"]).item())
    z0 = np.zeros(Ku+Kp) if z0 is None else np.asarray(z0, float)
    assert z0.size == Ku+Kp

    def wrap(z):
        # --- split ---
        Ku = int(np.array(offline["Ku"]).item())
        Kp = int(np.array(offline["Kp"]).item())
        a  = z[:Ku]
        b  = z[Ku:]

        # --- residual + 각 항 노름 ---
        Ru, Rp, nRu, nRp = residual_blocks_with_parts(
            z, Re, offline, use_boundary_term, scaled=True
        )
        r = np.concatenate([Ru, Rp])

        # --- 결합/발산 진단값 ---
        B = offline["B"]
        Bb = B @ b
        bnorm    = np.linalg.norm(b)
        Bb_norm  = np.linalg.norm(Bb)

        # div(u) = Σ_j a_j div(phi_j)  (offline.npz에 div_phi 저장해뒀다는 가정; 없으면 NaN)
        if ("div_phi" in offline):
            udiv = offline["div_phi"] @ a
            div_norm = np.linalg.norm(udiv)
        else:
            div_norm = np.nan

        # --- 헤더 ---
        if not hasattr(wrap, "it"):
            wrap.it = 0
            print("\n--- fsolve residual (scaled) ---")
            print(" iter |  ||Ru||   ||Rp||   ||r||  |  Ru terms [nuA+fA,c0,c_p,La,Qa,Bb]  |  Rp terms [Db,g0,G1a,G2aa,-nuN,F]  |  ||b||  ||B b||  ||div(u)||  |  smin(J)  cond(J)")
            print("-------------------------------------------------------------------------------------------------------------------------------------------------------------")

        # --- 요약 출력 (5회마다) + 야코비안 상태 ---
        if wrap.it % 5 == 0:
            smin = np.nan; cond = np.nan
            try:
                J = jacobian_split(z, Re, offline, use_boundary_term)
                s = np.linalg.svd(J, compute_uv=False)
                smin = s[-1]
                cond = s[0] / max(s[-1], 1e-300)
            except Exception:
                pass

            print(f"{wrap.it:5d} | {np.linalg.norm(Ru):7.2e} {np.linalg.norm(Rp):7.2e} {np.linalg.norm(r):7.2e} | "
                f"{nRu['nu(Aa+fA)']:7.2e} {nRu['c0']:7.2e} {nRu['c_p']:7.2e} {nRu['L a']:7.2e} {nRu['Q(a,a)']:7.2e} {nRu['B b']:7.2e} | "
                f"{nRp['D b']:7.2e} {nRp['g0']:7.2e} {nRp['G1 a']:7.2e} {nRp['G2(a,a)']:7.2e} {nRp['-nu*N']:7.2e} {nRp['F']:7.2e} | "
                f"{bnorm:7.2e} {Bb_norm:7.2e} {div_norm:7.2e} | {smin:7.2e} {cond:7.2e}")

        wrap.it += 1
        return r

    def jwrap(z):
        J, smin, cond = jacobian_with_cond(z, Re, offline, use_boundary_term)
        print(f"        [J] sigma_min={smin:.3e}  cond≈{cond:.3e}")
        return J

    sol, info, ier, msg = fsolve(wrap, z0, fprime=jwrap, full_output=True, xtol=tol, maxfev=max_iter)
    if ier != 1:
        print(f"[Warn] fsolve: {msg}")
        # 실패 시 최종 분해를 한 번 더 찍기
        Ru, Rp, nRu, nRp = residual_blocks_with_parts(sol, Re, offline, use_boundary_term, scaled=True)
        print(f"[Final] ||Ru||={np.linalg.norm(Ru):.3e}  ||Rp||={np.linalg.norm(Rp):.3e}  ||r||={np.linalg.norm(info['fvec']):.3e}")
        print(f"[Final] Ru parts: {nRu}")
        print(f"[Final] Rp parts: {nRp}")

    print(f"Final residual norm: {np.linalg.norm(info['fvec']):.6e}")
    return sol[:Ku], sol[Ku:]


def solve_steady_rom_ls(
    Re, offline, z0=None, tol=1e-10, max_nfev=20000,
    use_boundary_term=True, loss='linear', f_scale=1.0, verbose=2,
    method='trf'  # 'trf' 추천, 경계 쓸 때 특히 유리
):
    Ku = int(np.array(offline["Ku"]).item()); Kp = int(np.array(offline["Kp"]).item())
    z0 = np.zeros(Ku+Kp) if z0 is None else np.asarray(z0, float)
    assert z0.size == Ku + Kp

    def fun(z):
        return residual_steady_ppe_split(z, Re, offline, use_boundary_term)

    def jfun(z):
        return jacobian_split(z, Re, offline, use_boundary_term)

    res = least_squares(
        fun, z0,
        jac=jfun,
        method=method,
        loss=loss,
        f_scale=f_scale,
        x_scale='jac',
        ftol=tol, xtol=tol, gtol=tol,
        max_nfev=max_nfev,
        verbose=verbose,
        # bounds=(-np.inf, np.inf),
    )
    print(f"[least_squares] status={res.status}  message={res.message}")
    print(f"[least_squares] final ||r||_2 = {np.linalg.norm(res.fun):.6e}, nfev={res.nfev}")
    sol = res.x
    return sol[:Ku], sol[Ku:]



def reconstruct_fields(a,b,offline):
    u_modes=offline["u_modes"]; v_modes=offline["v_modes"]; chi=offline["chi"]
    u_bc=offline["u_bc"]; v_bc=offline["v_bc"]
    p_bar=offline["p_bar"] # p_bar 로드
    u = u_bc + (u_modes@a)
    v = v_bc + (v_modes@a)
    p = p_bar + (chi@b)  # 평균 압력 복원
    # p = p - p.mean() # 이 라인은 제거하는 것이 좋습니다.
    return p,u,v

def run_online_stage(re_values: Sequence[float] | float, output_dir: str|None=None, tol=1e-10, use_boundary_term=True , solver='fsolve'):
    output_dir = output_dir or DEFAULT_OUTPUT_DIRECTORY
    os.makedirs(output_dir, exist_ok=True)
    off = load_offline_data(DEFAULT_OFFLINE_DATA); coords=off["coords"]

    re_list = [float(re_values)] if isinstance(re_values,(int,float)) else [float(r) for r in re_values]
    Ku = int(np.array(off["Ku"]).item()); Kp = int(np.array(off["Kp"]).item())
    z_prev = np.zeros(Ku+Kp)

    for Re in re_list:
        print(f"\nSolving steady PPE-ROM (split, lifting) for Re={Re} ...")
        
        if solver == 'fsolve':
            a,b = solve_steady_rom(Re, off, z0=z_prev, tol=tol, use_boundary_term=use_boundary_term)
        elif solver == 'ls':
            a,b = solve_steady_rom_ls(
            Re, off, z0=z_prev, tol=tol, use_boundary_term=use_boundary_term,
            loss='linear',       # 필요시 'soft_l1'로 바꿔보면 수렴 더 잘 됨
            f_scale=1.0,         # soft_l1/huber 쓸 때만 의미 있음
            max_nfev=20000,
            verbose=2,
            )

        z_prev = np.concatenate([a,b])

        p,u,v = reconstruct_fields(a,b,off)
        c,p_,u_,v_ = to_physical_units(coords,p,u,v)
        save_solution_to_csv(c,p_,u_,v_,Re,output_dir)
        plot_solution_interpolated(c,p_,u_,v_,Re,output_dir)

if __name__=="__main__":

    re_list = np.arange(100, 1001, 100)
    re_list_test = [100,200,600,800,1000]
    run_online_stage(re_list,solver='fsolve')
