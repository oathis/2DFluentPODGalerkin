# online.py — Steady PPE-ROM (no p_bar, no F; pressure = chi@b)

import os
from typing import Sequence
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import fsolve, least_squares
from scipy.interpolate import griddata

# physical scales (match offline nondim)
RHO=998.2; U0=0.1; L0=0.01
OFFLINE_WAS_NONDIM = True
DEFAULT_OFFLINE_DATA = os.path.join(os.path.dirname(__file__), "ppe_rom_offline_data.npz")
DEFAULT_OUTPUT_DIRECTORY = os.path.join(os.path.dirname(__file__), "FinalResult")

def jacobian_split(z, Re, offline, use_boundary_term=True):
    Ku = int(np.asarray(offline["Ku"]).item())
    Kp = int(np.asarray(offline["Kp"]).item())
    a  = z[:Ku]
    nu = 1.0/float(Re)

    A=offline["A"]; B=offline["B"]
    Q=offline["Q"]; L=offline["L"]
    D=offline["D"]; G2=offline["G2"]; G1=offline["G1"]
    N1=offline["N1"]

    Ru_scale = float(np.asarray(offline.get("Ru_scale",1.0)))
    Rp_scale = float(np.asarray(offline.get("Rp_scale",1.0)))

    term1 = np.tensordot(Q, a, axes=(2, 0))      # (Ku, Ku)
    term2 = np.tensordot(Q, a, axes=(1, 0))      # (Ku, Ku)
    dRu_da = + nu * A + L + (term1 + term2)
    dRu_db = B

    H1 = np.tensordot(G2, a, axes=(2, 0))        # (Kp, Ku)
    H2 = np.tensordot(G2, a, axes=(1, 0))        # (Kp, Ku)
    dRp_da = G1 + (H1 + H2)
    if use_boundary_term:
        dRp_da = dRp_da + nu * N1
    dRp_db = D

    dRu_da /= max(Ru_scale, 1e-12)
    dRu_db /= max(Ru_scale, 1e-12)
    dRp_da /= max(Rp_scale, 1e-12)
    dRp_db /= max(Rp_scale, 1e-12)

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

# Residual (split form, no F)
def residual_steady_ppe_split(z, Re, offline, use_boundary_term=True):
    Ku = int(np.asarray(offline["Ku"]).item()); Kp = int(np.asarray(offline["Kp"]).item())
    a = z[:Ku]; b = z[Ku:]
    nu = 1.0/float(Re)

    A=offline["A"]; fA=offline["fA"]; B=offline["B"]
    Q=offline["Q"]; L=offline["L"]; c0=offline["c0"]
    D=offline["D"]; G2=offline["G2"]; G1=offline["G1"]; g0=offline["g0"]
    N1=offline["N1"]; N0=offline["N0"]

    Ru_scale = float(np.asarray(offline.get("Ru_scale",1.0)))
    Rp_scale = float(np.asarray(offline.get("Rp_scale",1.0)))

    Qa = np.einsum('mij,i,j->m', Q, a, a, optimize=True)
    Ru  = + nu*(A@a + fA) + (c0 + L@a + Qa) + (B@b)

    aGa = np.einsum('ijk,j,k->i', G2, a, a, optimize=True)
    Rp = (D@b) + (g0 + G1@a + aGa) + (nu*((N0 + (N1@a)) if use_boundary_term else 0.0))

    return np.concatenate([Ru/max(Ru_scale,1e-12), Rp/max(Rp_scale,1e-12)])

def solve_steady_rom(Re, offline, z0=None, tol=1e-10, max_iter=10000, use_boundary_term=True):
    Ku = int(np.asarray(offline["Ku"]).item())
    Kp = int(np.asarray(offline["Kp"]).item())
    z0 = np.zeros(Ku+Kp) if z0 is None else np.asarray(z0, float)

    def wrap(z):
        r = residual_steady_ppe_split(z, Re, offline, use_boundary_term)
        if not hasattr(wrap, "it"):
            wrap.it = 0
            print("\n--- fsolve residual (scaled) ---\n iter | ||res||_2\n--------------------")
        print(f"{wrap.it:5d} | {np.linalg.norm(r):.6e}")
        wrap.it += 1
        return r

    def jwrap(z):
        return jacobian_split(z, Re, offline, use_boundary_term)

    sol, info, ier, msg = fsolve(
        wrap, z0, fprime=jwrap, full_output=True, xtol=tol, maxfev=max_iter
    )
    if ier != 1:
        print(f"[Warn] fsolve: {msg}")
    print(f"Final residual norm: {np.linalg.norm(info['fvec']):.6e}")
    return sol[:Ku], sol[Ku:]

def solve_steady_rom_ls(Re, offline, z0=None, tol=1e-10, max_nfev=20000,
                        use_boundary_term=True, loss='linear', f_scale=1.0, verbose=2,
                        method='trf'):
    Ku = int(np.asarray(offline["Ku"]).item()); Kp = int(np.asarray(offline["Kp"]).item())
    z0 = np.zeros(Ku+Kp) if z0 is None else np.asarray(z0, float)

    def fun(z): return residual_steady_ppe_split(z, Re, offline, use_boundary_term)
    def jfun(z): return jacobian_split(z, Re, offline, use_boundary_term)

    res = least_squares(
        fun, z0, jac=jfun, method=method, loss=loss, f_scale=f_scale, x_scale='jac',
        ftol=tol, xtol=tol, gtol=tol, max_nfev=max_nfev, verbose=verbose,
    )
    print(f"[least_squares] status={res.status}  message={res.message}")
    print(f"[least_squares] final ||r||_2 = {np.linalg.norm(res.fun):.6e}, nfev={res.nfev}")
    sol = res.x
    return sol[:Ku], sol[Ku:]

def reconstruct_fields(a,b,offline):
    u_modes=offline["u_modes"]; v_modes=offline["v_modes"]; chi=offline["chi"]
    u_bc=offline["u_bc"]; v_bc=offline["v_bc"]
    u = u_bc + (u_modes@a)
    v = v_bc + (v_modes@a)
    p = chi @ b   # pure SVD pressure (no p_bar)
    return p,u,v

def run_online_stage(re_values: Sequence[float] | float, output_dir: str|None=None,
                     tol=1e-10, use_boundary_term=True , solver='fsolve'):
    output_dir = output_dir or DEFAULT_OUTPUT_DIRECTORY
    os.makedirs(output_dir, exist_ok=True)
    off = load_offline_data(DEFAULT_OFFLINE_DATA); coords=off["coords"]

    re_list = [float(re_values)] if isinstance(re_values,(int,float)) else [float(r) for r in re_values]
    Ku = int(np.asarray(off["Ku"]).item()); Kp = int(np.asarray(off["Kp"]).item())
    z_prev = np.zeros(Ku+Kp)

    for Re in re_list:
        print(f"\nSolving steady PPE-ROM (split, lifting) for Re={Re} ...")
        if solver == 'fsolve':
            a,b = solve_steady_rom(Re, off, z0=z_prev, tol=tol, use_boundary_term=use_boundary_term)
        else:
            a,b = solve_steady_rom_ls(Re, off, z0=z_prev, tol=tol, use_boundary_term=use_boundary_term,
                                      loss='soft_l1', f_scale=1.0, max_nfev=50000, verbose=2)
        z_prev = np.concatenate([a,b])

        p,u,v = reconstruct_fields(a,b,off)
        c,p_,u_,v_ = to_physical_units(coords,p,u,v)
        save_solution_to_csv(c,p_,u_,v_,Re,output_dir)
        plot_solution_interpolated(c,p_,u_,v_,Re,output_dir)

if __name__=="__main__":
    re_list = np.arange(100, 1001, 100)
    run_online_stage(re_list, solver='fsolve')
