import numpy as np
import pandas as pd
from scipy.optimize import fsolve
from scipy.interpolate import griddata
import os
import matplotlib.pyplot as plt



# ----- scales (choose once) -----
RHO = 998.2   # kg/m^3
U0  = 0.1     # m/s
L0  = 0.01    # m
OFFLINE_WAS_NONDIM = True


DEFAULT_OUTPUT_DIRECTORY = os.path.join(os.path.dirname(__file__), 'FinalResult')

def load_offline_data(filepath='rom_offline_data.npz'):
    """저장된 오프라인 데이터를 로드합니다."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"오프라인 데이터 파일 '{filepath}'을 찾을 수 없습니다. 먼저 offline.py를 실행하세요.")
    
    print(f"Loading offline data from '{filepath}'...")
    data = np.load(filepath)
    return {key: data[key] for key in data}

def rom_residual(alpha, Re, C1, C2, L1, L2, Q):
    """
    주어진 alpha와 Re에 대한 ROM 방정식의 잔차(residual)를 계산합니다.
    fsolve는 이 함수의 결과가 0이 되는 alpha를 찾습니다.
    """
    C = C1 + (1 / Re) * C2
    L = L1 + (1 / Re) * L2
    linear_term = L @ alpha
    quadratic_term = np.einsum('mij,i,j->m', Q, alpha, alpha)
    residual = C + linear_term + quadratic_term
    return residual

def residual_with_monitor(alpha, Re, C1, C2, L1, L2, Q):
    """
    rom_residual 함수를 감싸서 각 반복의 잔차 Norm을 출력합니다.
    SciPy 구 버전에 대한 호환성을 위해 callback 대신 이 방식을 사용합니다.
    """
    if not hasattr(residual_with_monitor, "iteration"):
        residual_with_monitor.iteration = 0
        print("\n--- fsolve convergence process ---")
        print(" Iter | Residual Norm (L2)")
        print("-----------------------------")

    residual = rom_residual(alpha, Re, C1, C2, L1, L2, Q)
    norm = np.linalg.norm(residual)
    print(f" {residual_with_monitor.iteration:4d} | {norm:.6e}")
    
    residual_with_monitor.iteration += 1
    return residual

def reconstruct_solution(alpha, modes_data):
    """계산된 계수 alpha를 사용하여 전체 유동장을 재구성합니다."""
    print("\nReconstructing full field solution...")
    p = modes_data['p_modes'] @ alpha
    u = modes_data['u_bc'] + (modes_data['u_modes'] @ alpha)
    v = modes_data['v_modes'] @ alpha
    return p, u, v

# --- ✨ 2. 함수가 저장 경로를 인자로 받도록 수정 ---
def save_solution_to_csv(coords, p, u, v, Re, output_dir):
    """결과를 지정된 경로에 CSV 파일로 저장합니다."""
    # os.path.join을 사용하여 운영체제에 맞는 파일 경로를 생성합니다.
    filename = os.path.join(output_dir, f'rom_solution_Re_{int(Re)}.csv')
    print(f"Saving solution to '{filename}'...")
    
    df = pd.DataFrame({
        'x-coordinate': coords[:, 0],
        'y-coordinate': coords[:, 1],
        'pressure': p,
        'x-velocity': u,
        'y-velocity': v
    })
    df.to_csv(filename, index=False)


def to_physical_units(coords, p, u, v, nondim=OFFLINE_WAS_NONDIM):
    """
    coords: (x*, y*) 또는 (x, y)
    p,u,v : 무차원(p*, u*, v*) 또는 유차원

    반환: (x[m], y[m]), p[Pa], u[m/s], v[m/s]
    """
    if nondim:
        coords_out = coords.astype(float).copy()
        coords_out[:, 0] *= L0
        coords_out[:, 1] *= L0
        p_out = p * (RHO * U0**2)
        u_out = u * U0
        v_out = v * U0
        return coords_out, p_out, u_out, v_out
    else:
        return coords, p, u, v

# --- ✨ 3. 함수가 저장 경로를 인자로 받도록 수정 ---
def plot_solution_interpolated(coords, p, u, v, Re, output_dir):
    """
    결과를 균일 격자에 보간하여 컨투어 플롯을 그리고 지정된 경로에 파일로 저장합니다.
    """
    # os.path.join을 사용하여 운영체제에 맞는 파일 경로를 생성합니다.
    filename = os.path.join(output_dir, f'rom_solution_Re_{int(Re)}_interpolated.png')
    print(f"Plotting interpolated solution and saving to '{filename}'...")

    x_orig = coords[:, 0]
    y_orig = coords[:, 1]

    x_min, x_max = x_orig.min(), x_orig.max()
    y_min, y_max = y_orig.min(), y_orig.max()
    grid_x, grid_y = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]

    grid_p = griddata(coords, p, (grid_x, grid_y), method='linear')
    grid_u = griddata(coords, u, (grid_x, grid_y), method='linear')
    grid_v = griddata(coords, v, (grid_x, grid_y), method='linear')

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'ROM Solution for Re = {Re} (Interpolated)', fontsize=16)

    contour1 = axes[0].contourf(grid_x, grid_y, grid_p, levels=50, cmap='viridis')
    fig.colorbar(contour1, ax=axes[0])
    axes[0].set_title('Pressure')
    axes[0].set_xlabel('x-coordinate')
    axes[0].set_ylabel('y-coordinate')
    axes[0].set_aspect('equal', 'box')

    contour2 = axes[1].contourf(grid_x, grid_y, grid_u, levels=50, cmap='viridis')
    fig.colorbar(contour2, ax=axes[1])
    axes[1].set_title('X-Velocity (u)')
    axes[1].set_xlabel('x-coordinate')
    axes[1].set_aspect('equal', 'box')

    contour3 = axes[2].contourf(grid_x, grid_y, grid_v, levels=50, cmap='viridis')
    fig.colorbar(contour3, ax=axes[2])
    axes[2].set_title('Y-Velocity (v)')
    axes[2].set_xlabel('x-coordinate')
    axes[2].set_aspect('equal', 'box')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(filename)
    plt.close()
    print("Plotting complete.")

def run_online_stage(re_input, output_dir=None):
    """ONLINE 단계 전체를 실행합니다."""
    try:
        output_dir = output_dir or DEFAULT_OUTPUT_DIRECTORY
        # --- ✨ 4. 저장할 디렉터리가 없으면 자동으로 생성 ---
        os.makedirs(output_dir, exist_ok=True)
        print(f"Output files will be saved to: '{output_dir}'")

        offline_data = load_offline_data()
        K = int(offline_data['K'])


        Re = float(re_input)

        print(f"\nSolving ROM for Re = {Re}...")
        alpha_initial_guess = np.zeros(K)
        args = (
            Re,
            offline_data['C1'], offline_data['C2'],
            offline_data['L1'], offline_data['L2'],
            offline_data['Q']
        )
        
        if hasattr(residual_with_monitor, "iteration"):
            del residual_with_monitor.iteration

        alpha_solution, info, ier, msg = fsolve(residual_with_monitor, alpha_initial_guess, args=args, full_output=True, xtol=1e-14)
        
        print("-----------------------------")

        if ier != 1:
            print(f"Warning: fsolve did not converge. Message: {msg}")
        else:
            print("fsolve converged successfully.")

        print(f"\nSolved alpha coefficients: {alpha_solution}")

        p, u, v = reconstruct_solution(alpha_solution, offline_data)
        coords_phys, p_out, u_out, v_out = to_physical_units(offline_data['coords'], p, u, v)

        # --- ✨ 5. 함수 호출 시 저장 경로를 전달 ---
        save_solution_to_csv(coords_phys, p_out, u_out, v_out, Re, output_dir)
        plot_solution_interpolated(coords_phys, p_out, u_out, v_out, Re, output_dir)
        
        print(f"\n--- ONLINE STAGE COMPLETE for Re = {Re} ---")

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except ValueError:
        print("Error: Invalid Reynolds number. Please enter a numeric value.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == '__main__':
    default_output_dir = os.environ.get('ONLINE_OUTPUT_DIRECTORY', DEFAULT_OUTPUT_DIRECTORY)
    # 100부터 1000까지 25씩 증가하는 Reynolds 수 리스트 생성
    # np.arange(start, stop, step)은 stop 값을 포함하지 않으므로 1001로 설정
    re_list = np.arange(100, 1001, 50)

    print(f"Starting online stage for {len(re_list)} Reynolds numbers...")
    print(re_list) # 생성된 리스트 확인 (선택 사항)

    for i in re_list:
        run_online_stage(i, output_dir=default_output_dir)

    print("\nAll simulations finished.")
