import numpy as np
import pandas as pd
from scipy.optimize import fsolve
from scipy.interpolate import griddata
import os
import matplotlib.pyplot as plt

# --- ✨ 1. 저장 경로를 지정하는 변수 추가 ---
# 저장할 디렉터리 경로를 정의합니다.
# raw string (r"...")을 사용하여 Windows 경로의 백슬래시 문제를 방지합니다.
OUTPUT_DIRECTORY = r'C:\Users\spearlab05\Desktop\Galerkin ROM\FinalResult'

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

def run_online_stage():
    """ONLINE 단계 전체를 실행합니다."""
    try:
        # --- ✨ 4. 저장할 디렉터리가 없으면 자동으로 생성 ---
        os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
        print(f"Output files will be saved to: '{OUTPUT_DIRECTORY}'")

        offline_data = load_offline_data()
        K = int(offline_data['K'])

        re_input = input("Enter the Reynolds number (Re) for prediction: ")
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

        alpha_solution, info, ier, msg = fsolve(residual_with_monitor, alpha_initial_guess, args=args, full_output=True, xtol=1e-9)
        
        print("-----------------------------")

        if ier != 1:
            print(f"Warning: fsolve did not converge. Message: {msg}")
        else:
            print("fsolve converged successfully.")

        print(f"\nSolved alpha coefficients: {alpha_solution}")

        p, u, v = reconstruct_solution(alpha_solution, offline_data)

        # --- ✨ 5. 함수 호출 시 저장 경로를 전달 ---
        save_solution_to_csv(offline_data['coords'], p, u, v, Re, OUTPUT_DIRECTORY)
        plot_solution_interpolated(offline_data['coords'], p, u, v, Re, OUTPUT_DIRECTORY)
        
        print(f"\n--- ONLINE STAGE COMPLETE for Re = {Re} ---")

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except ValueError:
        print("Error: Invalid Reynolds number. Please enter a numeric value.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == '__main__':
    run_online_stage()