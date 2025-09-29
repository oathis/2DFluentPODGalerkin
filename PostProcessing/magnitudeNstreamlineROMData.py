import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

def plot_velocity_and_streamlines(file_path, output_dir):
    """
    CSV 파일에서 데이터를 읽어 속도 크기(velocity magnitude)와 
    유선(streamline)을 계산하고, 이를 컨투어 플롯과 유선 플롯으로 시각화하여
    하나의 PNG 파일로 저장합니다.

    Args:
        file_path (str): 입력 CSV 파일의 전체 경로
        output_dir (str): 생성된 PNG 파일을 저장할 디렉토리 경로
    """
    try:
        # 파일 이름에서 Reynolds 수(Re) 추출
        filename = os.path.basename(file_path)
        re_value = int(filename.split('_')[-1].replace('.csv', ''))
        print(f"Processing file for Re = {re_value}...")
    except (IndexError, ValueError):
        print(f"Warning: Could not extract Reynolds number from '{filename}'. Skipping.")
        return

    # Pandas를 사용하여 CSV 파일 읽기
    data = pd.read_csv(file_path)

    # 필요한 열(column)이 모두 있는지 확인
    required_columns = ['x-coordinate', 'y-coordinate', 'x-velocity', 'y-velocity']
    if not all(col in data.columns for col in required_columns):
        print(f"Error: File '{filename}' is missing one of the required columns: {required_columns}. Skipping.")
        return
        
    coords = data[['x-coordinate', 'y-coordinate']].values
    u = data['x-velocity'].values
    v = data['y-velocity'].values

    # 속도 크기 계산: magnitude = sqrt(u^2 + v^2)
    velocity_magnitude = np.sqrt(u**2 + v**2)

    # ================================================================= #
    # 수정된 부분: np.meshgrid를 사용하여 격자 생성
    # ================================================================= #
    x_orig, y_orig = coords[:, 0], coords[:, 1]
    x_min, x_max = x_orig.min(), x_orig.max()
    y_min, y_max = y_orig.min(), y_orig.max()
    
    # 1. 각 축에 대해 1차원 배열 생성
    grid_xi = np.linspace(x_min, x_max, 200)
    grid_yi = np.linspace(y_min, y_max, 200)
    
    # 2. 1차원 배열을 이용해 2차원 격자 생성
    grid_x, grid_y = np.meshgrid(grid_xi, grid_yi)
    # ================================================================= #

    # 격자 위에 u, v, 속도 크기 보간
    grid_u = griddata(coords, u, (grid_x, grid_y), method='linear')
    grid_v = griddata(coords, v, (grid_x, grid_y), method='linear')
    grid_magnitude = griddata(coords, velocity_magnitude, (grid_x, grid_y), method='linear')

    # 시각화 (2개의 서브플롯)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f'Flow Visualization for Re = {re_value}', fontsize=16)

    # 1. 속도 크기 컨투어 플롯
    contour = axes[0].contourf(grid_x, grid_y, grid_magnitude, levels=50, cmap='viridis')
    fig.colorbar(contour, ax=axes[0], label='Velocity Magnitude')
    axes[0].set_title('Velocity Magnitude')
    axes[0].set_xlabel('x-coordinate')
    axes[0].set_ylabel('y-coordinate')
    axes[0].set_aspect('equal', 'box')

    # 2. 유선 플롯
    axes[1].contourf(grid_x, grid_y, grid_magnitude, levels=50, cmap='viridis')
    # streamplot은 NaN 값을 처리하지 못하므로 0으로 대체
    grid_u_filled = np.nan_to_num(grid_u)
    grid_v_filled = np.nan_to_num(grid_v)
    axes[1].streamplot(grid_x, grid_y, grid_u_filled, grid_v_filled, density=1.5, color='white', linewidth=0.7)
    axes[1].set_title('Streamlines')
    axes[1].set_xlabel('x-coordinate')
    axes[1].set_ylabel('') # y-label 중복 방지
    axes[1].set_aspect('equal', 'box')

    # 레이아웃 조정 및 파일 저장
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    output_filename = os.path.join(output_dir, f'flow_visualization_Re_{re_value}.png')
    plt.savefig(output_filename)
    plt.close(fig) # 메모리 해제를 위해 플롯 닫기
    print(f"Successfully saved plot to '{output_filename}'")


# --- 메인 실행 부분 ---

# 1. CSV 파일이 있는 기본 디렉토리 경로를 설정해주세요.
base_directory = r'C:\Users\spearlab05\Desktop\Galerkin ROM\FinalResult'

# 2. 생성된 PNG 이미지를 저장할 디렉토리 경로를 설정해주세요.
output_directory =r'C:\Users\spearlab05\Desktop\Galerkin ROM\FinalResult_Streamline'

# 3. 저장할 디렉토리가 없으면 새로 생성합니다.
if not os.path.exists(output_directory):
    os.makedirs(output_directory)
    print(f"Created output directory: {output_directory}")

# 4. Re 범위를 100부터 1000까지 50씩 증가시키며 반복합니다.
for re_number in range(100, 1001, 10):
    file_name = f'rom_solution_Re_{re_number}.csv'
    full_file_path = os.path.join(base_directory, file_name)

    # 해당 경로에 파일이 존재하는지 확인 후 함수 실행
    if os.path.exists(full_file_path):
        plot_velocity_and_streamlines(full_file_path, output_directory)
    else:
        print(f"File not found: {full_file_path}")

print("\nAll files processed.")