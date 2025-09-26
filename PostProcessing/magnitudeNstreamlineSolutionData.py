import os
import glob # 파일 경로 패턴을 검색하기 위해 glob 라이브러리 추가
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

def plot_velocity_and_streamlines(file_path, output_dir):
    """
    CSV 파일에서 데이터를 읽어 속도 크기(velocity magnitude)와 
    유선(streamline)을 계산하고, 이를 시각화하여 PNG 파일로 저장합니다.
    """
    try:
        # 파일 이름에서 'case'와 '_sorted.csv'를 제거하여 케이스 번호 추출
        filename = os.path.basename(file_path)
        case_number_str = filename.replace('case', '').replace('_sorted.csv', '')
        case_number = int(case_number_str)
        print(f"Processing file for Case {case_number}...")
    except (IndexError, ValueError):
        print(f"Warning: Could not extract case number from '{filename}'. Skipping.")
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

    # 속도 크기 계산
    velocity_magnitude = np.sqrt(u**2 + v**2)

    # np.meshgrid를 사용하여 보간을 위한 격자 생성
    x_orig, y_orig = coords[:, 0], coords[:, 1]
    x_min, x_max = x_orig.min(), x_orig.max()
    y_min, y_max = y_orig.min(), y_orig.max()
    grid_xi = np.linspace(x_min, x_max, 200)
    grid_yi = np.linspace(y_min, y_max, 200)
    grid_x, grid_y = np.meshgrid(grid_xi, grid_yi)

    # 격자 위에 u, v, 속도 크기 보간
    grid_u = griddata(coords, u, (grid_x, grid_y), method='linear')
    grid_v = griddata(coords, v, (grid_x, grid_y), method='linear')
    grid_magnitude = griddata(coords, velocity_magnitude, (grid_x, grid_y), method='linear')

    # 시각화 (2개의 서브플롯)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    # 제목을 케이스 번호로 변경
    fig.suptitle(f'Flow Visualization for Case {case_number}', fontsize=16)

    # 1. 속도 크기 컨투어 플롯
    contour = axes[0].contourf(grid_x, grid_y, grid_magnitude, levels=50, cmap='viridis')
    fig.colorbar(contour, ax=axes[0], label='Velocity Magnitude')
    axes[0].set_title('Velocity Magnitude')
    axes[0].set_xlabel('x-coordinate')
    axes[0].set_ylabel('y-coordinate')
    axes[0].set_aspect('equal', 'box')

    # 2. 유선 플롯
    axes[1].contourf(grid_x, grid_y, grid_magnitude, levels=50, cmap='viridis')
    grid_u_filled = np.nan_to_num(grid_u)
    grid_v_filled = np.nan_to_num(grid_v)
    axes[1].streamplot(grid_x, grid_y, grid_u_filled, grid_v_filled, density=1.5, color='white', linewidth=0.7)
    axes[1].set_title('Streamlines')
    axes[1].set_xlabel('x-coordinate')
    axes[1].set_ylabel('')
    axes[1].set_aspect('equal', 'box')

    # 저장될 파일 이름을 케이스 번호 형식으로 변경
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    output_filename = os.path.join(output_dir, f'flow_visualization_case_{case_number}.png')
    plt.savefig(output_filename)
    plt.close(fig)
    print(f"Successfully saved plot to '{output_filename}'")


# --- 메인 실행 부분 ---

# 1. CSV 파일이 있는 기본 디렉토리 경로를 설정해주세요.
base_directory = r'C:\Users\spearlab05\Desktop\Galerkin ROM\Data'

# 2. 생성된 PNG 이미지를 저장할 디렉토리 경로를 설정해주세요.
output_directory = r'C:\Users\spearlab05\Desktop\Galerkin ROM\StreamlineOfSOLUTIONDATA'

# 3. 저장할 디렉토리가 없으면 새로 생성합니다.
if not os.path.exists(output_directory):
    os.makedirs(output_directory)
    print(f"Created output directory: {output_directory}")

# 4. 'case*_sorted.csv' 패턴과 일치하는 모든 파일을 찾습니다.
file_pattern = os.path.join(base_directory, 'case*_sorted.csv')
file_list = glob.glob(file_pattern)

if not file_list:
    print(f"No files found matching the pattern: {file_pattern}")
else:
    print(f"Found {len(file_list)} files to process.")
    # 찾은 파일 목록을 순회하며 플롯 생성 함수를 실행합니다.
    for full_file_path in file_list:
        plot_velocity_and_streamlines(full_file_path, output_directory)

print("\nAll files processed.")