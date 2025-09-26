import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import os

# --- ✨ 1. 설정 영역 ---
# 🔥 여기 숫자만 바꾸면 모든 것이 자동으로 변경됩니다!
CASE_NUMBER = 19

# 기본 경로 설정
BASE_DATA_DIR = r'C:\Users\spearlab05\Desktop\Galerkin ROM\Data'
OUTPUT_DIRECTORY = r'C:\Users\spearlab05\Desktop\Galerkin ROM\OriginalPlot'

# 만약 지정된 경로에 폴더가 없다면 자동으로 생성합니다.
os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)


# --- ✨ 2. 동적 파일 경로 생성 및 데이터 불러오기 ---
# 설정된 CASE_NUMBER를 기반으로 파일명을 동적으로 만듭니다.
input_filename = f'case{CASE_NUMBER}_sorted.csv'
input_filepath = os.path.join(BASE_DATA_DIR, input_filename)

print(f"Loading data from: {input_filepath}")
# CSV 파일에서 데이터를 읽어옵니다.
try:
    df = pd.read_csv(input_filepath)
except FileNotFoundError:
    print(f"오류: 파일 '{input_filepath}'을(를) 찾을 수 없습니다. CASE_NUMBER를 확인하세요.")
    exit() # 파일이 없으면 프로그램 종료

# x, y 좌표 및 p, u, v 변수를 추출합니다.
x = df['x-coordinate'].values
y = df['y-coordinate'].values + 0.01 # y좌표에 오프셋 추가
p = df['pressure'].values
u = df['x-velocity'].values
v = df['y-velocity'].values

# 플롯할 변수들과 이름을 리스트로 묶어 관리합니다.
variables_to_plot = [p, u, v]
variable_names = ['Pressure', 'X-Velocity', 'Y-Velocity']


# --- 3. 데이터를 격자 형태로 보간 ---
# 원본 데이터의 범위를 기반으로 보간에 사용할 격자(grid)를 생성합니다.
grid_x, grid_y = np.mgrid[min(x):max(x):200j, min(y):max(y):200j]


# --- 4. 3개의 서브플롯(Subplot)으로 컨투어 플롯 생성 ---
# 1행 3열의 서브플롯을 생성하고, 전체 그림의 크기를 조절합니다.
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# for 반복문을 사용하여 각 변수(p, u, v)에 대한 플롯을 순서대로 그립니다.
for i, ax in enumerate(axes):
    # 현재 순서에 맞는 변수 데이터와 이름을 가져옵니다.
    current_variable_data = variables_to_plot[i]
    current_variable_name = variable_names[i]

    # griddata를 사용하여 데이터를 격자에 보간합니다.
    grid_z = griddata((x, y), current_variable_data, (grid_x, grid_y), method='cubic')

    # contourf 함수로 색상이 채워진 컨투어 플롯을 해당 서브플롯(ax)에 그립니다.
    contour = ax.contourf(grid_x, grid_y, grid_z, levels=100, cmap='viridis')

    # 각 서브플롯에 컬러바를 추가하고 레이블을 설정합니다.
    cbar = fig.colorbar(contour, ax=ax)
    cbar.set_label(current_variable_name)

    # 각 서브플롯의 제목과 축 레이블을 설정합니다.
    ax.set_title(f'Contour Plot of {current_variable_name}')
    ax.set_xlabel('X-coordinate')
    ax.set_ylabel('Y-coordinate')
    ax.grid(True, linestyle='--', alpha=0.6)

# --- ✨ 5. 동적 제목 설정 ---
# CASE_NUMBER에 맞는 제목을 자동으로 설정합니다.
fig.suptitle(f'Original Data Contour Plots (Case {CASE_NUMBER})', fontsize=16)

# 서브플롯들이 겹치지 않도록 레이아웃을 조정합니다.
plt.tight_layout(rect=[0, 0, 1, 0.95])


# --- ✨ 6. 동적 파일명으로 플롯 저장 ---
# CASE_NUMBER에 맞는 파일명을 자동으로 생성합니다.
output_filename = f'case{CASE_NUMBER}_original_data_plot.png'
output_filepath = os.path.join(OUTPUT_DIRECTORY, output_filename)

# plt.savefig()를 사용하여 그림을 파일로 저장합니다.
plt.savefig(output_filepath, dpi=300)
plt.close(fig) # 메모리에서 그림을 닫습니다.

print(f"Plot successfully saved to: {output_filepath}")