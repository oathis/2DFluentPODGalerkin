import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def calculate_l2_relative_error(true_values, pred_values):
    """L2 상대 오차를 계산하는 함수"""
    l2_norm_diff = np.linalg.norm(true_values - pred_values)
    l2_norm_true = np.linalg.norm(true_values)
    if l2_norm_true == 0:
        return 0
    return l2_norm_diff / l2_norm_true

def analyze_rom_error():
    """
    원본 솔루션과 ROM 솔루션의 오차를 분석하고 시각화합니다.
    """
    # --- 설정 ---
    # 파일 경로를 사용자 환경에 맞게 수정해주세요.
    original_data_path = r'C:\Users\spearlab05\Desktop\Galerkin ROM\Data'
    rom_data_path = r'C:\Users\spearlab05\Desktop\Galerkin ROM\FinalResult'

    # 분석할 Re 숫자 리스트
    reynolds_numbers = [100, 150, 200,250,300,350,400,450,500,550,600,650,700,750,800,850,900,950,1000]
    
    # 비교할 변수 리스트
    variables_to_compare = ['pressure', 'x-velocity', 'y-velocity']

    # 결과를 저장할 딕셔너리
    errors = {var: [] for var in variables_to_compare}

    # --- 데이터 처리 및 오차 계산 ---
    for i, re in enumerate(reynolds_numbers):
        case_num = i + 1
        
        # 파일 경로 설정
        original_solution_file = f"{original_data_path}\\case{case_num}_sorted.csv"
        rom_solution_file = f"{rom_data_path}\\rom_solution_Re_{re}.csv"

        print(f"--- Processing Re = {re} ---")
        print(f"Original solution file: {original_solution_file}")
        print(f"ROM solution file: {rom_solution_file}")

        try:
            # CSV 파일 불러오기
            df_orig = pd.read_csv(original_solution_file)
            df_rom = pd.read_csv(rom_solution_file)

            # 컬럼 이름의 공백 제거
            df_orig.columns = df_orig.columns.str.strip()
            df_rom.columns = df_rom.columns.str.strip()
            
            # 좌표 컬럼 이름 확인 및 설정 (필요시 수정)
            x_coord_col = 'x-coordinate'
            y_coord_col = 'y-coordinate'
            
            # 경계값 제외 (내부값만 선택)
            x_min, x_max = df_orig[x_coord_col].min(), df_orig[x_coord_col].max()
            y_min, y_max = df_orig[y_coord_col].min(), df_orig[y_coord_col].max()
            
            internal_nodes = df_orig[
                (df_orig[x_coord_col] > x_min) & (df_orig[x_coord_col] < x_max) &
                (df_orig[y_coord_col] > y_min) & (df_orig[y_coord_col] < y_max)
            ]

            # 두 데이터프레임을 좌표 기준으로 병합
            # 좌표값의 부동소수점 오차를 고려하여 근사값으로 병합 (필요에 따라 tolerance 조절)
            merged_df = pd.merge(internal_nodes, df_rom, on=[x_coord_col, y_coord_col], suffixes=('_orig', '_rom'))

            if merged_df.empty:
                print("Warning: No matching internal nodes found between the two files.")
                for var in variables_to_compare:
                    errors[var].append(np.nan) # 매칭되는 노드가 없으면 NaN으로 처리
                continue

            # 변수별 L2 상대 오차 계산
            for var in variables_to_compare:
                true_values = merged_df[f'{var}_orig']
                pred_values = merged_df[f'{var}_rom']
                error = calculate_l2_relative_error(true_values, pred_values)
                errors[var].append(error)
                print(f"L2 relative error for {var}: {error:.4e}")

        except FileNotFoundError as e:
            print(f"Error: {e}. Please check the file paths.")
            # 파일이 없을 경우, 해당 Re에 대한 데이터 포인트를 NaN으로 처리
            for var in variables_to_compare:
                errors[var].append(np.nan)
    
    # --- 결과 시각화 ---
    plt.figure(figsize=(10, 6))
    
    for var in variables_to_compare:
        plt.plot(reynolds_numbers, errors[var], marker='o', linestyle='-', label=f'Error in {var}')
        
    plt.title('ROM Solution Error vs. Reynolds Number')
    plt.xlabel('Reynolds Number')
    plt.ylabel('L2 Relative Error')
    plt.grid(True)
    plt.legend()
    plt.yscale('log') # 오차는 보통 로그 스케일로 보는 것이 유용합니다.
    
    # 그래프를 이미지 파일로 저장
    plt.savefig('rom_error_vs_re.png')
    
    plt.show()


if __name__ == '__main__':
    analyze_rom_error()