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
    original_data_path = r'C:\Users\spearlab05\Desktop\Galerkin ROM\Data'
    rom_data_path = r'C:\Users\spearlab05\Desktop\Galerkin ROM\FinalResult'
    #rom_data_path = r'C:\Users\spearlab05\Desktop\Galerkin ROM\RESULT STORAGE\Re_8_k_8\FinalResult'
    reynolds_numbers = [i for i in range(100, 1001, 50)]
    variables_to_compare = ['pressure', 'x-velocity', 'y-velocity']
    errors = {var: [] for var in variables_to_compare}

    # --- 데이터 처리 및 오차 계산 ---
    for i, re in enumerate(reynolds_numbers):
        case_num = i + 1
        original_solution_file = f"{original_data_path}\\case{case_num}_sorted.csv"
        rom_solution_file = f"{rom_data_path}\\rom_solution_Re_{re}.csv"

        print(f"--- Processing Re = {re} ---")
        try:
            df_orig = pd.read_csv(original_solution_file, skipinitialspace=True)
            df_rom = pd.read_csv(rom_solution_file, skipinitialspace=True)

            df_orig.columns = df_orig.columns.str.strip()
            df_rom.columns = df_rom.columns.str.strip()
            
            x_coord_col = 'x-coordinate'
            y_coord_col = 'y-coordinate'
            
            # --- ✨✨✨ 최종 해결 코드 ✨✨✨ ---
            # merge 전에 두 데이터프레임의 좌표를 모두 반올림하여 통일합니다.
            precision = 5  # 소수점 5자리까지 비교
            df_orig[x_coord_col] = df_orig[x_coord_col].round(decimals=precision)
            df_orig[y_coord_col] = df_orig[y_coord_col].round(decimals=precision)
            df_rom[x_coord_col] = df_rom[x_coord_col].round(decimals=precision)
            df_rom[y_coord_col] = df_rom[y_coord_col].round(decimals=precision)
            # ------------------------------------

            # 경계값 제외
            x_min, x_max = df_orig[x_coord_col].min(), df_orig[x_coord_col].max()
            y_min, y_max = df_orig[y_coord_col].min(), df_orig[y_coord_col].max()
            
            internal_nodes = df_orig[
                (df_orig[x_coord_col] > x_min) & (df_orig[x_coord_col] < x_max) &
                (df_orig[y_coord_col] > y_min) & (df_orig[y_coord_col] < y_max)
            ].copy() # SettingWithCopyWarning 방지를 위해 .copy() 사용

            # 이제 좌표값이 통일되었으므로 merge가 정상적으로 동작합니다.
            merged_df = pd.merge(internal_nodes, df_rom, on=[x_coord_col, y_coord_col], suffixes=('_orig', '_rom'))

            if merged_df.empty:
                print("Warning: No matching internal nodes found between the two files.")
                for var in variables_to_compare:
                    errors[var].append(np.nan)
                continue

            # (이하 오차 계산 및 시각화 코드는 동일)
            for var in variables_to_compare:
                true_values = merged_df[f'{var}_orig'].values
                pred_values = merged_df[f'{var}_rom'].values
                error = calculate_l2_relative_error(true_values, pred_values)
                errors[var].append(error)
                print(f"L2 relative error for {var}: {error:.4e}")

        except FileNotFoundError as e:
            print(f"Error: {e}. One of the files was not found.")
            for var in variables_to_compare:
                errors[var].append(np.nan)

    # --- ✨✨✨ 디버깅 코드 추가: 실제 오차값 확인 ✨✨✨ ---
    print("\n--- Final Calculated Error Values ---")
    for var in variables_to_compare:
        # np.nanmax를 사용하여 NaN 값을 무시하고 최대값을 찾습니다.
        if errors[var]: # 리스트가 비어있지 않은지 확인
            max_error = np.nanmax(errors[var])
            print(f"Maximum L2 Relative Error for '{var}': {max_error}")
        else:
            print(f"No valid errors calculated for '{var}'.")
    # ----------------------------------------------------
    
    # --- 결과 시각화 ---
    plt.figure(figsize=(10, 6))
    for var in variables_to_compare:
        plt.plot(reynolds_numbers, errors[var], marker='o', linestyle='-', label=f'Error in {var}')
    plt.title('ROM Solution Error vs. Reynolds Number')
    plt.xlabel('Reynolds Number')
    plt.ylabel('L2 Relative Error')
    plt.grid(True)
    plt.legend()
    plt.yscale('log')
    
    # --- 요청하신 코드 추가 ---
    # 그래프를 'rom_error_vs_re.png' 파일로 저장합니다.
    plt.savefig('rom_error_vs_re.png')
    # -----------------------
    
    plt.show()

if __name__ == '__main__':
    analyze_rom_error()
