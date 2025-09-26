import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.interpolate import interp1d, griddata

def online_phase_monolithic():
    """
    모든 물리 법칙을 통합한 단일 시스템을 fsolve로 풉니다.
    """
    # --- 1. 파라미터 설정 및 데이터 로드 ---
    Re_new = 1000.0
    print(f"--- [Online] 계산 시작 (Re = {Re_new}) ---")
    
    try:
        snapshot_matrix = np.load('trainsnapshot_full.npy')
        U, S, Vt = np.linalg.svd(snapshot_matrix, full_matrices=False)
        
        cumulative_energy_ratio = np.cumsum(S**2 / np.sum(S**2))
        k = np.argmax(cumulative_energy_ratio >= 0.99999) + 1
        Vt_k = Vt[:k, :]

        rom_data = np.load('rom_system_matrices.npz')
        L, C, P, V = rom_data['L'], rom_data['C'], rom_data['P'], rom_data['V']
        U_p, U_u, U_v = rom_data['U_p'], rom_data['U_u'], rom_data['U_v']
        print("✅ 'rom_system_matrices.npz' 파일 로드 완료.")
        print(f"✅ 현재 계산에 사용되는 모드의 수 k = {k}")
    except Exception as e:
        print(f"❌ 파일 로드 오류: {e}")
        return

    # --- 2. 통합 ROM 방정식 정의 ---
    def monolithic_rom_system(a):
        residual = (np.einsum('mij,i,j->m', L, a, a) +  # 비선형항
                    (P @ a) -                           # 압력항
                    (1/Re_new) * (V @ a) +              # 점성항
                    (C @ a))                            # 연속항
        return residual

    # --- 3. 스마트한 초기 추정값 생성 ---
    print("\n--- 스마트한 초기 추정값 생성 중 ---")
    traindatalist = [3, 5, 7, 9, 11, 13, 15, 17]
    Re_trainlist = [100 + (case_num - 1) * 50 for case_num in traindatalist]
    interpolator = interp1d(Re_trainlist, Vt_k, axis=1, fill_value="extrapolate")
    a_guess = interpolator(Re_new)
    print("✅ 초기 추정값 생성 완료.")

    # --- 4. 비선형 연립방정식 풀이 (fsolve 사용) ---
    print("\n비선형 시스템 해 찾기를 시작합니다 (fsolve 사용)...")
    
    a_solution, info, ier, msg = fsolve(monolithic_rom_system, a_guess, full_output=True)

    # --- 5. 최종 결과 분석 및 출력 ---
    if ier == 1:
        print(f"\n✅ 성공: 최적의 해를 찾았습니다.")
        final_residual_norm = np.linalg.norm(monolithic_rom_system(a_solution))
        print(f"  최종 잔차(Residual)의 크기: {final_residual_norm:.6e}")
    else:
        print(f"\n⚠️ 실패: 솔버가 해를 찾지 못했습니다. (메시지: {msg})")
        return

    # --- 6. 유동장 시각화 ---
    p_rom = U_p @ a_solution
    u_rom = U_u @ a_solution
    v_rom = U_v @ a_solution
    
    # ... (이하 시각화 코드는 이전과 동일) ...
    try:
        coords_df = pd.read_csv(r"C:\Users\spearlab05\Desktop\Galerkin ROM\Data\case3.csv")
        coords_df.columns = coords_df.columns.str.strip()
        coords = coords_df[['x-coordinate', 'y-coordinate']].values
    except FileNotFoundError: return
    
    # ... (이하 시각화 코드 붙여넣기) ...
    # (코드가 길어 생략합니다. 이전 코드의 시각화 부분을 그대로 사용하시면 됩니다.)
    plt.show() # 예시로 추가


if __name__ == '__main__':
    online_phase_monolithic()