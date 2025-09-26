import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from scipy.interpolate import griddata


def offline_phase_full_field():
    """
    평균장 제거 없이 전체 유동장(Full Field)을 사용하여 오프라인 계산을 수행합니다.
    """
    # ==================================================================
    # Part 1: 스냅샷 행렬 생성 (기존과 동일)
    # ==================================================================
    base_path = r"C:\Users\spearlab05\Desktop\Galerkin ROM\Data"
    traindatalist = [3, 5, 7, 9, 11, 13, 15, 17]
    snapshot_columns = []

    for case_num in traindatalist:
        file_path = os.path.join(base_path, f"case{case_num}.csv")
        try:
            df = pd.read_csv(file_path)
            df.columns = df.columns.str.strip()
            pressure_data = df['pressure'].values
            x_velocity_data = df['x-velocity'].values
            y_velocity_data = df['y-velocity'].values
            
            coords = df[['x-coordinate', 'y-coordinate']].values
            single_snapshot_column = np.concatenate([pressure_data, x_velocity_data, y_velocity_data])
            snapshot_columns.append(single_snapshot_column)
            print(f"성공: case {case_num}.csv 처리 완료.")
        except Exception as e:
            print(f"오류: {file_path} 처리 중 문제 발생 - {e}")
            return

    if not snapshot_columns:
        print("\n처리된 데이터가 없어 스냅샷 행렬을 생성하지 못했습니다.")
        return
        
    snapshot_matrix = np.column_stack(snapshot_columns)
    np.save('trainsnapshot_full.npy', snapshot_matrix)
    print(f"\n--- 최종 스냅샷 행렬 생성 완료 (Full Field) ---")
    print(f"Snapshot 행렬 형태: {snapshot_matrix.shape}")

    # ==================================================================
    # Part 2: SVD 및 모드 분리 (평균 제거 없음)
    # ==================================================================
    print("\n--- SVD 분해 시작 (평균 제거 없음) ---")
    
    # ⚠️ 중요: fluctuation_matrix 대신 snapshot_matrix를 직접 SVD합니다.
    U, S, Vt = np.linalg.svd(snapshot_matrix, full_matrices=False)

    print(f"U 행렬 (POD 모드) 형태: {U.shape}")
    
    # 에너지 분석을 통해 k 결정 (기존과 동일)
    modal_energy = S**2
    total_energy = np.sum(modal_energy)
    cumulative_energy_ratio = np.cumsum(modal_energy / total_energy)
    k = np.argmax(cumulative_energy_ratio >= 0.99999) + 1
    print(f"\n✅ 전체 에너지의 99.999% 이상을 설명하기 위한 모드의 수 k = {k}개 입니다.")

    # Truncated SVD (기존과 동일)
    U_k = U[:, :k]
    print(f"\n축소된 U_k (POD 모드) 형태: {U_k.shape}")
    
    # 모드 분리 (기존과 동일)
    n_points = 10000
    U_pressure = U_k[0:n_points, :]
    U_u = U_k[n_points : 2 * n_points, :]
    U_v = U_k[2 * n_points : 3 * n_points, :]

    # ==================================================================
    # Part 3: 공간 미분 및 시스템 행렬 계산 (기존과 동일)
    # ==================================================================
    



    def compute_derivatives_gridded(modes, coords, n_grid=100):
        """
        좌표(coords)를 이용해 비정형 모드(modes)를 정형 격자로 보간한 후,
        공간 미분을 계산합니다.
        """
        n_points, k = modes.shape
        x, y = coords[:, 0], coords[:, 1]

        # 1. 새로운 정형 격자 생성
        grid_x_vec = np.linspace(x.min(), x.max(), n_grid)
        grid_y_vec = np.linspace(y.min(), y.max(), n_grid)
        grid_xx, grid_yy = np.meshgrid(grid_x_vec, grid_y_vec)

        dx = grid_x_vec[1] - grid_x_vec[0]
        dy = grid_y_vec[1] - grid_y_vec[0]

        # 최종 결과를 저장할 배열
        all_grads_x_flat = np.zeros((n_points, k))
        all_grads_y_flat = np.zeros((n_points, k))
        all_laplacians_flat = np.zeros((n_points, k))

        print(f"\n✅ {k}개의 모드에 대해 격자 보간 및 미분 계산 시작...")

        for i in range(k):
            mode_vec = modes[:, i]

            # 2. griddata를 사용해 현재 모드를 정형 격자로 보간
            mode_grid = griddata((x, y), mode_vec, (grid_xx, grid_yy), method='cubic')
            mode_grid = np.nan_to_num(mode_grid) # NaN 값 처리

            # 3. 정형 격자 위에서 미분 계산
            grad_y_grid, grad_x_grid = np.gradient(mode_grid, dy, dx)

            grad_y_y_grid, _ = np.gradient(grad_y_grid, dy, dx)
            _, grad_x_x_grid = np.gradient(grad_x_grid, dy, dx)

            laplacian_grid = grad_x_x_grid + grad_y_y_grid

            # 4. 계산된 미분 값들을 다시 원본 좌표 위치로 역-보간
            grad_x_flat = griddata((grid_xx.ravel(), grid_yy.ravel()), grad_x_grid.ravel(), (x, y), method='cubic')
            grad_y_flat = griddata((grid_xx.ravel(), grid_yy.ravel()), grad_y_grid.ravel(), (x, y), method='cubic')
            laplacian_flat = griddata((grid_xx.ravel(), grid_yy.ravel()), laplacian_grid.ravel(), (x, y), method='cubic')

            all_grads_x_flat[:, i] = np.nan_to_num(grad_x_flat)
            all_grads_y_flat[:, i] = np.nan_to_num(grad_y_flat)
            all_laplacians_flat[:, i] = np.nan_to_num(laplacian_flat)

        print("✅ 모든 모드의 공간 미분 계산 완료.")

        return (all_grads_x_flat, all_grads_y_flat, all_laplacians_flat)



    
    Uu_dx, Uu_dy, Uu_lap = compute_derivatives_gridded(U_u, coords)
    Uv_dx, Uv_dy, Uv_lap = compute_derivatives_gridded(U_v, coords)
    Up_dx, Up_dy, _ = compute_derivatives_gridded(U_pressure, coords)
    print("\n✅ 모든 POD 모드의 공간 미분 계산 완료.")
    
    # 시스템 행렬 계산 (기존과 동일)
    print("\n--- [Offline] 시스템 행렬 계산 시작 ---")
    L_tensor = np.zeros((k, k, k))
    C_matrix = np.zeros((k, k))
    P_matrix = np.zeros((k, k))
    V_matrix = np.zeros((k, k))

    for i in tqdm(range(k), desc="Calculating L_tensor"):
        for j in range(k):
            for k_idx in range(k):
                integrand_L = (U_u[:, i] * Uu_dx[:, j] + U_v[:, i] * Uu_dy[:, j]) * U_u[:, k_idx] + \
                              (U_u[:, i] * Uv_dx[:, j] + U_v[:, i] * Uv_dy[:, j]) * U_v[:, k_idx]
                L_tensor[k_idx, i, j] = np.sum(integrand_L)

    for i in tqdm(range(k), desc="Calculating C, P, V matrices"):
        for k_idx in range(k):
            C_matrix[k_idx, i] = np.sum((Uu_dx[:, i] + Uv_dy[:, i]) * U_pressure[:, k_idx])
            P_matrix[k_idx, i] = np.sum(Up_dx[:, i] * U_u[:, k_idx] + Up_dy[:, i] * U_v[:, k_idx])
            V_matrix[k_idx, i] = np.sum(Uu_lap[:, i] * U_u[:, k_idx] + Uv_lap[:, i] * U_v[:, k_idx])

    # ⚠️ 중요: 이제 강제항(f)과 평균 스냅샷(mean_snap)은 저장할 필요가 없습니다.
    np.savez_compressed('rom_system_matrices.npz', 
                        L=L_tensor, C=C_matrix, P=P_matrix, V=V_matrix,
                        U_p=U_pressure, U_u=U_u, U_v=U_v)
    
    print("\n--- [Offline] 계산 완료 ---")
    print("✅ 시스템 행렬과 POD 모드를 'rom_system_matrices.npz' 파일에 성공적으로 저장했습니다.")

if __name__ == '__main__':
    offline_phase_full_field()