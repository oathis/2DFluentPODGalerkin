import numpy as np
import pandas as pd
from scipy.linalg import svd
import os
import glob
import time
import re # 숫자 추출을 위해 re 모듈 추가

# --- 설정값 ---
NUM_CASES = 8     # 스냅샷(케이스) 개수
NX, NY = 101, 101  # 격자 크기
N_NODES = NX * NY
K =8      # 사용할 모드의 개수 (최대 NUM_CASES)
DATA_DIRECTORY = r'C:\Users\spearlab05\Desktop\Galerkin ROM\offlineDATA' # 데이터 파일 경로 지정

# --- 유틸리티 함수 ---

def load_and_preprocess_data(filepath):
    """CSV 파일을 로드하고 전처리합니다."""
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()
    

    df['x-coordinate'] = df['x-coordinate'].round(decimals=5)
    df['y-coordinate'] = df['y-coordinate'].round(decimals=5)
    # ------------------------------------

    # 데이터의 물리적 순서를 보장하기 위해 좌표를 기준으로 정렬합니다.
    df = df.sort_values(by=['y-coordinate', 'x-coordinate'], ascending=[True, True])

    required_cols = ['x-coordinate', 'y-coordinate', 'pressure', 'x-velocity', 'y-velocity']
    
    missing_cols = set(required_cols) - set(df.columns)
    if missing_cols:
        raise ValueError(f"파일 '{os.path.basename(filepath)}'에 다음 열이 없습니다: {list(missing_cols)}")

    return df[required_cols]

class DerivativeHelper:
    """미분코드"""
    def __init__(self, nx, ny):
        self.nx, self.ny = nx, ny
        x = np.linspace(0, 0.01, nx)
        y = np.linspace(-0.01, 0, ny)
        self.delta_x = x[1] - x[0]
        self.delta_y = y[1] - y[0]

    def _to_2d(self, field_1d):
        return field_1d.reshape((self.ny, self.nx))

    def _to_1d(self, field_2d):
        return field_2d.flatten()

    def dx(self, field_1d):
        """x 방향 편미분"""
        field_2d = self._to_2d(field_1d)
        return self._to_1d(np.gradient(field_2d, self.delta_x, axis=1))

    def dy(self, field_1d):
        """y 방향 편미분"""
        field_2d = self._to_2d(field_1d)
        return self._to_1d(np.gradient(field_2d, self.delta_y, axis=0))

    def laplacian(self, field_1d):
        """라플라시안 연산"""
        field_2d = self._to_2d(field_1d)
        grad_x = np.gradient(field_2d, self.delta_x, axis=1)
        grad_y = np.gradient(field_2d, self.delta_y, axis=0)
        lap_x = np.gradient(grad_x, self.delta_x, axis=1)
        lap_y = np.gradient(grad_y, self.delta_y, axis=0)
        return self._to_1d(lap_x + lap_y)

def expand_modes(modes_interior, nx, ny):
    """내부 모드를 전체 도메인으로 확장합니다 (0-padding)."""
    nx_int, ny_int = nx - 2, ny - 2
    num_modes = modes_interior.shape[1]
    modes_full = np.zeros((nx * ny, num_modes))

    for k in range(num_modes):
        mode_k_int = modes_interior[:, k].reshape((ny_int, nx_int))
        mode_k_full = np.zeros((ny, nx))
        mode_k_full[1:-1, 1:-1] = mode_k_int
        modes_full[:, k] = mode_k_full.flatten()
    return modes_full


import matplotlib.pyplot as plt


def visualize_full_snapshot(Q_col, nx, ny, case_index, data_directory):
    """특정 전체 스냅샷의 압력 및 속도 필드를 시각화하고 파일로 저장합니다."""
    n_nodes = nx * ny

    # 1차원 벡터에서 p, u, v 분리
    p_full = Q_col[0*n_nodes : 1*n_nodes]
    u_full = Q_col[1*n_nodes : 2*n_nodes]
    v_full = Q_col[2*n_nodes : 3*n_nodes]

    # 플로팅을 위해 2D로 변환
    p_2d = p_full.reshape((ny, nx))
    u_2d = u_full.reshape((ny, nx))
    v_2d = v_full.reshape((ny, nx))

    # 시각화
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'Full Snapshot Visualization (from case {case_index + 1})', fontsize=16)

    im_p = axes[0].imshow(p_2d, cmap='viridis', origin='lower')
    axes[0].set_title('Full Pressure')
    fig.colorbar(im_p, ax=axes[0])

    im_u = axes[1].imshow(u_2d, cmap='viridis', origin='lower')
    axes[1].set_title('Full X-Velocity')
    fig.colorbar(im_u, ax=axes[1])

    im_v = axes[2].imshow(v_2d, cmap='viridis', origin='lower')
    axes[2].set_title('Full Y-Velocity')
    fig.colorbar(im_v, ax=axes[2])
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # 결과를 이미지 파일로 저장
    output_filename = os.path.join(data_directory, f'debug_full_snapshot_case_{case_index + 1}.png')
    plt.savefig(output_filename)
    print(f"   -> 디버깅 이미지 저장 완료: '{output_filename}'")
    plt.close(fig)


def visualize_interior_snapshot(Q_int_col, nx, ny, case_index, data_directory):
    """특정 내부 스냅샷의 압력 및 속도 필드를 시각화하고 파일로 저장합니다."""
    nx_int, ny_int = nx - 2, ny - 2
    n_nodes_int = nx_int * ny_int

    # 1차원 벡터에서 p, u, v 분리
    p_int = Q_int_col[0*n_nodes_int : 1*n_nodes_int]
    u_int = Q_int_col[1*n_nodes_int : 2*n_nodes_int]
    v_int = Q_int_col[2*n_nodes_int : 3*n_nodes_int]

    # 플로팅을 위해 2D로 변환
    p_2d = p_int.reshape((ny_int, nx_int))
    u_2d = u_int.reshape((ny_int, nx_int))
    v_2d = v_int.reshape((ny_int, nx_int))

    # 시각화
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'Interior Snapshot Visualization (from case {case_index + 1})', fontsize=16)

    im_p = axes[0].imshow(p_2d, cmap='viridis', origin='lower')
    axes[0].set_title('Interior Pressure')
    fig.colorbar(im_p, ax=axes[0])

    im_u = axes[1].imshow(u_2d, cmap='viridis', origin='lower')
    axes[1].set_title('Interior X-Velocity')
    fig.colorbar(im_u, ax=axes[1])

    im_v = axes[2].imshow(v_2d, cmap='viridis', origin='lower')
    axes[2].set_title('Interior Y-Velocity')
    fig.colorbar(im_v, ax=axes[2])
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # 결과를 이미지 파일로 저장
    output_filename = os.path.join(data_directory, f'debug_interior_snapshot_case_{case_index + 1}.png')
    plt.savefig(output_filename)
    print(f"   -> 디버깅 이미지 저장 완료: '{output_filename}'")
    plt.close(fig) # 창이 뜨는 것을 방지

# --- OFFLINE 계산 시작 ---

def run_offline_stage():
    """OFFLINE 단계 전체를 실행합니다."""
    print("--- OFFLINE STAGE START ---")
    start_time = time.time()

    # 1. 모든 스냅샷 행렬 Q 구성
    print(f"1. Loading snapshots from '{DATA_DIRECTORY}' and building Q matrix...")
    
    if not os.path.isdir(DATA_DIRECTORY):
        raise FileNotFoundError(f"지정된 디렉토리를 찾을 수 없습니다: '{DATA_DIRECTORY}'")

    file_pattern = os.path.join(DATA_DIRECTORY, 'case*_sorted.csv')
    filepaths = glob.glob(file_pattern)
    
    if len(filepaths) == 0:
        raise FileNotFoundError(f"디렉토리 '{DATA_DIRECTORY}'에서 'case*_sorted.csv' 패턴의 파일을 찾을 수 없습니다.")

    # --- 수정된 부분: 숫자 크기 순으로 파일 정렬 ---
    def get_case_number(path):
        """파일 이름에서 'case' 뒤의 숫자를 추출하여 정수로 반환합니다."""
        # 정규표현식을 사용하여 숫자 부분을 찾습니다.
        match = re.search(r'case(\d+)_sorted\.csv', os.path.basename(path))
        # 숫자를 찾으면 정수로 변환하여 반환하고, 없으면 정렬 순서에 영향을 주지 않도록 -1을 반환합니다.
        return int(match.group(1)) if match else -1
    
    filepaths.sort(key=get_case_number)
    # ---------------------------------------------
    
    if len(filepaths) != NUM_CASES:
        print(f"경고: {NUM_CASES}개의 파일을 예상했지만 {len(filepaths)}개를 찾았습니다. 찾은 파일로 계속 진행합니다.")
        actual_num_cases = len(filepaths)
    else:
        actual_num_cases = NUM_CASES

    Q = np.zeros((N_NODES * 3, actual_num_cases))

    for i, fpath in enumerate(filepaths):
        print(f"  - Loading {os.path.basename(fpath)}...")
        df = load_and_preprocess_data(fpath)
        q_n = np.concatenate([df['pressure'].values, df['x-velocity'].values, df['y-velocity'].values])
        Q[:, i] = q_n
    
    coords = df[['x-coordinate', 'y-coordinate']].values

    # --- ✨1단계 디버깅: 전체 스냅샷 확인✨ ---
    print("1a. [Debug] Visualizing the first FULL snapshot...")
    case_to_visualize = 0 
    visualize_full_snapshot(Q[:, case_to_visualize], NX, NY, case_to_visualize, DATA_DIRECTORY)
    # -----------------------------------------



    # 2. Q_interior 구성
    print("2. Building Q_interior matrix...")
    nx_int, ny_int = NX - 2, NY - 2
    Q_interior = np.zeros((nx_int * ny_int * 3, actual_num_cases))
    for i in range(actual_num_cases):
        p, u, v = Q[0:N_NODES, i], Q[N_NODES:2*N_NODES, i], Q[2*N_NODES:3*N_NODES, i]
        p_int = p.reshape(NY, NX)[1:-1, 1:-1].flatten()
        u_int = u.reshape(NY, NX)[1:-1, 1:-1].flatten()
        v_int = v.reshape(NY, NX)[1:-1, 1:-1].flatten()
        Q_interior[:, i] = np.concatenate([p_int, u_int, v_int])
        
    # --- ✨여기에 코드를 추가하세요✨ ---
    print("2a. [Debug] Visualizing the first interior snapshot...")
    # 첫 번째 스냅샷(case 1)의 내부 유동장을 이미지로 저장
    case_to_visualize = 0 
    visualize_interior_snapshot(Q_interior[:, case_to_visualize], NX, NY, case_to_visualize, DATA_DIRECTORY)
    # ---------------------------------


    # 3. POD 수행
    print(f"3. Performing POD for k={K} modes...")
    U, s, Vh = svd(Q_interior, full_matrices=False)
    Phi_interior = U[:, :K]



    print("\n3a. [Verification] Analyzing energy distribution within each mode...")

    nx_int, ny_int = NX - 2, NY - 2
    n_nodes_int = nx_int * ny_int

    # 각 모드를 p, u, v 부분으로 분리
    p_modes_int = Phi_interior[0*n_nodes_int : 1*n_nodes_int, :]
    u_modes_int = Phi_interior[1*n_nodes_int : 2*n_nodes_int, :]
    v_modes_int = Phi_interior[2*n_nodes_int : 3*n_nodes_int, :]

    print("-" * 50)
    print(" Mode # | Pressure Energy | Velocity Energy | Total")
    print("-" * 50)

    for k in range(K):
        # 각 성분의 에너지 (L2 norm 제곱) 계산
        p_energy = np.linalg.norm(p_modes_int[:, k])**2
        u_energy = np.linalg.norm(u_modes_int[:, k])**2
        v_energy = np.linalg.norm(v_modes_int[:, k])**2
        
        total_energy = p_energy + u_energy + v_energy
        
        # 속도 에너지 = u 에너지 + v 에너지
        velocity_energy = u_energy + v_energy
        
        # 각 성분의 에너지 기여도 (%) 계산
        p_percentage = (p_energy / total_energy) * 100
        velocity_percentage = (velocity_energy / total_energy) * 100
        
        print(f" {k+1:^6} | {p_percentage:^15.2f}% | {velocity_percentage:^15.2f}% | 100.00%")

    print("-" * 50)

    # 4. 모드 분리 및 확장
    print("4. Separating and expanding modes...")
    p_modes_int = Phi_interior[0*(nx_int*ny_int) : 1*(nx_int*ny_int), :]
    u_modes_int = Phi_interior[1*(nx_int*ny_int) : 2*(nx_int*ny_int), :]
    v_modes_int = Phi_interior[2*(nx_int*ny_int) : 3*(nx_int*ny_int), :]
    p_modes, u_modes, v_modes = (expand_modes(m, NX, NY) for m in [p_modes_int, u_modes_int, v_modes_int])

    # 5. 경계 조건(u_bc) 생성
    print("5. Creating boundary condition vector u_bc...")
    u_bc = np.zeros(N_NODES)
    y_coords = coords[:, 1].reshape((NY, NX))
    u_bc[(y_coords == y_coords.max()).flatten()] = 0.1

    # 6. Galerkin 투영을 위한 텐서 계산 (실무용 완전한 버전)
    print("6. Pre-calculating Galerkin tensors (Full System)...")
    deriv = DerivativeHelper(NX, NY)
    
    d_dx = {'p': [], 'u': [], 'v': []}
    d_dy = {'p': [], 'u': [], 'v': []}
    lap = {'u': [], 'v': []}
    for i in range(K):
        d_dx['p'].append(deriv.dx(p_modes[:, i]))
        d_dy['p'].append(deriv.dy(p_modes[:, i]))
        d_dx['u'].append(deriv.dx(u_modes[:, i]))
        d_dy['u'].append(deriv.dy(u_modes[:, i]))
        d_dx['v'].append(deriv.dx(v_modes[:, i]))
        d_dy['v'].append(deriv.dy(v_modes[:, i]))
        lap['u'].append(deriv.laplacian(u_modes[:, i]))
        lap['v'].append(deriv.laplacian(v_modes[:, i]))
        
    u_bc_dx = deriv.dx(u_bc)
    u_bc_dy = deriv.dy(u_bc)
    u_bc_lap = deriv.laplacian(u_bc)

    C1 = np.zeros(K)
    C2 = np.zeros(K)
    L1 = np.zeros((K, K))
    L2 = np.zeros((K, K))
    Q = np.zeros((K, K, K))

    print("   Calculating tensors C, L, Q. This may take a while...")
    for m in range(K):
        C1[m] = np.dot(u_bc_dx, p_modes[:, m]) + np.dot(u_bc * u_bc_dx, u_modes[:, m])
        C2[m] = np.dot(-u_bc_lap, u_modes[:, m])
        
        for j in range(K):
            L1_rc = np.dot(d_dx['u'][j] + d_dy['v'][j], p_modes[:, m])
            L1_ru = np.dot(u_bc * d_dx['u'][j] + u_modes[:, j] * u_bc_dx + v_modes[:, j] * u_bc_dy + d_dx['p'][j], u_modes[:, m])
            L1_rv = np.dot(u_bc * d_dx['v'][j] + d_dy['p'][j], v_modes[:, m])
            L1[m, j] = L1_rc + L1_ru + L1_rv
            
            L2_ru = np.dot(-lap['u'][j], u_modes[:, m])
            L2_rv = np.dot(-lap['v'][j], v_modes[:, m])
            L2[m, j] = L2_ru + L2_rv

            for i in range(K):
                ru_quad_term = u_modes[:, i] * d_dx['u'][j] + v_modes[:, i] * d_dy['u'][j]
                Q_ru = np.dot(ru_quad_term, u_modes[:, m])
                
                rv_quad_term = u_modes[:, i] * d_dx['v'][j] + v_modes[:, i] * d_dy['v'][j]
                Q_rv = np.dot(rv_quad_term, v_modes[:, m])
                
                Q[m, i, j] = Q_ru + Q_rv

    # 7. 오프라인 데이터 저장
    print("7. Saving offline data to 'rom_offline_data.npz'...")
    np.savez('rom_offline_data.npz',
             p_modes=p_modes, u_modes=u_modes, v_modes=v_modes,
             u_bc=u_bc, coords=coords,
             C1=C1, C2=C2, L1=L1, L2=L2, Q=Q, K=K, NX=NX, NY=NY
            )
    
    end_time = time.time()
    print(f"--- OFFLINE STAGE COMPLETE (Total time: {end_time - start_time:.2f} seconds) ---")




if __name__ == '__main__':
    run_offline_stage()

