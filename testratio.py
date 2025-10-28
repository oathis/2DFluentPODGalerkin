import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

EPS = 1e-12  # 0 나눗셈/극소치 마스크용

def _round_coords(df, xcol, ycol, precision=5):
    df[xcol] = df[xcol].round(precision)
    df[ycol] = df[ycol].round(precision)
    return df

def _internal_nodes(df, xcol, ycol):
    x_min, x_max = df[xcol].min(), df[xcol].max()
    y_min, y_max = df[ycol].min(), df[ycol].max()
    return df[(df[xcol] > x_min) & (df[xcol] < x_max) &
              (df[ycol] > y_min) & (df[ycol] < y_max)].copy()

def _safe_stats(x_fom, x_rom):
    """노드별 비율/노름비/회귀계수 등 계산"""
    # 마스크: FOM 기준 유효 노드
    mask = np.isfinite(x_fom) & np.isfinite(x_rom) & (np.abs(x_fom) > EPS)
    if not np.any(mask):
        return dict(ratio_mean_abs=np.nan, ratio_l2=np.nan,
                    alpha_origin=np.nan, alpha=np.nan, intercept=np.nan, R2=np.nan)

    xf = x_fom[mask].astype(float)
    xr = x_rom[mask].astype(float)

    # (1) 노드별 절대비의 평균
    ratio_mean_abs = np.nanmean(np.abs(xr) / np.abs(xf))

    # (2) L2 노름 비
    ratio_l2 = (np.linalg.norm(xr) / max(np.linalg.norm(xf), EPS))

    # (3) 원점 통과 기울기: xr ≈ α * xf
    denom = max(np.dot(xf, xf), EPS)
    alpha_origin = float(np.dot(xf, xr) / denom)

    # (4) 절편 포함 회귀: xr ≈ α * xf + c
    A = np.vstack([xf, np.ones_like(xf)]).T
    sol, *_ = np.linalg.lstsq(A, xr, rcond=None)
    alpha, intercept = float(sol[0]), float(sol[1])

    # R^2
    xr_hat = alpha * xf + intercept
    sse = float(np.sum((xr - xr_hat) ** 2))
    sst = float(np.sum((xr - np.mean(xr)) ** 2)) + EPS
    R2 = 1.0 - sse / sst

    return dict(ratio_mean_abs=ratio_mean_abs, ratio_l2=ratio_l2,
                alpha_origin=alpha_origin, alpha=alpha, intercept=intercept, R2=R2)

def analyze_global_ratios():
    # --- 경로 설정 ---
    original_data_path = r'C:\Users\spearlab05\Desktop\Galerkin ROM\TestData'
    rom_data_path      = r'C:\Users\spearlab05\Desktop\Galerkin ROM\FinalResult'
    reynolds_numbers   = [i for i in range(100, 1001, 10)]

    x_col, y_col = 'x-coordinate', 'y-coordinate'
    vars_map = {
        'pressure':   ('pressure_orig',   'pressure_rom'),
        'x-velocity': ('x-velocity_orig', 'x-velocity_rom'),
        'y-velocity': ('y-velocity_orig', 'y-velocity_rom'),
    }

    rows = []  # 결과 누적

    for i, Re in enumerate(reynolds_numbers):
        case_num = i + 1
        f_fom = Path(original_data_path) / f'case{case_num}_sorted.csv'
        f_rom = Path(rom_data_path)      / f'rom_solution_Re_{Re}.csv'
        print(f'\n--- Re={Re} ---')
        try:
            df_fom = pd.read_csv(f_fom, skipinitialspace=True)
            df_rom = pd.read_csv(f_rom, skipinitialspace=True)
        except FileNotFoundError as e:
            print(f'  [warn] missing file: {e}')
            rows.append({'Re': Re, 'field': 'pressure',   **{k: np.nan for k in ['ratio_mean_abs','ratio_l2','alpha_origin','alpha','intercept','R2']}})
            rows.append({'Re': Re, 'field': 'x-velocity', **{k: np.nan for k in ['ratio_mean_abs','ratio_l2','alpha_origin','alpha','intercept','R2']}})
            rows.append({'Re': Re, 'field': 'y-velocity', **{k: np.nan for k in ['ratio_mean_abs','ratio_l2','alpha_origin','alpha','intercept','R2']}})
            continue

        df_fom.columns = df_fom.columns.str.strip()
        df_rom.columns = df_rom.columns.str.strip()
        _round_coords(df_fom, x_col, y_col, 5)
        _round_coords(df_rom, x_col, y_col, 5)

        # 내부 노드만, 좌표로 매칭
        df_fom_in = _internal_nodes(df_fom, x_col, y_col)
        merged = pd.merge(df_fom_in, df_rom, on=[x_col, y_col], suffixes=('_orig', '_rom'))

        if merged.empty:
            print('  [warn] no matching internal nodes')
            for fld in vars_map:
                rows.append({'Re': Re, 'field': fld, **{k: np.nan for k in ['ratio_mean_abs','ratio_l2','alpha_origin','alpha','intercept','R2']}})
            continue

        # 각 필드별 전역 비율 지표 계산
        for fld, (col_fom, col_rom) in vars_map.items():
            stats = _safe_stats(merged[col_fom].values, merged[col_rom].values)
            print(f"  {fld:11s} | ratio_mean_abs={stats['ratio_mean_abs']:.4f}  "
                  f"ratio_l2={stats['ratio_l2']:.4f}  alpha0={stats['alpha_origin']:.4f}  "
                  f"alpha={stats['alpha']:.4f}  c={stats['intercept']:.3e}  R2={stats['R2']:.3f}")
            rows.append({'Re': Re, 'field': fld, **stats})

    # 결과 집계/저장
    res_df = pd.DataFrame(rows)
    out_csv = 'global_ratio_results.csv'
    res_df.to_csv(out_csv, index=False)
    print(f'\nSaved: {out_csv}')

    # 요약(압력)
    prs = res_df[res_df['field'] == 'pressure'].copy()
    mean_ratio = prs['ratio_mean_abs'].mean(skipna=True)
    mean_l2    = prs['ratio_l2'].mean(skipna=True)
    mean_alpha = prs['alpha_origin'].mean(skipna=True)
    print(f"\n[pressure] mean ratio_mean_abs={mean_ratio:.4f}  "
          f"mean ratio_l2={mean_l2:.4f}  mean alpha_origin={mean_alpha:.4f}")

    # 간단 진단 메시지
    if np.isfinite(mean_ratio) and 0.45 <= mean_ratio <= 0.55:
        print(">>> Hint: 압력 전역비율이 ~0.5 → Cp 스케일 또는 U0 미스매치(√2) 가능성이 큼.")

    # 플롯: Re별 전역 비율
    plt.figure(figsize=(10,6))
    for fld, color, marker in [('pressure', 'tab:blue', 'o'),
                               ('x-velocity', 'tab:orange', 's'),
                               ('y-velocity', 'tab:green', '^')]:
        sub = res_df[res_df['field']==fld]
        plt.plot(sub['Re'], sub['ratio_l2'], marker=marker, linestyle='-', label=f'{fld}  (L2 norm ratio)')

    plt.title('Global Ratio vs Reynolds number (‖ROM‖ / ‖FOM‖)')
    plt.xlabel('Re')
    plt.ylabel('global L2 norm ratio')
    plt.grid(True, alpha=0.4)
    plt.legend()
    plt.savefig('global_ratio_vs_re.png')
    plt.show()

if __name__ == '__main__':
    analyze_global_ratios()
