import pandas as pd
import glob
import os

def batch_sort_grid_data(folder_path=r"C:\Users\spearlab05\Desktop\Galerkin ROM\Data"):
    """
    지정된 폴더에서 'case*.csv' 파일을 모두 찾아
    각각을 행 우선 순서(y-x)로 정렬하고 '_sorted.csv'를 붙여 새로 저장합니다.
    """
    # 1. 'case*.csv' 패턴에 맞는 파일 목록을 찾습니다.
    #    이미 정렬된 파일이 다시 처리되는 것을 막기 위해 '_sorted'가 없는 파일만 대상으로 합니다.
    input_files = glob.glob(os.path.join(folder_path, 'case*.csv'))
    files_to_process = [f for f in input_files if '_sorted' not in f]

    if not files_to_process:
        print("정렬할 'case*.csv' 파일을 찾을 수 없습니다. (이미 정렬된 파일은 제외됩니다)")
        return

    print(f"총 {len(files_to_process)}개의 파일을 정렬합니다.")
    print("-" * 30)

    # ❗ 실제 CSV 파일의 좌표 컬럼 이름 (필요시 수정)
    x_col = 'x-coordinate'
    y_col = 'y-coordinate'
    
    # 2. 찾은 파일들을 하나씩 순회하며 정렬 작업을 수행합니다.
    for input_file in files_to_process:
        try:
            # 3. 출력 파일 이름을 생성합니다. (예: 'case1.csv' -> 'case1_sorted.csv')
            base_name, ext = os.path.splitext(input_file)
            output_file = f"{base_name}_sorted{ext}"

            print(f"처리 중: '{os.path.basename(input_file)}'  ->  '{os.path.basename(output_file)}'")

            # 4. CSV 파일을 읽고, 컬럼 이름의 공백을 제거합니다.
            df = pd.read_csv(input_file)
            df.columns = df.columns.str.strip()

            # 5. y좌표 우선, 그 다음 x좌표 순으로 데이터를 정렬합니다.
            sorted_df = df.sort_values(by=[y_col, x_col], kind='mergesort')

            # 6. 정렬된 데이터를 새로운 CSV 파일로 저장합니다.
            sorted_df.to_csv(output_file, index=False)

        except FileNotFoundError:
            print(f"  -> 오류: 파일을 찾을 수 없어 건너뜁니다.")
            continue
        except KeyError:
            print(f"  -> 오류: 파일에 '{y_col}' 또는 '{x_col}' 컬럼이 없어 건너뜁니다.")
            continue
        except Exception as e:
            print(f"  -> 알 수 없는 오류 발생: {e}. 건너뜁니다.")
            continue

    print("-" * 30)
    print("✅ 모든 파일의 정렬 작업이 완료되었습니다.")
    print("이제 '_sorted.csv'로 끝나는 파일들을 사용하여 ROM을 구성하세요.")


if __name__ == '__main__':
    # 스크립트가 위치한 현재 폴더에서 작업을 실행합니다.
    batch_sort_grid_data()