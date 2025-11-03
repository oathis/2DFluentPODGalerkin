import pandas as pd
import glob
import os

def batch_sort_grid_data(folder_path=r"C:\Users\spearlab05\Desktop\Testy데이터저장소\FOM_origin_Data\40000cell\RawData"):
    """
    지정된 폴더에서 'case*.csv' 파일을 모두 찾아
    각각을 행 우선 순서(y-x)로 정렬하고,
    'sorted'라는 하위 폴더 안에 '_sorted.csv'를 붙여 새로 저장합니다.
    """
    # 1. 'case*.csv' 패턴에 맞는 파일 목록을 찾습니다.
    #    이미 정렬된 파일이 다시 처리되는 것을 막기 위해 '_sorted'가 없는 파일만 대상으로 합니다.
    input_files = glob.glob(os.path.join(folder_path, 'case*.csv'))
    files_to_process = [f for f in input_files if '_sorted' not in f]

    if not files_to_process:
        print("정렬할 'case*.csv' 파일을 찾을 수 없습니다. (이미 정렬된 파일은 제외됩니다)")
        return

    # ❗ [수정됨] 'sorted' 하위 폴더 경로를 생성하고, 폴더가 없으면 만듭니다.
    output_folder_path = os.path.join(folder_path, 'sorted')
    os.makedirs(output_folder_path, exist_ok=True)
    print(f"'sorted' 폴더에 결과를 저장합니다: {output_folder_path}")

    print(f"총 {len(files_to_process)}개의 파일을 정렬합니다.")
    print("-" * 30)

    # ❗ 실제 CSV 파일의 좌표 컬럼 이름 (필요시 수정)
    x_col = 'x-coordinate'
    y_col = 'y-coordinate'
    
    # 2. 찾은 파일들을 하나씩 순회하며 정렬 작업을 수행합니다.
    for input_file in files_to_process:
        try:
            # 3. ❗ [수정됨] 입력 파일 이름과 출력 파일 경로를 재구성합니다.
            input_file_name = os.path.basename(input_file) # 예: 'case1.csv'
            base_name, ext = os.path.splitext(input_file_name) # 예: 'case1', '.csv'
            
            output_file_name = f"{base_name}_sorted{ext}" # 예: 'case1_sorted.csv'
            # 최종 저장 경로는 'sorted' 폴더 안이 됩니다.
            output_file_path = os.path.join(output_folder_path, output_file_name) 

            # ❗ [수정됨] 출력 경로를 'sorted/...' 형식으로 표시
            print(f"처리 중: '{input_file_name}'  ->  '{os.path.join('sorted', output_file_name)}'")

            # 4. CSV 파일을 읽고, 컬럼 이름의 공백을 제거합니다.
            df = pd.read_csv(input_file)
            df.columns = df.columns.str.strip()

            # 5. y좌표 우선, 그 다음 x좌표 순으로 데이터를 정렬합니다.
            sorted_df = df.sort_values(by=[y_col, x_col], kind='mergesort')

            # 6. ❗ [수정됨] 정렬된 데이터를 'sorted' 폴더의 새 CSV 파일로 저장합니다.
            sorted_df.to_csv(output_file_path, index=False)

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
    # ❗ [수정됨] 최종 메시지 변경
    print("이제 'sorted' 폴더 안의 '_sorted.csv' 파일들을 사용하여 ROM을 구성하세요.")


if __name__ == '__main__':
    # 스크립트가 위치한 현재 폴더에서 작업을 실행합니다.
    batch_sort_grid_data()