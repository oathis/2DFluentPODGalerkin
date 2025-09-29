import cv2
import os
import glob
import re

# --- ✨ 설정 ✨ ---

# 1. PNG 이미지 파일들이 있는 폴더 경로를 지정하세요.
IMAGE_FOLDER = r'C:\Users\spearlab05\Desktop\Galerkin ROM\FinalResult_Streamline' # 예시 경로입니다. 실제 폴더 경로로 변경해주세요.

# 2. 동영상을 만들 이미지 파일명의 시작 부분을 지정하세요.
#    사용자님의 파일 이름이 'flow_visualization_Re_620.png' 이므로, 'flow_visualization'으로 설정합니다.
FILE_PREFIX = 'flow_visualization'

# 3. 생성될 동영상 파일의 이름을 지정하세요. (FILE_PREFIX를 기반으로 자동 생성됩니다)
VIDEO_NAME = f'{FILE_PREFIX}_animation.mp4'

# 4. 동영상의 초당 프레임 수(FPS)를 지정하세요. (숫자가 클수록 빠르게 재생됩니다)
FPS = 5

# --- 코드 시작 ---

def create_video_from_images():
    """지정된 폴더의 PNG 이미지들을 찾아 동영상으로 만듭니다."""

    # 1. 이미지 파일 목록 찾기 및 Re 숫자를 기준으로 정렬
    print(f"'{IMAGE_FOLDER}' 폴더에서 이미지 파일을 검색합니다...")
    
    # 파일 이름 패턴을 FILE_PREFIX를 사용하여 동적으로 만듭니다.
    # 예: 'flow_visualization_Re_*.png'
    file_pattern = os.path.join(IMAGE_FOLDER, f'{FILE_PREFIX}_Re_*.png')
    image_files = glob.glob(file_pattern)

    if not image_files:
        print(f"오류: '{IMAGE_FOLDER}' 폴더에서 '{FILE_PREFIX}_Re_*.png' 패턴의 파일을 찾을 수 없습니다.")
        print("-> IMAGE_FOLDER 경로와 FILE_PREFIX가 올바른지 확인하세요.")
        return

    # 파일 이름에서 Reynolds 수를 추출하여 숫자로 정렬하는 함수
    def get_re_number_from_filename(filename):
        # 정규표현식을 사용하여 'Re_숫자.png' 패턴에서 숫자 부분만 추출
        match = re.search(r'Re_(\d+)\.png$', os.path.basename(filename))
        if match:
            return int(match.group(1))
        return -1 # 매칭되지 않으면 맨 뒤로 보냄

    image_files.sort(key=get_re_number_from_filename)
    print(f"총 {len(image_files)}개의 이미지를 순서대로 정렬했습니다.")
    
    # 정렬된 파일 중 처음 5개 출력 (확인용)
    print("정렬된 파일 (상위 5개):")
    for f in image_files[:5]:
        print(f" - {os.path.basename(f)}")


    # 2. 동영상 설정을 위해 첫 번째 이미지의 크기 읽기
    first_image = cv2.imread(image_files[0])
    if first_image is None:
        print(f"오류: 첫 번째 이미지 파일 '{image_files[0]}'을 읽을 수 없습니다.")
        return
        
    height, width, layers = first_image.shape
    print(f"동영상 해상도를 {width}x{height}으로 설정합니다.")

    # 3. 동영상 writer 객체 생성
    # 'mp4v'는 MP4 파일을 위한 표준 코덱입니다.
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video_path = os.path.join(IMAGE_FOLDER, VIDEO_NAME)
    video = cv2.VideoWriter(output_video_path, fourcc, FPS, (width, height))

    # 4. 각 이미지를 프레임으로 동영상에 추가
    print("\n동영상 변환을 시작합니다...")
    for i, image_path in enumerate(image_files):
        # 진행 상황 출력
        print(f"  - 프레임 추가 중 ({i + 1}/{len(image_files)}): {os.path.basename(image_path)}")
        frame = cv2.imread(image_path)
        if frame is not None:
            video.write(frame)
        else:
            print(f"    경고: '{os.path.basename(image_path)}' 파일을 읽을 수 없어 건너뜁니다.")


    # 5. 작업 완료 후 객체 해제
    video.release()
    print("\n" + "="*40)
    print(f"✅ 동영상 생성이 완료되었습니다!")
    print(f"   -> 저장 위치: {output_video_path}")
    print("="*40)


if __name__ == '__main__':
    create_video_from_images()
