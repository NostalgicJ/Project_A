import os
import cv2
import numpy as np
import glob # 폴더 내의 파일 목록을 가져오기 위해 import
from utils.delete_file import delete_files_in_directory


# --- (1) 설치 확인 ---
# 터미널에 pip install opencv-python numpy 를 입력해 설치하세요.
# ---

def crop_and_warp_polygons(image_path, txt_path, output_dir):
    """
    (이 함수는 이전 스크립트와 동일합니다)
    원본 이미지와 텍스트 파일의 폴리곤 좌표를 읽어
    각 단어 영역을 반듯한 직사각형 이미지로 잘라내어 저장합니다.
    """
    
    # 1. 원본 이미지 읽기
    try:
        image = cv2.imread(image_path)
        if image is None:
            print(f"  [오류] 원본 이미지를 불러올 수 없습니다: {image_path}")
            return 0 # 처리한 파일 개수 0 반환
    except Exception as e:
        print(f"  [오류] 이미지 로드 중: {e}")
        return 0

   
    
    # 2. 출력 폴더 생성 (이미 메인에서 생성했지만, 안전을 위해 한 번 더)
    os.makedirs(output_dir, exist_ok=True)

    # 3. 출력 파일명 베이스 설정
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    # 4. 좌표 텍스트 파일 읽기
    try:
        with open(txt_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"  [오류] 좌표 파일 읽기 중: {e}")
        return 0
        
    # 5. 각 라인(폴리곤)을 순회하며 자르기
    crop_count = 0
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue

        try:
            # 5-1. 좌표 파싱 (x1,y1,x2,y2,x3,y3,x4,y4)
            parts = line.split(',')
            if len(parts) != 8:
                continue
            
            points = np.array([float(p) for p in parts]).reshape(-1, 2).astype(np.float32)

            # 5-2. 변환할 사각형의 너비(width)와 높이(height) 계산
            width_top = np.linalg.norm(points[0] - points[1])
            width_bottom = np.linalg.norm(points[3] - points[2])
            target_width = int(max(width_top, width_bottom))
            
            height_left = np.linalg.norm(points[0] - points[3])
            height_right = np.linalg.norm(points[1] - points[2])
            target_height = int(max(height_left, height_right))

            if target_width <= 0 or target_height <= 0:
                continue

            # 5-3. 목표 사각형(Destination) 좌표 정의
            dst_points = np.array([
                [0, 0],
                [target_width - 1, 0],
                [target_width - 1, target_height - 1],
                [0, target_height - 1]
            ], dtype=np.float32)

            # 5-4. 원근 변환 매트릭스 계산
            matrix = cv2.getPerspectiveTransform(points, dst_points)
            
            # 5-5. 원근 변환 적용 (이미지 "펴기")
            warped_image = cv2.warpPerspective(image, matrix, (target_width, target_height))

            # 5-6. (선택 사항) 세로로 긴 이미지(세로 쓰기 단어)를 90도 회전
            if target_height > target_width * 1.5:
                warped_image = cv2.rotate(warped_image, cv2.ROTATE_90_CLOCKWISE)

            # 5-7. 파일로 저장
            crop_count += 1
            output_filename = f"{base_name}_{crop_count}.jpg"
            output_path = os.path.join(output_dir, output_filename)
            
            cv2.imwrite(output_path, warped_image)

        except Exception as e:
            # 라인 처리 중 오류가 나도 다음 라인은 계속 처리
            print(f"    [경고] {i+1}번째 라인 처리 중 문제 발생 ({e}). 건너뜁니다.")
            
    # 총 몇 개의 crop 이미지를 만들었는지 반환
    return crop_count


# --- (2) 메인 실행 부분 (폴더 전체 처리 로직) ---
if __name__ == "__main__":
    
    # --- (중요) 사용자의 환경에 맞게 3개의 "폴더" 경로를 수정하세요 ---
    
    # 1. 원본 이미지가 들어있는 폴더
    # (예: "My_images" 또는 "CRAFT_Make_Polugon/my_test_images")
    source_image_folder = "./CRAFT_Make_Polygon/my_test_images"

    # 2. 좌표 텍스트 파일(res_...txt)이 들어있는 폴더
    # (예: "result")
    source_txt_folder = "./CRAFT_Make_Polygon/result"
    
    # 3. 잘라낸 이미지를 저장할 폴더 경로
    target_output_dir = "./inference/inference_data/cropped_images_for_inference"
    
     # 기존 생성된 이미지 삭제
    delete_files_in_directory(target_output_dir, ["*.jpg"])
    

    # --- (수정 불필요) 스크립트 실행 ---
    
    print(f"이미지 폴더: {source_image_folder}")
    print(f"텍스트 폴더: {source_txt_folder}")
    print(f"출력 폴더: {target_output_dir}")
    
    # 3-1. 출력 폴더 생성
    os.makedirs(target_output_dir, exist_ok=True)
    
    # 3-2. 원본 이미지 폴더에서 .jpg, .png, .jpeg 파일 목록을 모두 찾음
    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(source_image_folder, ext)))
        
    if not image_files:
        print(f"\n오류: '{source_image_folder}'에서 처리할 이미지를 찾을 수 없습니다.")
        print("이미지 폴더 경로를 확인하세요.")
    else:
        print(f"\n총 {len(image_files)}개의 원본 이미지를 찾았습니다. 변환을 시작합니다...")

    total_cropped_count = 0
    
    # 3-3. 찾은 이미지 파일들을 하나씩 순회
    for img_path in image_files:
        
        # 3-4. 이미지명 기준으로 텍스트 파일 경로 추정
        # (예: .../cosmetics_01993.jpg -> cosmetics_01993)
        img_basename = os.path.basename(img_path)
        img_name_only = os.path.splitext(img_basename)[0]
        
        # (예: .../result/res_cosmetics_01993.txt)
        txt_filename = f"res_{img_name_only}.txt"
        txt_path = os.path.join(source_txt_folder, txt_filename)
        
        print(f"\n--- 처리 시작: {img_basename} ---")
        
        # 3-5. 짝이 맞는 텍스트 파일이 존재하는지 확인
        if not os.path.exists(txt_path):
            print(f"  [경고] 짝이 맞는 좌표 파일이 없습니다. 건너뜁니다.")
            print(f"  (찾은 경로: {txt_path})")
            continue
            
        # 3-6. 짝이 맞으면 자르기 함수 실행
        print(f"  [OK] 좌표 파일 찾음: {txt_filename}")
        
        count = crop_and_warp_polygons(img_path, txt_path, target_output_dir)
        
        print(f"  [완료] {img_basename} -> {count}개의 이미지 저장 완료.")
        total_cropped_count += count

    print(f"\n--- 최종 완료 ---")
    print(f"총 {len(image_files)}개의 원본 이미지 처리.")
    print(f"총 {total_cropped_count}개의 잘라낸 이미지를 '{target_output_dir}'에 저장했습니다.")
