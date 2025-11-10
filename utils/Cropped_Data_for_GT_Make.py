import os

# --- 1. [사용자 설정] 검사할 이미지 폴더 ---
IMAGE_DIR = "/home/yys/ai_hub_package_for_github/validation_result/Validation_Cropped_Images"

# --- 2. [사용자 설정] 원본 라벨 목록 파일 ---
GT_FILE_PATH = "/home/yys/ai_hub_package_for_github/utils/gt_cropped_for_validation.txt"

# --- 3. [사용자 설정] 청소된 레코드만 저장할 "새" 라벨 목록 파일 ---
OUTPUT_CLEANED_GT_FILE = "/home/yys/ai_hub_package_for_github/utils/gt_cropped_for_validation_NEW.txt"

# -----------------------------------------------------------------
# (메인 스크립트)
# -----------------------------------------------------------------

def clean_gt_file():
    
    total_lines = 0
    kept_lines = 0
    deleted_lines = 0
    
    print(f"이미지 폴더: {IMAGE_DIR}")
    print(f"라벨 파일: {GT_FILE_PATH}")
    
    # 1. 원본 gt.txt 파일 열기
    try:
        with open(GT_FILE_PATH, 'r', encoding='utf-8') as infile, \
             open(OUTPUT_CLEANED_GT_FILE, 'w', encoding='utf-8') as outfile:
            
            # 2. 원본 파일을 한 줄씩 읽기
            for line in infile:
                total_lines += 1
                
                # 3. 줄바꿈 문자 제거 및 탭으로 분리
                try:
                    line_stripped = line.strip()
                    if not line_stripped:
                        continue # 빈 줄은 건너뛰기
                        
                    parts = line_stripped.split('\t')
                    
                    if len(parts) != 2:
                        print(f"[경고] 형식 오류 (탭 1개 아님): {line_stripped[:50]}... -> 삭제")
                        deleted_lines += 1
                        continue
                        
                    image_filename = parts[0]
                    
                except Exception as e:
                    print(f"[경고] 줄 처리 중 오류: {e} -> 삭제")
                    deleted_lines += 1
                    continue

                # 4. [핵심] 이미지 파일이 실제로 존재하는지 확인
                full_image_path = os.path.join(IMAGE_DIR, image_filename)
                
                if os.path.exists(full_image_path):
                    # 5. 파일이 존재하면, 새 파일에 원본 줄을 그대로 쓴다.
                    outfile.write(line)
                    kept_lines += 1
                else:
                    # 6. 파일이 존재하지 않으면, 이 레코드를 삭제(쓰지 않음)한다.
                    # print(f"[삭제] {image_filename} 파일이 존재하지 않음.")
                    deleted_lines += 1
                    
                if total_lines % 10000 == 0:
                    print(f"... {total_lines} 줄 처리 중 ... (유지: {kept_lines}, 삭제: {deleted_lines}) ...", end='\r')

        print(f"\n\n--- ✅ 작업 완료 ---")
        print(f"총 {total_lines}개 레코드 처리 완료.")
        print(f"  [유지] {kept_lines}개 (파일 존재)")
        print(f"  [삭제] {deleted_lines}개 (파일 없음 또는 형식 오류)")
        print(f"\n결과가 '{OUTPUT_CLEANED_GT_FILE}' 파일에 저장되었습니다.")
        
    except FileNotFoundError:
        print(f"❌ [오류] 파일을 찾을 수 없습니다. 경로를 확인하세요:")
        print(f"  {GT_FILE_PATH}")
        print(f"  {IMAGE_DIR}")
    except Exception as e:
        print(f"❌ [오류] 스크립트 실행 중 오류 발생: {e}")


# 스크립트 실행
if __name__ == "__main__":
    clean_gt_file()
