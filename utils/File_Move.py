import os
import shutil
import random

# --- 1. [사용자 설정] 원본 파일이 있는 폴더 ---
# (예시: 크롭된 이미지 폴더)
SOURCE_FOLDER = "/home/yys/ai_hub_package/utils/cropped_images"

# --- 2. [사용자 설정] N%의 파일을 "이동"시킬 새 폴더 ---
# (예시: 검증용 이미지 폴더)
DEST_FOLDER = "/home/yys/ai_hub_package/validation_result/Validation_Cropped_Images"

# --- 3. [사용자 설정] 이동할 파일의 비율 (예: 10 = 10%) ---
PERCENT_TO_MOVE = 20

# -----------------------------------------------------------------
# (메인 스크립트)
# -----------------------------------------------------------------

def move_random_files():
    # 1. 대상 폴더 생성 (없으면)
    os.makedirs(DEST_FOLDER, exist_ok=True)

    # 2. 원본 폴더에서 "파일" 목록만 가져오기 (하위 폴더는 제외)
    try:
        all_files = [f for f in os.listdir(SOURCE_FOLDER) 
                     if os.path.isfile(os.path.join(SOURCE_FOLDER, f))]
    except FileNotFoundError:
        print(f"❌ [오류] 원본 폴더를 찾을 수 없습니다: {SOURCE_FOLDER}")
        return
    except Exception as e:
        print(f"❌ [오류] 파일 목록을 읽는 중 오류 발생: {e}")
        return

    total_files = len(all_files)
    
    if total_files == 0:
        print(f"ℹ️ [정보] 원본 폴더에 파일이 없습니다: {SOURCE_FOLDER}")
        return

    # 3. 이동할 파일 개수 계산 (N%)
    num_to_move = int(total_files * (PERCENT_TO_MOVE / 100.0))
    
    if num_to_move == 0 and total_files > 0:
        print(f"ℹ️ [정보] 이동할 파일이 0개입니다. (총 {total_files}개의 {PERCENT_TO_MOVE}%)")
        return

    print(f"원본 폴더: {SOURCE_FOLDER} (총 {total_files}개 파일)")
    print(f"대상 폴더: {DEST_FOLDER}")
    print(f" -> {PERCENT_TO_MOVE}%에 해당하는 {num_to_move}개의 파일을 이동합니다...")

    # 4. 이동할 파일 랜덤 선택
    # random.sample은 리스트에서 중복 없이 N개의 항목을 뽑습니다.
    files_to_move = random.sample(all_files, num_to_move)

    moved_count = 0
    
    # 5. 파일 이동 실행
    for file_name in files_to_move:
        src_path = os.path.join(SOURCE_FOLDER, file_name)
        dest_path = os.path.join(DEST_FOLDER, file_name)
        
        try:
            # shutil.move가 "이동"을 수행합니다.
            shutil.move(src_path, dest_path)
            moved_count += 1
        except Exception as e:
            print(f"\n[경고] {file_name} 파일 이동 실패: {e}")
            
    print(f"\n--- ✅ 작업 완료 ---")
    print(f"총 {moved_count}개의 파일 이동을 완료했습니다.")

# 스크립트 실행
if __name__ == "__main__":
    move_random_files()
