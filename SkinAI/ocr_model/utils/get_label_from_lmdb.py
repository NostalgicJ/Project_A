import lmdb
import sys
import os
from tqdm import tqdm

# lmdb로 부터 분류된 레이블을 가져옴
# 이 레이블들을 lo.txt에 저장
def extract_unique_characters(lmdb_path):
    """
    LMDB에서 모든 라벨을 읽어와 중복 제거된 글자 목록을 만듭니다.
    """
    all_labels = []
    env = None
    
    try:
        # LMDB 환경(폴더) 열기
        env = lmdb.open(lmdb_path, readonly=True, max_readers=1, lock=False, readahead=False, meminit=False)
        if not env:
            print(f"오류: LMDB를 열 수 없습니다. 경로를 확인하세요: {lmdb_path}")
            return
            
        with env.begin(write=False) as txn:
            # 총 샘플 개수 가져오기
            num_samples_key = b'num-samples'
            num_samples_data = txn.get(num_samples_key)
            if num_samples_data is None:
                print(f"오류: 'num-samples' 키를 LMDB에서 찾을 수 없습니다.")
                return
                
            num_samples = int(num_samples_data.decode())
            print(f"LMDB에서 총 {num_samples}개의 샘플을 발견했습니다.")

            # 모든 라벨 읽기
            for index in tqdm(range(1, num_samples + 1), desc="라벨 읽는 중"):
                label_key = f'label-{index:09d}'.encode()
                label_data = txn.get(label_key)
                
                if label_data is None:
                    print(f"경고: {label_key}에 해당하는 라벨이 없습니다.")
                    continue
                
                # 라벨을 utf-8로 디코딩
                label = label_data.decode('utf-8')
                all_labels.append(label)

    except lmdb.Error as e:
        print(f"LMDB 오류 발생: {e}")
        return
    finally:
        if env:
            env.close()

    # --- 중복 제거 ---
    if not all_labels:
        print("읽어온 라벨이 없습니다.")
        return

    # 1. 모든 라벨을 하나의 긴 문자열로 합침
    full_text = "".join(all_labels)
    
    # 2. set()을 사용해 중복 제거
    unique_chars = set(full_text)
    
    # 3. 정렬
    sorted_unique_chars = sorted(list(unique_chars))
    
    # 4. 다시 하나의 문자열로 합침
    final_char_string = "".join(sorted_unique_chars)
    
    num_unique = len(final_char_string)
    print(f"\n중복 제거된 고유 글자 개수: {num_unique}개")

    # --- 파일로 저장 ---
    output_filename = 'ko_correct.txt'
    try:
        with open(output_filename, 'w', encoding='utf-8') as f:
            f.write(final_char_string)
        print(f"성공: '{output_filename}' 파일에 {num_unique}개의 글자를 저장했습니다.")
        print("이 파일의 이름을 'ko.txt'로 변경한 뒤 demo.py를 다시 실행하세요.")
        
    except IOError as e:
        print(f"파일 저장 오류: {e}")

# --- 스크립트 실행 ---
if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("="*50)
        print(" [사용법]")
        print(" python extract_chars.py [학습용_LMDB_폴더_경로]")
        print("\n [예시]")
        print(" python extract_chars.py /home/yys/ai_hub_package/data/Training_LMDB")
        print("="*50)
    else:
        lmdb_folder_path = sys.argv[1]
        if not os.path.isdir(lmdb_folder_path):
            print(f"오류: '{lmdb_folder_path}'는 폴더가 아닙니다.")
        else:
            extract_unique_characters(lmdb_folder_path)
