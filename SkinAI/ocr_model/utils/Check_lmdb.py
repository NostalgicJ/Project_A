import lmdb
import sys
import os

# --- 1. 확인할 LMDB 폴더 경로 ---
# lmdb_convert.py의 --outputPath와 동일한 경로를 적어주세요.
LMDB_PATH = "/home/yys/ai_hub_package/data_set/MJ"
# --------------------------------

def check_lmdb_status(lmdb_path):
    if not os.path.exists(lmdb_path):
        print(f"❌ [오류] 폴더를 찾을 수 없습니다: {lmdb_path}")
        return

    try:
        # 1. LMDB 환경을 읽기 전용으로 엽니다.
        env = lmdb.open(lmdb_path, readonly=True, max_readers=1, lock=False)
    except lmdb.Error as e:
        print(f"❌ [비정상] LMDB 파일을 여는 데 실패했습니다 (파일이 손상되었을 수 있음).")
        print(f"LMDB 오류: {e}")
        return

    nSamples = 0
    with env.begin() as txn:
        try:
            # 2. 'num-samples' 키를 읽어옵니다.
            num_samples_bin = txn.get('num-samples'.encode())
            
            if num_samples_bin is None:
                print(f"❌ [비정상] LMDB가 불완전합니다.")
                print(f"원인: 'num-samples' 키를 찾을 수 없습니다.")
                print(f"추정: lmdb_convert.py가 끝까지 실행되지 못하고 중단되었습니다.")
            else:
                # 3. 키가 존재하면 총 샘플 수를 확인합니다.
                nSamples = int(num_samples_bin.decode())
                print(f"✅ [정상] 'num-samples' 키를 찾았습니다.")
                print(f"   총 샘플 수: {nSamples} 개")
                
                # 4. (추가 검증) 첫 번째와 마지막 샘플 키를 확인합니다.
                if nSamples > 0:
                    first_img = txn.get('image-%09d'.encode() % 1)
                    last_img = txn.get('image-%09d'.encode() % nSamples)
                    
                    if first_img is None:
                        print(f"❌ [비정상] 첫 번째 이미지(image-000000001) 키가 없습니다.")
                    elif last_img is None:
                         print(f"❌ [비정상] 마지막 이미지(image-{nSamples:09d}) 키가 없습니다.")
                    else:
                        print(f"✅ [정상] 첫 번째와 마지막 샘플 키가 모두 존재합니다.")

        except Exception as e:
            print(f"❌ [비정상] LMDB 데이터를 읽는 중 오류 발생: {e}")

    env.close()

if __name__ == "__main__":
    check_lmdb_status(LMDB_PATH)
