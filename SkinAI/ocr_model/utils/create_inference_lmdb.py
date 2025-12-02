import lmdb
import json
import base64  # 이미지를 텍스트(JSON)에 저장하기 위해 Base64 인코딩 사용
import os
import argparse
import glob
from tqdm import tqdm

def create_initial_lmdb(image_folder_path, lmdb_path):
    """
    지정된 폴더의 이미지들을 읽어 초기 LMDB를 생성합니다.
    Key: 파일 이름 (e.g., "IMG_3660.jpg")
    Value: {
        "image_data_base64": (Base64 인코딩된 이미지 텍스트),
        "prediction": null,
        "confidence": null
    }
    """
    
    # 1. 대상 폴더에서 이미지 파일 목록을 가져옵니다 (jpg, jpeg, png)
    image_extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff')
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(image_folder_path, ext)))
        
    if not image_files:
        print(f"오류: '{image_folder_path}' 폴더에서 이미지 파일을 찾을 수 없습니다.")
        return

    print(f"총 {len(image_files)}개의 이미지 파일을 LMDB로 변환합니다...")

    # 2. LMDB 환경 생성 (넉넉한 map_size 설정)
    # (이미지 파일 크기에 따라 map_size를 조절해야 할 수 있습니다)
    map_size = 10 * 1024 * 1024 * 1024  # 10 GB
    env = lmdb.open(lmdb_path, map_size=map_size)

    # 3. 쓰기 트랜잭션 시작
    with env.begin(write=True) as txn:
        for image_path in tqdm(image_files, desc="LMDB 생성 중"):
            try:
                # 3-1. Key: 파일 이름 (e.g., "IMG_3660.jpg")
                key = os.path.basename(image_path)
                
                # 3-2. Value: JSON 생성을 위한 데이터 준비
                
                # (a) 이미지를 바이너리로 읽기
                with open(image_path, "rb") as f:
                    image_binary = f.read()
                    
                # (b) 바이너리를 Base64 텍스트로 인코딩
                image_base64_string = base64.b64encode(image_binary).decode('utf-8')
                
                # (c) Python 딕셔너리 생성 (None은 JSON의 null이 됨)
                value_dict = {
                    "image_data_base64": image_base64_string,
                    "prediction": None,
                    "confidence": None
                }
                
                # (d) 딕셔너리를 JSON 문자열로 변환
                value_json_string = json.dumps(value_dict)
                
                # 3-3. LMDB에 저장 (Key와 Value는 모두 bytes여야 함)
                txn.put(key.encode('utf-8'), value_json_string.encode('utf-8'))
                
            except Exception as e:
                print(f"오류: {image_path} 처리 중 문제 발생: {e}")

    env.close()
    print(f"\n성공: '{lmdb_path}'에 초기 LMDB 생성이 완료되었습니다.")


def update_lmdb_record(lmdb_path, key, prediction, confidence):
    """
    (추후 사용 예시) LMDB의 특정 키(파일 이름) 레코드를 업데이트합니다.
    """
    env = lmdb.open(lmdb_path, map_size=10 * 1024 * 1024 * 1024)
    
    with env.begin(write=True) as txn:
        # 1. 기존 Value (JSON 문자열)를 가져옵니다.
        value_json_string = txn.get(key.encode('utf-8'))
        
        if value_json_string is None:
            print(f"경고: 키 '{key}'를 찾을 수 없어 업데이트에 실패했습니다.")
            return

        # 2. JSON 문자열을 Python 딕셔너리로 파싱
        data = json.loads(value_json_string.decode('utf-8'))
        
        # 3. 딕셔너리의 'prediction'과 'confidence' 값을 업데이트
        data['prediction'] = prediction
        data['confidence'] = confidence
        
        # 4. 수정된 딕셔너리를 다시 JSON 문자열로 변환
        updated_value_json_string = json.dumps(data)
        
        # 5. LMDB에 덮어쓰기 (put)
        txn.put(key.encode('utf-8'), updated_value_json_string.encode('utf-8'))
        
    env.close()
    print(f"업데이트: 키 '{key}'의 예측값이 '{prediction}'으로 업데이트되었습니다.")


def read_lmdb_record(lmdb_path, key):
    """
    (확인용 예시) LMDB의 특정 키 레코드를 읽어 출력합니다.
    """
    env = lmdb.open(lmdb_path, readonly=True)
    
    with env.begin(write=False) as txn:
        value_json_string = txn.get(key.encode('utf-8'))
        
        if value_json_string is None:
            print(f"조회 실패: 키 '{key}'를 찾을 수 없습니다.")
            return

        data = json.loads(value_json_string.decode('utf-8'))
        
        print(f"\n--- [키: {key} 레코드 조회] ---")
        print(f"  Prediction: {data['prediction']}")
        print(f"  Confidence: {data['confidence']}")
        print(f"  Image Data (Base64): {data['image_data_base64'][:50]}...") # (앞 50자만)
        print("-" * (26 + len(key)))
        
    env.close()


# -----------------------------------------------------------------
# [✅ 메인(main) 실행 블록]
# -----------------------------------------------------------------
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Create an LMDB database for inference.")
    
    
    parser.add_argument('--image_folder', type=str, default ='/home/yys/ai_hub_package/inference/inference_data/cropped_images_for_inference', 
                        help='(입력) 원본 이미지 파일들이 있는 폴더 경로 (e.g., ./my_test_images)')
    parser.add_argument('--lmdb_path', type=str, default = '/home/yys/ai_hub_package/inference/inference_data', 
                        help='(출력) 생성할 LMDB 폴더 경로 (e.g., ./inference_lmdb)')

    args = parser.parse_args()


    #0 lmdb 기존 내용 모두 지우기 
    
    lmdb_path = './inference_lmdb' 

    if not os.path.exists(lmdb_path):
       print(f"'{lmdb_path}' 폴더가 이미 없습니다.")
    else:
       try:
        # 1. LMDB 환경(폴더) 열기
          env = lmdb.open(lmdb_path)
        
        # 2. 쓰기 트랜잭션 시작
          with env.begin(write=True) as txn:
            
            # 3. (핵심) 데이터베이스의 모든 내용 삭제
            main_db = env.open_db()     # 1: 메인 데이터베이스 핸들 가져오기
            txn.drop(db=main_db)      # 2: 'drop' 메서드를 사용해 내용 삭제
            
          env.close()
          print(f"성공: '{lmdb_path}' LMDB의 모든 데이터(키/값)를 삭제했습니다.")
         
       except lmdb.Error as e:
         print(f"LMDB 오류 발생: {e}")
        
         
    # 1. (필수) 초기 LMDB 생성
    # --------------------------
    create_initial_lmdb(args.image_folder, args.lmdb_path)


    ''' # 2. (선택/예시) LMDB 확인 및 업데이트 테스트
    # -------------------------------------------
    # (IMG_3660.jpg 파일이 image_folder에 있다고 가정)
    test_key = 'IMG_3660.jpg' 
    
    # (a) 생성 직후 레코드 읽기 (예측값이 None인지 확인)
    print("\n--- [생성 직후 확인] ---")
    read_lmdb_record(args.lmdb_path, test_key)
    
    # (b) 추후 예측값을 업데이트하는 예시 (Inference.py 실행 후)
    # (실제로는 Inference.py의 결과로 이 함수를 호출해야 함)
    print("\n--- [업데이트 테스트] ---")
    update_lmdb_record(args.lmdb_path, test_key, "입출력", 0.95)
    
    # (c) 업데이트 후 레코드 다시 읽기
    print("\n--- [업데이트 후 확인] ---")
    read_lmdb_record(args.lmdb_path, test_key)
    '''
