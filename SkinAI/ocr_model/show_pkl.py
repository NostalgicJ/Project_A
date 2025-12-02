import pickle
import sys
import os

# --- (중요) 원본 클래스 import (필수) ---
# 이 import가 성공해야 .pkl 파일을 열 수 있습니다.
# (이전 로그에서 'utils.etc.AttnLabelConverter'로 떴지만,
#  Inference_copyed_from_demo.py를 보면 
#  'utils.converter'가 맞을 수 있습니다. 이전 실행에서 성공한 
#  'utils.converter'를 사용합니다.)
try:
    from utils.converter import AttnLabelConverter
    print("Import 성공: AttnLabelConverter (utils.converter)")
except ImportError as e:
    print(f"!!! Import 실패: {e}")
    print("이 스크립트는 utils 폴더와 같은 위치 (ai_hub_package)에서 실행해야 합니다.")
    print("이로 인해 아래에서 오류가 날 수 있습니다.")
    pass
# ---

# (1) 확인할 파일 경로
file_path = 'data.pkl' 

# (2) 파일 존재 여부 확인
if not os.path.exists(file_path):
    print(f"\n!!! 치명적 에러: 파일을 찾을 수 없습니다 !!!")
    print(f"파일 경로: {os.path.abspath(file_path)}")
    sys.exit()

print(f"\n파일을 찾았습니다: {file_path}")
print("순수 pickle로 객체를 로드합니다...")

try:
    with open(file_path, 'rb') as f:
        # latin1 인코딩이 이전 실행에서 성공했습니다.
        data_object = pickle.load(f, encoding='latin1')
    
    print("\n--- 객체 로드 성공 ---")
    print(f"객체 타입: {type(data_object)}")

    # ---
    # (핵심) 객체 내부의 속성을 딕셔너리로 보여줍니다.
    # ---
    print("\n--- [1] 객체가 가진 모든 속성 (vars) ---")
    try:
        attributes = vars(data_object)
        print(attributes)
        
        # ---
        # (핵심2) AttnLabelConverter의 핵심은 'character' 리스트입니다.
        # 이 속성을 직접 출력해 봅니다.
        # ---
        print("\n--- [2] 'character' 속성 직접 접근 ---")
        if hasattr(data_object, 'character'):
            print("data_object.character 의 내용:")
            print(data_object.character)
        else:
            print("'character' 속성을 찾을 수 없습니다.")
            print("위의 [1]번 딕셔너리에서 문자 리스트가 들어있는 키(key)를 확인해보세요.")
            
    except TypeError:
        print("vars()를 사용할 수 없는 객체입니다.")

except Exception as e:
    print(f"\n--- [최종 오류] pickle 로드 또는 속성 접근 실패 ---")
    print(f"오류 메시지: {e}")
