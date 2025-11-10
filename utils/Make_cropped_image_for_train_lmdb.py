import os
import glob
import json
import cv2
import numpy as np
import argparse  # 1. argparse 라이브러리 추가
from delete_file import delete_files_in_directory

# --- 1. 사용자 설정 (이제 argparse의 기본값으로 사용됨) ---
# 기존 변수 선언은 삭제하고, 나중에 main 함수에서 args로 처리합니다.


# --- 2. 메인 스크립트 (수정됨) ---

# 2. 메인 함수가 args 인수를 받도록 수정
def crop_and_generate_gt(args):
    
    # 3. 하드코딩된 경로 대신 args의 속성 사용
    os.makedirs(args.output_crop_dir, exist_ok=True)
    
    json_files = glob.glob(os.path.join(args.label_dir, '*.json'))
    
    if not json_files:
        print(f"오류: '{args.label_dir}' 폴더에서 .json 파일을 찾을 수 없습니다.")
        return

    print(f"총 {len(json_files)}개의 원본 JSON/이미지 파일을 처리합니다...")
    
    
    # 미리 기존에 생성된 파일을 삭제함
    delete_files_in_directory(args.output_crop_dir, ["*.jpg"])
    delete_files_in_directory(args.output_gt_file_directory, ["*.txt"])

    global_crop_count = 1
    
    # 3. 하드코딩된 경로 대신 args의 속성 사용
    with open(args.output_gt_file, 'w', encoding='utf-8') as gt_outfile:
        
        for json_path in json_files:
            base_name = os.path.basename(json_path)
            image_name, _ = os.path.splitext(base_name)
            image_filename = image_name + ".jpg"
            
            # 3. 하드코딩된 경로 대신 args의 속성 사용
            image_path = os.path.join(args.image_dir, image_filename)
            
            if not os.path.exists(image_path):
                # print(f"[정보] {image_filename} 이미지를 찾을 수 없어 건너뜁니다.")
                continue

            try:
                image = cv2.imread(image_path)
                if image is None:
                    print(f"[오류] {image_filename} 이미지 파일을 읽을 수 없습니다.")
                    continue
                    
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if (not data.get('annotations') or 
                    len(data['annotations']) == 0 or 
                    not data['annotations'][0].get('polygons')):
                    # print(f"[정보] {base_name}: 'annotations' 또는 'polygons' 키가 없거나 비어있어 건너뜁니다.")
                    continue

                for polygon_data in data['annotations'][0]['polygons']:
                    try:
                        text = polygon_data['text']
                        points = polygon_data['points']

                        if (not text or 
                            text.strip() == "" or 
                            text.strip() == "|" or 
                            text.strip() == "·"):
                            continue
                            
                        np_points = np.array(points, dtype=np.int32)
                        
                        rect = cv2.boundingRect(np_points)
                        x, y, w, h = rect
                        
                        y_max, x_max = image.shape[:2]
                        x = max(0, x)
                        y = max(0, y)
                        w = min(x_max - x, w)
                        h = min(y_max - y, h)

                        cropped_img = image[y:y+h, x:x+w]
                        
                        if cropped_img.shape[0] == 0 or cropped_img.shape[1] == 0:
                            # print(f"[경고] {base_name}에서 '{text}' 크롭 실패 (영역 크기 0)")
                            continue

                        new_image_name = f"crop_{global_crop_count:09d}.jpg"
                        # 3. 하드코딩된 경로 대신 args의 속성 사용
                        new_image_path = os.path.join(args.output_crop_dir, new_image_name)
                        cv2.imwrite(new_image_path, cropped_img)
                        
                        gt_outfile.write(f"{new_image_name}\t{text.strip()}\n")
                        
                        global_crop_count += 1

                    except (KeyError, IndexError, TypeError) as e:
                        print(f"[경고] {base_name}의 일부 annotation 처리 실패 (구조 오류?): {e}") # base_Nmae 오타 수정
                        continue
                        
            except Exception as e:
                print(f"[오류] {base_name} 처리 중 예외 발생: {e}")
                continue

    print(f"\n--- 작업 완료 ---")
    # 3. 하드코딩된 경로 대신 args의 속성 사용
    print(f"총 {global_crop_count - 1}개의 잘린 이미지를 '{args.output_crop_dir}'에 저장했습니다.")
    print(f"LMDB 생성을 위한 '{args.output_gt_file}' 파일을 생성했습니다.")


# 스크립트 실행
if __name__ == "__main__":
    
    # 4. argparse 설정 추가
    parser = argparse.ArgumentParser(description='OCR 데이터셋 크롭 및 gt.txt 생성 스크립트')
    
    # 원본 파일의 경로를 기본값(default)으로 사용합니다.
    parser.add_argument('--image_dir', type=str, 
                        default="../data_set/Original_Image_Data_for_Train",
                        help='원본 .jpg 이미지 파일이 들어있는 폴더')
    
    parser.add_argument('--label_dir', type=str, 
                        default="../data_set/Original_JSON_Data_for_Train",
                        help='원본 .json 라벨 파일이 들어있는 폴더')
    
    parser.add_argument('--output_crop_dir', type=str, 
                        default="../data_set/Original_Image_Data_for_Train/Cropped_Images_for_Train",
                        help='잘려진(Cropped) 이미지들이 저장될 새 폴더')
                       
                        
    parser.add_argument('--output_gt_file', type=str, 
                        default="../data_set/Original_Image_Data_for_Train/gtFile.txt",
                        help='새로 생성될 라벨 목록 파일 (lmdb_convert.py의 gtFile로 사용)')
                        
                        
                       
    args = parser.parse_args()
    
    args.output_gt_file_directory = str(os.path.dirname(args.output_gt_file))
    
    
    
    # 5. args를 메인 함수로 전달
    crop_and_generate_gt(args)
