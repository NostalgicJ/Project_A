import re
import time
import torch
import torch.nn.functional as F
from collections import OrderedDict
import argparse
import os

# --- 추론에 필요한 PyTorch/PIL 모듈 ---
import torch.utils.data
from torch.utils.data import Dataset

from torchvision import transforms
from PIL import Image
import lmdb
from io import BytesIO

# --- demo.py에서 가져온 모듈 ---
# (utils 폴더가 ai_hub_package 안에 있어야 합니다)
from utils.converter import AttnLabelConverter
from models.ops import ModelContainer
from utils.create_inference_lmdb import create_initial_lmdb
from utils.delete_file import delete_files_in_directory
from Change_Result import process_inference_file



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def Inference(model, evaluation_loader, converter, opt):
    """ 
    Inference 전용 함수 (demo.py의 validation 함수 변형)
    Loader가 (file_names, image_tensors)를 반환한다고 가정합니다.
    """
    
       
    
    length_of_data = 0
    infer_time = 0
    
    # 모든 배치의 결과를 저장할 리스트
    total_file_names = []  # (✅ 추가) 이미지 파일 이름을 저장할 리스트
    total_preds_str = []
    total_confidence_scores = []
    
    print("Starting Inference...")
    
    # [✅ 수정] 로더가 (파일이름_리스트, 이미지_텐서_배치)를 반환
    for i, (file_names, image_tensors) in enumerate(evaluation_loader):
        
        batch_size = image_tensors.size(0)
        length_of_data = length_of_data + batch_size
        image = image_tensors.to(device)
        
        # (✅ 추가) 파일 이름(키) 리스트 저장
        # file_names는 ('img1.jpg', 'img2.jpg', ...) 형태의 튜플/리스트
        total_file_names.extend(file_names)
        
        # For max length prediction
        length_for_pred = torch.IntTensor([opt.batch_max_length] * batch_size).to(device)
        text_for_pred = torch.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0).to(device)

        start_time = time.time()
        
        # [✅ 수정] torch.no_grad()로 추론 최적화
        with torch.no_grad():
            preds = model(image, text_for_pred, is_train=False)
            
        forward_time = time.time() - start_time
        
        # [❌ 삭제] 라벨 및 손실(loss) 계산 로직 전부 제거
        
        _, preds_index = preds.max(2)
        preds_str = converter.decode(preds_index, length_for_pred)

        infer_time += forward_time

        # calculate accuracy & confidence score
        preds_prob = F.softmax(preds, dim=2)
        preds_max_prob, _ = preds_prob.max(dim=2)
        confidence_score_list = []
        
        # [✅ 수정] 라벨(gt) 없이 예측(pred)만 처리
        for pred, pred_max_prob in zip(preds_str, preds_max_prob):
            
            pred_EOS = pred.find('[s]')
            pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
            pred_max_prob = pred_max_prob[:pred_EOS]

            # [❌ 삭제] 정확도 및 Edit Distance 계산 로직 전부 제거

            try:
                # .item()을 사용해 텐서에서 float 값 추출
                confidence_score = pred_max_prob.cumprod(dim=0)[-1].item()
            except:
                confidence_score = 0  # for empty pred case
            
            confidence_score_list.append(confidence_score)
            
            # [✅ 수정] total_preds_str 리스트에는 [s]가 제거된 최종 문자열을 저장
            total_preds_str.append(pred)

        total_confidence_scores.extend(confidence_score_list)

    print(f"Inference complete. Processed {length_of_data} images in {infer_time:.2f}s.")

    # [✅ 수정] 파일이름 목록, 예측 문자열, 신뢰도 점수를 반환
    return total_file_names, total_preds_str, total_confidence_scores



# -----------------------------------------------------------------
# [✅ 추론용 데이터셋 클래스 (수정본)]
# (create_inference_lmdb.py로 생성한 LMDB 전용)
# -----------------------------------------------------------------
import base64 # (파일 맨 위 import 영역에 이미 있어야 함)
import json   # (json을 파싱하기 위해 import 필요)

class InferenceDataset(Dataset):
    def __init__(self, root, opt):
        self.root = root
        self.opt = opt
        
        print(f"Loading LMDB from: {root}")
        self.env = lmdb.open(root, readonly=True, max_readers=1, lock=False, readahead=False, meminit=False)
        if not self.env:
            raise IOError('Cannot create lmdb environment', root)

        self.keys = []
        with self.env.begin(write=False) as txn:
            # [✅ 수정] 'num-samples' 대신 모든 키를 순회하여 리스트로 저장
            cursor = txn.cursor()
            for key, _ in cursor:
                self.keys.append(key)
        
        self.nSamples = len(self.keys)
        if self.nSamples == 0:
             raise ValueError(f"LMDB '{root}' is empty.")
             
        print(f"Found {self.nSamples} samples in LMDB.")

        # 이미지 전처리 (demo.py의 single_image_inference와 동일하게)
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1 if not opt.rgb else 3),
            transforms.Resize((opt.imgH, opt.imgW)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]) # Grayscale 기준
        ])

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        
        # [✅ 수정] 인덱스를 사용해 키(파일이름)를 가져옴
        key = self.keys[index]
        file_name = key.decode('utf-8')
        
        with self.env.begin(write=False) as txn:
            # 1. 키로 Value (JSON 문자열)를 가져옵니다.
            value_json_string = txn.get(key)
            
            if value_json_string is None:
                print(f"Error: Key {file_name} not found in LMDB.")
                return file_name, torch.zeros(1 if not self.opt.rgb else 3, self.opt.imgH, self.opt.W)

            try:
                # 2. JSON 파싱
                data = json.loads(value_json_string.decode('utf-8'))
                
                # 3. Base64 디코딩
                image_base64_string = data['image_data_base64']
                image_binary = base64.b64decode(image_base64_string)
                
                # 4. BytesIO를 사용해 바이너리 데이터를 PIL 이미지로 변환
                image_pil = Image.open(BytesIO(image_binary)).convert('L') # 그레이스케일
                
            except Exception as e:
                print(f"Error reading/decoding image {file_name}: {e}")
                # 오류 발생 시 빈 이미지와 파일이름 반환
                return file_name, torch.zeros(1 if not self.opt.rgb else 3, self.opt.imgH, self.opt.imgW)

            # 5. 이미지 전처리
            image_tensor = self.transform(image_pil)
            
            # (파일이름, 이미지 텐서) 반환
            return file_name, image_tensor

# -----------------------------------------------------------------
# [✅ 추론용 Collate 함수]
# (파일이름은 리스트로, 이미지는 텐서로 묶음)
# -----------------------------------------------------------------
def AlignCollateInference(batch):
    file_names, images = zip(*batch)
    image_tensors = torch.cat([img.unsqueeze(0) for img in images], 0)
    return file_names, image_tensors

# -----------------------------------------------------------------
# [✅ 메인(main) 실행 블록]
# -----------------------------------------------------------------
if __name__ == '__main__':
    
    # --- 옵션(Argument) 파서 ---
    parser = argparse.ArgumentParser()
    # [✅ 수정] 추론용 LMDB 경로를 받도록 변경
    parser.add_argument('--inference_data', type=str, default ='./inference/inference_data', help='path to inference LMDB')
    parser.add_argument('--saved_model', type=str, default='./saved_models/93992a6c-5cc0-4f24-bc4a-17d2681d9c37-Seed1234/best_accuracy.pth',  help='path to saved_model')
    
    parser.add_argument('--inference_result', type=str, default ='./inference/inference_result', help='path to inference result folder')
    
    parser.add_argument('--batch_size', type=int, default=3, help='input batch size')
    parser.add_argument('--workers', type=int, default=0, help='number of data loading workers')
    
    # --- cfg.py / demo.py의 필수 옵션들 ---
    parser.add_argument('--exp_name', type=str, default='Inference', help='Where to store logs and models')
    parser.add_argument('--manualSeed', type=int, default=1234, help='for reproducibility')
    parser.add_argument('--batch_max_length', type=int, default=25, help='maximum-label-length') #최대 레이블 글자수
    parser.add_argument('--imgH', type=int, default=32, help='the height of the input image')
    parser.add_argument('--imgW', type=int, default=100, help='the width of the input image')
    parser.add_argument('--rgb', action='store_true', help='use rgb input')
    parser.add_argument('--character', type=str, default='ko', help='character label')
    parser.add_argument('--sensitive', action='store_true', help='for sensitive character mode')
    parser.add_argument('--PAD', action='store_true', help='whether to keep ratio then pad for image resize')
    
    # --- ops.py의 ModelContainer 옵션들 ---
    parser.add_argument('--input_channel', type=int, default=1, help='the number of input channel of Feature extractor')
    parser.add_argument('--output_channel', type=int, default=512, help='the number of output channel of Feature extractor')
    parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')
    
    # for create inference_lmdb 만들기 위한것
    
    parser.add_argument('--inference_cropped_image_folder', type=str, default ='./inference/inference_data/cropped_images_for_inference', 
                        help='(입력) 원본 이미지 파일들이 있는 폴더 경로 (e.g., ./my_test_images)')
    parser.add_argument('--inference_lmdb_path', type=str, default = './inference/inference_data', 
                        help='(출력) 생성할 LMDB 폴더 경로 (e.g., ./inference_lmdb)')
                        

    opt = parser.parse_args()
     # 기존 생성된 이미지 삭제
    delete_files_in_directory(opt.inference_lmdb_path, ["*.mdb"])
    delete_files_in_directory(opt.inference_result, ["*.txt"])
    




 #0 inference_lmdb 기존 내용 모두 지우기 
    
    lmdb_path = opt.inference_lmdb_path

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
        
         
    # 1. (필수) 초기 LMDB 생성 및 cropped image 정보 입력
    # --------------------------
    create_initial_lmdb(opt.inference_cropped_image_folder, opt.inference_lmdb_path)
    
    
    
    
    # [ ✅ 새 코드: 저장된 converter.pth 로드 ]
    # saved_model 경로에서 converter 경로를 추정
    converter_path = os.path.join(os.path.dirname(opt.saved_model), 'converter.pth')
    
    
    
    
    if not os.path.exists(converter_path):
        raise FileNotFoundError(f"Error: 'converter.pth' not found in model directory.")
        
    print(f"Loading converter from: {converter_path}")
    
    converter = torch.load(converter_path, map_location=device, weights_only=False)
    
    opt.num_class = len(converter.character)
    
       # -------------------------------
    # 2. 모델(Model) 준비
    # -------------------------------
    model = ModelContainer(opt)
    model = model.to(device)
    
    print(f'Loading pretrained model from {opt.saved_model}')
    # 'module.' 접두사 제거 로직 (demo.py와 동일)
    saved_state_dict = torch.load(opt.saved_model, map_location=device)
    new_state_dict = OrderedDict()
    for k, v in saved_state_dict.items():
        name = k.replace('module.', '', 1) # 'module.' 접두사 제거
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    model.eval()

    # -------------------------------
    # 3. 데이터 로더(DataLoader) 준비
    # -------------------------------
    inference_dataset = InferenceDataset(root=opt.inference_data, opt=opt)
    evaluation_loader = torch.utils.data.DataLoader(
        inference_dataset, batch_size=opt.batch_size,
        shuffle=False,  # 추론 시에는 섞을 필요 없음 (순서대로)
        num_workers=int(opt.workers),
        collate_fn=AlignCollateInference, pin_memory=True)

    # -------------------------------
    # 4. 추론(Inference) 실행
    # -------------------------------
    file_names, preds, scores = Inference(model, evaluation_loader, converter, opt)

    # -------------------------------
    # 5. 결과 저장 및 출력
    # -------------------------------
    output_filename = opt.inference_result+"/" + "inference_results.txt"
    with open(output_filename, 'w', encoding='utf-8') as f:
        f.write("ImageName\tPrediction\tConfidence\n")
        for name, pred_text, score in zip(file_names, preds, scores):
            f.write(f"{name}\t{pred_text}\t{score:.4f}\n")
     process_inference_file('./inference/inference_result/inference_results.txt', './inference/inference_result/inference_results.txt')
     
    print(f"\nSuccess: Inference results saved to '{output_filename}'")
  
