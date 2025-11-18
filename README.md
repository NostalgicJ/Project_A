#################################################################
학습 프로그램 구성



  1. 학습을 위한 Configure 세트 # 이미 지정되어 생략 가능
         프로그램 : cfg.py 에서 수정
      Saved model 불러오기 :  FT = true # 처음 학습시 false로 
      saved model path : --saved_model 지정
          
          
          
          
  2. 원본 이미지 전처리 : 원본 이미지를 Cropped Image로 만들고, lmdb 생성을 위한 gtFile 생성
            프로그램 : Make_cropped_image_for_train_lmdb.py, 실행 디렉토리 : "/ai_hub_package_for_github/utils
           실행 : python Make_cropped_image_for_train_lmdb.py
           # 원본 .jpg 이미지 파일이 들어있는 폴더
           IMAGE_DIR = "../data_set/Original_Image_Data_for_Train"
           
           # 원본 .json 라벨 파일이 들어있는 폴더
            LABEL_DIR = "../data_set/Original_JSON_Data_for_Train"

           # [출력 1] 잘려진(Cropped) 이미지들이 저장될 새 폴더
              OUTPUT_CROP_DIR = "../data_set/Original_Image_Data_for_Train/Cropped_Images_for_Train"

           # [출력 2] 새로 생성될 라벨 목록 파일 (lmdb_convert.py의 gtFile로 사용)

              OUTPUT_GT_FILE = ../data_set/Original_Image_Data_for_Train/gtFile.txt
              
              
              
              
  3. lmdb 만드는 과정 : 2번 결과를 이용 lmdb 만듬   
        
       # cropped 이미지와 한글레이블 2개만을 이용하여 lmdb 만듬
       # 실행 디렉토리 : "/ai_hub_package_for_github/utils
       # lmdb 만드는 프로그램:  
       
             python lmdb_convert.py \
              ../data_set/Original_Image_Data_for_Train/Cropped_Images_for_Train \
              ../data_set/Original_Image_Data_for_Train/gtFile.txt \
              ../data_set/MJ    
             
             
      # 2.에의해  gtFile이 JSON 파일 이용하여 먼저 만들어짐
         
          
          ARGS:
        inputPath  : input folder path where starts imagePath
        outputPath : LMDB output path
        gtFile     : list of image path and label # JSON 파일 이용하여 먼저 만듬
        checkValid : if true, check the validity of every image
    
                
  4. Validation data 만들기
   # 원본 데이터 중 Train에 사용하지 안은 데이트를  ./validation_result/val_img_ORIGINAL 폴더로 이동
     레이블 데이터 : ./validation_result/val_label_ORIGINAL에 JSON 데이터 저장
      
      위의 2, 3의 과정을 반복
      
      2. 원본 이미지 전처리 : 원본 이미지를 Cropped Image로 만들고, lmdb 생성을 위한 gtFile 생성
            프로그램 : Make_cropped_image_for_train_lmdb.py, 실행 디렉토리 : "/ai_hub_package_for_github/utils
           실행 : 
          
           python Make_cropped_image_for_train_lmdb.py \
                  --image_dir ../validation_result/val_img_ORIGINAL \
                  --label_dir ../validation_result/val_label_ORIGINAL \
                  --output_crop_dir ../validation_result/Validation_Cropped_Images \
                  --output_gt_file ../validation_result/gtFile.txt
           
           # 원본 .jpg 이미지 파일이 들어있는 폴더
           IMAGE_DIR = "../validation_result/val_img_ORIGINAL"
           
           # 원본 .json 라벨 파일이 들어있는 폴더
            LABEL_DIR = "../validation_result/val_label_ORIGINAL"

           # [출력 1] 잘려진(Cropped) 이미지들이 저장될 새 폴더
              OUTPUT_CROP_DIR = "../validation_result/Validation_Cropped_Images"

           # [출력 2] 새로 생성될 라벨 목록 파일 (lmdb_convert.py의 gtFile로 사용)

              OUTPUT_GT_FILE = "../validation_result/gtFile.txt"
              
              
              
       3. lmdb 만드는 과정 : 2번 결과를 이용 lmdb 만듬   
        
       # cropped 이미지와 한글레이블 2개만을 이용하여 lmdb 만듬
       # 실행 디렉토리 : "/ai_hub_package_for_github/utils
       # lmdb 만드는 프로그램:  
       
             python lmdb_convert.py \
              ../validation_result/Validation_Cropped_Images \
              ../validation_result/gtFile.txt \
              ../validation_result/Validation_LMDB/MJ_Validation
              
                  
      
      
                   
  5. 학습실행 : 학습하는 과정
  
  
    실행파일 :  train.py , 실행 디렉토리 ai_hub_package_for_github
             python train.py
    train 데이터 저장소 : './data_set/MJ' 폴더에 lmdb 데이터
   
       
  
   
   6. 학습 실행 결과물 : 베스트 모델 ./saved_models/332b5140-81f7-44ce-a17f-10020db47a8c-Seed1234(예시)  
              # converter.pth는 Inference할 때 결과물에 대한 레이블 찾을 때 사용하기 위함(레이블은 1813개의 개별 글자로 되어 있음)
              
            best 모델을 위해서는 최근의 bestmodel.pth이용 필요,   매번 최근의 학습결과 이용 cfg.py에서 parser.add_argument('--saved_model', default='./saved_models/4027fd29-300d-442e-ac5f-a955af1dafa6-Seed1234/best_accuracy.pth', help="path to model to continue training")
             최신 best 모델 가져오기 위해서는 최신 폴더 지정 필요
              
        
        
        
        
        
        
##################################################################################    
Inference 프로그램 구성 :  학습하고 나서 실제 데이터 적용 위한 


 
1. Polygon 만들기

  실행파일 :  make_Polygon.py 
  실행  :   python make_Polygon.py # 실행디렉토리 : /ai_hub_package_for_github
   테스트할 원본 이미지 저장소 : './CRAFT_Make_Polygon/my_test_images 폴더에 저장
   
   실행 결과물 : 원본 이미지에 polygon 그리고  및 polygon 좌표가 있는  txt 파일 생성
   실행 결과물 저장소 : ./ai_hub_package/CRAFT_Make_Polygon/result 폴더에 있음

> python make_polygon.py
  
  
  
  
  2. Polygon jpg 파일을 polygon별 cropped image 파일 생성하기
  
 # 원본 이미지와 텍스트 파일의 폴리곤 좌표를 읽어   각 단어 영역을 반듯한 직사각형 이미지로 잘라내어 저장합니다.
    
    
  실행파일 :  Crop_Polygon.py 
  실행 : python Crop_Polygons.py # /ai_hub_package_for_github 디렉토리에서 실행
  
   테스트할 원본 이미지 저장소 : "./CRAFT_Make_Polygon/my_test_images" 폴더에 저장(위의 make_Polygon.py 실행 결과)
              폴리곤 정보 저장소 : ./CRAFT_Make_Polygon/result 폴더에 폴리곤 정보 저장(위의 make_Polygon.py 실행 결과)
   
   실행 결과물 : 폴리곤 이미지별 cropped jpg 생성
   실행 결과물 저장소 : "./inference/inference_data/cropped_images_for_inference 폴더에 있음
  
> python make Crop_Polygons.py
  
  
  3. Inference 단계 : cropped Images들을 LMDB로 입력하는 과정과 LMDB 데이터를 이용하여 인식하는 과정으로 구성
    
    
   실행파일 :  Images_to_LMDB_and_Inference.py
   
  실행 python Images_to_LMDB_and_Inference.py # /ai_hub_package_for_github 디렉토리에서 실행
  
             # Images_to_LMDB_and_Inference.py 실행시 이 안에서 create_inference_lmdb 함수 실행시킴
             # create_inference_lmdb 따로 실행 불필요, lmdb에는 이미지와 파일명만 들어가 있음
             
      create_inference_lmdb.py 실행을 위한 원본 데이터 :  './inference/inference_data/cropped_images_for_inference'
             # 이안에 단어별 image 파일 있음 , cropped 파일
             
      lmdb 저장폴더 :        './inference/inference_data 이 안에 data.mdb, lock.mdb 있음
        # create_inference_lmdb 실행전 lmdb 내용 자동으로 다 삭제함
                                  
             
      Inference 입력 데이터 저장소 : './inference/inference_data 이 안에 data.mdb
          
              
      Inference 실행 결과물 : 이미지 파일이름, 레이블, confidence 값
          실행 결과물 저장소 : './inference/inference_result 폴더에 있음

>  python Images_to_LMDB_and_Inference,.py
  
  
  
  


