import re
import os

def process_inference_file(input_file, output_file):
    """
    inference 결과 파일을 읽어 조건을 적용한 뒤 새로운 파일로 저장합니다.
    """
    try:
        with open(input_file, 'r', encoding='utf-8') as f_in:
            lines = f_in.readlines()
            
        processed_lines = []
        
        # 헤더 처리 (첫 줄)
        if lines:
            header = lines[0].strip().split('\t')
            # 헤더는 그대로 유지하거나 필요에 따라 수정 (여기서는 그대로 저장)
            processed_lines.append('\t'.join(header) + '\n')
            
        # 데이터 처리 (두 번째 줄부터)
        for line in lines[1:]:
            if not line.strip():
                continue
                
            parts = line.strip().split('\t')
            
            # 데이터 형식이 맞지 않는 경우(컬럼 부족 등) 건너뜀
            if len(parts) < 3:
                continue

            image_name = parts[0]
            prediction = parts[1]
            confidence = parts[2]

            # 1. ImageName 컬럼: 끝에 있는 .jpg 삭제
            if image_name.endswith('.jpg'):
                image_name = image_name[:-4]

            # 2. Prediction 컬럼: 시작점과 끝점에서 한글/숫자/영어가 아닌 기호 삭제
            # 정규표현식 설명:
            # ^[^가-힣0-9a-zA-Z]+ : 시작 부분이 한글, 숫자, 영어가 아닌 문자로 시작하면 제거
            # | : 또는
            # [^가-힣0-9a-zA-Z]+$ : 끝 부분이 한글, 숫자, 영어가 아닌 문자로 끝나면 제거
            prediction = re.sub(r'^[^가-힣0-9a-zA-Z]+|[^가-힣0-9a-zA-Z]+$', '', prediction)

            # 3. Confidence 컬럼: 그대로 복제 (수정 없음)

            # 처리된 라인 생성 (탭으로 구분)
            new_line = f"{image_name}\t{prediction}\t{confidence}\n"
            processed_lines.append(new_line)

        # 결과 파일 저장
        with open(output_file, 'w', encoding='utf-8') as f_out:
            f_out.writelines(processed_lines)
            
        print(f"처리가 완료되었습니다. 결과 파일: {output_file}")
        print(f"총 {len(processed_lines)-1}개의 데이터가 처리되었습니다.") # 헤더 제외

    except FileNotFoundError:
        print(f"오류: '{input_file}' 파일을 찾을 수 없습니다.")
    except Exception as e:
        print(f"오류 발생: {e}")

# 프로그램 실행
if __name__ == "__main__":
    input_filename = './inference/inference_result/inference_results.txt'
    output_filename = './inference/inference_result/Inference_result_New.txt'
    
    process_inference_file(input_filename, output_filename)
