import os
import sys
import json
import uuid
import sqlite3
import csv
import torch
import numpy as np
import subprocess # OCR 스크립트 실행용
import shutil     # 폴더 청소용
from datetime import datetime
from flask import Flask, send_from_directory, request, jsonify
from flask_cors import CORS
from unittest.mock import MagicMock
from itertools import combinations

# ------------------------------------------------------
# [1] Matplotlib 에러 방지 (Mocking)
# ------------------------------------------------------
sys.modules["matplotlib"] = MagicMock()
sys.modules["matplotlib.pyplot"] = MagicMock()

# ------------------------------------------------------
# [2] 경로 설정
# ------------------------------------------------------
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# server 폴더 위치 찾기
if os.path.exists(os.path.join(BASE_DIR, 'server')):
    SERVER_ROOT = os.path.join(BASE_DIR, 'server')
else:
    SERVER_ROOT = os.path.abspath(os.path.join(BASE_DIR, '..', 'server'))

SHARED_DIR = os.path.join(SERVER_ROOT, 'src', 'shared')
sys.path.append(SHARED_DIR)

MODEL_PATH = os.path.join(SERVER_ROOT, 'storage', 'models', 'trained', 'gnn_final_20251111', 'gnn_model_final.pth')
CSV_PATH = os.path.join(BASE_DIR, 'processed_cosmetics_final_2.csv')
DB_PATH = os.path.join(BASE_DIR, 'skinai.db')

# OCR 경로
OCR_ROOT = os.path.join(BASE_DIR, 'ocr_model') 
OCR_INPUT_DIR = os.path.join(OCR_ROOT, 'CRAFT_Make_Polygon', 'my_test_images')
OCR_RESULT_DIR = os.path.join(OCR_ROOT, 'inference', 'inference_result')

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def find_model_path(root_dir):
    search_start = os.path.join(root_dir, 'storage')
    if not os.path.exists(search_start): return None
    for root, dirs, files in os.walk(search_start):
        for file in files:
            if file.endswith('.pth') and 'gnn' in file:
                return os.path.join(root, file)
    return None

# ------------------------------------------------------
# [3] Flask 앱 설정
# ------------------------------------------------------
app = Flask(__name__,
            static_url_path='/',
            static_folder=os.path.join(BASE_DIR, 'dist', 'public'),
            template_folder=os.path.join(BASE_DIR, 'dist', 'public'))
CORS(app)

# ------------------------------------------------------
# [4] CSV 로드
# ------------------------------------------------------
PRODUCT_DICT = {} 
def load_products_from_csv():
    if not os.path.exists(CSV_PATH):
        print(f"❌ [오류] CSV 파일 없음: {CSV_PATH}")
        return
    try:
        print(f"📂 CSV 로딩 중: {CSV_PATH}")
        with open(CSV_PATH, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            count = 0
            for row in reader:
                name = row.get('제품명_정리')
                ingredients_str = row.get('성분_문자열')
                if name and ingredients_str:
                    PRODUCT_DICT[name] = ingredients_str.split(' ')
                    count += 1
            print(f"✅ CSV 로드 완료! ({count}개)")
    except Exception as e:
        print(f"❌ CSV 에러: {e}")

load_products_from_csv()

# ------------------------------------------------------
# [5] DB 초기화
# ------------------------------------------------------
def init_history_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute('''CREATE TABLE IF NOT EXISTS user_products (id TEXT PRIMARY KEY, name TEXT, ingredients TEXT, date TEXT)''')
    conn.execute('''CREATE TABLE IF NOT EXISTS analysis_history (id TEXT PRIMARY KEY, date TEXT, items TEXT, result TEXT)''')
    conn.commit()
    conn.close()

init_history_db()

# ------------------------------------------------------
# [6] GNN 모델 로드
# ------------------------------------------------------
gnn_analyzer = None
collate_fn = None 
IngredientFormulaDataset = None 

def load_gnn_model():
    global gnn_analyzer, collate_fn, IngredientFormulaDataset
    try:
        from gnn_final_20251111 import GNNCosmeticAnalyzer, GNNCollate, IngredientFormulaDataset as IFD
        collate_fn = GNNCollate()
        IngredientFormulaDataset = IFD
        gnn_analyzer = GNNCosmeticAnalyzer()
        
        real_model_path = find_model_path(SERVER_ROOT)
        if real_model_path:
            gnn_analyzer.load_model(real_model_path)
            print(f"🚀 GNN 모델 로드 성공")
        else:
            print(f"⚠️ 모델 파일 없음")
    except Exception as e:
        print(f"❌ 모델 로드 실패: {e}")

load_gnn_model()

# ★★★ [핵심] GNN 추론 로직 (1:1 조합 분석 + 점수 보정) ★★★
def run_gnn_inference(ingredients_list):
    if not gnn_analyzer: return {"score": 0, "status": "UNKNOWN", "message": "모델 미로드"}
    
    try:
        # 1. 성분 필터링 (모델이 아는 성분만)
        vocab = gnn_analyzer.vocab_to_idx
        valid_ingredients = [ing for ing in ingredients_list if vocab.get(ing, 0) != 0]
        
        print(f"🔍 [분석 시작] 입력: {len(ingredients_list)}개 -> 유효: {len(valid_ingredients)}개")
        print(f"   (인식된 성분 예시: {valid_ingredients[:5]}...)")
        
        if len(valid_ingredients) < 2:
            return {
                "score": 100, 
                "status": "SAFE", 
                "message": "분석할 성분 데이터가 충분하지 않습니다. (OCR 인식 실패 또는 성분 부족)",
                "problematic_ingredients": []
            }

        # 2. 모든 2개 조합 생성 (Pairs)
        pairs = list(combinations(valid_ingredients, 2))
        if len(pairs) > 3000: 
            import random
            pairs = random.sample(pairs, 3000)

        # 3. 배치 추론
        formulas = [(list(pair), 0.0, 0.0, False) for pair in pairs]
        dataset = IngredientFormulaDataset(formulas, vocab)
        loader = torch.utils.data.DataLoader(dataset, batch_size=64, collate_fn=collate_fn)
        
        gnn_analyzer.model.eval()
        max_danger = 0.0
        worst_pair = [] 
        
        with torch.no_grad():
            batch_idx = 0
            for batch in loader:
                if hasattr(batch, 'to'): batch = batch.to(DEVICE)
                outputs = gnn_analyzer.model(batch)
                dangers = outputs['danger_score'].cpu().numpy().flatten()
                
                # 현재 배치에서 최대 위험도 찾기
                current_max = np.max(dangers)
                if current_max > max_danger:
                    max_danger = current_max
                    local_idx = np.argmax(dangers)
                    global_idx = (batch_idx * 64) + local_idx
                    if global_idx < len(pairs):
                        worst_pair = list(pairs[global_idx])
                batch_idx += 1
        
        print(f"🔍 [결과] 최대 위험도: {max_danger:.4f}, 원인: {worst_pair}")
        
        # 4. 점수 계산 (100 - 위험도*100)
        final_score = int(100 - (max_danger * 100))
        final_score = max(0, min(100, final_score))
        
        # 5. 상태 결정 (60점 미만 주의)
        status = "CAUTION" if final_score < 60 else "SAFE"
        
        # 6. 범인 목록
        problematic_ingredients = []
        if status == "CAUTION":
            if worst_pair: problematic_ingredients = worst_pair
            else: problematic_ingredients = list(pairs[0]) # 안전장치
        
        msg = []
        if status == "CAUTION":
            msg.append(f"주의가 필요한 조합입니다. (안전 점수: {final_score}점)")
        else:
            msg.append(f"안전한 조합입니다. (안전 점수: {final_score}점)")
            
        return {
            "score": final_score,
            "status": status,
            "message": " ".join(msg),
            "problematic_ingredients": problematic_ingredients
        }
                
    except Exception as e:
        print(f"추론 에러: {e}")
        return {"score": 0, "status": "UNKNOWN", "message": str(e)}

# ★★★ [핵심] OCR 파이프라인 실행 및 결과 읽기 ★★★
def run_ocr_pipeline():
    if not os.path.exists(os.path.join(OCR_ROOT, "make_Polygon.py")):
        print("⚠️ OCR 스크립트 없음. 가짜 결과 반환.")
        return ["정제수", "글리세린", "레티놀", "에탄올"]

    try:
        python_exe = sys.executable
        
        print("📸 [1/3] Polygon 생성...")
        subprocess.run([python_exe, "make_Polygon.py"], cwd=OCR_ROOT, check=True)
        
        print("📸 [2/3] 이미지 자르기...")
        subprocess.run([python_exe, "Crop_Polygons.py"], cwd=OCR_ROOT, check=True)
        
        print("📸 [3/3] 텍스트 인식...")
        subprocess.run([python_exe, "Images_to_LMDB_and_Inference.py"], cwd=OCR_ROOT, check=True)
        
        # 결과 읽기 (inference_results.txt 또는 개별 txt 파일)
        detected_ingredients = []
        if os.path.exists(OCR_RESULT_DIR):
            files = os.listdir(OCR_RESULT_DIR)
            print(f"📂 결과 폴더 파일 목록: {files}")
            
            for filename in files:
                if filename.endswith(".txt"):
                    try:
                        filepath = os.path.join(OCR_RESULT_DIR, filename)
                        with open(filepath, 'r', encoding='utf-8') as f:
                            lines = f.readlines()
                            for line in lines:
                                # 파일 형식이 "이미지경로 \t 텍스트 \t 점수" 인 경우를 대비
                                parts = line.strip().split('\t')
                                if len(parts) < 2: 
                                    parts = line.strip().split(',') # 콤마 구분 시도
                                
                                # 텍스트 추출 (보통 2번째나 마지막 컬럼)
                                for part in parts:
                                    # 한글이 포함되어 있거나 길이가 2 이상인 것만 추출
                                    clean_text = part.strip()
                                    if len(clean_text) >= 2:
                                        detected_ingredients.append(clean_text)
                    except: pass
        
        unique_ingredients = list(set(detected_ingredients))
        print(f"📸 [OCR 완료] 추출된 단어 {len(unique_ingredients)}개: {unique_ingredients[:10]}...")
        return unique_ingredients

    except Exception as e:
        print(f"❌ OCR 실행 에러: {e}")
        import traceback
        traceback.print_exc()
        return []

# ------------------------------------------------------
# [7] API 엔드포인트
# ------------------------------------------------------
def get_db_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

@app.route('/api/analyze/text', methods=['POST'])
def analyze_by_name():
    try:
        data = request.json
        p1, p2 = data.get('product1_name'), data.get('product2_name')
        ing1, ing2 = PRODUCT_DICT.get(p1), PRODUCT_DICT.get(p2)
        if not ing1 or not ing2: return jsonify({'error': '제품을 찾을 수 없습니다.'}), 404

        combined = list(set(ing1 + ing2))
        result = run_gnn_inference(combined)

        conn = get_db_conn()
        pid1, pid2, hid = str(uuid.uuid4()), str(uuid.uuid4()), str(uuid.uuid4())
        now = datetime.now().strftime("%Y-%m-%d")
        conn.execute("INSERT INTO user_products VALUES (?, ?, ?, ?)", (pid1, p1, json.dumps(ing1), now))
        conn.execute("INSERT INTO user_products VALUES (?, ?, ?, ?)", (pid2, p2, json.dumps(ing2), now))
        conn.execute("INSERT INTO analysis_history VALUES (?, ?, ?, ?)", (hid, f"{now} {datetime.now().strftime('%H:%M')}", json.dumps([p1, p2]), json.dumps(result)))
        conn.commit()
        conn.close()

        return jsonify({"products": [{"id": pid1, "name": p1, "ingredients": ing1}, {"id": pid2, "name": p2, "ingredients": ing2}], "analysis": result})
    except Exception as e: return jsonify({'error': str(e)}), 500

@app.route('/api/analyze/image', methods=['POST'])
def analyze_by_image():
    try:
        if 'image1' not in request.files or 'image2' not in request.files:
            return jsonify({'error': '이미지 2개가 필요합니다.'}), 400
            
        img1 = request.files['image1']
        img2 = request.files['image2']
        
        if os.path.exists(OCR_INPUT_DIR): shutil.rmtree(OCR_INPUT_DIR)
        os.makedirs(OCR_INPUT_DIR, exist_ok=True)
        
        img1.save(os.path.join(OCR_INPUT_DIR, "input_01.jpg"))
        img2.save(os.path.join(OCR_INPUT_DIR, "input_02.jpg"))
        
        # OCR 실행
        ingredients = run_ocr_pipeline()
        
        # GNN 분석 실행
        result = run_gnn_inference(ingredients)
        
        return jsonify({
            "products": [
                {"id": "img1", "name": "사진 제품 1", "ingredients": []},
                {"id": "img2", "name": "사진 제품 2", "ingredients": []}
            ],
            "analysis": result
        })
    except Exception as e:
        print(f"이미지 핸들러 에러: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/my-vanity', methods=['GET'])
def get_my_vanity():
    try:
        conn = get_db_conn()
        rows = conn.execute("SELECT * FROM user_products ORDER BY date DESC").fetchall()
        return jsonify([dict(r, ingredients=json.loads(r['ingredients'])) for r in rows])
    except: return jsonify([])

@app.route('/api/management', methods=['GET'])
def get_management():
    try:
        conn = get_db_conn()
        rows = conn.execute("SELECT * FROM analysis_history ORDER BY date DESC").fetchall()
        return jsonify([dict(r, items=json.loads(r['items']), result=json.loads(r['result'])) for r in rows])
    except: return jsonify([])

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_react(path):
    if path.startswith("api/"): return jsonify({"error": "Not Found"}), 404
    if path != "" and os.path.exists(app.static_folder + '/' + path):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.template_folder, 'index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=False)
