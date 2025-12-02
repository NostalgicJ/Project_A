"""
실제 제품 데이터를 사용한 데이터 로더 및 분할 스크립트
훈련/테스트/검증 세트를 8:1:1 비율로 분할
"""
import pandas as pd
import numpy as np
import pickle
import json
from typing import List, Dict, Tuple, Set
from pathlib import Path
import random
from collections import Counter
from datetime import datetime

class RealDataLoader:
    """실제 제품 데이터 로더"""
    
    def __init__(self, 
                 products_file: str = "data/processed/processed_cosmetics_final_2.csv",
                 ingredients_file: str = "data/processed/integrated_product_ingredient_normalized_2.csv",
                 master_ingredients_file: str = "data/processed/coos_master_ingredients_cleaned.csv",
                 public_ingredients_file: str = "data/raw/public_ingredients.json"): # [수정] 공용 성분 파일 경로 추가
        self.products_file = products_file
        self.ingredients_file = ingredients_file
        self.master_ingredients_file = master_ingredients_file
        self.public_ingredients_file = public_ingredients_file # [수정]
        
        self.products_df = None
        self.ingredients_df = None
        self.master_ingredients_df = None
        self.public_ingredients_list = None # [수정] JSON 데이터 저장용
        
        self.vocab = None
        self.vocab_to_idx = None
        self.idx_to_vocab = None
        
    def load_data(self):
        """데이터 로드"""
        print("📊 데이터 로드 중...")
        
        # 제품 데이터 로드
        self.products_df = pd.read_csv(self.products_file)
        print(f"✅ 제품 데이터: {len(self.products_df)}개")
        
        # 성분 데이터 로드
        if Path(self.ingredients_file).exists():
            self.ingredients_df = pd.read_csv(self.ingredients_file)
            print(f"✅ 성분 데이터: {len(self.ingredients_df)}개")
        
        # 마스터 성분 로드 (참고용으로 로드는 유지)
        if Path(self.master_ingredients_file).exists():
            self.master_ingredients_df = pd.read_csv(self.master_ingredients_file)
            print(f"✅ 마스터 성분: {len(self.master_ingredients_df)}개")

        # --- [수정] ---
        # 공용 성분 사전 로드 (JSON)
        if Path(self.public_ingredients_file).exists():
            try:
                with open(self.public_ingredients_file, 'r', encoding='utf-8') as f:
                    self.public_ingredients_list = json.load(f)
                print(f"✅ 공용 성분 사전: {len(self.public_ingredients_list)}개")
            except json.JSONDecodeError:
                print(f"⚠️ 공용 성분 사전 파일({self.public_ingredients_file})이 올바른 JSON 형식이 아닙니다.")
                self.public_ingredients_list = []
            except Exception as e:
                print(f"⚠️ 공용 성분 사전 파일 로드 중 오류: {e}")
                self.public_ingredients_list = []
        else:
            print(f"⚠️ 공용 성분 사전 파일을 찾을 수 없습니다: {self.public_ingredients_file}")
            self.public_ingredients_list = []
        # --- [수정 완료] ---
        
        return self
    
    # --- [수정] ---
    # 어휘 사전 구축 로직을 public_ingredients.json 기준으로 변경
    def build_vocabulary(self, min_freq: int = 1) -> List[str]:
        """공용 성분 사전(JSON)을 기반으로 어휘 사전 구축"""
        print("🔤 성분 어휘 사전 구축 중 (공용 성분 사전 기준)...")
        
        ingredient_counts = Counter()
        
        if self.public_ingredients_list is None or len(self.public_ingredients_list) == 0:
            print("⚠️ 'public_ingredients.json'이 로드되지 않았거나 비어있습니다.")
        else:
            # public_ingredients.json의 '한글명'을 어휘 사전으로 사용
            for item in self.public_ingredients_list:
                ing = item.get('한글명')
                if ing:
                    ing = str(ing).strip()
                    if len(ing) > 1:
                        ingredient_counts[ing] += 1 # 1회 카운트
        
        # 최소 빈도 필터링 (사실상 min_freq=1이면 모든 성분 포함)
        filtered_ingredients = [
            ing for ing, count in ingredient_counts.items() 
            if count >= min_freq and len(ing) > 1
        ]
        
        # 빈도순 정렬
        filtered_ingredients.sort(key=lambda x: ingredient_counts[x], reverse=True)
        
        # <UNK> 토큰 추가
        self.vocab = ['<UNK>'] + filtered_ingredients
        self.vocab_to_idx = {ing: idx for idx, ing in enumerate(self.vocab)}
        self.idx_to_vocab = {idx: ing for ing, idx in self.vocab_to_idx.items()}
        
        print(f"✅ 어휘 사전 구축 완료: {len(self.vocab)}개 성분")
        print(f"   - <UNK> 토큰: 1개")
        print(f"   - 실제 성분: {len(filtered_ingredients)}개")
        print(f"   - 최소 빈도: {min_freq}회 이상 (JSON 기준)")
        
        return self.vocab
    # --- [수정 완료] ---
    
    # --- [수정] ---
    # ingredients_df의 실제 컬럼명을 사용하도록 수정
    def extract_product_formulas(self) -> List[Tuple[List[str], str]]:
        """제품별 성분 포뮬러 추출 (정제된 'ingredients_df' 기준)"""
        print("📝 제품별 성분 포뮬러 추출 중...")
        
        if self.ingredients_df is None:
            print("⚠️ 'ingredients_df'가 로드되지 않았습니다.")
            return []

        # [수정] 'ingredients_df'의 실제 컬럼명으로 변경
        # '사용_원료명'이 'public_ingredients.json'의 '한글명'과 일치한다고 가정
        product_id_cols = ['브랜드명', '제품명'] 
        ingredient_col = '사용_원료명' # [수정] COOS_원료명 -> 사용_원료명
        
        required_cols = product_id_cols + [ingredient_col]
        
        if not all(col in self.ingredients_df.columns for col in required_cols):
            print(f"⚠️ 'ingredients_df'에 {required_cols} 컬럼이 모두 필요합니다. 컬럼명을 확인해주세요.")
            print(f"   (현재 컬럼: {self.ingredients_df.columns.tolist()})")
            return []

        # 메모리 효율을 위해 필요한 컬럼만 선택
        df_subset = self.ingredients_df[required_cols].dropna()

        # 어휘 사전에 있는 유효한 성분만 필터링
        print("   - 어휘 사전 기준으로 성분 필터링 중...")
        
        # 어휘 사전(self.vocab_to_idx)이 Set이면 더 빠릅니다.
        vocab_set = set(self.vocab_to_idx.keys())
        
        valid_ingredients_mask = df_subset[ingredient_col].apply(
            lambda ing: str(ing).strip() in vocab_set and len(str(ing).strip()) > 1
        )
        df_filtered = df_subset[valid_ingredients_mask].copy() # SettingWithCopyWarning 방지

        # 제품 ID별로 성분 그룹핑
        print("   - 제품별 성분 그룹핑 중...")
        
        def create_product_id(row):
            # 원본 코드의 ID 생성 방식
            return f"{row.get(product_id_cols[0], 'Unknown')}_{row.get(product_id_cols[1], 'Unknown')}"

        # 고유 ID 생성
        df_filtered['product_id'] = df_filtered.apply(create_product_id, axis=1)

        # 그룹핑하여 set으로 만든 후 list로 변환 (중복 성분 제거)
        grouped = df_filtered.groupby('product_id')[ingredient_col].apply(set).apply(list)
        
        # 2개 이상의 성분을 가진 제품만 포뮬러로 인정 (쌍을 만들어야 하므로)
        formulas = [
            (ingredients, product_id) 
            for product_id, ingredients in grouped.items()
            if len(ingredients) > 1
        ]
        
        print(f"✅ 성분 포뮬러 추출 완료: {len(formulas)}개 제품 (유효 성분 2개 이상)")
        return formulas
    # --- [수정 완료] ---

    def create_ingredient_pairs(self, formulas: List[Tuple[List[str], str]]) -> List[Tuple[str, str, float, float]]:
        """성분 쌍 데이터 생성 (위험도 및 시너지 라벨 포함)"""
        print("🔗 성분 쌍 데이터 생성 중...")
        
        # 규칙 파일 로드
        rules_file = Path("config/ingredient_rules.json")
        dangerous_combinations = {}
        synergy_combinations = {}
        
        if rules_file.exists():
            try:
                with open(rules_file, 'r', encoding='utf-8') as f:
                    rules = json.load(f)
            except Exception as e:
                print(f"⚠️ 'ingredient_rules.json' 파일 로드 중 오류: {e}")
                rules = {}
            
            # 성분 계열 매핑
            ingredient_families = rules.get('ingredient_families', {})
            vocab_set = set(self.vocab_to_idx.keys()) # 빠른 조회를 위해 Set 사용

            # 위험한 조합
            for combo in rules.get('dangerous_combinations', []):
                family1 = combo.get('family1')
                family2 = combo.get('family2')
                danger_level = combo.get('danger_level', 0.5)
                
                if not family1 or not family2: continue
                
                ing1_list_from_family = ingredient_families.get(family1, [])
                ing2_list_from_family = ingredient_families.get(family2, [])

                for ing1 in ing1_list_from_family:
                    if ing1 in vocab_set:
                        for ing2 in ing2_list_from_family:
                            if ing2 in vocab_set:
                                pair = tuple(sorted([ing1, ing2]))
                                dangerous_combinations[pair] = danger_level
            
            # 시너지 조합
            for combo in rules.get('synergy_combinations', []):
                family1 = combo.get('family1')
                family2 = combo.get('family2')
                synergy_level = combo.get('synergy_level', 0.5)

                if not family1 or not family2: continue

                ing1_list_from_family = ingredient_families.get(family1, [])
                ing2_list_from_family = ingredient_families.get(family2, [])

                for ing1 in ing1_list_from_family:
                    if ing1 in vocab_set:
                        for ing2 in ing2_list_from_family:
                            if ing2 in vocab_set:
                                pair = tuple(sorted([ing1, ing2]))
                                synergy_combinations[pair] = synergy_level
        
        else:
            print("⚠️ 'config/ingredient_rules.json' 파일을 찾을 수 없습니다. 위험/시너지 라벨이 0.0으로 설정됩니다.")

        pairs = []
        
        for formula, product_id in formulas:
            for i, ing1 in enumerate(formula):
                for ing2 in formula[i+1:]:
                    pair = tuple(sorted([ing1, ing2]))
                    
                    # 위험도 및 시너지 라벨 (ingredient_rules.json에 있는 것만 정답 레이블)
                    danger = dangerous_combinations.get(pair, 0.0)
                    synergy = synergy_combinations.get(pair, 0.0)
                    
                    # 라벨이 있는지 확인 (위험 또는 시너지가 0보다 크면 라벨 있음)
                    has_label = (danger > 0.0) or (synergy > 0.0)
                    
                    # 라벨이 없는 경우 -1로 표시 (미확인 상태)
                    if not has_label:
                        danger = -1.0  # 라벨 없음 표시
                        synergy = -1.0  # 라벨 없음 표시
                    
                    pairs.append((ing1, ing2, danger, synergy))
        
        labeled_count = sum(1 for _, _, d, s in pairs if d > 0 or s > 0)
        unlabeled_count = len(pairs) - labeled_count
        
        print(f"✅ 성분 쌍 데이터 생성 완료: {len(pairs)}개 쌍")
        print(f"   - 라벨이 있는 조합: {labeled_count}개 (정답 레이블)")
        print(f"     * 위험한 조합: {sum(1 for _, _, d, _ in pairs if d > 0)}개")
        print(f"     * 시너지 조합: {sum(1 for _, _, _, s in pairs if s > 0)}개")
        print(f"   - 라벨이 없는 조합: {unlabeled_count}개 (미확인 상태)")
        print(f"   ⚠️  라벨이 없는 조합은 학습 시 손실 계산에서 제외됩니다.")
        
        return pairs
    
    def split_data(self, 
                   pairs: List[Tuple[str, str, float, float]],
                   train_ratio: float = 0.8,
                   test_ratio: float = 0.1,
                   val_ratio: float = 0.1,
                   random_seed: int = 42) -> Tuple[List, List, List]:
        """데이터 분할 (8:1:1)"""
        print(f"📊 데이터 분할 중... (훈련:{train_ratio}, 테스트:{test_ratio}, 검증:{val_ratio})")
        
        total = len(pairs)
        if total == 0:
            print("⚠️ 분할할 데이터(쌍)가 없습니다. 빈 리스트를 반환합니다.")
            print(f"✅ 데이터 분할 완료:")
            print(f"   - 훈련 세트: 0개 (0.0%)")
            print(f"   - 테스트 세트: 0개 (0.0%)")
            print(f"   - 검증 세트: 0개 (0.0%)")
            return [], [], []
        
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        # 셔플
        shuffled = pairs.copy()
        random.shuffle(shuffled)
        
        # 분할
        train_end = int(total * train_ratio)
        test_end = train_end + int(total * test_ratio)
        
        train_data = shuffled[:train_end]
        test_data = shuffled[train_end:test_end]
        val_data = shuffled[test_end:]
        
        print(f"✅ 데이터 분할 완료:")
        print(f"   - 훈련 세트: {len(train_data)}개 ({len(train_data)/total:.1%})")
        print(f"   - 테스트 세트: {len(test_data)}개 ({len(test_data)/total:.1%})")
        print(f"   - 검증 세트: {len(val_data)}개 ({len(val_data)/total:.1%})")
        
        return train_data, test_data, val_data
    
    def save_data_splits(self, 
                        train_data: List,
                        test_data: List,
                        val_data: List,
                        output_dir: str = "data/splits"):
        """데이터 분할 저장"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 데이터 저장
        with open(output_path / f"train_data_{timestamp}.pkl", 'wb') as f:
            pickle.dump(train_data, f)
        
        with open(output_path / f"test_data_{timestamp}.pkl", 'wb') as f:
            pickle.dump(test_data, f)
        
        with open(output_path / f"val_data_{timestamp}.pkl", 'wb') as f:
            pickle.dump(val_data, f)
        
        # 어휘 사전 저장
        with open(output_path / f"vocab_{timestamp}.pkl", 'wb') as f:
            pickle.dump({
                'vocab': self.vocab,
                'vocab_to_idx': self.vocab_to_idx,
                'idx_to_vocab': self.idx_to_vocab
            }, f)
        
        # 메타데이터 저장
        metadata = {
            'timestamp': timestamp,
            'vocab_size': len(self.vocab),
            'train_size': len(train_data),
            'test_size': len(test_data),
            'val_size': len(val_data),
            'total_pairs': len(train_data) + len(test_data) + len(val_data)
        }
        
        with open(output_path / f"metadata_{timestamp}.json", 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 데이터 분할 저장 완료: {output_dir}")
        print(f"   - 타임스탬프: {timestamp}")
        
        return timestamp


if __name__ == "__main__":
    # 데이터 로드 및 분할 실행ㅈ
    loader = RealDataLoader(
        # 필요한 경우 파일 경로를 여기서 직접 지정
        # products_file="...",
        # ingredients_file="...",
        public_ingredients_file="data/raw/public_ingredients.json"
    )
    loader.load_data()
    vocab = loader.build_vocabulary(min_freq=1)
    
    formulas = loader.extract_product_formulas()
    pairs = loader.create_ingredient_pairs(formulas)
    
    train_data, test_data, val_data = loader.split_data(pairs, train_ratio=0.8, test_ratio=0.1, val_ratio=0.1)
    
    if len(train_data) > 0:
        timestamp = loader.save_data_splits(train_data, test_data, val_data)
        print(f"\n🎉 데이터 준비 완료! 타임스탬프: {timestamp}")
    else:
        print("\n⚠️ 생성된 데이터가 없어 파일 저장을 스킵합니다.")
