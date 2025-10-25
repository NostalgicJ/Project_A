"""
화장품 성분 데이터 전처리 모듈
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import re
from collections import Counter
import pickle
import os


class CosmeticDataProcessor:
    """화장품 데이터 전처리 클래스"""
    
    def __init__(self, data_dir: str = "data/processed"):
        self.data_dir = data_dir
        self.raw_data_dir = "data/raw"
        self.products_df = None
        self.ingredients_df = None
        self.master_ingredients = None
        self.ingredient_vocab = None
        self.ingredient_embeddings = None
        
    def load_data(self) -> None:
        """데이터 로드"""
        print("📊 데이터 로딩 중...")
        
        # 화장품 제품 데이터 로드
        self.products_df = pd.read_csv(f"{self.raw_data_dir}/processed_cosmetics_final_2.csv")
        print(f"✅ 제품 데이터 로드 완료: {len(self.products_df)}개 제품")
        
        # 성분 정규화 데이터 로드
        self.ingredients_df = pd.read_csv(f"{self.raw_data_dir}/integrated_product_ingredient_normalized_2.csv")
        print(f"✅ 성분 데이터 로드 완료: {len(self.ingredients_df)}개 성분 레코드")
        
        # 마스터 성분 데이터 로드
        self.master_ingredients = pd.read_csv(f"{self.raw_data_dir}/coos_master_ingredients_cleaned.csv")
        print(f"✅ 마스터 성분 데이터 로드 완료: {len(self.master_ingredients)}개 성분")
        
    def clean_ingredient_names(self, ingredient_text: str) -> List[str]:
        """성분명 정제"""
        if pd.isna(ingredient_text):
            return []
            
        # 쉼표로 분리하고 각 성분 정제
        ingredients = [ing.strip() for ing in str(ingredient_text).split(',')]
        cleaned_ingredients = []
        
        for ingredient in ingredients:
            if ingredient and len(ingredient) > 1:  # 빈 문자열이나 단일 문자 제외
                # 특수문자 제거 및 정규화
                cleaned = re.sub(r'[^\w가-힣]', '', ingredient)
                if cleaned:
                    cleaned_ingredients.append(cleaned)
                    
        return cleaned_ingredients
    
    def build_ingredient_vocabulary(self) -> None:
        """성분 어휘 사전 구축"""
        print("🔤 성분 어휘 사전 구축 중...")
        
        all_ingredients = set()
        
        # 제품 데이터에서 성분 추출
        for idx, row in self.products_df.iterrows():
            ingredients = self.clean_ingredient_names(row['성분_문자열'])
            all_ingredients.update(ingredients)
        
        # 정규화된 성분 데이터에서 추출
        for idx, row in self.ingredients_df.iterrows():
            if pd.notna(row['사용_원료명']):
                cleaned = re.sub(r'[^\w가-힣]', '', str(row['사용_원료명']))
                if cleaned and len(cleaned) > 1:
                    all_ingredients.add(cleaned)
        
        # 마스터 성분에서 추출
        for idx, row in self.master_ingredients.iterrows():
            if pd.notna(row['원료명_정제됨']):
                cleaned = re.sub(r'[^\w가-힣]', '', str(row['원료명_정제됨']))
                if cleaned and len(cleaned) > 1:
                    all_ingredients.add(cleaned)
        
        self.ingredient_vocab = sorted(list(all_ingredients))
        print(f"✅ 성분 어휘 사전 구축 완료: {len(self.ingredient_vocab)}개 성분")
        
        # 어휘 사전 저장
        with open(f"{self.data_dir}/ingredient_vocab.pkl", "wb") as f:
            pickle.dump(self.ingredient_vocab, f)
    
    def create_ingredient_matrix(self) -> pd.DataFrame:
        """성분-제품 매트릭스 생성"""
        print("📊 성분-제품 매트릭스 생성 중...")
        
        # 제품별 성분 리스트 생성
        product_ingredients = {}
        
        for idx, row in self.products_df.iterrows():
            product_id = f"{row['브랜드명_정리']}_{row['제품명_정리']}"
            ingredients = self.clean_ingredient_names(row['성분_문자열'])
            product_ingredients[product_id] = ingredients
        
        # 매트릭스 생성
        ingredient_matrix = pd.DataFrame(0, 
                                      index=list(product_ingredients.keys()),
                                      columns=self.ingredient_vocab)
        
        for product_id, ingredients in product_ingredients.items():
            for ingredient in ingredients:
                if ingredient in self.ingredient_vocab:
                    ingredient_matrix.loc[product_id, ingredient] = 1
        
        print(f"✅ 성분-제품 매트릭스 생성 완료: {ingredient_matrix.shape}")
        return ingredient_matrix
    
    def analyze_ingredient_combinations(self, ingredient_matrix: pd.DataFrame) -> Dict:
        """성분 조합 분석"""
        print("🔍 성분 조합 분석 중...")
        
        # 자주 함께 사용되는 성분 조합 찾기
        ingredient_pairs = {}
        ingredient_frequency = {}
        
        for product_id, row in ingredient_matrix.iterrows():
            # 해당 제품에 포함된 성분들
            present_ingredients = row[row == 1].index.tolist()
            
            # 성분 빈도 계산
            for ingredient in present_ingredients:
                ingredient_frequency[ingredient] = ingredient_frequency.get(ingredient, 0) + 1
            
            # 성분 쌍 계산
            for i, ing1 in enumerate(present_ingredients):
                for ing2 in present_ingredients[i+1:]:
                    pair = tuple(sorted([ing1, ing2]))
                    ingredient_pairs[pair] = ingredient_pairs.get(pair, 0) + 1
        
        # 상위 조합 추출
        top_pairs = sorted(ingredient_pairs.items(), key=lambda x: x[1], reverse=True)[:100]
        top_ingredients = sorted(ingredient_frequency.items(), key=lambda x: x[1], reverse=True)[:50]
        
        analysis_result = {
            'top_ingredient_pairs': top_pairs,
            'top_ingredients': top_ingredients,
            'total_combinations': len(ingredient_pairs),
            'total_ingredients': len(ingredient_frequency)
        }
        
        print(f"✅ 성분 조합 분석 완료: {len(ingredient_pairs)}개 조합")
        return analysis_result
    
    def create_ingredient_embeddings(self, ingredient_matrix: pd.DataFrame) -> np.ndarray:
        """성분 임베딩 생성 (간단한 TF-IDF 기반)"""
        print("🧠 성분 임베딩 생성 중...")
        
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD
        
        # 제품별 성분을 텍스트로 변환
        product_texts = []
        for product_id, row in ingredient_matrix.iterrows():
            ingredients = row[row == 1].index.tolist()
            product_texts.append(' '.join(ingredients))
        
        # TF-IDF 벡터화
        tfidf = TfidfVectorizer(max_features=1000)
        tfidf_matrix = tfidf.fit_transform(product_texts)
        
        # 차원 축소 (SVD)
        svd = TruncatedSVD(n_components=128)
        embeddings = svd.fit_transform(tfidf_matrix)
        
        self.ingredient_embeddings = embeddings
        print(f"✅ 성분 임베딩 생성 완료: {embeddings.shape}")
        
        return embeddings
    
    def save_processed_data(self, ingredient_matrix: pd.DataFrame, 
                          analysis_result: Dict, embeddings: np.ndarray) -> None:
        """전처리된 데이터 저장"""
        print("💾 전처리된 데이터 저장 중...")
        
        # 매트릭스 저장
        ingredient_matrix.to_csv(f"{self.data_dir}/ingredient_matrix.csv")
        
        # 분석 결과 저장
        with open(f"{self.data_dir}/ingredient_analysis.pkl", "wb") as f:
            pickle.dump(analysis_result, f)
        
        # 임베딩 저장
        np.save(f"{self.data_dir}/ingredient_embeddings.npy", embeddings)
        
        print("✅ 전처리된 데이터 저장 완료")
    
    def process_all(self) -> Tuple[pd.DataFrame, Dict, np.ndarray]:
        """전체 데이터 전처리 파이프라인 실행"""
        print("🚀 화장품 데이터 전처리 시작...")
        
        # 1. 데이터 로드
        self.load_data()
        
        # 2. 어휘 사전 구축
        self.build_ingredient_vocabulary()
        
        # 3. 성분-제품 매트릭스 생성
        ingredient_matrix = self.create_ingredient_matrix()
        
        # 4. 성분 조합 분석
        analysis_result = self.analyze_ingredient_combinations(ingredient_matrix)
        
        # 5. 성분 임베딩 생성
        embeddings = self.create_ingredient_embeddings(ingredient_matrix)
        
        # 6. 데이터 저장
        self.save_processed_data(ingredient_matrix, analysis_result, embeddings)
        
        print("🎉 데이터 전처리 완료!")
        return ingredient_matrix, analysis_result, embeddings


if __name__ == "__main__":
    # 데이터 전처리 실행
    processor = CosmeticDataProcessor()
    ingredient_matrix, analysis_result, embeddings = processor.process_all()
    
    # 결과 요약 출력
    print("\n📊 처리 결과 요약:")
    print(f"- 총 제품 수: {len(processor.products_df)}")
    print(f"- 총 성분 수: {len(processor.ingredient_vocab)}")
    print(f"- 매트릭스 크기: {ingredient_matrix.shape}")
    print(f"- 임베딩 크기: {embeddings.shape}")
    print(f"- 상위 성분 조합: {len(analysis_result['top_ingredient_pairs'])}")

