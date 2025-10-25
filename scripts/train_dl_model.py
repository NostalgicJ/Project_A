#!/usr/bin/env python3
"""
딥러닝 모델 훈련 스크립트
실제 화장품 데이터를 학습하여 위험한 성분 조합을 예측하는 모델
"""
import sys
import os
sys.path.append('src')

import pandas as pd
import numpy as np
from models.advanced_ingredient_analyzer import AdvancedCosmeticAnalyzer
import pickle

def load_real_data():
    """실제 화장품 데이터 로드"""
    print("📊 실제 화장품 데이터 로드 중...")
    
    try:
        # 제품 데이터 로드
        products_df = pd.read_csv('data/processed_cosmetics_final_2.csv')
        print(f"✅ 제품 데이터: {len(products_df)}개")
        
        # 성분 데이터 로드
        ingredients_df = pd.read_csv('data/integrated_product_ingredient_normalized_2.csv')
        print(f"✅ 성분 데이터: {len(ingredients_df)}개")
        
        # 마스터 성분 로드
        master_ingredients = pd.read_csv('data/coos_master_ingredients_cleaned.csv')
        print(f"✅ 마스터 성분: {len(master_ingredients)}개")
        
        return products_df, ingredients_df, master_ingredients
        
    except FileNotFoundError as e:
        print(f"❌ 데이터 파일을 찾을 수 없습니다: {e}")
        return None, None, None

def create_ingredient_vocab(products_df, master_ingredients):
    """성분 어휘 사전 생성"""
    print("🔤 성분 어휘 사전 생성 중...")
    
    all_ingredients = set()
    
    # 제품 데이터에서 성분 추출
    for idx, row in products_df.iterrows():
        if pd.notna(row['성분_문자열']):
            ingredients = [ing.strip() for ing in str(row['성분_문자열']).split(',')]
            all_ingredients.update(ingredients)
    
    # 마스터 성분에서 추출
    for idx, row in master_ingredients.iterrows():
        if pd.notna(row['원료명_정제됨']):
            all_ingredients.add(str(row['원료명_정제됨']).strip())
    
    # 상위 1000개 성분만 사용 (메모리 절약)
    ingredient_counts = {}
    for ingredient in all_ingredients:
        if len(ingredient) > 2:  # 너무 짧은 성분 제외
            ingredient_counts[ingredient] = ingredient_counts.get(ingredient, 0) + 1
    
    # 빈도순으로 정렬하여 상위 1000개 선택
    sorted_ingredients = sorted(ingredient_counts.items(), key=lambda x: x[1], reverse=True)
    top_ingredients = [ing[0] for ing in sorted_ingredients[:1000]]
    
    print(f"✅ 성분 어휘 사전 생성 완료: {len(top_ingredients)}개")
    return top_ingredients

def create_training_data_from_real_data(products_df, vocab):
    """실제 데이터에서 학습 데이터 생성"""
    print("📚 실제 데이터 기반 학습 데이터 생성 중...")
    
    # 제품별 성분 조합 생성
    product_ingredients = {}
    for idx, row in products_df.iterrows():
        # 성분_문자열 컬럼 확인
        if '성분_문자열' in row and pd.notna(row['성분_문자열']):
            ingredients = [ing.strip() for ing in str(row['성분_문자열']).split(',')]
            # 어휘에 있는 성분만 필터링
            filtered_ingredients = [ing for ing in ingredients if ing in vocab]
            if len(filtered_ingredients) > 1:
                product_ingredients[f"{row.get('브랜드명_정리', 'Unknown')}_{row.get('제품명_정리', 'Unknown')}"] = filtered_ingredients
    
    print(f"✅ 제품별 성분 데이터: {len(product_ingredients)}개")
    
    # 만약 실제 데이터가 없다면 가상 데이터 생성
    if len(product_ingredients) == 0:
        print("⚠️ 실제 데이터에서 성분을 찾을 수 없어 가상 데이터를 생성합니다.")
        
        # 주요 성분들로 가상 제품 생성
        main_ingredients = ['비타민C', '레티놀', '히알루론산', '세라마이드', '나이아신아마이드', 
                           'AHA', 'BHA', '판테놀', '아연', '니아신아마이드']
        
        # 가상 제품들 생성
        virtual_products = {
            '제품1': ['비타민C', '히알루론산', '판테놀'],
            '제품2': ['레티놀', '세라마이드', '나이아신아마이드'],
            '제품3': ['비타민C', '레티놀', 'AHA'],  # 위험한 조합
            '제품4': ['히알루론산', '세라마이드', '판테놀'],  # 시너지 조합
            '제품5': ['BHA', '레티놀', '아연'],  # 위험한 조합
            '제품6': ['비타민C', '비타민E', '히알루론산'],  # 시너지 조합
        }
        
        product_ingredients = virtual_products
    
    # 성분 조합 분석을 위한 데이터 생성
    ingredient_pairs = []
    labels = []
    
    # 실제 제품에서 성분 조합 추출
    for product_id, ingredients in product_ingredients.items():
        for i, ing1 in enumerate(ingredients):
            for ing2 in ingredients[i+1:]:
                ingredient_pairs.append((ing1, ing2))
                
                # 라벨 결정 (간단한 규칙 기반)
                if any(keyword in ing1.lower() for keyword in ['비타민c', 'ascorbic', '레티놀', 'retinol']):
                    if any(keyword in ing2.lower() for keyword in ['비타민c', 'ascorbic', '레티놀', 'retinol']):
                        labels.append(2)  # 위험
                    else:
                        labels.append(1)  # 주의
                elif any(keyword in ing2.lower() for keyword in ['비타민c', 'ascorbic', '레티놀', 'retinol']):
                    labels.append(1)  # 주의
                else:
                    labels.append(0)  # 안전
    
    print(f"✅ 학습 데이터 생성 완료: {len(ingredient_pairs)}개 조합")
    return ingredient_pairs, labels

def train_model():
    """딥러닝 모델 훈련"""
    print("🚀 화장품 성분 조합 분석 딥러닝 모델 훈련 시작")
    print("=" * 60)
    
    # 1. 데이터 로드
    products_df, ingredients_df, master_ingredients = load_real_data()
    if products_df is None:
        print("❌ 데이터 로드 실패")
        return
    
    # 2. 성분 어휘 사전 생성
    vocab = create_ingredient_vocab(products_df, master_ingredients)
    
    # 3. 고급 분석기 초기화
    analyzer = AdvancedCosmeticAnalyzer()
    
    # 4. 실제 데이터로 학습 데이터 생성
    ingredient_pairs, labels = create_training_data_from_real_data(products_df, vocab)
    
    # 5. 모델 훈련 (어휘 사전과 학습 데이터를 직접 전달)
    print("\n🧠 딥러닝 모델 훈련 중...")
    model = analyzer.train_model_with_data(vocab, ingredient_pairs, labels, num_epochs=30)
    
    # 6. 모델 저장
    model_path = "models/advanced_ingredient_analyzer.pth"
    os.makedirs("models", exist_ok=True)
    analyzer.save_model(model_path)
    
    # 7. 테스트
    print("\n🧪 모델 테스트")
    test_cases = [
        ['비타민C', '레티놀', '히알루론산'],
        ['비타민C', '비타민E', '히알루론산'],
        ['히알루론산', '세라마이드', '판테놀']
    ]
    
    for i, test_ingredients in enumerate(test_cases, 1):
        print(f"\n테스트 {i}: {test_ingredients}")
        result = analyzer.analyze_combination(test_ingredients)
        print(f"  분류: {result['predicted_class']}")
        print(f"  위험도: {result['danger_score']:.1%}")
        print(f"  시너지: {result['synergy_score']:.1%}")
        print(f"  분석: {result['analysis']}")
    
    print("\n🎉 딥러닝 모델 훈련 완료!")
    print("이제 실제 데이터를 학습한 모델이 위험한 성분 조합을 예측할 수 있습니다!")

if __name__ == "__main__":
    train_model()
