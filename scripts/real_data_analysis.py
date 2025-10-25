#!/usr/bin/env python3
"""
실제 데이터에서 구달 청귤 비타민C 세럼 성분 정보를 찾아서 분석
"""
import sys
import os
sys.path.append('src')

import pandas as pd
from models.ingredient_analyzer import CosmeticIngredientAnalyzer

def find_real_products():
    print("🔍 실제 데이터에서 구달 청귤 비타민C 세럼 찾기")
    print("=" * 60)
    
    # 데이터 로드
    try:
        products_df = pd.read_csv('data/processed_cosmetics_final_2.csv')
        print(f"✅ 제품 데이터 로드 완료: {len(products_df)}개 제품")
        
        # 구달 청귤 비타민C 세럼 찾기
        gudal_products = products_df[
            (products_df['브랜드명_정리'].str.contains('구달', na=False)) &
            (products_df['제품명_정리'].str.contains('청귤', na=False))
        ]
        
        print(f"\n📊 구달 청귤 관련 제품: {len(gudal_products)}개")
        
        if len(gudal_products) > 0:
            for idx, row in gudal_products.iterrows():
                print(f"\n🔍 제품 {idx+1}:")
                print(f"  브랜드: {row['브랜드명_정리']}")
                print(f"  제품명: {row['제품명_정리']}")
                print(f"  카테고리: {row['카테고리']}")
                
                # 성분 정보
                if pd.notna(row['성분_문자열']):
                    ingredients = [ing.strip() for ing in str(row['성분_문자열']).split(',')]
                    print(f"  성분 수: {len(ingredients)}개")
                    print(f"  주요 성분: {', '.join(ingredients[:10])}...")
                    
                    # 실제 성분으로 분석
                    analyze_real_ingredients(ingredients, row['제품명_정리'])
                else:
                    print("  성분 정보 없음")
        else:
            print("❌ 구달 청귤 제품을 찾을 수 없습니다.")
            print("📋 사용 가능한 구달 제품들:")
            gudal_all = products_df[products_df['브랜드명_정리'].str.contains('구달', na=False)]
            for idx, row in gudal_all.head(5).iterrows():
                print(f"  - {row['제품명_정리']}")
                
    except FileNotFoundError:
        print("❌ 데이터 파일을 찾을 수 없습니다.")
        print("📁 데이터 파일 위치 확인 중...")
        print(f"현재 디렉토리: {os.getcwd()}")
        print(f"data 폴더 내용: {os.listdir('data') if os.path.exists('data') else 'data 폴더 없음'}")

def analyze_real_ingredients(ingredients, product_name):
    """실제 성분으로 분석"""
    print(f"\n🧪 {product_name} 성분 조합 분석")
    print("-" * 40)
    
    # 분석기 초기화
    analyzer = CosmeticIngredientAnalyzer()
    
    # 성분 조합 분석
    result = analyzer.analyze_combination(ingredients)
    
    print(f"📊 분석 결과:")
    print(f"  🎯 분류: {result['predicted_class']}")
    print(f"  📈 신뢰도: {result['confidence']:.1%}")
    print(f"  🛡️ 안전성 점수: {result['safety_score']:.1%}")
    print(f"  ✨ 시너지 점수: {result['synergy_score']:.1%}")
    print(f"  💡 분석: {result['analysis']}")
    
    # 안전성 이슈 표시
    if 'safety_issues' in result and result['safety_issues']:
        print(f"\n⚠️ 주의사항:")
        for issue in result['safety_issues']:
            print(f"    • {issue}")
    
    # 시너지 효과 표시
    if 'synergy_benefits' in result and result['synergy_benefits']:
        print(f"\n✨ 시너지 효과:")
        for benefit in result['synergy_benefits']:
            print(f"    • {benefit}")
    
    # 주요 성분 분석
    print(f"\n🔬 주요 성분 분석:")
    key_ingredients = ['비타민C', '레티놀', '히알루론산', '세라마이드', '나이아신아마이드']
    found_ingredients = [key for key in key_ingredients if any(key in ing for ing in ingredients)]
    
    if found_ingredients:
        print(f"  발견된 주요 성분: {', '.join(found_ingredients)}")
    else:
        print(f"  주요 성분을 찾을 수 없습니다.")
        print(f"  실제 성분 샘플: {', '.join(ingredients[:5])}")

def find_retinol_products():
    """레티놀 제품 찾기"""
    print(f"\n🔍 레티놀 제품 찾기")
    print("=" * 40)
    
    try:
        products_df = pd.read_csv('data/processed_cosmetics_final_2.csv')
        
        # 레티놀 관련 제품 찾기
        retinol_products = products_df[
            products_df['성분_문자열'].str.contains('레티놀', na=False)
        ]
        
        print(f"📊 레티놀 제품: {len(retinol_products)}개")
        
        if len(retinol_products) > 0:
            print(f"\n🔍 레티놀 제품 샘플:")
            for idx, row in retinol_products.head(3).iterrows():
                print(f"  {idx+1}. {row['브랜드명_정리']} - {row['제품명_정리']}")
                if pd.notna(row['성분_문자열']):
                    ingredients = [ing.strip() for ing in str(row['성분_문자열']).split(',')]
                    print(f"     성분 수: {len(ingredients)}개")
                    print(f"     주요 성분: {', '.join(ingredients[:5])}...")
        else:
            print("❌ 레티놀 제품을 찾을 수 없습니다.")
            
    except FileNotFoundError:
        print("❌ 데이터 파일을 찾을 수 없습니다.")

if __name__ == "__main__":
    find_real_products()
    find_retinol_products()
