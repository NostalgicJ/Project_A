#!/usr/bin/env python3
"""
제품 조합 테스트 스크립트

사용 방법:
    python scripts/test_product_combination.py

기능:
    - 제품명 입력
    - 제품 검색 및 성분 추출
    - 성분 조합 분석
    - 결과 출력
"""

import sys
sys.path.append('../src')

import pandas as pd
import json
from pathlib import Path
import os

class ProductCombinationTester:
    """제품 조합 테스트 클래스"""
    
    def __init__(self):
        self.products_file = "data/processed/oliveyoung_products_cleaned.csv"
        self.products_df = None
        self.load_products()
        
        # 규칙 기반 분석기를 사용
        sys.path.append('src')
        from models.product_combination_analyzer import ProductIngredientMatcher
        self.analyzer = ProductIngredientMatcher()
    
    def load_products(self):
        """제품 데이터 로드"""
        try:
            self.products_df = pd.read_csv(self.products_file)
            print(f"✅ 제품 데이터 로드: {len(self.products_df)}개")
        except FileNotFoundError:
            print(f"❌ 제품 데이터 파일을 찾을 수 없습니다: {self.products_file}")
            print("먼저 전처리를 실행하세요: python scripts/preprocess_oliveyoung_data.py")
            sys.exit(1)
    
    def search_product(self, query):
        """제품 검색"""
        # 브랜드명과 제품명으로 검색
        query_lower = query.lower()
        
        # 정확히 일치하는 제품
        exact_match = self.products_df[
            (self.products_df['brand'].str.lower() == query_lower) |
            (self.products_df['product_name'].str.lower() == query_lower) |
            (self.products_df['product_name'].str.lower().str.contains(query_lower))
        ]
        
        if len(exact_match) > 0:
            return exact_match
        
        # 부분 일치
        partial_match = self.products_df[
            (self.products_df['brand'].str.contains(query, case=False)) |
            (self.products_df['product_name'].str.contains(query, case=False))
        ]
        
        return partial_match
    
    def display_product_info(self, product):
        """제품 정보 표시"""
        print("\n" + "="*60)
        print("제품 정보")
        print("="*60)
        print(f"브랜드: {product['brand']}")
        print(f"제품명: {product['product_name']}")
        print(f"카테고리: {product['category']}")
        print(f"URL: {product['url']}")
        
        # 성분 정보
        ingredients = product['all_ingredients']
        if pd.notna(ingredients):
            ing_list = ingredients.split(',')
            print(f"\n성분 수: {len(ing_list)}개")
            print(f"성분: {', '.join(ing_list[:10])}")
            if len(ing_list) > 10:
                print(f"... 외 {len(ing_list)-10}개")
    
    def parse_ingredients(self, ingredients_str):
        """성분 문자열을 리스트로 변환"""
        if pd.isna(ingredients_str):
            return []
        return [ing.strip() for ing in str(ingredients_str).split(',')]
    
    def analyze_combination(self, product_a_ingredients, product_b_ingredients):
        """두 제품의 성분 조합 분석"""
        print("\n" + "="*60)
        print("성분 조합 분석")
        print("="*60)
        
        result = self.analyzer.analyze_product_pair(
            product_a_ingredients,
            product_b_ingredients
        )
        
        # 결과 출력
        print(self.analyzer.format_analysis_result(result))
        
        return result
    
    def interactive_test(self):
        """대화형 테스트"""
        print("\n" + "="*60)
        print("제품 조합 테스트")
        print("="*60)
        print("\n사용 방법:")
        print("1. 첫 번째 제품명 입력")
        print("2. 검색 결과에서 선택")
        print("3. 두 번째 제품명 입력")
        print("4. 검색 결과에서 선택")
        print("5. 조합 분석 결과 확인")
        
        # 첫 번째 제품
        print("\n" + "-"*60)
        print("[1단계] 첫 번째 제품 검색")
        query_a = input("\n제품명 또는 브랜드명을 입력하세요: ").strip()
        
        results_a = self.search_product(query_a)
        
        if len(results_a) == 0:
            print("❌ 검색 결과가 없습니다.")
            return
        
        print(f"\n검색 결과: {len(results_a)}개")
        for idx, (_, product) in enumerate(results_a.head(10).iterrows(), 1):
            print(f"\n[{idx}] {product['brand']} {product['product_name']}")
            print(f"    카테고리: {product['category']}")
        
        choice_a = int(input(f"\n선택 (1-{min(10, len(results_a))}): ")) - 1
        product_a = results_a.iloc[choice_a]
        
        self.display_product_info(product_a)
        
        # 두 번째 제품
        print("\n" + "-"*60)
        print("[2단계] 두 번째 제품 검색")
        query_b = input("\n제품명 또는 브랜드명을 입력하세요: ").strip()
        
        results_b = self.search_product(query_b)
        
        if len(results_b) == 0:
            print("❌ 검색 결과가 없습니다.")
            return
        
        print(f"\n검색 결과: {len(results_b)}개")
        for idx, (_, product) in enumerate(results_b.head(10).iterrows(), 1):
            print(f"\n[{idx}] {product['brand']} {product['product_name']}")
            print(f"    카테고리: {product['category']}")
        
        choice_b = int(input(f"\n선택 (1-{min(10, len(results_b))}): ")) - 1
        product_b = results_b.iloc[choice_b]
        
        self.display_product_info(product_b)
        
        # 성분 추출
        ingredients_a = self.parse_ingredients(product_a['all_ingredients'])
        ingredients_b = self.parse_ingredients(product_b['all_ingredients'])
        
        print(f"\n제품 A 성분: {len(ingredients_a)}개")
        print(f"제품 B 성분: {len(ingredients_b)}개")
        
        # 조합 분석
        result = self.analyze_combination(ingredients_a, ingredients_b)
        
        # 요약
        print("\n" + "="*60)
        print("📊 최종 요약")
        print("="*60)
        print(f"제품 A: {product_a['brand']} {product_a['product_name']}")
        print(f"제품 B: {product_b['brand']} {product_b['product_name']}")
        print(f"\n종합 평가: {result['overall_assessment']}")
        print(f"최대 위험도: {result['max_danger_level']:.1%}")
        print(f"평균 시너지: {result['avg_synergy_level']:.1%}")
        
        if result['max_danger_level'] > 0.7:
            print("\n⚠️ 경고: 함께 사용 시 주의가 필요합니다.")
        elif result['avg_synergy_level'] > 0.5:
            print("\n✅ 시너지: 함께 사용 시 효과적입니다.")


def main():
    """메인 함수"""
    tester = ProductCombinationTester()
    tester.interactive_test()


if __name__ == "__main__":
    main()
