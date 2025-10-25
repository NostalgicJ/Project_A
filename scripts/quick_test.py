#!/usr/bin/env python3
"""
빠른 테스트 스크립트

사용 예시:
    # 직접 성분 리스트 입력
    python scripts/quick_test.py
    
    성분 리스트 1: 아스코빅애씨드,히알루론산
    성분 리스트 2: 레티놀,세라마이드
"""

import sys
sys.path.append('src')

from models.product_combination_analyzer import ProductIngredientMatcher

def quick_test():
    """빠른 테스트"""
    print("="*60)
    print("빠른 성분 조합 테스트")
    print("="*60)
    
    # 분석기 초기화
    analyzer = ProductIngredientMatcher()
    
    # 첫 번째 성분 리스트
    print("\n[첫 번째 성분 리스트]")
    ing1_str = input("성분을 쉼표로 구분하여 입력: ").strip()
    ingredients1 = [ing.strip() for ing in ing1_str.split(',')]
    
    # 두 번째 성분 리스트
    print("\n[두 번째 성분 리스트]")
    ing2_str = input("성분을 쉼표로 구분하여 입력: ").strip()
    ingredients2 = [ing.strip() for ing in ing2_str.split(',')]
    
    # 분석
    result = analyzer.analyze_product_pair(ingredients1, ingredients2)
    
    # 결과 출력
    print(analyzer.format_analysis_result(result))
    
    # 간단 요약
    print("\n" + "="*60)
    print("요약")
    print("="*60)
    print(f"종합 평가: {result['overall_assessment']}")
    print(f"최대 위험도: {result['max_danger_level']:.1%}")
    print(f"평균 시너지: {result['avg_synergy_level']:.1%}")

if __name__ == "__main__":
    quick_test()
