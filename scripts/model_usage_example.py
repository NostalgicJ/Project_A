#!/usr/bin/env python3
"""
화장품 성분 조합 분석 모델 사용 예시
"""
import sys
import os
sys.path.append('src')

from models.ingredient_analyzer import CosmeticIngredientAnalyzer
import pandas as pd
import numpy as np

def main():
    print("🧪 화장품 성분 조합 분석 모델 사용 예시")
    print("=" * 50)
    
    # 1. 분석기 초기화
    analyzer = CosmeticIngredientAnalyzer()
    
    # 2. 테스트 케이스들
    test_cases = [
        {
            "name": "위험한 조합",
            "ingredients": ["비타민C", "레티놀", "히알루론산"],
            "description": "비타민C와 레티놀은 함께 사용하면 산화 반응으로 효과가 상쇄됩니다."
        },
        {
            "name": "좋은 조합",
            "ingredients": ["비타민C", "비타민E", "히알루론산"],
            "description": "비타민C와 비타민E는 함께 사용하면 항산화 효과가 증대됩니다."
        },
        {
            "name": "시너지 조합",
            "ingredients": ["나이아신아마이드", "아연", "세라마이드"],
            "description": "나이아신아마이드와 아연은 모공 관리에 시너지 효과를 낼 수 있습니다."
        },
        {
            "name": "안전한 조합",
            "ingredients": ["히알루론산", "세라마이드", "판테놀"],
            "description": "이 성분들은 모두 보습에 도움을 주는 안전한 조합입니다."
        }
    ]
    
    # 3. 각 케이스 분석
    for i, case in enumerate(test_cases, 1):
        print(f"\n🔍 테스트 {i}: {case['name']}")
        print(f"성분: {', '.join(case['ingredients'])}")
        print(f"설명: {case['description']}")
        
        # 성분 조합 분석
        result = analyzer.analyze_combination(case['ingredients'])
        
        print(f"📊 분석 결과:")
        print(f"  - 분류: {result['predicted_class']}")
        print(f"  - 신뢰도: {result['confidence']:.3f}")
        print(f"  - 안전성 점수: {result['safety_score']:.3f}")
        print(f"  - 시너지 점수: {result['synergy_score']:.3f}")
        print(f"  - 분석: {result['analysis']}")
        
        # 안전성 및 시너지 이슈 표시
        if 'safety_issues' in result and result['safety_issues']:
            print(f"⚠️ 안전성 이슈:")
            for issue in result['safety_issues']:
                print(f"    - {issue}")
        
        if 'synergy_benefits' in result and result['synergy_benefits']:
            print(f"✨ 시너지 효과:")
            for benefit in result['synergy_benefits']:
                print(f"    - {benefit}")
        
        print("-" * 50)
    
    # 4. 성분 추천 예시
    print(f"\n🎯 성분 추천 예시")
    print("=" * 50)
    
    current_ingredients = ["비타민C", "히알루론산"]
    recommendations = analyzer.get_ingredient_recommendations(current_ingredients)
    
    print(f"현재 사용 중인 성분: {', '.join(current_ingredients)}")
    print(f"추천 성분:")
    
    for i, rec in enumerate(recommendations, 1):
        print(f"  {i}. {rec['ingredient']}")
        print(f"     이유: {rec['reason']}")
        print(f"     시너지 점수: {rec['synergy_score']:.3f}")
    
    print("\n🎉 모델 사용 예시 완료!")

if __name__ == "__main__":
    main()
