#!/usr/bin/env python3
"""
실제 화장품 제품으로 성분 조합 분석 테스트
"""
import sys
import os
sys.path.append('src')

from models.ingredient_analyzer import CosmeticIngredientAnalyzer

def analyze_real_products():
    print("🧴 실제 화장품 제품 성분 조합 분석")
    print("=" * 60)
    
    # 분석기 초기화
    analyzer = CosmeticIngredientAnalyzer()
    
    # 실제 제품 테스트 케이스
    test_cases = [
        {
            "name": "구달 청귤 비타민C 세럼 + 다이소 레티놀 크림",
            "ingredients": [
                # 구달 청귤 비타민C 세럼의 주요 성분들
                "3-O-에틸아스코빅애씨드",  # 비타민C 유도체
                "아스코빅애씨드",  # 비타민C
                "아스코빌글루코사이드",  # 비타민C 유도체
                "아스코빌팔미테이트",  # 비타민C 유도체
                "레티놀",  # 레티놀
                "레티닐팔미테이트",  # 레티놀 유도체
                "히알루론산",
                "세라마이드",
                "나이아신아마이드"
            ],
            "description": "구달 청귤 비타민C 세럼과 다이소 레티놀 크림을 함께 사용하는 경우"
        },
        {
            "name": "구달 청귤 비타민C 세럼 단독 사용",
            "ingredients": [
                "3-O-에틸아스코빅애씨드",
                "아스코빅애씨드", 
                "아스코빌글루코사이드",
                "히알루론산",
                "세라마이드",
                "나이아신아마이드"
            ],
            "description": "구달 청귤 비타민C 세럼만 단독으로 사용하는 경우"
        },
        {
            "name": "다이소 레티놀 크림 단독 사용",
            "ingredients": [
                "레티놀",
                "레티닐팔미테이트",
                "히알루론산",
                "세라마이드"
            ],
            "description": "다이소 레티놀 크림만 단독으로 사용하는 경우"
        }
    ]
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n🔍 테스트 {i}: {case['name']}")
        print(f"설명: {case['description']}")
        print(f"주요 성분: {', '.join(case['ingredients'][:5])}...")
        
        # 성분 조합 분석
        result = analyzer.analyze_combination(case['ingredients'])
        
        print(f"\n📊 분석 결과:")
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
        
        # 구체적인 조언
        if case['name'] == "구달 청귤 비타민C 세럼 + 다이소 레티놀 크림":
            print(f"\n🎯 전문가 조언:")
            if result['safety_score'] < 0.7:
                print("    ⚠️ 비타민C와 레티놀을 함께 사용하면 산화 반응으로 효과가 상쇄될 수 있습니다.")
                print("    💡 권장사항: 아침에는 비타민C, 저녁에는 레티놀을 사용하세요.")
                print("    ⏰ 시간 간격: 최소 12시간 이상 간격을 두고 사용하세요.")
            else:
                print("    ✅ 이 조합은 안전하게 사용할 수 있습니다.")
                print("    💡 팁: 순서대로 사용하면 더 좋은 효과를 얻을 수 있습니다.")
        
        print("-" * 60)
    
    # 추가 추천
    print(f"\n🎯 추가 추천 성분:")
    current_ingredients = ["아스코빅애씨드", "레티놀", "히알루론산"]
    recommendations = analyzer.get_ingredient_recommendations(current_ingredients)
    
    print(f"현재 사용 중인 성분: {', '.join(current_ingredients)}")
    for i, rec in enumerate(recommendations[:3], 1):
        print(f"  {i}. {rec['ingredient']} - {rec['reason']}")
    
    print(f"\n🎉 실제 제품 분석 완료!")

if __name__ == "__main__":
    analyze_real_products()



