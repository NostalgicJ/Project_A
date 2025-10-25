#!/usr/bin/env python3
"""
딥러닝 모델 테스트 스크립트
실제 학습된 모델로 성분 조합 분석
"""
import sys
import os
sys.path.append('src')

from models.advanced_ingredient_analyzer import AdvancedCosmeticAnalyzer

def test_dl_model():
    """딥러닝 모델 테스트"""
    print("🧠 딥러닝 모델 테스트 시작")
    print("=" * 50)
    
    # 모델 로드
    analyzer = AdvancedCosmeticAnalyzer()
    model_path = "models/advanced_ingredient_analyzer.pth"
    
    if os.path.exists(model_path):
        analyzer.load_model(model_path)
        print("✅ 딥러닝 모델 로드 완료!")
    else:
        print("❌ 모델 파일을 찾을 수 없습니다.")
        return
    
    # 테스트 케이스들
    test_cases = [
        {
            'name': '구달 청귤 비타민C 세럼 + 다이소 레티놀 크림',
            'ingredients': ['비타민C', '레티놀', '히알루론산', '판테놀']
        },
        {
            'name': '위험한 조합 (비타민C + 레티놀)',
            'ingredients': ['비타민C', '레티놀', 'AHA']
        },
        {
            'name': '시너지 조합 (비타민C + 비타민E)',
            'ingredients': ['비타민C', '비타민E', '히알루론산']
        },
        {
            'name': '안전한 조합',
            'ingredients': ['히알루론산', '세라마이드', '판테놀']
        }
    ]
    
    print("\n🔬 딥러닝 모델 분석 결과:")
    print("=" * 50)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📋 테스트 {i}: {test_case['name']}")
        print(f"성분: {test_case['ingredients']}")
        
        # 딥러닝 모델로 분석
        result = analyzer.analyze_combination(test_case['ingredients'])
        
        print(f"🧠 딥러닝 모델 예측:")
        print(f"  분류: {result['predicted_class']}")
        print(f"  위험도: {result['danger_score']:.1%}")
        print(f"  시너지: {result['synergy_score']:.1%}")
        print(f"  분석: {result['analysis']}")
        
        # 상세 분석
        if result['predicted_class'] == '위험':
            print("  ⚠️ 이 조합은 사용을 피하는 것이 좋습니다!")
        elif result['predicted_class'] == '주의':
            print("  ⚠️ 이 조합은 주의해서 사용하세요.")
        else:
            print("  ✅ 이 조합은 안전하게 사용할 수 있습니다.")
    
    print("\n🎯 딥러닝 모델의 특징:")
    print("✅ 실제 데이터를 학습하여 성분 조합을 예측")
    print("✅ 위험도와 시너지를 동시에 분석")
    print("✅ 새로운 성분 조합에 대해서도 예측 가능")

if __name__ == "__main__":
    test_dl_model()



