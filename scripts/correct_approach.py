#!/usr/bin/env python3
"""
올바른 화장품 성분 조합 분석 시스템
실제 제품별 성분 리스트와 성분 계열 정보를 기반으로 한 정확한 분석
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import json

class CosmeticProduct:
    """화장품 제품 정보"""
    def __init__(self, brand: str, product_name: str, ingredients: List[str]):
        self.brand = brand
        self.product_name = product_name
        self.ingredients = ingredients
        self.full_name = f"{brand} {product_name}"
    
    def __str__(self):
        return f"{self.full_name}: {', '.join(self.ingredients[:3])}{'...' if len(self.ingredients) > 3 else ''}"

class IngredientInfo:
    """성분 정보"""
    def __init__(self, korean_name: str, english_name: str, 
                 ingredient_family: str, ph: Optional[float] = None,
                 dangerous_combinations: List[str] = None,
                 synergy_combinations: List[str] = None):
        self.korean_name = korean_name
        self.english_name = english_name
        self.ingredient_family = ingredient_family  # 비타민C계열, 레티놀계열 등
        self.ph = ph
        self.dangerous_combinations = dangerous_combinations or []
        self.synergy_combinations = synergy_combinations or []
    
    def __str__(self):
        return f"{self.korean_name} ({self.ingredient_family})"

class CorrectCosmeticAnalyzer:
    """올바른 화장품 성분 조합 분석기"""
    
    def __init__(self):
        self.products = {}  # 제품 정보 저장
        self.ingredient_database = {}  # 성분 정보 저장
        self.ingredient_family_mapping = {}  # 성분명 -> 계열 매핑
        
    def load_product_data(self, products_data: List[Dict]):
        """제품 데이터 로드"""
        print("📦 제품 데이터 로딩 중...")
        
        for product_data in products_data:
            product = CosmeticProduct(
                brand=product_data['brand'],
                product_name=product_data['product_name'],
                ingredients=product_data['ingredients']
            )
            self.products[product.full_name] = product
        
        print(f"✅ {len(self.products)}개 제품 로드 완료")
    
    def load_ingredient_database(self, ingredients_data: List[Dict]):
        """성분 데이터베이스 로드"""
        print("🧪 성분 데이터베이스 로딩 중...")
        
        for ingredient_data in ingredients_data:
            ingredient = IngredientInfo(
                korean_name=ingredient_data['korean_name'],
                english_name=ingredient_data['english_name'],
                ingredient_family=ingredient_data['ingredient_family'],
                ph=ingredient_data.get('ph'),
                dangerous_combinations=ingredient_data.get('dangerous_combinations', []),
                synergy_combinations=ingredient_data.get('synergy_combinations', [])
            )
            
            self.ingredient_database[ingredient.korean_name] = ingredient
            self.ingredient_database[ingredient.english_name] = ingredient
            
            # 성분명 -> 계열 매핑
            self.ingredient_family_mapping[ingredient.korean_name] = ingredient.ingredient_family
            self.ingredient_family_mapping[ingredient.english_name] = ingredient.ingredient_family
        
        print(f"✅ {len(self.ingredient_database)}개 성분 로드 완료")
    
    def get_ingredient_family(self, ingredient_name: str) -> Optional[str]:
        """성분의 계열 찾기"""
        print(f"🔍 성분 계열 찾기: {ingredient_name}")
        
        # 정확한 매칭
        if ingredient_name in self.ingredient_family_mapping:
            family = self.ingredient_family_mapping[ingredient_name]
            print(f"  ✅ 정확 매칭: {family}")
            return family
        
        # 부분 매칭 (예: "아스코빅애씨드" -> "비타민C계열")
        for known_ingredient, family in self.ingredient_family_mapping.items():
            if ingredient_name.lower() in known_ingredient.lower() or known_ingredient.lower() in ingredient_name.lower():
                print(f"  ✅ 부분 매칭: {ingredient_name} -> {family}")
                return family
        
        print(f"  ❌ 매칭 실패: {ingredient_name}")
        return None
    
    def analyze_product_combination(self, product_names: List[str]) -> Dict:
        """제품 조합 분석"""
        print(f"\n🔬 제품 조합 분석: {', '.join(product_names)}")
        
        # 1. 각 제품의 성분 리스트 추출
        product_ingredients = {}
        for product_name in product_names:
            if product_name in self.products:
                product = self.products[product_name]
                product_ingredients[product_name] = product.ingredients
                print(f"📦 {product_name}: {len(product.ingredients)}개 성분")
            else:
                print(f"❌ 제품을 찾을 수 없습니다: {product_name}")
                return None
        
        # 2. 각 제품의 성분 계열 분석
        product_families = {}
        for product_name, ingredients in product_ingredients.items():
            families = {}
            for ingredient in ingredients:
                family = self.get_ingredient_family(ingredient)
                if family:
                    if family not in families:
                        families[family] = []
                    families[family].append(ingredient)
            product_families[product_name] = families
        
        # 3. 제품 간 성분 계열 조합 분석
        results = {
            'dangerous_combinations': [],
            'synergy_combinations': [],
            'safe_combinations': []
        }
        
        products = list(product_families.keys())
        print(f"\n🔍 제품 간 조합 분석 시작: {len(products)}개 제품")
        
        for i in range(len(products)):
            for j in range(i + 1, len(products)):
                product1, product2 = products[i], products[j]
                families1, families2 = product_families[product1], product_families[product2]
                
                print(f"\n📊 {product1} vs {product2}")
                print(f"  {product1} 계열: {list(families1.keys())}")
                print(f"  {product2} 계열: {list(families2.keys())}")
                
                # 각 계열 조합 분석
                for family1, ingredients1 in families1.items():
                    for family2, ingredients2 in families2.items():
                        print(f"  🔬 조합 분석: {family1} vs {family2}")
                        combination_result = self._analyze_family_combination(
                            product1, product2, family1, family2, 
                            ingredients1, ingredients2
                        )
                        
                        print(f"    결과: {combination_result['type']}")
                        
                        if combination_result['type'] == 'dangerous':
                            results['dangerous_combinations'].append(combination_result)
                        elif combination_result['type'] == 'synergy':
                            results['synergy_combinations'].append(combination_result)
                        else:
                            results['safe_combinations'].append(combination_result)
        
        return results
    
    def _analyze_family_combination(self, product1: str, product2: str, 
                                  family1: str, family2: str,
                                  ingredients1: List[str], ingredients2: List[str]) -> Dict:
        """성분 계열 조합 분석"""
        
        # 위험한 조합 체크 (순서 무관하게 매칭)
        dangerous_combinations = {
            ('비타민C계열', '레티놀계열'): {
                'reason': 'pH 불일치로 효과 상쇄',
                'detail': '비타민C는 산성(pH 3-4), 레티놀은 중성(pH 6-7)으로 함께 사용 시 효과가 상쇄됩니다.'
            },
            ('레티놀계열', '비타민C계열'): {
                'reason': 'pH 불일치로 효과 상쇄',
                'detail': '비타민C는 산성(pH 3-4), 레티놀은 중성(pH 6-7)으로 함께 사용 시 효과가 상쇄됩니다.'
            },
            ('AHA계열', '레티놀계열'): {
                'reason': '과도한 각질 제거',
                'detail': '두 성분 모두 각질 제거 효과가 강해 피부 자극을 일으킬 수 있습니다.'
            },
            ('레티놀계열', 'AHA계열'): {
                'reason': '과도한 각질 제거',
                'detail': '두 성분 모두 각질 제거 효과가 강해 피부 자극을 일으킬 수 있습니다.'
            },
            ('BHA계열', '레티놀계열'): {
                'reason': '과도한 각질 제거',
                'detail': '살리실릭애씨드와 레티놀의 조합은 피부를 과도하게 자극할 수 있습니다.'
            },
            ('레티놀계열', 'BHA계열'): {
                'reason': '과도한 각질 제거',
                'detail': '살리실릭애씨드와 레티놀의 조합은 피부를 과도하게 자극할 수 있습니다.'
            }
        }
        
        # 시너지 조합 체크 (순서 무관하게 매칭)
        synergy_combinations = {
            ('비타민C계열', '비타민E계열'): {
                'reason': '항산화 효과 증대',
                'detail': '비타민C와 비타민E의 조합은 상호 보완적으로 항산화 효과를 증대시킵니다.'
            },
            ('비타민E계열', '비타민C계열'): {
                'reason': '항산화 효과 증대',
                'detail': '비타민C와 비타민E의 조합은 상호 보완적으로 항산화 효과를 증대시킵니다.'
            },
            ('히알루론산계열', '세라마이드계열'): {
                'reason': '보습 효과 증대',
                'detail': '히알루론산의 수분 공급과 세라마이드의 수분 보존 효과가 시너지를 만듭니다.'
            },
            ('세라마이드계열', '히알루론산계열'): {
                'reason': '보습 효과 증대',
                'detail': '히알루론산의 수분 공급과 세라마이드의 수분 보존 효과가 시너지를 만듭니다.'
            }
        }
        
        # 조합 분석
        combination_key = tuple(sorted([family1, family2]))
        
        if combination_key in dangerous_combinations:
            info = dangerous_combinations[combination_key]
            return {
                'type': 'dangerous',
                'product1': product1,
                'product2': product2,
                'family1': family1,
                'family2': family2,
                'ingredients1': ingredients1,
                'ingredients2': ingredients2,
                'reason': info['reason'],
                'detail': info['detail'],
                'message': f"⚠️ {product1}의 '{ingredients1[0]}'과 {product2}의 '{ingredients2[0]}'로 인해 {info['reason']}이 발생할 수 있으니 함께 사용하지 마세요!"
            }
        elif combination_key in synergy_combinations:
            info = synergy_combinations[combination_key]
            return {
                'type': 'synergy',
                'product1': product1,
                'product2': product2,
                'family1': family1,
                'family2': family2,
                'ingredients1': ingredients1,
                'ingredients2': ingredients2,
                'reason': info['reason'],
                'detail': info['detail'],
                'message': f"✅ {product1}의 '{ingredients1[0]}'과 {product2}의 '{ingredients2[0]}'로 인해 {info['reason']}을 낼 수 있어요!"
            }
        else:
            return {
                'type': 'safe',
                'product1': product1,
                'product2': product2,
                'family1': family1,
                'family2': family2,
                'ingredients1': ingredients1,
                'ingredients2': ingredients2,
                'message': f"✅ {product1}과 {product2}는 안전하게 함께 사용할 수 있습니다."
            }
    
    def print_analysis_results(self, results: Dict):
        """분석 결과 출력"""
        print("\n" + "="*60)
        print("🔬 화장품 성분 조합 분석 결과")
        print("="*60)
        
        if results['dangerous_combinations']:
            print("\n⚠️ 위험한 조합:")
            for combo in results['dangerous_combinations']:
                print(f"  {combo['message']}")
                print(f"    상세: {combo['detail']}")
        
        if results['synergy_combinations']:
            print("\n✅ 시너지 조합:")
            for combo in results['synergy_combinations']:
                print(f"  {combo['message']}")
                print(f"    상세: {combo['detail']}")
        
        if results['safe_combinations']:
            print(f"\n✅ 안전한 조합: {len(results['safe_combinations'])}개")


def create_sample_data():
    """샘플 데이터 생성"""
    
    # 제품 데이터
    products_data = [
        {
            'brand': '구달',
            'product_name': '청귤 비타민C 세럼',
            'ingredients': ['아스코빅애씨드', '히알루론산', '판테놀', '비타민E', '레티놀']
        },
        {
            'brand': '다이소',
            'product_name': '레티놀 크림',
            'ingredients': ['레티놀', '세라마이드', '히알루론산', '니아신아마이드']
        },
        {
            'brand': '더바디샵',
            'product_name': '비타민E 크림',
            'ingredients': ['토코페롤', '히알루론산', '세라마이드', '판테놀']
        }
    ]
    
    # 성분 데이터베이스
    ingredients_data = [
        {
            'korean_name': '아스코빅애씨드',
            'english_name': 'Ascorbic Acid',
            'ingredient_family': '비타민C계열',
            'ph': 3.5,
            'dangerous_combinations': ['레티놀계열', 'AHA계열'],
            'synergy_combinations': ['비타민E계열', '페룰릭애씨드계열']
        },
        {
            'korean_name': '레티놀',
            'english_name': 'Retinol',
            'ingredient_family': '레티놀계열',
            'ph': 6.5,
            'dangerous_combinations': ['비타민C계열', 'AHA계열', 'BHA계열'],
            'synergy_combinations': ['세라마이드계열', '히알루론산계열']
        },
        {
            'korean_name': '히알루론산',
            'english_name': 'Hyaluronic Acid',
            'ingredient_family': '히알루론산계열',
            'ph': 7.0,
            'dangerous_combinations': [],
            'synergy_combinations': ['세라마이드계열', '레티놀계열']
        },
        {
            'korean_name': '세라마이드',
            'english_name': 'Ceramide',
            'ingredient_family': '세라마이드계열',
            'ph': 7.0,
            'dangerous_combinations': [],
            'synergy_combinations': ['히알루론산계열', '레티놀계열']
        },
        {
            'korean_name': '토코페롤',
            'english_name': 'Tocopherol',
            'ingredient_family': '비타민E계열',
            'ph': 7.0,
            'dangerous_combinations': [],
            'synergy_combinations': ['비타민C계열']
        }
    ]
    
    return products_data, ingredients_data


if __name__ == "__main__":
    print("🎯 올바른 화장품 성분 조합 분석 시스템")
    print("="*50)
    
    # 분석기 초기화
    analyzer = CorrectCosmeticAnalyzer()
    
    # 샘플 데이터 생성
    products_data, ingredients_data = create_sample_data()
    
    # 데이터 로드
    analyzer.load_product_data(products_data)
    analyzer.load_ingredient_database(ingredients_data)
    
    # 제품 조합 분석
    test_products = ['구달 청귤 비타민C 세럼', '다이소 레티놀 크림']
    results = analyzer.analyze_product_combination(test_products)
    
    # 결과 출력
    if results:
        analyzer.print_analysis_results(results)
