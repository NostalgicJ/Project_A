#!/usr/bin/env python3
"""
성분 조합 규칙 관리 스크립트

성분 계열과 위험 조합을 추가/수정/삭제할 수 있습니다.
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List

class IngredientRuleManager:
    """성분 규칙 관리 클래스"""
    
    def __init__(self):
        self.rules_file = "config/ingredient_rules.json"
        self.load_rules()
    
    def load_rules(self):
        """규칙 파일 로드"""
        if Path(self.rules_file).exists():
            with open(self.rules_file, 'r', encoding='utf-8') as f:
                self.rules = json.load(f)
        else:
            # 기본 규칙 생성
            self.rules = {
                'ingredient_families': {
                    '비타민C계열': ['아스코빅애씨드', '아스코빌글루코사이드', '소듐아스코빌포스페이트'],
                    '레티놀계열': ['레티놀', '레티놀아세테이트', '레티날'],
                    'AHA계열': ['글라이콜릭애씨드', '젖산', '시트릭애씨드'],
                    'BHA계열': ['살리실릭애씨드', '베타하이드록시애씨드'],
                    '비타민E계열': ['토코페롤', '토코페릴아세테이트'],
                    '히알루론산계열': ['하이알루로닉애씨드', '소듐하이알루로네이트'],
                    '세라마이드계열': ['세라마이드', '세라마이드엔피', '세라마이드3'],
                },
                'dangerous_combinations': [
                    {
                        'family1': '비타민C계열',
                        'family2': '레티놀계열',
                        'danger_level': 0.9,
                        'reason': 'pH 불일치로 효과 상쇄',
                        'detail': '비타민C는 산성(pH 3-4), 레티놀은 중성(pH 6-7)으로 함께 사용 시 효과가 상쇄됩니다.'
                    },
                    {
                        'family1': 'AHA계열',
                        'family2': '레티놀계열',
                        'danger_level': 0.8,
                        'reason': '과도한 각질 제거',
                        'detail': '두 성분 모두 각질 제거 효과가 강해 피부 자극을 일으킬 수 있습니다.'
                    },
                    {
                        'family1': 'BHA계열',
                        'family2': '레티놀계열',
                        'danger_level': 0.8,
                        'reason': '과도한 각질 제거',
                        'detail': '살리실릭애씨드와 레티놀의 조합은 피부를 과도하게 자극할 수 있습니다.'
                    }
                ],
                'synergy_combinations': [
                    {
                        'family1': '비타민C계열',
                        'family2': '비타민E계열',
                        'synergy_level': 0.8,
                        'reason': '항산화 효과 증대',
                        'detail': '비타민C와 비타민E의 조합은 상호 보완적으로 항산화 효과를 증대시킵니다.'
                    },
                    {
                        'family1': '히알루론산계열',
                        'family2': '세라마이드계열',
                        'synergy_level': 0.7,
                        'reason': '보습 효과 증대',
                        'detail': '히알루론산의 수분 공급과 세라마이드의 수분 보존 효과가 시너지를 만듭니다.'
                    }
                ]
            }
            self.save_rules()
    
    def save_rules(self):
        """규칙 파일 저장"""
        Path(self.rules_file).parent.mkdir(parents=True, exist_ok=True)
        with open(self.rules_file, 'w', encoding='utf-8') as f:
            json.dump(self.rules, f, ensure_ascii=False, indent=2)
        print(f"✅ 규칙 저장: {self.rules_file}")
    
    def display_rules(self):
        """현재 규칙 표시"""
        print("\n" + "="*60)
        print("📋 현재 성분 규칙")
        print("="*60)
        
        print("\n[성분 계열]")
        for family, ingredients in self.rules['ingredient_families'].items():
            print(f"  {family}: {len(ingredients)}개 성분")
            for ing in ingredients[:3]:
                print(f"    - {ing}")
            if len(ingredients) > 3:
                print(f"    ... 외 {len(ingredients)-3}개")
        
        print("\n[위험한 조합]")
        for combo in self.rules['dangerous_combinations']:
            print(f"  ⚠️ {combo['family1']} + {combo['family2']}")
            print(f"    위험도: {combo['danger_level']}, 이유: {combo['reason']}")
        
        print("\n[시너지 조합]")
        for combo in self.rules['synergy_combinations']:
            print(f"  ✅ {combo['family1']} + {combo['family2']}")
            print(f"    시너지: {combo['synergy_level']}, 이유: {combo['reason']}")
    
    def add_ingredient_family(self, family_name, ingredients):
        """성분 계열 추가"""
        if family_name in self.rules['ingredient_families']:
            print(f"⚠️ 이미 존재하는 계열입니다: {family_name}")
            response = input("덮어쓰시겠습니까? (y/n): ")
            if response.lower() != 'y':
                return
        
        self.rules['ingredient_families'][family_name] = ingredients
        self.save_rules()
        print(f"✅ 성분 계열 추가: {family_name} ({len(ingredients)}개 성분)")
    
    def add_dangerous_combination(self):
        """위험한 조합 추가"""
        print("\n위험한 조합 추가:")
        family1 = input("계열 1: ").strip()
        family2 = input("계열 2: ").strip()
        danger_level = float(input("위험도 (0-1): "))
        reason = input("이유: ").strip()
        detail = input("상세 설명: ").strip()
        
        combo = {
            'family1': family1,
            'family2': family2,
            'danger_level': danger_level,
            'reason': reason,
            'detail': detail
        }
        
        # 중복 체크
        for existing in self.rules['dangerous_combinations']:
            if (existing['family1'] == family1 and existing['family2'] == family2) or \
               (existing['family1'] == family2 and existing['family2'] == family1):
                print("⚠️ 이미 존재하는 조합입니다.")
                response = input("덮어쓰시겠습니까? (y/n): ")
                if response.lower() != 'y':
                    return
                self.rules['dangerous_combinations'].remove(existing)
                break
        
        self.rules['dangerous_combinations'].append(combo)
        self.save_rules()
        print(f"✅ 위험한 조합 추가: {family1} + {family2}")
    
    def add_synergy_combination(self):
        """시너지 조합 추가"""
        print("\n시너지 조합 추가:")
        family1 = input("계열 1: ").strip()
        family2 = input("계열 2: ").strip()
        synergy_level = float(input("시너지 정도 (0-1): "))
        reason = input("이유: ").strip()
        detail = input("상세 설명: ").strip()
        
        combo = {
            'family1': family1,
            'family2': family2,
            'synergy_level': synergy_level,
            'reason': reason,
            'detail': detail
        }
        
        # 중복 체크
        for existing in self.rules['synergy_combinations']:
            if (existing['family1'] == family1 and existing['family2'] == family2) or \
               (existing['family1'] == family2 and existing['family2'] == family1):
                print("⚠️ 이미 존재하는 조합입니다.")
                response = input("덮어쓰시겠습니까? (y/n): ")
                if response.lower() != 'y':
                    return
                self.rules['synergy_combinations'].remove(existing)
                break
        
        self.rules['synergy_combinations'].append(combo)
        self.save_rules()
        print(f"✅ 시너지 조합 추가: {family1} + {family2}")
    
    def interactive_manage(self):
        """대화형 관리"""
        while True:
            self.display_rules()
            
            print("\n" + "="*60)
            print("작업 선택:")
            print("  1. 성분 계열 추가")
            print("  2. 위험한 조합 추가")
            print("  3. 시너지 조합 추가")
            print("  4. 규칙 삭제")
            print("  5. 종료")
            
            choice = input("\n선택: ").strip()
            
            if choice == '1':
                family = input("\n계열명: ").strip()
                ingredients_str = input("성분 리스트 (쉼표로 구분): ").strip()
                ingredients = [ing.strip() for ing in ingredients_str.split(',')]
                self.add_ingredient_family(family, ingredients)
            
            elif choice == '2':
                self.add_dangerous_combination()
            
            elif choice == '3':
                self.add_synergy_combination()
            
            elif choice == '4':
                print("\n⚠️ 규칙 삭제 기능은 추후 구현 예정")
            
            elif choice == '5':
                print("\n✅ 종료")
                break
            
            else:
                print("잘못된 선택입니다.")


def main():
    """메인 함수"""
    manager = IngredientRuleManager()
    manager.interactive_manage()


if __name__ == "__main__":
    main()
