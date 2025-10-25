#!/usr/bin/env python3
"""
미확인 성분 수동 검토 및 수정 스크립트

사용 방법:
1. 미확인 성분 CSV 파일 열기
2. 각 미확인 성분을 확인하고 올바른 성분명으로 수정
3. 수정 내용을 자동으로 전체 데이터에 반영
"""

import pandas as pd
import json
from pathlib import Path
from typing import Dict, List
from collections import defaultdict

class UnconfirmedIngredientReviewer:
    """미확인 성분 검토 및 수정 클래스"""
    
    def __init__(self):
        self.unconfirmed_file = "data/processed/oliveyoung_products_cleaned_unconfirmed.csv"
        self.main_file = "data/processed/oliveyoung_products_cleaned.csv"
        self.changes_file = "data/processed/ingredient_review_changes.json"
        
    def load_unconfirmed(self):
        """미확인 성분 데이터 로드"""
        try:
            df = pd.read_csv(self.unconfirmed_file)
            print(f"✅ 미확인 성분 데이터 로드: {len(df)}개 제품")
            return df
        except FileNotFoundError:
            print("❌ 미확인 성분 파일이 없습니다.")
            return None
    
    def parse_ingredients(self, ingredients_str):
        """성분 문자열을 리스트로 변환"""
        if pd.isna(ingredients_str):
            return []
        return [ing.strip() for ing in str(ingredients_str).split(',')]
    
    def collect_unique_ingredients(self, df):
        """모든 미확인 성분 수집"""
        all_unconfirmed = set()
        for idx, row in df.iterrows():
            unconfirmed = self.parse_ingredients(row['unconfirmed_ingredients'])
            all_unconfirmed.update(unconfirmed)
        return sorted(all_unconfirmed)
    
    def display_statistics(self, df):
        """통계 출력"""
        print("\n" + "="*60)
        print("📊 미확인 성분 통계")
        print("="*60)
        
        all_ingredients = self.collect_unique_ingredients(df)
        print(f"고유 미확인 성분 수: {len(all_ingredients)}개")
        
        # 성분별 출현 빈도
        ingredient_freq = defaultdict(int)
        for idx, row in df.iterrows():
            unconfirmed = self.parse_ingredients(row['unconfirmed_ingredients'])
            for ing in unconfirmed:
                ingredient_freq[ing] += 1
        
        print(f"\n가장 많이 나타나는 미확인 성분 (상위 10개):")
        sorted_freq = sorted(ingredient_freq.items(), key=lambda x: x[1], reverse=True)
        for ing, count in sorted_freq[:10]:
            print(f"  - {ing}: {count}회")
    
    def review_ingredients(self, unconfirmed_list):
        """성분 리스트 검토 및 수정"""
        print("\n성분 검토 및 수정:")
        print("(입력하지 않으면 그대로 유지, 'skip'은 건너뛰기)")
        
        corrections = {}
        for ing in unconfirmed_list:
            correction = input(f"  '{ing}' → (올바른 성분명 또는 엔터): ").strip()
            if correction.lower() == 'skip':
                break
            if correction:
                corrections[ing] = correction
            else:
                print(f"    유지: {ing}")
        
        return corrections
    
    def save_corrections(self, corrections):
        """수정 사항 저장"""
        changes = {
            'type': 'ingredient_corrections',
            'corrections': corrections,
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        if Path(self.changes_file).exists():
            with open(self.changes_file, 'r', encoding='utf-8') as f:
                all_changes = json.load(f)
        else:
            all_changes = []
        
        all_changes.append(changes)
        
        with open(self.changes_file, 'w', encoding='utf-8') as f:
            json.dump(all_changes, f, ensure_ascii=False, indent=2)
        
        print(f"✅ {len(corrections)}개 성분 수정 사항 저장")
    
    def apply_corrections(self):
        """수정 사항을 전체 데이터에 적용"""
        print("\n" + "="*60)
        print("📝 수정사항 적용 중...")
        print("="*60)
        
        if not Path(self.changes_file).exists():
            print("⚠️ 수정사항이 없습니다.")
            return
        
        # 수정사항 로드
        with open(self.changes_file, 'r', encoding='utf-8') as f:
            all_changes = json.load(f)
        
        # 모든 수정사항 통합
        all_corrections = {}
        for change in all_changes:
            if change['type'] == 'ingredient_corrections':
                all_corrections.update(change['corrections'])
        
        if not all_corrections:
            print("⚠️ 적용할 수정사항이 없습니다.")
            return
        
        # 메인 데이터 로드
        main_df = pd.read_csv(self.main_file)
        
        print(f"\n{len(all_corrections)}개 성분 수정 적용 중...")
        corrections_applied = 0
        
        # 각 성분 문자열 수정
        for idx, row in main_df.iterrows():
            updated = False
            
            # unconfirmed_ingredients 수정
            if pd.notna(row['unconfirmed_ingredients']):
                unconfirmed = self.parse_ingredients(row['unconfirmed_ingredients'])
                new_unconfirmed = []
                new_confirmed = []
                
                for ing in unconfirmed:
                    if ing in all_corrections:
                        # 수정된 성분은 confirmed로 이동
                        new_confirmed.append(all_corrections[ing])
                        updated = True
                        corrections_applied += 1
                    else:
                        # 그대로 유지
                        new_unconfirmed.append(ing)
                
                if updated:
                    main_df.at[idx, 'unconfirmed_ingredients'] = ','.join(new_unconfirmed)
                    
                    # confirmed에 추가
                    if new_confirmed:
                        existing_confirmed = self.parse_ingredients(row['confirmed_ingredients'])
                        existing_confirmed.extend(new_confirmed)
                        main_df.at[idx, 'confirmed_ingredients'] = ','.join(existing_confirmed)
                        main_df.at[idx, 'confirmed_ingredients_count'] = len(existing_confirmed)
                        main_df.at[idx, 'unconfirmed_ingredients_count'] = len(new_unconfirmed)
        
        # 저장
        main_df.to_csv(self.main_file, index=False, encoding='utf-8-sig')
        
        print(f"✅ {corrections_applied}개 성분 수정 적용 완료")
        print(f"  - 저장된 파일: {self.main_file}")
    
    def interactive_review(self):
        """대화형 검토"""
        df = self.load_unconfirmed()
        if df is None:
            return
        
        # 통계 출력
        self.display_statistics(df)
        
        # 모든 미확인 성분 수집
        all_unconfirmed = self.collect_unique_ingredients(df)
        
        if not all_unconfirmed:
            print("\n✅ 미확인 성분이 없습니다!")
            return
        
        print(f"\n총 {len(all_unconfirmed)}개 미확인 성분을 검토합니다.")
        print("\n검토 방법:")
        print("  1. 각 성분에 대해 올바른 성분명을 입력")
        print("  2. 그대로 유지하려면 엔터")
        print("  3. 나중에 확인하려면 'skip'")
        print("  4. 종료하려면 'q'\n")
        
        # 검토 및 수정
        corrections = {}
        for i, ing in enumerate(all_unconfirmed, 1):
            if (i-1) % 10 == 0 and i > 1:
                apply_now = input("\n지금까지 수정사항을 적용하시겠습니까? (y/n): ")
                if apply_now.lower() == 'y':
                    if corrections:
                        self.save_corrections(corrections)
                        self.apply_corrections()
                        corrections = {}
            
            correction = input(f"[{i}/{len(all_unconfirmed)}] '{ing}' → ").strip()
            
            if correction.lower() == 'q':
                break
            elif correction.lower() == 'skip':
                print(f"  건너뜀")
                continue
            elif correction:
                corrections[ing] = correction
                print(f"  ✅ '{ing}' → '{correction}'")
        
        # 남은 수정사항 저장
        if corrections:
            self.save_corrections(corrections)
        
        # 최종 적용
        final_apply = input("\n모든 수정사항을 적용하시겠습니까? (y/n): ")
        if final_apply.lower() == 'y':
            self.apply_corrections()
            print("\n✅ 검토 완료!")


def main():
    """메인 함수"""
    reviewer = UnconfirmedIngredientReviewer()
    reviewer.interactive_review()


if __name__ == "__main__":
    main()
