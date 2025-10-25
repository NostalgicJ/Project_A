#!/usr/bin/env python3
"""
기획 상품 수동 검토 및 수정 스크립트

사용 방법:
1. 기획 상품 CSV 파일 열기
2. 각 기획 상품을 개별 제품으로 분리
3. 수정 내용을 자동으로 전체 데이터에 반영
"""

import pandas as pd
import json
from pathlib import Path
from typing import Dict, List

class PackageProductReviewer:
    """기획 상품 검토 및 수정 클래스"""
    
    def __init__(self):
        self.packages_file = "data/processed/oliveyoung_products_cleaned_packages.csv"
        self.main_file = "data/processed/oliveyoung_products_cleaned.csv"
        self.changes_file = "data/processed/package_review_changes.json"
        
    def load_packages(self):
        """기획 상품 데이터 로드"""
        try:
            df = pd.read_csv(self.packages_file)
            print(f"✅ 기획 상품 데이터 로드: {len(df)}개")
            return df
        except FileNotFoundError:
            print("❌ 기획 상품 파일이 없습니다.")
            return None
    
    def display_package(self, row):
        """기획 상품 정보 표시"""
        print("\n" + "="*60)
        print(f"제품 ID: {row.name}")
        print(f"카테고리: {row['category']}")
        print(f"브랜드: {row['brand']}")
        print(f"제품명: {row['product_name']}")
        print(f"원본 제품명: {row['original_name']}")
        print(f"URL: {row['url']}")
        print(f"성분 수: {row['total_ingredients']}")
        print(f"성분: {row['all_ingredients'][:100]}...")
        print("="*60)
    
    def parse_ingredients(self, ingredients_str):
        """성분 문자열을 리스트로 변환"""
        if pd.isna(ingredients_str):
            return []
        return [ing.strip() for ing in str(ingredients_str).split(',')]
    
    def split_package_manually(self, package_id, split_data: List[Dict]):
        """
        기획 상품을 수동으로 분리
        
        Args:
            package_id: 기획 상품 ID
            split_data: 분리할 제품 정보 리스트
                [{
                    'brand': '브랜드명',
                    'product_name': '제품명',
                    'ingredients': ['성분1', '성분2', ...],
                    'category': '카테고리' (선택)
                }, ...]
        """
        changes = {
            'type': 'split_package',
            'package_id': str(package_id),
            'new_products': split_data,
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        # 변경사항 저장
        self.save_changes(changes)
        
        print(f"✅ 기획 상품 {package_id}를 {len(split_data)}개 제품으로 분리")
        for i, product in enumerate(split_data, 1):
            print(f"  {i}. {product['brand']} {product['product_name']} ({len(product['ingredients'])}개 성분)")
    
    def remove_package(self, package_id, reason=""):
        """기획 상품 제거"""
        changes = {
            'type': 'remove_package',
            'package_id': str(package_id),
            'reason': reason,
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        self.save_changes(changes)
        print(f"✅ 기획 상품 {package_id} 제거 (사유: {reason})")
    
    def keep_package(self, package_id):
        """기획 상품 유지"""
        changes = {
            'type': 'keep_package',
            'package_id': str(package_id),
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        self.save_changes(changes)
        print(f"✅ 기획 상품 {package_id} 유지")
    
    def save_changes(self, change):
        """변경사항 저장"""
        if Path(self.changes_file).exists():
            with open(self.changes_file, 'r', encoding='utf-8') as f:
                changes = json.load(f)
        else:
            changes = []
        
        changes.append(change)
        
        with open(self.changes_file, 'w', encoding='utf-8') as f:
            json.dump(changes, f, ensure_ascii=False, indent=2)
    
    def apply_changes(self):
        """변경사항을 전체 데이터에 적용"""
        print("\n" + "="*60)
        print("📝 변경사항 적용 중...")
        print("="*60)
        
        if not Path(self.changes_file).exists():
            print("⚠️ 변경사항이 없습니다.")
            return
        
        # 변경사항 로드
        with open(self.changes_file, 'r', encoding='utf-8') as f:
            changes = json.load(f)
        
        # 메인 데이터 로드
        main_df = pd.read_csv(self.main_file)
        packages_df = pd.read_csv(self.packages_file)
        
        new_products = []
        packages_to_remove = []
        
        # 변경사항 적용
        for change in changes:
            if change['type'] == 'split_package':
                # 기획 상품을 개별 제품으로 추가
                for product in change['new_products']:
                    new_row = {
                        'category': product.get('category', packages_df.loc[int(change['package_id']), 'category']),
                        'brand': product['brand'],
                        'product_name': product['product_name'],
                        'original_name': f"[분리됨] {product['brand']} {product['product_name']}",
                        'url': packages_df.loc[int(change['package_id']), 'url'],
                        'is_package': False,
                        'total_ingredients': len(product['ingredients']),
                        'confirmed_ingredients_count': len(product['ingredients']),
                        'unconfirmed_ingredients_count': 0,
                        'confirmed_ingredients': ','.join(product['ingredients']),
                        'unconfirmed_ingredients': '',
                        'all_ingredients': ','.join(product['ingredients']),
                    }
                    new_products.append(new_row)
                
                packages_to_remove.append(int(change['package_id']))
            
            elif change['type'] == 'remove_package':
                packages_to_remove.append(int(change['package_id']))
        
        # 메인 데이터에 새 제품 추가
        if new_products:
            new_df = pd.DataFrame(new_products)
            main_df = pd.concat([main_df, new_df], ignore_index=True)
            print(f"✅ {len(new_products)}개 제품 추가")
        
        # 기획 상품에서 제거
        if packages_to_remove:
            packages_df = packages_df.drop(index=packages_to_remove)
            print(f"✅ {len(packages_to_remove)}개 기획 상품 제거")
        
        # 저장
        main_df.to_csv(self.main_file, index=False, encoding='utf-8-sig')
        packages_df.to_csv(self.packages_file, index=False, encoding='utf-8-sig')
        
        print(f"\n✅ 변경사항 적용 완료")
        print(f"  - 메인 데이터: {len(main_df)}개 제품")
        print(f"  - 기획 상품: {len(packages_df)}개 남음")
    
    def interactive_review(self):
        """대화형 검토"""
        df = self.load_packages()
        if df is None:
            return
        
        print(f"\n총 {len(df)}개 기획 상품을 검토합니다.")
        print("\n사용 방법:")
        print("  - s: 제품 분리")
        print("  - r: 제품 제거")
        print("  - k: 제품 유지")
        print("  - n: 다음으로")
        print("  - q: 종료\n")
        
        for idx, row in df.iterrows():
            self.display_package(row)
            
            while True:
                action = input("작업 선택 (s/r/k/n/q): ").lower()
                
                if action == 's':
                    # 제품 분리
                    print("\n분리할 제품 정보를 입력하세요 (종료: 빈 줄)")
                    products = []
                    while True:
                        brand = input("브랜드명: ")
                        if not brand:
                            break
                        product_name = input("제품명: ")
                        ingredients_str = input("성분 리스트 (쉼표로 구분): ")
                        ingredients = [ing.strip() for ing in ingredients_str.split(',')]
                        
                        products.append({
                            'brand': brand,
                            'product_name': product_name,
                            'ingredients': ingredients,
                            'category': row['category']
                        })
                        print("✅ 제품 추가됨")
                    
                    if products:
                        self.split_package_manually(idx, products)
                    break
                
                elif action == 'r':
                    reason = input("제거 사유: ")
                    self.remove_package(idx, reason)
                    break
                
                elif action == 'k':
                    self.keep_package(idx)
                    break
                
                elif action == 'n':
                    break
                
                elif action == 'q':
                    print("검토를 종료합니다.")
                    return
                
                else:
                    print("잘못된 입력입니다.")
            
            apply_now = input("\n지금 변경사항을 적용하시겠습니까? (y/n): ")
            if apply_now.lower() == 'y':
                self.apply_changes()
        
        # 최종 적용 확인
        final_apply = input("\n모든 변경사항을 최종 적용하시겠습니까? (y/n): ")
        if final_apply.lower() == 'y':
            self.apply_changes()
            print("\n✅ 검토 완료!")


def main():
    """메인 함수"""
    reviewer = PackageProductReviewer()
    reviewer.interactive_review()


if __name__ == "__main__":
    main()
