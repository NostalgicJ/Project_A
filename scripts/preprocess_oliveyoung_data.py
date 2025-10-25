#!/usr/bin/env python3
"""
올리브영 제품 데이터 전처리 스크립트

기능:
1. JSON 파일에서 제품 정보 파싱 (브랜드명, 제품명, 성분리스트)
2. 제품명에서 불필요한 키워드 제거 (기획, 증정, 올영픽 등)
3. 성분 정규화 - 공공데이터포털 API 성분 데이터와 매칭
4. 미확인 성분 분리
5. 기획 상품 처리 (한 제품에 여러 아이템 포함된 경우)
"""

import json
import pandas as pd
import re
from typing import List, Dict, Tuple, Set
from pathlib import Path
import requests
import time

class OliveYoungDataPreprocessor:
    """올리브영 데이터 전처리기"""
    
    def __init__(self):
        self.unwanted_keywords = [
            '기획', '증정', '올영픽', 'PICK', '화잘먹', '리필기획',
            '더블기획', '한정기획', '단독', '1+1', '2+1', '3+1',
            '리필', '기프트', '증정품', '사은품', '프리미엄',
            '연예인', '유명인', '인기', '베스트', 'NEW',
            '신상', '출시', '런칭', '오픈', '특가',
            '[', ']', '(', ')'
        ]
        
        self.all_ingredients = set()  # 공공데이터 API에서 받은 전체 성분
        self.api_ingredients = None
        
    def load_public_api_ingredients(self):
        """
        공공데이터 포털 API에서 전체 성분 데이터 로드
        """
        print("📡 성분 데이터 로드 중...")
        
        # 1. 공공데이터 API에서 다운로드한 데이터 우선 사용
        try:
            public_df = pd.read_csv('data/raw/public_ingredients.csv')
            # 한글명과 영문명을 모두 포함
            ingredients = set()
            ingredients.update(public_df['한글명'].dropna().astype(str))
            ingredients.update(public_df['영문명'].dropna().astype(str))
            
            if len(ingredients) > 0:
                self.all_ingredients = ingredients
                print(f"✅ 공공데이터 API 성분 로드: {len(self.all_ingredients)}개")
                return
        except Exception as e:
            print(f"⚠️ 공공데이터 파일 없음: {e}")
        
        # 2. COOS 마스터 데이터 사용 (백업)
        try:
            master_df = pd.read_csv('data/raw/coos_master_ingredients_cleaned.csv')
            self.all_ingredients = set(master_df['원료명_정제됨'].dropna())
            print(f"✅ COOS 마스터 데이터 사용: {len(self.all_ingredients)}개")
        except Exception as e:
            print(f"❌ 성분 데이터 로드 실패: {e}")
            self.all_ingredients = set()
    
    def clean_product_name(self, product_name: str) -> str:
        """
        제품명에서 불필요한 키워드 제거
        
        Args:
            product_name: 원본 제품명
            
        Returns:
            정제된 제품명
        """
        cleaned = product_name
        
        # 대괄호 내용 제거
        cleaned = re.sub(r'\[.*?\]', '', cleaned)
        
        # 괄호 내용 제거 (단, 사이즈 정보는 유지)
        cleaned = re.sub(r'\([^)]*ml[^)]*\)', '', cleaned)
        cleaned = re.sub(r'\([^)]*\)', '', cleaned)
        
        # 불필요한 키워드 제거
        for keyword in self.unwanted_keywords:
            cleaned = cleaned.replace(keyword, '')
        
        # 다중 공백 정리
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        # 마지막에 남는 특수문자 제거
        cleaned = re.sub(r'[\+\-]+$', '', cleaned).strip()
        
        return cleaned
    
    def extract_brand_and_name(self, product_name: str) -> Tuple[str, str]:
        """
        브랜드명과 제품명 추출
        
        Args:
            product_name: 제품명 (정제 전 또는 후)
            
        Returns:
            (브랜드명, 제품명)
        """
        cleaned = self.clean_product_name(product_name)
        
        # 첫 번째 단어를 브랜드로 추정 (공백 기준)
        parts = cleaned.split()
        if len(parts) > 0:
            brand = parts[0]
            name = ' '.join(parts[1:]) if len(parts) > 1 else parts[0]
        else:
            brand = 'Unknown'
            name = cleaned
        
        return brand, name
    
    def check_if_package_product(self, product_name: str, ingredients: List[str]) -> bool:
        """
        기획 상품인지 확인 (한 제품에 여러 아이템이 포함된 경우)
        
        Args:
            product_name: 제품명
            ingredients: 성분 리스트
            
        Returns:
            기획 상품 여부
        """
        # 제품명에 기획 관련 키워드가 있는지 확인
        package_keywords = ['더블', '트리플', '2개입', '3개입', '세트']
        for keyword in package_keywords:
            if keyword in product_name:
                return True
        
        # 성분 수가 비정상적으로 많은 경우 (샘플링 필요)
        if len(ingredients) > 50:
            return True
        
        return False
    
    def split_package_product(self, product_name: str, ingredients: List[str]) -> List[Dict]:
        """
        기획 상품을 개별 제품으로 분리
        
        Args:
            product_name: 제품명
            ingredients: 성분 리스트
            
        Returns:
            분리된 제품 리스트
        """
        # TODO: 복잡한 기획 상품 분리 로직 구현
        # 현재는 원본 그대로 반환 (수동 검토 필요)
        
        brand, name = self.extract_brand_and_name(product_name)
        
        return [{
            'brand': brand,
            'product_name': name,
            'original_name': product_name,
            'is_package': True,
            'ingredients': ingredients,
            'note': '기획 상품 - 수동 검토 필요'
        }]
    
    def categorize_ingredients(self, ingredients: List[str]) -> Dict[str, List[str]]:
        """
        성분을 확인된 성분과 미확인 성분으로 분류
        
        Args:
            ingredients: 성분 리스트
            
        Returns:
            {'confirmed': [...], 'unconfirmed': [...]}
        """
        confirmed = []
        unconfirmed = []
        
        for ing in ingredients:
            ing_clean = ing.strip()
            
            # 빈 문자열이나 숫자만 있는 경우 제외
            if not ing_clean or ing_clean.isdigit():
                continue
            
            # 전체 성분 데이터에 존재하는지 확인
            if ing_clean in self.all_ingredients:
                confirmed.append(ing_clean)
            else:
                # 유사도 검사 (추후 구현)
                unconfirmed.append(ing_clean)
        
        return {
            'confirmed': confirmed,
            'unconfirmed': unconfirmed
        }
    
    def process_json_file(self, file_path: str, category: str) -> pd.DataFrame:
        """
        JSON 파일에서 제품 데이터 추출 및 정제
        
        Args:
            file_path: JSON 파일 경로
            category: 제품 카테고리
            
        Returns:
            정제된 데이터 DataFrame
        """
        print(f"\n📂 파일 처리 중: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"   원본 데이터: {len(data)}개 제품")
        
        processed_products = []
        
        for item in data:
            product_name = item.get('제품명', '')
            url = item.get('URL', '')
            ingredients = item.get('성분리스트', [])
            
            # 제품명 정제
            cleaned_name = self.clean_product_name(product_name)
            brand, name = self.extract_brand_and_name(cleaned_name)
            
            # 기획 상품 여부 확인
            is_package = self.check_if_package_product(product_name, ingredients)
            
            # 성분 분류
            categorized = self.categorize_ingredients(ingredients)
            
            # 제품 데이터 생성
            product_data = {
                'category': category,
                'brand': brand,
                'product_name': name,
                'original_name': product_name,
                'url': url,
                'is_package': is_package,
                'total_ingredients': len(ingredients),
                'confirmed_ingredients_count': len(categorized['confirmed']),
                'unconfirmed_ingredients_count': len(categorized['unconfirmed']),
                'confirmed_ingredients': ','.join(categorized['confirmed']),
                'unconfirmed_ingredients': ','.join(categorized['unconfirmed']),
                'all_ingredients': ','.join(ingredients),
            }
            
            processed_products.append(product_data)
            
            # 기획 상품인 경우 별도 처리
            if is_package:
                package_products = self.split_package_product(product_name, ingredients)
                for pkg_product in package_products:
                    # 추가 검토용 데이터 저장
                    pass
        
        df = pd.DataFrame(processed_products)
        print(f"   처리 완료: {len(df)}개 제품")
        
        return df
    
    def process_all_files(self, output_file: str = 'data/processed/oliveyoung_products_cleaned.csv'):
        """
        모든 올리브영 JSON 파일 처리
        
        Args:
            output_file: 출력 파일 경로
        """
        print("=" * 60)
        print("🧴 올리브영 제품 데이터 전처리 시작")
        print("=" * 60)
        
        # 1. 전체 성분 데이터 로드
        self.load_public_api_ingredients()
        
        # 2. JSON 파일 목록
        json_files = [
            ('data/raw/oliveyoung_스킨_토너_raw_limited.json', '스킨_토너'),
            ('data/raw/oliveyoung_에센스_세럼_앰플_raw_limited.json', '에센스_세럼_앰플'),
            ('data/raw/oliveyoung_크림_raw_limited.json', '크림'),
            ('data/raw/oliveyoung_로션_raw_limited.json', '로션'),
            ('data/raw/oliveyoung_미스트_오일_raw_limited.json', '미스트_오일'),
        ]
        
        # 3. 각 파일 처리
        all_dataframes = []
        
        for file_path, category in json_files:
            if Path(file_path).exists():
                df = self.process_json_file(file_path, category)
                all_dataframes.append(df)
            else:
                print(f"⚠️ 파일 없음: {file_path}")
        
        # 4. 모든 데이터 통합
        if all_dataframes:
            combined_df = pd.concat(all_dataframes, ignore_index=True)
            
            # 5. 통계 출력
            print("\n" + "=" * 60)
            print("📊 전처리 결과 통계")
            print("=" * 60)
            print(f"총 제품 수: {len(combined_df)}")
            print(f"\n카테고리별:")
            print(combined_df['category'].value_counts())
            print(f"\n기획 상품 수: {combined_df['is_package'].sum()}개")
            print(f"\n성분 매칭률:")
            print(f"  - 전체 성분: {combined_df['total_ingredients'].sum():,}")
            print(f"  - 확인된 성분: {combined_df['confirmed_ingredients_count'].sum():,}")
            print(f"  - 미확인 성분: {combined_df['unconfirmed_ingredients_count'].sum():,}")
            match_rate = (combined_df['confirmed_ingredients_count'].sum() / 
                         combined_df['total_ingredients'].sum() * 100) if combined_df['total_ingredients'].sum() > 0 else 0
            print(f"  - 매칭률: {match_rate:.1f}%")
            
            # 6. CSV 저장
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            combined_df.to_csv(output_file, index=False, encoding='utf-8-sig')
            print(f"\n✅ 저장 완료: {output_file}")
            
            # 7. 미확인 성분 별도 저장
            unconfirmed_file = output_file.replace('.csv', '_unconfirmed.csv')
            unconfirmed_df = combined_df[combined_df['unconfirmed_ingredients_count'] > 0].copy()
            if len(unconfirmed_df) > 0:
                unconfirmed_df.to_csv(unconfirmed_file, index=False, encoding='utf-8-sig')
                print(f"✅ 미확인 성분 저장: {unconfirmed_file}")
            
            # 8. 기획 상품 별도 저장
            package_file = output_file.replace('.csv', '_packages.csv')
            package_df = combined_df[combined_df['is_package'] == True].copy()
            if len(package_df) > 0:
                package_df.to_csv(package_file, index=False, encoding='utf-8-sig')
                print(f"✅ 기획 상품 저장: {package_file}")
            
            print("\n" + "=" * 60)
            print("🎉 전처리 완료!")
            print("=" * 60)
            
            return combined_df
        else:
            print("❌ 처리할 데이터가 없습니다.")
            return None


def main():
    """메인 실행 함수"""
    preprocessor = OliveYoungDataPreprocessor()
    result = preprocessor.process_all_files()
    
    if result is not None:
        print("\n📝 다음 단계:")
        print("1. 미확인 성분 파일을 확인하여 수동으로 정리")
        print("2. 기획 상품 파일을 확인하여 수동으로 분리")
        print("3. 전체 성분 데이터베이스 업데이트")


if __name__ == "__main__":
    main()
