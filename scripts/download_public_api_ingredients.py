#!/usr/bin/env python3
"""
공공데이터 포털 API를 사용하여 화장품 원료 성분 데이터 다운로드

API: 식품의약품안전처 화장품 원료정보
"""

import requests
import pandas as pd
import time
from pathlib import Path
import json

class PublicAPIHandler:
    """공공데이터 API 핸들러"""
    
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "http://apis.data.go.kr/1471000/GoodAtcpInfoService02"
        
    def get_all_ingredients(self, save_path="data/raw/public_ingredients.csv"):
        """
        전체 원료 성분 데이터 다운로드
        
        Args:
            save_path: 저장 경로
        """
        print("📡 공공데이터 포털 API에서 원료 성분 데이터 다운로드 중...")
        print("=" * 60)
        
        all_ingredients = []
        page = 1
        per_page = 100  # 한 번에 가져올 데이터 수
        
        try:
            while True:
                # API 엔드포인트: 원료별 정보 조회
                url = f"{self.base_url}/getGoodAtcpInfoService02"
                
                params = {
                    'serviceKey': self.api_key,
                    'pageNo': page,
                    'numOfRows': per_page,
                    'type': 'json'
                }
                
                print(f"  📄 페이지 {page} 요청 중...", end=' ')
                
                response = requests.get(url, params=params, timeout=30)
                
                if response.status_code != 200:
                    print(f"❌ 에러: HTTP {response.status_code}")
                    break
                
                data = response.json()
                
                # 응답 구조 확인
                if 'body' not in data or 'items' not in data['body']:
                    print("⚠️ 더 이상 데이터가 없습니다.")
                    break
                
                items = data['body']['items']
                
                if not items:
                    print("완료")
                    break
                
                # 데이터 추출
                for item in items:
                    ingredient = {
                        '한글명': item.get('INCI_NM_KO', ''),
                        '영문명': item.get('INCI_NM', ''),
                        'CAS번호': item.get('CAS_NO', ''),
                        '용도': item.get('USAGE', ''),
                        '제한사항': item.get('LIMIT_YN', ''),
                        '농도제한': item.get('LIMIT_CONTENT', ''),
                        '주의사항': item.get('NOTICE_ITEM', ''),
                        '비고': item.get('REMARK', ''),
                    }
                    all_ingredients.append(ingredient)
                
                print(f"✅ {len(items)}개 데이터 수신 (누적: {len(all_ingredients)}개)")
                
                # API 호출 제한 대기
                time.sleep(0.5)
                page += 1
                
                # 최대 페이지 제한 (안전장치)
                if page > 100:
                    print("⚠️ 최대 페이지 제한 도달 (100페이지)")
                    break
            
            # 데이터프레임 생성
            if all_ingredients:
                df = pd.DataFrame(all_ingredients)
                
                # CSV 저장
                Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(save_path, index=False, encoding='utf-8-sig')
                
                print("\n" + "=" * 60)
                print("📊 다운로드 결과")
                print("=" * 60)
                print(f"총 원료 수: {len(df)}개")
                print(f"저장 위치: {save_path}")
                print("\n데이터 샘플:")
                print(df.head())
                
                # JSON으로도 저장 (임베딩용)
                json_path = save_path.replace('.csv', '.json')
                df.to_json(json_path, orient='records', force_ascii=False, indent=2)
                print(f"\nJSON 저장: {json_path}")
                
                return df
            else:
                print("❌ 데이터를 가져오지 못했습니다.")
                return None
                
        except Exception as e:
            print(f"\n❌ 오류 발생: {e}")
            return None


def main():
    """메인 실행 함수"""
    # API 키 (Decoding)
    api_key = "50hjvXuloV4qFNdrIUOglZZ6RGV7uq7pvpP0oxT+EV57bvEGnWfvqbjL939z/yfj9ta/H2Cn382mGmHpm4wmcw=="
    
    print("🧪 공공데이터 포털 화장품 원료 정보 다운로드")
    print("=" * 60)
    
    # 이미 파일이 있는지 확인
    save_path = "data/raw/public_ingredients.csv"
    if Path(save_path).exists():
        print(f"⚠️ 이미 다운로드된 파일이 존재합니다: {save_path}")
        response = input("다시 다운로드하시겠습니까? (y/n): ")
        if response.lower() != 'y':
            print("기존 파일을 사용합니다.")
            return
    
    # API 핸들러 초기화
    handler = PublicAPIHandler(api_key)
    
    # 데이터 다운로드
    df = handler.get_all_ingredients(save_path)
    
    if df is not None:
        print("\n✅ 완료!")
        print("\n다음 단계:")
        print("1. 다운로드한 데이터 검토")
        print("2. 전처리 스크립트 실행: python scripts/preprocess_oliveyoung_data.py")
    else:
        print("\n❌ 다운로드 실패")


if __name__ == "__main__":
    main()
