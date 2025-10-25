# 데이터 전처리 가이드

## 📋 개요

이 문서는 화장품 성분 데이터 전처리 프로세스에 대해 설명합니다.

## 🔄 전처리 프로세스

### 1단계: 올리브영 원본 데이터 수집
- 스크래핑으로 수집한 JSON 파일
- 위치: `data/raw/oliveyoung_*.json`
- 형식: 제품명, URL, 성분리스트

### 2단계: 전체 성분 데이터 수집

#### 옵션 A: 공공데이터포털 API 사용 (권장)
```python
# API 호출 예시
api_url = "공공데이터API_URL"
response = requests.get(api_url)
data = response.json()
```

#### 옵션 B: COOS 데이터 사용 (임시)
- 현재: `data/raw/coos_master_ingredients_cleaned.csv`
- 25,426개 성분 데이터

### 3단계: 데이터 정제

#### 3-1. 제품명 정제
- 불필요한 키워드 제거: 기획, 증정, 올영픽, PICK 등
- 대괄호, 괄호 내용 제거
- 특수문자 정리

**처리 전:**
```
[10월 올영픽/화잘먹/미스트 기획] 라네즈 크림 스킨 170ml 리필기획 (+170ml 리필+50ml+미스트펌프)
```

**처리 후:**
```
라네즈 크림 스킨
```

#### 3-2. 브랜드/제품명 분리
- 첫 번째 단어 → 브랜드명
- 나머지 → 제품명

#### 3-3. 성분 정규화
각 성분을 전체 성분 데이터베이스와 매칭:
- ✅ 확인된 성분: 공공데이터에 존재
- ⚠️ 미확인 성분: 수동 검토 필요

#### 3-4. 기획 상품 처리
- 키워드 기반 감지: 더블, 트리플, 2개입 등
- 성분 수 기준: 50개 이상이면 의심
- 별도 파일로 저장하여 수동 검토

## 📊 출력 파일

### 메인 파일: `oliveyoung_products_cleaned.csv`
- `category`: 카테고리 (스킨_토너, 크림 등)
- `brand`: 브랜드명
- `product_name`: 정제된 제품명
- `original_name`: 원본 제품명
- `url`: 제품 URL
- `is_package`: 기획 상품 여부
- `total_ingredients`: 전체 성분 수
- `confirmed_ingredients`: 확인된 성분 (쉼표로 구분)
- `unconfirmed_ingredients`: 미확인 성분 (쉼표로 구분)

### 추가 파일
- `*_unconfirmed.csv`: 미확인 성분이 있는 제품만
- `*_packages.csv`: 기획 상품만

## 🚀 실행 방법

### 전처리 실행
```bash
python scripts/preprocess_oliveyoung_data.py
```

### 결과 확인
```bash
# 메인 결과
head data/processed/oliveyoung_products_cleaned.csv

# 미확인 성분
head data/processed/oliveyoung_products_cleaned_unconfirmed.csv

# 기획 상품
head data/processed/oliveyoung_products_cleaned_packages.csv
```

## 📝 다음 단계

### 1. 미확인 성분 수동 검토
1. `*_unconfirmed.csv` 파일 열기
2. 각 미확인 성분 검토
3. 올바른 성분명으로 수정하거나 새 성분 추가
4. 전체 성분 데이터베이스 업데이트

### 2. 기획 상품 수동 분리
1. `*_packages.csv` 파일 열기
2. 각 기획 상품의 실제 포함 제품 확인
3. 개별 제품으로 분리하거나 제외

### 3. 데이터베이스 업데이트
- 새로운 성분 데이터 추가
- 기존 데이터 품질 개선

## 🔍 품질 관리

### 성분 매칭률
- 목표: 90% 이상
- 현재 COOS 데이터 기준으로 계산

### 제품명 정확도
- 원본 제품명 유지
- 정제된 제품명과 비교 가능

### 기획 상품 처리
- 자동 감지 + 수동 검토
- 별도 파일로 관리

## 💡 팁

1. **단계별 확인**: 각 단계마다 결과를 확인하고 필요시 수정
2. **백업**: 원본 데이터는 항상 보관
3. **반복 개선**: 전처리 로직을 지속적으로 개선
4. **문서화**: 모든 변경사항을 문서로 기록

## 📞 지원

문제가 발생하면 이슈를 생성하거나 문서를 확인하세요.
