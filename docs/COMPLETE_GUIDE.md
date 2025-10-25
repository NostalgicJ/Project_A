# 🧴 화장품 성분 조합 분석 프로젝트 - 완전 가이드

## 📚 목차

1. [프로젝트 개요](#1-프로젝트-개요)
2. [데이터 수집](#2-데이터-수집)
3. [데이터 전처리](#3-데이터-전처리)
4. [모델 학습](#4-모델-학습)
5. [서비스 실행](#5-서비스-실행)

## 1. 프로젝트 개요

화장품 성분 조합의 안전성과 시너지를 분석하는 AI 시스템

### 주요 기능
- ✅ 화장품 제품 검색
- ✅ 성분 조합 분석
- ✅ 제품 간 조합 비교
- ✅ 위험 조합 경고
- ✅ 시너지 추천

## 2. 데이터 수집

### 2-1. 공공데이터 API로 전체 성분 데이터 다운로드

```bash
# API 키로 성분 데이터 다운로드 (최초 1회만)
python scripts/download_public_api_ingredients.py
```

**출력:**
- `data/raw/public_ingredients.csv`: 전체 성분 데이터

**특징:**
- 최초 다운로드 후 로컬 저장
- 파일이 있으면 재다운로드 안함
- 필요시 수동으로 업데이트

### 2-2. 올리브영 제품 데이터 (이미 스크래핑 완료)

`data/raw/oliveyoung_*.json` 파일들

## 3. 데이터 전처리

### 3-1. 전처리 실행

```bash
# 올리브영 제품 데이터 전처리
python scripts/preprocess_oliveyoung_data.py
```

**출력:**
- `oliveyoung_products_cleaned.csv`: 정제된 제품 데이터
- `*_packages.csv`: 기획 상품만
- `*_unconfirmed.csv`: 미확인 성분 제품만

### 3-2. 기획 상품 수동 검토 및 분리

```bash
python scripts/manual_review_packages.py
```

**기능:**
- 기획 상품을 개별 제품으로 분리
- 브랜드명, 제품명, 성분 리스트 입력
- 전체 데이터에 자동 반영

**작업 종류:**
- `s`: 제품 분리
- `r`: 제품 제거
- `k`: 제품 유지
- `n`: 다음으로

### 3-3. 미확인 성분 수동 수정

```bash
python scripts/manual_review_unconfirmed_ingredients.py
```

**기능:**
- 미확인 성분을 올바른 성분명으로 수정
- 통계 확인 (출현 빈도)
- 전체 데이터에 자동 반영

### 3-4. 성분 규칙 관리

```bash
python scripts/manage_ingredient_rules.py
```

**기능:**
- 성분 계열 추가 (예: 비타민C계열, 레티놀계열)
- 위험한 조합 규칙 추가
- 시너지 조합 규칙 추가

**예시:**
```
성분 계열 추가
  계열명: 비타민C계열
  성분 리스트: 아스코빅애씨드, 소듐아스코빌포스페이트, ...

위험한 조합 추가
  계열 1: 비타민C계열
  계열 2: 레티놀계열
  위험도: 0.9
  이유: pH 불일치로 효과 상쇄
```

## 4. 모델 학습

### 4-1. 모델 구조

#### 규칙 기반 모델 (빠름, 정확도 보통)
- 성분 계열 기반 매칭
- 위험 조합 규칙 적용

#### 딥러닝 모델 (느림, 정확도 높음)
- 성분 쌍 임베딩
- 상호작용 분석
- 위험도/시너지 예측

### 4-2. 모델 학습

```bash
python scripts/train_dl_model.py
```

### 4-3. 모델 테스트

```bash
python scripts/test_dl_model.py
```

### 4-4. 제품 간 조합 분석 (새 모델)

제품 A의 성분 20개와 제품 B의 성분 30개를 비교하여
위험한 조합을 찾아냅니다.

**모델 구조:**
```
제품 A 성분 리스트 → 임베딩 → 제품 A 표현
제품 B 성분 리스트 → 임베딩 → 제품 B 표현
                    ↓
              상호작용 분석
                    ↓
            위험도 + 시너지 점수
```

**사용 방법:**
```python
from src.models.product_combination_analyzer import ProductIngredientMatcher

matcher = ProductIngredientMatcher()

product_a_ingredients = ['비타민C', '히알루론산', ...]
product_b_ingredients = ['레티놀', '세라마이드', ...]

result = matcher.analyze_product_pair(
    product_a_ingredients, 
    product_b_ingredients
)

print(matcher.format_analysis_result(result))
```

## 5. 서비스 실행

### 5-1. API 서버 실행

```bash
python scripts/run_server.py
```

**접속:**
- 웹 UI: http://localhost:8000
- API 문서: http://localhost:8000/docs

### 5-2. API 엔드포인트

- `POST /search/products`: 제품 검색
- `POST /analyze/ingredients`: 성분 조합 분석
- `POST /analyze/products`: 제품 조합 분석
- `POST /recommend/ingredients`: 성분 추천

## 🔄 전체 워크플로우

```
1. 공공데이터 API 다운로드 (최초 1회)
   python scripts/download_public_api_ingredients.py
   
2. 제품 데이터 전처리
   python scripts/preprocess_oliveyoung_data.py
   
3. 기획 상품 검토 및 분리
   python scripts/manual_review_packages.py
   
4. 미확인 성분 수정
   python scripts/manual_review_unconfirmed_ingredients.py
   
5. 성분 규칙 설정
   python scripts/manage_ingredient_rules.py
   
6. 모델 학습 (선택)
   python scripts/train_dl_model.py
   
7. 서버 실행
   python scripts/run_server.py
```

## 📋 주요 스크립트 요약

| 스크립트 | 용도 | 실행 빈도 |
|---------|------|----------|
| `download_public_api_ingredients.py` | 공공데이터 API 다운로드 | 최초 1회 |
| `preprocess_oliveyoung_data.py` | 제품 데이터 전처리 | 데이터 변경 시 |
| `manual_review_packages.py` | 기획 상품 검토 | 필요시 |
| `manual_review_unconfirmed_ingredients.py` | 미확인 성분 수정 | 필요시 |
| `manage_ingredient_rules.py` | 성분 규칙 관리 | 필요시 |
| `train_dl_model.py` | 모델 학습 | 모델 개선 시 |
| `test_dl_model.py` | 모델 테스트 | 모델 평가 시 |
| `run_server.py` | 서버 실행 | 항상 |

## 📁 출력 파일 구조

```
data/processed/
├── oliveyoung_products_cleaned.csv           # 최종 제품 데이터
├── oliveyoung_products_cleaned_packages.csv  # 기획 상품
├── oliveyoung_products_cleaned_unconfirmed.csv # 미확인 성분
├── package_review_changes.json              # 기획 상품 수정 내역
└── ingredient_review_changes.json           # 성분 수정 내역

config/
└── ingredient_rules.json                    # 성분 규칙

models/
└── advanced_ingredient_analyzer.pth         # 학습된 모델
```

## 💡 추가 팁

1. **데이터 업데이트**: 공공데이터가 업데이트되면 다시 다운로드
2. **규칙 추가**: 새로운 위험 조합 발견 시 규칙 추가
3. **모델 재학습**: 데이터 증가 시 모델 재학습
4. **백업**: 중요 수정 전 데이터 백업

## 📞 참고 문서

- [데이터 전처리 가이드](DATA_PREPROCESSING_GUIDE.md)
- [데이터 검토 가이드](DATA_REVIEW_GUIDE.md)
- [모델 학습 가이드](MODEL_TRAINING_GUIDE.md)
- [API 통합 가이드](PUBLIC_API_INTEGRATION.md)
- [프로젝트 구조](PROJECT_STRUCTURE.md)
