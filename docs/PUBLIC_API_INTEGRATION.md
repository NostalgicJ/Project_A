# 공공데이터 API 통합 가이드

## 📋 개요

공공데이터 포털 API를 사용하여 화장품 원료 성분 데이터를 다운로드하는 방법을 설명합니다.

## 🔑 API 키

```
일반 인증키 (Decoding):
50hjvXuloV4qFNdrIUOglZZ6RGV7uq7pvpP0oxT+EV57bvEGnWfvqbjL939z/yfj9ta/H2Cn382mGmHpm4wmcw==
```

## 🚀 사용 방법

### 1. API 키로 성분 데이터 다운로드

```bash
python scripts/download_public_api_ingredients.py
```

### 2. 전처리 스크립트 실행

```bash
python scripts/preprocess_oliveyoung_data.py
```

## 📊 데이터 구조

### 다운로드된 데이터 (`public_ingredients.csv`)

| 컬럼명 | 설명 |
|--------|------|
| 한글명 | 원료의 한글 이름 |
| 영문명 | 원료의 INCI 명 |
| CAS번호 | CAS 등록번호 |
| 용도 | 원료 용도 |
| 제한사항 | 사용 제한 여부 |
| 농도제한 | 농도 제한 내용 |
| 주의사항 | 사용시 주의사항 |
| 비고 | 기타 정보 |

## 🔄 데이터 흐름

```
공공데이터 API
    ↓
다운로드 (download_public_api_ingredients.py)
    ↓
public_ingredients.csv
    ↓
전처리 (preprocess_oliveyoung_data.py)
    ↓
oliveyoung_products_cleaned.csv
```

## 💡 참고

- API 키는 `scripts/download_public_api_ingredients.py`에 포함되어 있습니다.
- API 호출 제한이 있을 수 있으니 주의하세요.
- 다운로드한 데이터는 `data/raw/` 디렉토리에 저장됩니다.
