# 프로젝트 구조 설명

## 📁 전체 구조

```
Project_A/
├── src/                      # 소스 코드
│   ├── api/                  # FastAPI 백엔드
│   │   └── main.py           # API 서버 메인
│   ├── data/                 # 데이터 처리
│   │   └── data_processor.py # 데이터 전처리
│   ├── models/               # 딥러닝 모델
│   │   ├── ingredient_analyzer.py           # 규칙 기반 분석기
│   │   └── advanced_ingredient_analyzer.py  # 딥러닝 분석기
│   ├── pipeline/             # 데이터 파이프라인
│   │   ├── build_master_index.py   # 마스터 인덱스 생성
│   │   ├── ingredient_parser.py    # 성분 파서
│   │   └── run_parse_products.py   # 제품 파싱 실행
│   └── frontend/             # 웹 인터페이스
│       └── index.html        # React 웹앱
│
├── scripts/                  # 실행 스크립트
│   ├── run_server.py         # 서버 실행
│   ├── train_dl_model.py     # 딥러닝 모델 훈련
│   ├── test_dl_model.py      # 모델 테스트
│   ├── model_usage_example.py
│   ├── correct_approach.py
│   └── real_*.py
│
├── data/                     # 데이터 파일
│   ├── raw/                  # 원본 데이터
│   └── processed/            # 전처리된 데이터
│
├── models/                   # 학습된 모델
│   └── *.pth                 # PyTorch 모델 파일
│
├── notebooks/                # Jupyter 노트북
│   ├── data_analysis/
│   └── model_training/
│
├── docs/                     # 문서
│   ├── README.md
│   ├── quick_start.md
│   ├── architecture_design.md
│   └── deploy/               # 배포 설정
│       ├── Dockerfile
│       ├── docker-compose.yml
│       ├── nginx.conf
│       └── deploy_aws.sh
│
├── config/                   # 설정 파일
├── static/                   # 정적 파일
├── logs/                     # 로그 파일
├── requirements.txt          # Python 패키지
└── .gitignore               # Git 제외 파일

```

## 📝 주요 디렉토리 설명

### `src/` - 소스 코드
- **api/**: FastAPI 백엔드 서버 코드
- **data/**: 데이터 전처리 및 로드 로직
- **models/**: 성분 분석 딥러닝 모델
- **pipeline/**: 데이터 파이프라인 (성분 파싱, 인덱싱 등)
- **frontend/**: 웹 인터페이스 (HTML/CSS/JS)

### `scripts/` - 실행 스크립트
프로젝트 실행에 필요한 모든 스크립트들

### `data/`
- **raw/**: 원본 CSV, JSON 파일들
- **processed/**: 전처리된 데이터 (매트릭스, 임베딩, 파싱 결과 등)

### `models/`
학습된 딥러닝 모델 파일 (.pth)

### `docs/`
프로젝트 문서 및 배포 설정

## 🚀 주요 실행 방법

### 1. 서버 실행
```bash
python scripts/run_server.py
```

### 2. 모델 훈련
```bash
python scripts/train_dl_model.py
```

### 3. 모델 테스트
```bash
python scripts/test_dl_model.py
```

### 4. 웹 인터페이스 접속
- URL: http://localhost:8000
- API 문서: http://localhost:8000/docs

## 🔧 데이터 경로 설정

모든 데이터 경로는 `data/` 디렉토리를 기준으로 설정되어 있습니다:
- 원본 데이터: `data/raw/`
- 전처리 데이터: `data/processed/`
- 모델: `models/`

## 📦 배포

Docker를 사용한 배포:
```bash
cd docs/deploy
docker-compose up -d
```

## 🔍 파일 역할 요약

### API 서버
- `src/api/main.py`: FastAPI 메인 애플리케이션

### 데이터 처리
- `src/data/data_processor.py`: 데이터 전처리 파이프라인
- `src/pipeline/`: 성분 파싱 및 인덱싱

### 모델
- `src/models/ingredient_analyzer.py`: 규칙 기반 분석기
- `src/models/advanced_ingredient_analyzer.py`: 딥러닝 분석기

### 실행 스크립트
- `scripts/run_server.py`: 서버 실행
- `scripts/train_dl_model.py`: 모델 훈련
- `scripts/test_dl_model.py`: 모델 테스트

## 📚 문서
- `docs/README.md`: 프로젝트 개요
- `docs/quick_start.md`: 빠른 시작 가이드
- `docs/architecture_design.md`: 아키텍처 설계

