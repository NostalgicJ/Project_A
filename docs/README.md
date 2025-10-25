# 🧴 화장품 성분 조합 분석 딥러닝 프로젝트

AI 기반 화장품 성분 조합 분석 및 추천 시스템

## 🎯 프로젝트 개요

이 프로젝트는 딥러닝을 활용하여 화장품 성분 조합의 안전성과 시너지 효과를 분석하고, 사용자에게 맞춤형 추천을 제공하는 웹 애플리케이션입니다.

### 주요 기능
- 🔍 **화장품 검색**: 브랜드명, 제품명으로 화장품 검색
- 🧪 **성분 조합 분석**: AI가 분석하는 안전성 및 시너지 효과
- ⚠️ **안전성 검사**: 사용하면 안 되는 성분 조합 경고
- ✨ **시너지 추천**: 함께 사용하면 좋은 성분 조합 추천
- 📊 **시각화**: 분석 결과를 직관적으로 표시

## 🏗️ 시스템 아키텍처

```
📁 Project_A/
├── 📁 src/                    # 소스 코드
│   ├── 📁 data/               # 데이터 처리
│   │   └── data_processor.py  # 데이터 전처리
│   ├── 📁 models/             # 딥러닝 모델
│   │   └── ingredient_analyzer.py  # 성분 분석 모델
│   ├── 📁 api/                # API 서버
│   │   └── main.py            # FastAPI 서버
│   └── 📁 frontend/           # 웹 인터페이스
│       └── index.html         # React 웹앱
├── 📁 data/                   # 데이터 파일
├── 📁 models/                 # 학습된 모델
├── 📁 notebooks/              # Jupyter 노트북
├── requirements.txt           # Python 패키지
├── run_server.py             # 서버 실행 스크립트
└── README.md                 # 프로젝트 문서
```

## 🚀 빠른 시작

### 1. 환경 설정

```bash
# 프로젝트 클론
git clone <repository-url>
cd Project_A

# 가상환경 생성 (선택사항)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 또는
venv\Scripts\activate     # Windows

# 패키지 설치
pip install -r requirements.txt
```

### 2. 서버 실행

```bash
# 서버 실행
python run_server.py
```

### 3. 웹 인터페이스 접속

- **웹 애플리케이션**: http://localhost:8000
- **API 문서**: http://localhost:8000/docs
- **헬스 체크**: http://localhost:8000/health

## 📊 데이터 구조

### 입력 데이터
- `processed_cosmetics_final_2.csv`: 2,169개 화장품 제품 정보
- `integrated_product_ingredient_normalized_2.csv`: 92,809개 성분 레코드
- `coos_master_ingredients_cleaned.csv`: 25,426개 정제된 성분명

### 처리된 데이터
- `ingredient_matrix.csv`: 성분-제품 매트릭스
- `ingredient_vocab.pkl`: 성분 어휘 사전
- `ingredient_embeddings.npy`: 성분 임베딩 벡터

## 🧠 딥러닝 모델

### 1. 성분 임베딩 모델
- **목적**: 화장품 성분을 벡터로 변환
- **기법**: Word2Vec 또는 BERT 기반 임베딩
- **출력**: 128차원 성분 벡터

### 2. 조합 분석 모델
- **아키텍처**: Transformer + CNN
- **입력**: 성분 조합 벡터
- **출력**: 
  - 안전성 점수 (0-1)
  - 시너지 점수 (0-1)
  - 분류 결과 (안전/주의/위험)

### 3. 추천 모델
- **기법**: Graph Neural Network (GNN)
- **입력**: 사용자 선택 화장품들
- **출력**: 추천 화장품 및 조합 분석

## 🔬 성분 조합 분석 규칙

### ⚠️ 안전하지 않은 조합
1. **비타민C + 레티놀**: 산화 반응으로 효과 상쇄
2. **AHA/BHA + 레티놀**: 과도한 각질 제거
3. **니아신아마이드 + 비타민C**: pH 불일치
4. **벤조일퍼옥사이드 + 레티놀**: 과도한 각질 제거

### ✨ 시너지 조합
1. **비타민C + 비타민E**: 항산화 효과 증대
2. **히알루론산 + 세라마이드**: 보습 효과 증대
3. **나이아신아마이드 + 아연**: 모공 관리 효과 증대
4. **레티놀 + 하이드로퀴논**: 미백 효과 증대
5. **펩타이드 + 레티놀**: 주름 개선 효과 증대

## 🛠️ API 엔드포인트

### 제품 검색
```http
POST /search/products
Content-Type: application/json

{
  "query": "토리든",
  "limit": 10
}
```

### 성분 조합 분석
```http
POST /analyze/ingredients
Content-Type: application/json

{
  "ingredients": ["비타민C", "레티놀", "히알루론산"]
}
```

### 제품 조합 분석
```http
POST /analyze/products
Content-Type: application/json

{
  "product_ids": ["토리든_다이브인저분자히알루론산세럼", "메디힐_마데카소사이드리페어세럼"]
}
```

### 성분 추천
```http
POST /recommend/ingredients
Content-Type: application/json

{
  "ingredients": ["비타민C", "히알루론산"]
}
```

## 📈 성능 지표

- **응답 시간**: API 응답 < 2초
- **정확도**: 조합 분석 정확도 > 85%
- **동시 사용자**: 100명 동시 접속 지원
- **가용성**: 99.9% 업타임

## 🔒 보안 고려사항

- **데이터 암호화**: 민감한 사용자 데이터 암호화
- **API 인증**: JWT 토큰 기반 인증 (향후 구현)
- **입력 검증**: 사용자 입력 데이터 검증
- **HTTPS**: SSL/TLS 암호화 통신 (프로덕션)

## 🚀 배포 (AWS EC2)

### 1. EC2 인스턴스 설정
```bash
# Ubuntu 20.04 LTS 인스턴스 생성
# t3.medium 이상 권장 (2 vCPU, 4GB RAM)

# 시스템 업데이트
sudo apt update && sudo apt upgrade -y

# Python 3.9 설치
sudo apt install python3.9 python3.9-pip python3.9-venv -y

# Git 설치
sudo apt install git -y
```

### 2. 애플리케이션 배포
```bash
# 프로젝트 클론
git clone <repository-url>
cd Project_A

# 가상환경 생성
python3.9 -m venv venv
source venv/bin/activate

# 패키지 설치
pip install -r requirements.txt

# 서버 실행
python run_server.py
```

### 3. Nginx 설정 (선택사항)
```nginx
server {
    listen 80;
    server_name your-domain.com;
    
    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## 🧪 테스트

### 단위 테스트
```bash
# 테스트 실행
pytest tests/

# 커버리지 확인
pytest --cov=src tests/
```

### API 테스트
```bash
# API 문서 확인
curl http://localhost:8000/docs

# 헬스 체크
curl http://localhost:8000/health
```

## 📝 개발 로그

### v1.0.0 (2024-01-XX)
- ✅ 기본 데이터 전처리 파이프라인 구현
- ✅ 규칙 기반 성분 조합 분석
- ✅ FastAPI 백엔드 구현
- ✅ React 웹 인터페이스 구현
- ✅ 기본 API 엔드포인트 구현

### 향후 계획
- 🔄 딥러닝 모델 훈련 및 최적화
- 🔄 사용자 피드백 수집 시스템
- 🔄 모바일 앱 개발
- 🔄 다국어 지원

## 🤝 기여하기

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 `LICENSE` 파일을 참조하세요.

## 📞 문의

프로젝트에 대한 문의사항이 있으시면 이슈를 생성해주세요.

---

**🎯 화장품 성분 조합 분석기** - AI로 더 안전하고 효과적인 화장품 사용법을 찾아보세요!
