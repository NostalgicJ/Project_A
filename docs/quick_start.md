# 🚀 빠른 시작 가이드

## 로컬 개발 환경에서 실행하기

### 1. 환경 설정
```bash
# 프로젝트 디렉토리로 이동
cd /Users/yeojung/Desktop/github/Project_A

# 가상환경 활성화 (이미 생성되어 있음)
source pyenv/cosmetics-fix/bin/activate

# 또는 새로운 가상환경 생성
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate     # Windows
```

### 2. 패키지 설치
```bash
# 필요한 패키지 설치
pip install -r requirements.txt
```

### 3. 데이터 전처리 (선택사항)
```bash
# 데이터 전처리 실행
python src/data/data_processor.py
```

### 4. 서버 실행
```bash
# 서버 실행
python run_server.py
```

### 5. 웹 애플리케이션 접속
- **웹 애플리케이션**: http://localhost:8000
- **API 문서**: http://localhost:8000/docs
- **헬스 체크**: http://localhost:8000/health

## 🧪 테스트 방법

### 1. API 테스트
```bash
# 헬스 체크
curl http://localhost:8000/health

# 제품 검색
curl -X POST "http://localhost:8000/search/products" \
     -H "Content-Type: application/json" \
     -d '{"query": "토리든", "limit": 5}'

# 성분 조합 분석
curl -X POST "http://localhost:8000/analyze/ingredients" \
     -H "Content-Type: application/json" \
     -d '{"ingredients": ["비타민C", "레티놀", "히알루론산"]}'
```

### 2. 웹 인터페이스 테스트
1. 브라우저에서 http://localhost:8000 접속
2. 화장품 검색 (예: "토리든", "메디힐")
3. 제품 선택 후 조합 분석 실행
4. 결과 확인

## 🔧 문제 해결

### 서버가 시작되지 않는 경우
```bash
# 포트 확인
lsof -i :8000

# 프로세스 종료
kill -9 $(lsof -t -i:8000)

# 다시 시작
python run_server.py
```

### 패키지 설치 오류
```bash
# pip 업그레이드
pip install --upgrade pip

# 개별 패키지 설치
pip install fastapi uvicorn pandas numpy torch
```

### 데이터 로드 오류
```bash
# 데이터 파일 확인
ls -la *.csv

# 데이터 전처리 강제 실행
python -c "from src.data.data_processor import CosmeticDataProcessor; processor = CosmeticDataProcessor(); processor.process_all()"
```

## 📊 성능 모니터링

### 시스템 리소스 확인
```bash
# CPU 및 메모리 사용량
htop

# 디스크 사용량
df -h

# 네트워크 연결
netstat -tulpn | grep :8000
```

### 로그 확인
```bash
# 서버 로그 (터미널에서 확인)
# 또는 별도 로그 파일이 있다면
tail -f logs/server.log
```

## 🚀 프로덕션 배포

### Docker 사용
```bash
# Docker 이미지 빌드
docker build -t cosmetic-analyzer .

# 컨테이너 실행
docker run -p 8000:8000 cosmetic-analyzer

# Docker Compose 사용
docker-compose up -d
```

### AWS EC2 배포
```bash
# 배포 스크립트 실행
chmod +x deploy_aws.sh
./deploy_aws.sh
```

## 📝 개발 팁

### 코드 수정 후 재시작
```bash
# 서버 재시작
pkill -f "python run_server.py"
python run_server.py &
```

### 데이터베이스 초기화
```bash
# 데이터 파일 삭제 후 재생성
rm -rf data/ingredient_*.pkl data/ingredient_matrix.csv
python src/data/data_processor.py
```

### 모델 재훈련
```bash
# Jupyter 노트북 실행
jupyter notebook notebooks/model_training/train_ingredient_model.ipynb
```

## 🆘 도움말

문제가 발생하면 다음을 확인하세요:

1. **Python 버전**: Python 3.9 이상 필요
2. **포트 충돌**: 8000번 포트가 사용 중인지 확인
3. **데이터 파일**: CSV 파일들이 올바른 위치에 있는지 확인
4. **권한**: 파일 읽기/쓰기 권한 확인
5. **메모리**: 충분한 메모리가 있는지 확인 (최소 4GB 권장)

더 자세한 정보는 `README.md`를 참조하세요.



