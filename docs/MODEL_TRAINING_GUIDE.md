# 모델 학습 가이드

## 📋 개요

이 문서는 화장품 성분 조합 분석 딥러닝 모델의 구조와 학습 방법을 설명합니다.

## 🏗️ 모델 아키텍처

### 전체 구조

```
성분 쌍 (ingredient1, ingredient2)
    ↓
임베딩 레이어 (Embedding)
    ↓
성분1 임베딩 + 성분2 임베딩
    ↓
상호작용 분석 네트워크 (MLP)
    ↓
분류 헤드 + 위험도 점수 + 시너지 점수
    ↓
출력 (분류, 위험도, 시너지)
```

### 모델 계층 구조

```python
IngredientInteractionModel(
    embedding_dim=128,      # 임베딩 차원
    hidden_dim=256,         # 히든 레이어 차원
    
    # 레이어 구성:
    ingredient_embedding: nn.Embedding    # 성분 → 벡터
    interaction_net:      Sequential       # 상호작용 분석
        - Linear + ReLU + Dropout (x3)
    classifier:           Linear           # 분류 (3개 클래스)
    danger_score:         Linear + Sigmoid # 위험도 (0-1)
    synergy_score:        Linear + Sigmoid # 시너지 (0-1)
)
```

## 📊 데이터 흐름

### 1. 입력 데이터
- **성분 쌍**: (ingredient1, ingredient2)
- **라벨**: 0(안전), 1(주의), 2(위험)

### 2. 전처리
- 성분명 → 인덱스 매핑
- 어휘 사전 구축

### 3. 모델 학습
- 임베딩 학습
- 상호작용 패턴 학습
- 분류 및 점수 예측

### 4. 출력
- 분류: 안전/주의/위험
- 점수: 위험도(0-1), 시너지(0-1)

## 🚀 학습 방법

### 방법 1: 스크립트로 학습

```bash
# 모델 학습 실행
python scripts/train_dl_model.py
```

### 방법 2: Python 코드로 학습

```python
from models.advanced_ingredient_analyzer import AdvancedCosmeticAnalyzer
import pandas as pd

# 1. 데이터 로드
products_df = pd.read_csv('data/processed/processed_cosmetics_final_2.csv')
vocab = [...]  # 성분 어휘 사전

# 2. 학습 데이터 생성
analyzer = AdvancedCosmeticAnalyzer()
ingredient_pairs, labels = analyzer.create_training_data(vocab)

# 3. 모델 학습
model = analyzer.train_model(vocab, num_epochs=50)

# 4. 모델 저장
analyzer.save_model('models/advanced_ingredient_analyzer.pth')
```

### 방법 3: Jupyter 노트북으로 학습

`notebooks/model_visualization_and_training.ipynb` 참조

## 📈 학습 과정 단계

### Step 1: 환경 설정
```bash
pip install -r requirements.txt
```

### Step 2: 데이터 준비
- 제품 데이터: `data/processed/processed_cosmetics_final_2.csv`
- 어휘 사전: `data/processed/ingredient_vocab.pkl`

### Step 3: 학습 실행
```bash
python scripts/train_dl_model.py
```

### Step 4: 모델 저장
- 위치: `models/advanced_ingredient_analyzer.pth`
- 포함: 모델 가중치, 어휘 사전

### Step 5: 모델 평가
```bash
python scripts/test_dl_model.py
```

## 🔧 하이퍼파라미터

### 현재 설정
```python
{
    'embedding_dim': 128,      # 임베딩 차원
    'hidden_dim': 256,         # 히든 레이어 크기
    'num_epochs': 50,          # 학습 에포크
    'batch_size': 32,          # 배치 크기
    'learning_rate': 0.001,    # 학습률
    'dropout': 0.3             # 드롭아웃 비율
}
```

### 튜닝 방법
- `embedding_dim`: 성분 표현력 (64-256)
- `hidden_dim`: 모델 복잡도 (128-512)
- `learning_rate`: 학습 속도 (0.0001-0.01)
- `num_epochs`: 학습 반복 (30-100)

## 📊 학습 모니터링

### 손실 함수
- 분류: CrossEntropyLoss
- 회귀: MSE Loss

### 평가 지표
- 정확도 (Accuracy)
- F1 Score
- ROC AUC

### 학습 곡선
```
Epoch 0: Loss = 0.8234
Epoch 10: Loss = 0.4567
Epoch 20: Loss = 0.2345
...
Epoch 50: Loss = 0.1234  (수렴)
```

## 🎯 사용 방법

### 모델 로드
```python
from models.advanced_ingredient_analyzer import AdvancedCosmeticAnalyzer

analyzer = AdvancedCosmeticAnalyzer()
analyzer.load_model('models/advanced_ingredient_analyzer.pth')
```

### 성분 조합 분석
```python
ingredients = ['비타민C', '레티놀', '히알루론산']
result = analyzer.analyze_combination(ingredients)

print(f"분류: {result['predicted_class']}")
print(f"위험도: {result['danger_score']:.1%}")
print(f"시너지: {result['synergy_score']:.1%}")
```

## 🐛 문제 해결

### 학습이 수렴하지 않는 경우
- 학습률 낮추기
- 배치 크기 조정
- 학습 데이터 품질 확인

### 메모리 부족
- batch_size 감소
- embedding_dim 감소
- GPU 사용 고려

### 정확도가 낮은 경우
- 학습 데이터 증가
- 모델 크기 증가
- 특징 엔지니어링

## 📝 참고

- 모델 코드: `src/models/advanced_ingredient_analyzer.py`
- 학습 스크립트: `scripts/train_dl_model.py`
- 테스트 스크립트: `scripts/test_dl_model.py`
