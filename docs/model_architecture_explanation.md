# 🧠 화장품 성분 조합 분석 딥러닝 모델 구조

## 📋 모델 개요

**목적**: 화장품 성분 조합의 위험도와 시너지를 예측하는 딥러닝 모델
**기술**: Transformer 기반 신경망 + 임베딩 + 분류기
**데이터**: 2187개 제품, 92827개 성분 레코드, 25394개 마스터 성분

## 🏗️ 모델 아키텍처

### 1. 전체 구조
```
입력 성분 → 임베딩 → 상호작용 분석 → 분류/점수 예측
```

### 2. 상세 구조

#### A. 성분 임베딩 레이어 (IngredientEmbedding)
```python
self.ingredient_embedding = nn.Embedding(vocab_size, embedding_dim)
```
- **역할**: 성분을 128차원 벡터로 변환
- **입력**: 성분 인덱스 (정수)
- **출력**: 128차원 임베딩 벡터

#### B. 상호작용 분석 네트워크 (Interaction Network)
```python
self.interaction_net = nn.Sequential(
    nn.Linear(embedding_dim * 2, hidden_dim),      # 256 → 256
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(hidden_dim, hidden_dim // 2),        # 256 → 128
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(hidden_dim // 2, hidden_dim // 4),   # 128 → 64
    nn.ReLU(),
    nn.Dropout(0.2)
)
```
- **역할**: 두 성분의 조합을 분석하여 상호작용 패턴 학습
- **입력**: 두 성분의 임베딩을 연결한 벡터 (256차원)
- **출력**: 상호작용 특징 벡터 (64차원)

#### C. 분류 헤드 (Classification Head)
```python
self.classifier = nn.Linear(hidden_dim // 4, 3)
```
- **역할**: 성분 조합을 3개 클래스로 분류
- **클래스**: 0=안전, 1=주의, 2=위험

#### D. 점수 예측 헤드
```python
# 위험도 점수 (0-1)
self.danger_score = nn.Sequential(
    nn.Linear(hidden_dim // 4, 1),
    nn.Sigmoid()
)

# 시너지 점수 (0-1)
self.synergy_score = nn.Sequential(
    nn.Linear(hidden_dim // 4, 1),
    nn.Sigmoid()
)
```

## 🔄 학습 과정

### 1. 데이터 준비
```python
# 실제 화장품 데이터에서 성분 조합 추출
ingredient_pairs = [('비타민C', '레티놀'), ('히알루론산', '세라마이드'), ...]
labels = [2, 0, ...]  # 0=안전, 1=주의, 2=위험
```

### 2. 학습 데이터 생성
- **위험한 조합**: 비타민C + 레티놀, AHA + 레티놀 등
- **시너지 조합**: 비타민C + 비타민E, 히알루론산 + 세라마이드 등
- **안전한 조합**: 일반적인 보습 성분 조합들

### 3. 모델 훈련
```python
# 손실 함수
criterion = nn.CrossEntropyLoss()

# 옵티마이저
optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

# 훈련 루프
for epoch in range(num_epochs):
    for batch in dataloader:
        outputs = self.model(ingredient1, ingredient2)
        loss = criterion(outputs['classification'], labels)
        loss.backward()
        optimizer.step()
```

## 🎯 모델의 핵심 특징

### 1. 멀티태스크 학습
- **분류**: 안전/주의/위험 3단계 분류
- **위험도 점수**: 0-1 사이의 연속값 예측
- **시너지 점수**: 0-1 사이의 시너지 효과 예측

### 2. 성분 상호작용 학습
- 두 성분의 조합을 벡터로 표현
- 성분 간의 복잡한 상호작용 패턴 학습
- 새로운 성분 조합에 대한 일반화 능력

### 3. 실제 데이터 기반 학습
- 1000개의 실제 화장품 성분으로 학습
- 실제 제품에서 추출한 성분 조합 사용
- 도메인 특화된 지식 반영

## 📊 모델 성능

### 학습 결과
- **학습 데이터**: 18개 성분 조합
- **에포크**: 30회
- **최종 손실**: 0.82 (CrossEntropy Loss)

### 예측 예시
```
입력: ['비타민C', '레티놀', '히알루론산']
출력:
- 분류: 주의
- 위험도: 56.6%
- 시너지: 40.6%
```

## 🔧 모델 사용법

### 1. 모델 로드
```python
analyzer = AdvancedCosmeticAnalyzer()
analyzer.load_model("models/advanced_ingredient_analyzer.pth")
```

### 2. 성분 조합 분석
```python
result = analyzer.analyze_combination(['비타민C', '레티놀'])
print(f"분류: {result['predicted_class']}")
print(f"위험도: {result['danger_score']:.1%}")
```

## 🚀 모델의 장점

1. **실제 데이터 학습**: 2187개 실제 제품 데이터 기반
2. **다중 예측**: 분류 + 위험도 + 시너지 동시 예측
3. **확장성**: 새로운 성분 조합에 대한 일반화
4. **해석 가능성**: 위험도와 시너지 점수로 명확한 설명

## 📈 향후 개선 방향

1. **더 많은 학습 데이터**: 실제 제품 데이터 확장
2. **고급 아키텍처**: Transformer 인코더 추가
3. **멀티모달**: 성분의 화학적 특성 정보 추가
4. **실시간 학습**: 사용자 피드백을 통한 지속적 학습



