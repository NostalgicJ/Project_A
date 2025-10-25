# 데이터 검토 및 수정 가이드

## 📋 개요

이 문서는 전처리된 데이터를 검토하고 수정하는 방법을 설명합니다.

## 🔄 워크플로우

```
1. 데이터 전처리 실행
   → 스크립트: scripts/preprocess_oliveyoung_data.py
   
2. 기획 상품 검토 및 수정
   → 스크립트: scripts/manual_review_packages.py
   
3. 미확인 성분 검토 및 수정
   → 스크립트: scripts/manual_review_unconfirmed_ingredients.py
   
4. 성분 규칙 관리
   → 스크립트: scripts/manage_ingredient_rules.py
   
5. 최종 데이터 확인
```

## 📝 1. 기획 상품 검토

### 실행
```bash
python scripts/manual_review_packages.py
```

### 기능
- 기획 상품을 개별 제품으로 분리
- 분리 시 브랜드명, 제품명, 성분 리스트 입력
- 전체 데이터에 자동 반영

### 사용 예시
```
제품 ID: 123
제품명: [더블기획] A 제품 + B 제품

작업 선택 (s/r/k/n/q): s
  → 제품 분리

브랜드명: 브랜드A
제품명: A 제품
성분 리스트: 성분1, 성분2, 성분3
✅ 제품 추가됨

브랜드명: 브랜드B  
제품명: B 제품
성분 리스트: 성분4, 성분5, 성분6
✅ 제품 추가됨
```

## 📝 2. 미확인 성분 검토

### 실행
```bash
python scripts/manual_review_unconfirmed_ingredients.py
```

### 기능
- 미확인 성분을 올바른 성분명으로 수정
- 통계 확인 (출현 빈도)
- 전체 데이터에 자동 반영

### 사용 예시
```
[1/50] 'XXX성분' → 아스코빅애씨드
  ✅ 'XXX성분' → '아스코빅애씨드'

[2/50] 'YYY성분' → 
  유지: YYY성분
```

### 출력 파일
- `data/processed/ingredient_review_changes.json`: 수정 내역

## 📝 3. 성분 규칙 관리

### 실행
```bash
python scripts/manage_ingredient_rules.py
```

### 기능
- 성분 계열 추가/수정
- 위험한 조합 규칙 추가
- 시너지 조합 규칙 추가

### 규칙 파일 구조
```json
{
  "ingredient_families": {
    "비타민C계열": ["아스코빅애씨드", "소듐아스코빌포스페이트"]
  },
  "dangerous_combinations": [
    {
      "family1": "비타민C계열",
      "family2": "레티놀계열",
      "danger_level": 0.9,
      "reason": "pH 불일치",
      "detail": "상세 설명"
    }
  ]
}
```

### 사용 예시
```
성분 계열 추가
  계열명: 나이아신계열
  성분 리스트: 나이아신아마이드, 나이아신
  
위험한 조합 추가
  계열 1: AHA계열
  계열 2: BHA계열
  위험도 (0-1): 0.7
  이유: 과도한 각질 제거
```

## 🔄 4. 자동 반영 메커니즘

### 변경사항 저장
- 기획 상품: `package_review_changes.json`
- 미확인 성분: `ingredient_review_changes.json`
- 성분 규칙: `config/ingredient_rules.json`

### 반영 과정
1. 변경사항 JSON 파일에 저장
2. `apply_changes()` 함수로 전체 데이터 수정
3. CSV 파일 자동 업데이트

## 📊 출력 파일

### 메인 파일
- `oliveyoung_products_cleaned.csv`: 최종 정제된 제품 데이터

### 검토용 파일
- `*_packages.csv`: 기획 상품만 모음
- `*_unconfirmed.csv`: 미확인 성분 제품만 모음

### 규칙 파일
- `config/ingredient_rules.json`: 성분 계열 및 조합 규칙

## 💡 팁

1. **단계별 진행**: 한 번에 모든 작업을 하지 말고 단계별로 진행
2. **백업**: 중요한 수정 전에 데이터 백업
3. **반영 확인**: 변경사항 적용 후 결과 확인
4. **일괄 처리**: 여러 항목을 한 번에 수정 가능

## 🎯 체크리스트

- [ ] 기획 상품 검토 완료
- [ ] 미확인 성분 수정 완료
- [ ] 성분 규칙 설정 완료
- [ ] 변경사항 적용 확인
- [ ] 데이터 품질 검증
