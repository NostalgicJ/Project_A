"""
고급 화장품 성분 조합 분석 딥러닝 모델
실제 데이터를 학습하여 위험한 조합을 예측하는 모델
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
import pickle
import os
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings('ignore')


class IngredientInteractionDataset(Dataset):
    """성분 상호작용 데이터셋"""
    
    def __init__(self, ingredient_pairs, labels, vocab_to_idx):
        self.ingredient_pairs = ingredient_pairs
        self.labels = labels
        self.vocab_to_idx = vocab_to_idx
        
    def __len__(self):
        return len(self.ingredient_pairs)
    
    def __getitem__(self, idx):
        pair = self.ingredient_pairs[idx]
        label = self.labels[idx]
        
        # 성분을 인덱스로 변환
        ing1_idx = self.vocab_to_idx.get(pair[0], 0)
        ing2_idx = self.vocab_to_idx.get(pair[1], 0)
        
        return {
            'ingredient1': torch.tensor(ing1_idx, dtype=torch.long),
            'ingredient2': torch.tensor(ing2_idx, dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.long)
        }


class IngredientInteractionModel(nn.Module):
    """성분 상호작용 분석 딥러닝 모델"""
    
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256):
        super().__init__()
        
        # 성분 임베딩
        self.ingredient_embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # 성분 간 상호작용 분석을 위한 네트워크
        self.interaction_net = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # 분류 헤드 (0: 안전, 1: 주의, 2: 위험)
        self.classifier = nn.Linear(hidden_dim // 4, 3)
        
        # 위험도 점수 헤드 (0-1)
        self.danger_score = nn.Sequential(
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )
        
        # 시너지 점수 헤드 (0-1)
        self.synergy_score = nn.Sequential(
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )
        
    def forward(self, ingredient1, ingredient2):
        # 성분 임베딩
        emb1 = self.ingredient_embedding(ingredient1)
        emb2 = self.ingredient_embedding(ingredient2)
        
        # 성분 조합 벡터
        combined = torch.cat([emb1, emb2], dim=-1)
        
        # 상호작용 분석
        interaction = self.interaction_net(combined)
        
        # 분류 및 점수
        classification = self.classifier(interaction)
        danger_score = self.danger_score(interaction)
        synergy_score = self.synergy_score(interaction)
        
        return {
            'classification': classification,
            'danger_score': danger_score,
            'synergy_score': synergy_score
        }


class AdvancedCosmeticAnalyzer:
    """고급 화장품 성분 분석기 (딥러닝 기반)"""
    
    def __init__(self, model_path=None):
        self.model = None
        self.vocab = None
        self.vocab_to_idx = None
        self.idx_to_vocab = None
        self.model_path = model_path
        
        # 알려진 위험한 조합 (학습 데이터)
        self.known_dangerous_combinations = {
            ('비타민C', '레티놀'): {'danger': 0.9, 'reason': 'pH 불일치로 효과 상쇄'},
            ('AHA', '레티놀'): {'danger': 0.8, 'reason': '과도한 각질 제거'},
            ('BHA', '레티놀'): {'danger': 0.8, 'reason': '과도한 각질 제거'},
            ('니아신아마이드', '비타민C'): {'danger': 0.7, 'reason': 'pH 불일치'},
            ('벤조일퍼옥사이드', '레티놀'): {'danger': 0.9, 'reason': '과도한 각질 제거'},
            ('하이드로퀴논', '레티놀'): {'danger': 0.6, 'reason': '피부 자극 위험'},
            ('살리실릭애씨드', '레티놀'): {'danger': 0.7, 'reason': '과도한 각질 제거'},
        }
        
        # 알려진 시너지 조합
        self.known_synergy_combinations = {
            ('비타민C', '비타민E'): {'synergy': 0.8, 'reason': '항산화 효과 증대'},
            ('히알루론산', '세라마이드'): {'synergy': 0.7, 'reason': '보습 효과 증대'},
            ('나이아신아마이드', '아연'): {'synergy': 0.6, 'reason': '모공 관리 효과 증대'},
            ('레티놀', '하이드로퀴논'): {'synergy': 0.5, 'reason': '미백 효과 증대'},
            ('펩타이드', '레티놀'): {'synergy': 0.6, 'reason': '주름 개선 효과 증대'},
        }
    
    def create_training_data(self, vocab):
        """학습 데이터 생성"""
        print("📊 학습 데이터 생성 중...")
        
        # 성분 쌍과 라벨 생성
        ingredient_pairs = []
        labels = []
        
        # 위험한 조합 (라벨: 2)
        for (ing1, ing2), info in self.known_dangerous_combinations.items():
            if ing1 in vocab and ing2 in vocab:
                ingredient_pairs.append((ing1, ing2))
                labels.append(2)  # 위험
        
        # 시너지 조합 (라벨: 0)
        for (ing1, ing2), info in self.known_synergy_combinations.items():
            if ing1 in vocab and ing2 in vocab:
                ingredient_pairs.append((ing1, ing2))
                labels.append(0)  # 안전
        
        # 랜덤 조합 (라벨: 1 - 주의)
        import random
        random.seed(42)
        for _ in range(len(ingredient_pairs)):
            ing1 = random.choice(vocab)
            ing2 = random.choice(vocab)
            if ing1 != ing2 and (ing1, ing2) not in self.known_dangerous_combinations:
                ingredient_pairs.append((ing1, ing2))
                labels.append(1)  # 주의
        
        print(f"✅ 학습 데이터 생성 완료: {len(ingredient_pairs)}개 조합")
        return ingredient_pairs, labels
    
    def train_model(self, vocab, num_epochs=50):
        """모델 훈련"""
        print("🧠 딥러닝 모델 훈련 시작...")
        
        # 어휘 사전 설정
        self.vocab = vocab
        self.vocab_to_idx = {ing: idx for idx, ing in enumerate(vocab)}
        self.idx_to_vocab = {idx: ing for ing, idx in self.vocab_to_idx.items()}
        
        # 학습 데이터 생성
        ingredient_pairs, labels = self.create_training_data(vocab)
        
        # 데이터셋 생성
        dataset = IngredientInteractionDataset(ingredient_pairs, labels, self.vocab_to_idx)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        # 모델 초기화
        self.model = IngredientInteractionModel(len(vocab))
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        # 훈련
        self.model.train()
        for epoch in range(num_epochs):
            total_loss = 0
            for batch in dataloader:
                ingredient1 = batch['ingredient1']
                ingredient2 = batch['ingredient2']
                labels = batch['label']
                
                # 순전파
                outputs = self.model(ingredient1, ingredient2)
                loss = criterion(outputs['classification'], labels)
                
                # 역전파
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Loss = {total_loss/len(dataloader):.4f}")
        
        print("✅ 모델 훈련 완료!")
        return self.model
    
    def train_model_with_data(self, vocab, ingredient_pairs, labels, num_epochs=50):
        """외부 데이터로 모델 훈련"""
        print("🧠 딥러닝 모델 훈련 시작...")
        
        # 어휘 사전 설정
        self.vocab = vocab
        self.vocab_to_idx = {ing: idx for idx, ing in enumerate(vocab)}
        self.idx_to_vocab = {idx: ing for ing, idx in self.vocab_to_idx.items()}
        
        print(f"📊 학습 데이터: {len(ingredient_pairs)}개 조합")
        
        # 데이터셋 생성
        dataset = IngredientInteractionDataset(ingredient_pairs, labels, self.vocab_to_idx)
        dataloader = DataLoader(dataset, batch_size=min(32, len(ingredient_pairs)), shuffle=True)
        
        # 모델 초기화
        self.model = IngredientInteractionModel(len(vocab))
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        # 훈련
        self.model.train()
        for epoch in range(num_epochs):
            total_loss = 0
            for batch in dataloader:
                ingredient1 = batch['ingredient1']
                ingredient2 = batch['ingredient2']
                labels = batch['label']
                
                # 순전파
                outputs = self.model(ingredient1, ingredient2)
                loss = criterion(outputs['classification'], labels)
                
                # 역전파
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Loss = {total_loss/len(dataloader):.4f}")
        
        print("✅ 모델 훈련 완료!")
        return self.model
    
    def analyze_combination(self, ingredients):
        """성분 조합 분석 (딥러닝 기반)"""
        if self.model is None:
            return self._rule_based_analysis(ingredients)
        
        self.model.eval()
        
        # 모든 성분 쌍에 대해 분석
        danger_scores = []
        synergy_scores = []
        classifications = []
        
        with torch.no_grad():
            for i, ing1 in enumerate(ingredients):
                for ing2 in ingredients[i+1:]:
                    # 성분을 인덱스로 변환
                    ing1_idx = self.vocab_to_idx.get(ing1, 0)
                    ing2_idx = self.vocab_to_idx.get(ing2, 0)
                    
                    # 모델 예측
                    outputs = self.model(
                        torch.tensor([ing1_idx], dtype=torch.long),
                        torch.tensor([ing2_idx], dtype=torch.long)
                    )
                    
                    classification = torch.softmax(outputs['classification'], dim=1)
                    predicted_class = torch.argmax(classification, dim=1).item()
                    danger_score = outputs['danger_score'].item()
                    synergy_score = outputs['synergy_score'].item()
                    
                    classifications.append(predicted_class)
                    danger_scores.append(danger_score)
                    synergy_scores.append(synergy_score)
        
        # 전체 조합 분석
        if danger_scores:
            max_danger = max(danger_scores)
            avg_synergy = np.mean(synergy_scores)
            most_dangerous_class = max(set(classifications), key=classifications.count)
        else:
            max_danger = 0.0
            avg_synergy = 0.0
            most_dangerous_class = 0
        
        # 분류 결정
        if max_danger > 0.7:
            predicted_class = '위험'
            confidence = max_danger
        elif max_danger > 0.4:
            predicted_class = '주의'
            confidence = max_danger
        else:
            predicted_class = '안전'
            confidence = 1 - max_danger
        
        # 분석 텍스트 생성
        if predicted_class == '위험':
            analysis = f"⚠️ 주의: 이 성분 조합은 위험할 수 있습니다. (위험도: {max_danger:.1%})"
        elif predicted_class == '주의':
            analysis = f"⚠️ 주의: 이 성분 조합은 주의가 필요합니다. (위험도: {max_danger:.1%})"
        else:
            analysis = f"✅ 안전: 이 성분 조합은 안전합니다."
        
        return {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'safety_score': 1 - max_danger,
            'synergy_score': avg_synergy,
            'analysis': analysis,
            'danger_score': max_danger
        }
    
    def _rule_based_analysis(self, ingredients):
        """규칙 기반 분석 (백업)"""
        safety_issues = []
        synergy_benefits = []
        
        for i, ing1 in enumerate(ingredients):
            for ing2 in ingredients[i+1:]:
                # 위험한 조합 체크
                if (ing1, ing2) in self.known_dangerous_combinations:
                    info = self.known_dangerous_combinations[(ing1, ing2)]
                    safety_issues.append(f"{ing1} + {ing2}: {info['reason']}")
                elif (ing2, ing1) in self.known_dangerous_combinations:
                    info = self.known_dangerous_combinations[(ing2, ing1)]
                    safety_issues.append(f"{ing1} + {ing2}: {info['reason']}")
                
                # 시너지 조합 체크
                if (ing1, ing2) in self.known_synergy_combinations:
                    info = self.known_synergy_combinations[(ing1, ing2)]
                    synergy_benefits.append(f"{ing1} + {ing2}: {info['reason']}")
                elif (ing2, ing1) in self.known_synergy_combinations:
                    info = self.known_synergy_combinations[(ing2, ing1)]
                    synergy_benefits.append(f"{ing1} + {ing2}: {info['reason']}")
        
        # 점수 계산
        danger_score = len(safety_issues) * 0.3
        synergy_score = len(synergy_benefits) * 0.2
        
        if danger_score > 0.6:
            predicted_class = '위험'
        elif danger_score > 0.3:
            predicted_class = '주의'
        else:
            predicted_class = '안전'
        
        return {
            'predicted_class': predicted_class,
            'confidence': 0.8,
            'safety_score': max(0, 1 - danger_score),
            'synergy_score': min(1, synergy_score),
            'safety_issues': safety_issues,
            'synergy_benefits': synergy_benefits,
            'analysis': f"규칙 기반 분석: {predicted_class}"
        }
    
    def save_model(self, model_path):
        """모델 저장"""
        if self.model is not None:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'vocab': self.vocab,
                'vocab_to_idx': self.vocab_to_idx
            }, model_path)
            print(f"✅ 모델 저장 완료: {model_path}")
    
    def load_model(self, model_path):
        """모델 로드"""
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location='cpu')
            self.vocab = checkpoint['vocab']
            self.vocab_to_idx = checkpoint['vocab_to_idx']
            self.idx_to_vocab = {idx: ing for ing, idx in self.vocab_to_idx.items()}
            
            self.model = IngredientInteractionModel(len(self.vocab))
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            print(f"✅ 모델 로드 완료: {model_path}")
        else:
            print(f"❌ 모델 파일을 찾을 수 없습니다: {model_path}")


if __name__ == "__main__":
    # 테스트
    analyzer = AdvancedCosmeticAnalyzer()
    
    # 간단한 어휘로 테스트
    test_vocab = ['비타민C', '레티놀', '히알루론산', '세라마이드', '나이아신아마이드', 'AHA', 'BHA']
    
    # 모델 훈련
    model = analyzer.train_model(test_vocab, num_epochs=20)
    
    # 테스트
    test_ingredients = ['비타민C', '레티놀', '히알루론산']
    result = analyzer.analyze_combination(test_ingredients)
    
    print(f"\n🧪 딥러닝 모델 테스트 결과:")
    print(f"성분: {test_ingredients}")
    print(f"분류: {result['predicted_class']}")
    print(f"위험도: {result['danger_score']:.1%}")
    print(f"시너지: {result['synergy_score']:.1%}")
    print(f"분석: {result['analysis']}")
