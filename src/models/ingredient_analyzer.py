"""
화장품 성분 조합 분석 딥러닝 모델
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os


class IngredientEmbedding(nn.Module):
    """성분 임베딩 레이어"""
    
    def __init__(self, vocab_size: int, embedding_dim: int = 128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        return self.dropout(self.embedding(x))


class IngredientCombinationAnalyzer(nn.Module):
    """성분 조합 분석 모델"""
    
    def __init__(self, vocab_size: int, embedding_dim: int = 128, 
                 hidden_dim: int = 256, num_classes: int = 3):
        super().__init__()
        
        # 성분 임베딩
        self.ingredient_embedding = IngredientEmbedding(vocab_size, embedding_dim)
        
        # 조합 분석을 위한 Transformer
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=8,
                dim_feedforward=hidden_dim,
                dropout=0.1
            ),
            num_layers=4
        )
        
        # 분류 헤드
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        # 안전성 점수 헤드
        self.safety_head = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # 시너지 점수 헤드
        self.synergy_head = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, ingredient_ids, attention_mask=None):
        # 성분 임베딩
        embedded = self.ingredient_embedding(ingredient_ids)
        
        # Transformer 인코더
        if attention_mask is not None:
            # 패딩 마스크 적용
            embedded = embedded.transpose(0, 1)  # (seq_len, batch, embed_dim)
            output = self.transformer(embedded, src_key_padding_mask=attention_mask)
            output = output.transpose(0, 1)  # (batch, seq_len, embed_dim)
        else:
            embedded = embedded.transpose(0, 1)
            output = self.transformer(embedded)
            output = output.transpose(0, 1)
        
        # 평균 풀링
        pooled = output.mean(dim=1)  # (batch, embed_dim)
        
        # 분류 결과
        classification = self.classifier(pooled)
        safety_score = self.safety_head(pooled)
        synergy_score = self.synergy_head(pooled)
        
        return {
            'classification': classification,
            'safety_score': safety_score,
            'synergy_score': synergy_score
        }


class CosmeticIngredientAnalyzer:
    """화장품 성분 분석기"""
    
    def __init__(self, model_path: str = None):
        self.model = None
        self.vocab = None
        self.ingredient_to_idx = None
        self.idx_to_ingredient = None
        self.model_path = model_path
        
        # 성분 조합 규칙 (도메인 지식 기반)
        self.unsafe_combinations = {
            ('비타민C', '레티놀'): '산화 반응으로 효과 상쇄',
            ('AHA', '레티놀'): '과도한 각질 제거로 피부 자극',
            ('BHA', '레티놀'): '과도한 각질 제거로 피부 자극',
            ('니아신아마이드', '비타민C'): 'pH 불일치로 효과 감소',
            ('벤조일퍼옥사이드', '레티놀'): '과도한 각질 제거',
        }
        
        self.synergy_combinations = {
            ('비타민C', '비타민E'): '항산화 효과 증대',
            ('히알루론산', '세라마이드'): '보습 효과 증대',
            ('나이아신아마이드', '아연'): '모공 관리 효과 증대',
            ('레티놀', '하이드로퀴논'): '미백 효과 증대',
            ('펩타이드', '레티놀'): '주름 개선 효과 증대',
        }
    
    def load_vocabulary(self, vocab_path: str):
        """어휘 사전 로드"""
        with open(vocab_path, 'rb') as f:
            self.vocab = pickle.load(f)
        
        self.ingredient_to_idx = {ing: idx for idx, ing in enumerate(self.vocab)}
        self.idx_to_ingredient = {idx: ing for ing, idx in self.ingredient_to_idx.items()}
        
        print(f"✅ 어휘 사전 로드 완료: {len(self.vocab)}개 성분")
    
    def load_model(self, model_path: str):
        """모델 로드"""
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location='cpu')
            self.model = IngredientCombinationAnalyzer(
                vocab_size=len(self.vocab),
                embedding_dim=128,
                hidden_dim=256,
                num_classes=3
            )
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            print(f"✅ 모델 로드 완료: {model_path}")
        else:
            print(f"❌ 모델 파일을 찾을 수 없습니다: {model_path}")
    
    def preprocess_ingredients(self, ingredients: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """성분 리스트 전처리"""
        # 성분명을 인덱스로 변환
        ingredient_ids = []
        for ingredient in ingredients:
            if ingredient in self.ingredient_to_idx:
                ingredient_ids.append(self.ingredient_to_idx[ingredient])
            else:
                # OOV 처리: 0으로 매핑
                ingredient_ids.append(0)
        
        # 패딩 및 마스크 생성
        max_length = 50  # 최대 성분 수
        if len(ingredient_ids) > max_length:
            ingredient_ids = ingredient_ids[:max_length]
        
        # 패딩
        attention_mask = [1] * len(ingredient_ids) + [0] * (max_length - len(ingredient_ids))
        ingredient_ids += [0] * (max_length - len(ingredient_ids))
        
        return torch.tensor([ingredient_ids]), torch.tensor([attention_mask])
    
    def analyze_combination(self, ingredients: List[str]) -> Dict:
        """성분 조합 분석"""
        if self.model is None:
            return self._rule_based_analysis(ingredients)
        
        # 전처리
        ingredient_tensor, attention_mask = self.preprocess_ingredients(ingredients)
        
        # 모델 예측
        with torch.no_grad():
            outputs = self.model(ingredient_tensor, attention_mask)
            
            # 결과 해석
            classification = F.softmax(outputs['classification'], dim=1)
            safety_score = outputs['safety_score'].item()
            synergy_score = outputs['synergy_score'].item()
            
            # 분류 결과
            class_labels = ['안전', '주의', '위험']
            predicted_class = torch.argmax(classification, dim=1).item()
            confidence = classification[0][predicted_class].item()
            
            return {
                'predicted_class': class_labels[predicted_class],
                'confidence': confidence,
                'safety_score': safety_score,
                'synergy_score': synergy_score,
                'analysis': self._generate_analysis_text(ingredients, safety_score, synergy_score)
            }
    
    def _rule_based_analysis(self, ingredients: List[str]) -> Dict:
        """규칙 기반 분석 (모델이 없을 때)"""
        safety_issues = []
        synergy_benefits = []
        
        # 안전하지 않은 조합 체크
        for i, ing1 in enumerate(ingredients):
            for ing2 in ingredients[i+1:]:
                # 정확한 매칭
                if (ing1, ing2) in self.unsafe_combinations:
                    safety_issues.append(f"{ing1} + {ing2}: {self.unsafe_combinations[(ing1, ing2)]}")
                elif (ing2, ing1) in self.unsafe_combinations:
                    safety_issues.append(f"{ing1} + {ing2}: {self.unsafe_combinations[(ing2, ing1)]}")
                
                # 시너지 조합 체크
                if (ing1, ing2) in self.synergy_combinations:
                    synergy_benefits.append(f"{ing1} + {ing2}: {self.synergy_combinations[(ing1, ing2)]}")
                elif (ing2, ing1) in self.synergy_combinations:
                    synergy_benefits.append(f"{ing1} + {ing2}: {self.synergy_combinations[(ing2, ing1)]}")
        
        # 점수 계산
        safety_score = max(0, 1 - len(safety_issues) * 0.3)
        synergy_score = min(1, len(synergy_benefits) * 0.2)
        
        # 분류
        if safety_issues:
            predicted_class = '위험' if len(safety_issues) > 2 else '주의'
        else:
            predicted_class = '안전'
        
        return {
            'predicted_class': predicted_class,
            'confidence': 0.8,  # 규칙 기반이므로 높은 신뢰도
            'safety_score': safety_score,
            'synergy_score': synergy_score,
            'safety_issues': safety_issues,
            'synergy_benefits': synergy_benefits,
            'analysis': self._generate_analysis_text(ingredients, safety_score, synergy_score)
        }
    
    def _generate_analysis_text(self, ingredients: List[str], safety_score: float, synergy_score: float) -> str:
        """분석 결과 텍스트 생성"""
        if safety_score < 0.5:
            return f"⚠️ 주의: 이 성분 조합은 피부에 자극을 줄 수 있습니다. 사용을 자제해주세요."
        elif synergy_score > 0.7:
            return f"✨ 좋은 조합: 이 성분들은 함께 사용하면 시너지 효과를 낼 수 있습니다!"
        elif synergy_score > 0.4:
            return f"👍 괜찮은 조합: 이 성분들은 안전하게 함께 사용할 수 있습니다."
        else:
            return f"✅ 안전한 조합: 이 성분들은 문제없이 함께 사용할 수 있습니다."
    
    def get_ingredient_recommendations(self, current_ingredients: List[str], 
                                     num_recommendations: int = 5) -> List[Dict]:
        """성분 추천"""
        recommendations = []
        
        # 현재 성분과 시너지를 낼 수 있는 성분들 찾기
        for ingredient, benefit in self.synergy_combinations.items():
            if ingredient[0] in current_ingredients and ingredient[1] not in current_ingredients:
                recommendations.append({
                    'ingredient': ingredient[1],
                    'reason': f"{ingredient[0]}와 함께 사용하면 {benefit}",
                    'synergy_score': 0.8
                })
            elif ingredient[1] in current_ingredients and ingredient[0] not in current_ingredients:
                recommendations.append({
                    'ingredient': ingredient[0],
                    'reason': f"{ingredient[1]}와 함께 사용하면 {benefit}",
                    'synergy_score': 0.8
                })
        
        # 점수순으로 정렬하고 상위 N개 반환
        recommendations.sort(key=lambda x: x['synergy_score'], reverse=True)
        return recommendations[:num_recommendations]
    
    def save_model(self, model_path: str):
        """모델 저장"""
        if self.model is not None:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'vocab_size': len(self.vocab),
                'vocab': self.vocab
            }, model_path)
            print(f"✅ 모델 저장 완료: {model_path}")


if __name__ == "__main__":
    # 테스트 코드
    analyzer = CosmeticIngredientAnalyzer()
    
    # 테스트 성분 조합
    test_ingredients = ['비타민C', '레티놀', '히알루론산']
    result = analyzer.analyze_combination(test_ingredients)
    
    print("🧪 테스트 결과:")
    print(f"성분: {test_ingredients}")
    print(f"분류: {result['predicted_class']}")
    print(f"안전성 점수: {result['safety_score']:.2f}")
    print(f"시너지 점수: {result['synergy_score']:.2f}")
    print(f"분석: {result['analysis']}")

