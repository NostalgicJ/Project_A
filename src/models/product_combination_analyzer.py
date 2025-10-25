"""
제품 간 성분 조합 비교 분석 모델

A 제품의 성분 리스트와 B 제품의 성분 리스트를 비교하여
위험한 조합을 찾아냅니다.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from typing import List, Dict, Set
import json
from pathlib import Path


class ProductIngredientMatcher:
    """제품 간 성분 조합 분석기 (규칙 기반)"""
    
    def __init__(self, rules_file="config/ingredient_rules.json"):
        self.rules_file = rules_file
        self.ingredient_families = {}
        self.dangerous_combinations = []
        self.synergy_combinations = []
        
        self.load_rules()
    
    def load_rules(self):
        """성분 규칙 로드"""
        if Path(self.rules_file).exists():
            with open(self.rules_file, 'r', encoding='utf-8') as f:
                rules = json.load(f)
            
            self.ingredient_families = rules.get('ingredient_families', {})
            self.dangerous_combinations = rules.get('dangerous_combinations', [])
            self.synergy_combinations = rules.get('synergy_combinations', [])
            
            print(f"✅ 성분 규칙 로드: {len(self.ingredient_families)}개 계열")
        else:
            print("⚠️ 규칙 파일이 없습니다. 기본 규칙을 사용합니다.")
            self._load_default_rules()
    
    def _load_default_rules(self):
        """기본 규칙"""
        self.ingredient_families = {
            '비타민C계열': ['아스코빅애씨드', '아스코빌글루코사이드', '소듐아스코빌포스페이트'],
            '레티놀계열': ['레티놀', '레티놀아세테이트'],
            'AHA계열': ['글라이콜릭애씨드', '젖산'],
            'BHA계열': ['살리실릭애씨드'],
            '비타민E계열': ['토코페롤', '토코페릴아세테이트'],
        }
    
    def get_ingredient_family(self, ingredient: str) -> str:
        """성분의 계열 찾기"""
        for family, ingredients in self.ingredient_families.items():
            for ing in ingredients:
                if ingredient.lower() == ing.lower():
                    return family
        return None
    
    def analyze_product_pair(self, product_a_ingredients: List[str], 
                            product_b_ingredients: List[str]) -> Dict:
        """
        두 제품의 성분 리스트 비교 분석
        
        Args:
            product_a_ingredients: A 제품의 성분 리스트
            product_b_ingredients: B 제품의 성분 리스트
            
        Returns:
            분석 결과 딕셔너리
        """
        # 각 제품의 성분 계열 추출
        a_families = set()
        b_families = set()
        
        for ing in product_a_ingredients:
            family = self.get_ingredient_family(ing)
            if family:
                a_families.add(family)
        
        for ing in product_b_ingredients:
            family = self.get_ingredient_family(ing)
            if family:
                b_families.add(family)
        
        # 위험한 조합 찾기
        dangerous_matches = []
        for combo in self.dangerous_combinations:
            family1 = combo['family1']
            family2 = combo['family2']
            
            # A에 family1이 있고 B에 family2가 있거나
            # A에 family2가 있고 B에 family1이 있는 경우
            if (family1 in a_families and family2 in b_families) or \
               (family2 in a_families and family1 in b_families):
                
                dangerous_matches.append({
                    'family1': family1,
                    'family2': family2,
                    'danger_level': combo['danger_level'],
                    'reason': combo['reason'],
                    'detail': combo['detail']
                })
        
        # 시너지 조합 찾기
        synergy_matches = []
        for combo in self.synergy_combinations:
            family1 = combo['family1']
            family2 = combo['family2']
            
            if (family1 in a_families and family2 in b_families) or \
               (family2 in a_families and family1 in b_families):
                
                synergy_matches.append({
                    'family1': family1,
                    'family2': family2,
                    'synergy_level': combo['synergy_level'],
                    'reason': combo['reason'],
                    'detail': combo['detail']
                })
        
        # 위험도 계산
        max_danger = max([m['danger_level'] for m in dangerous_matches]) if dangerous_matches else 0
        avg_synergy = sum([m['synergy_level'] for m in synergy_matches]) / len(synergy_matches) if synergy_matches else 0
        
        # 결과 생성
        result = {
            'dangerous_matches': dangerous_matches,
            'synergy_matches': synergy_matches,
            'max_danger_level': max_danger,
            'avg_synergy_level': avg_synergy,
            'product_a_families': list(a_families),
            'product_b_families': list(b_families),
        }
        
        # 종합 평가
        if max_danger > 0.7:
            result['overall_assessment'] = '위험'
        elif max_danger > 0.4:
            result['overall_assessment'] = '주의'
        elif avg_synergy > 0.5:
            result['overall_assessment'] = '시너지'
        else:
            result['overall_assessment'] = '안전'
        
        return result
    
    def format_analysis_result(self, result: Dict) -> str:
        """분석 결과를 읽기 쉬운 형태로 포맷"""
        lines = []
        
        lines.append("\n" + "="*60)
        lines.append("제품 간 성분 조합 분석 결과")
        lines.append("="*60)
        
        if result['dangerous_matches']:
            lines.append("\n⚠️ 위험한 조합:")
            for match in result['dangerous_matches']:
                lines.append(f"  - {match['family1']} + {match['family2']}")
                lines.append(f"    위험도: {match['danger_level']:.1%}")
                lines.append(f"    이유: {match['reason']}")
                lines.append(f"    상세: {match['detail']}")
        
        if result['synergy_matches']:
            lines.append("\n✅ 시너지 조합:")
            for match in result['synergy_matches']:
                lines.append(f"  - {match['family1']} + {match['family2']}")
                lines.append(f"    시너지: {match['synergy_level']:.1%}")
                lines.append(f"    이유: {match['reason']}")
        
        lines.append(f"\n종합 평가: {result['overall_assessment']}")
        lines.append("="*60)
        
        return "\n".join(lines)


class AdvancedProductAnalyzer:
    """
    딥러닝 기반 제품 간 성분 조합 분석 모델
    
    구조:
    - 제품 A, B의 성분 리스트를 받아서
    - 각 성분을 임베딩하고
    - 두 제품 간의 성분 상호작용을 분석
    - 위험도와 시너지를 예측
    """
    
    def __init__(self, vocab_size: int, embedding_dim: int = 128):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.model = None
        
    def build_model(self):
        """모델 구축"""
        
        class ProductInteractionModel(nn.Module):
            def __init__(self, vocab_size, embedding_dim):
                super().__init__()
                
                # 성분 임베딩
                self.embedding = nn.Embedding(vocab_size, embedding_dim)
                
                # 각 제품의 성분 리스트를 집계
                self.product_aggregator = nn.Sequential(
                    nn.Linear(embedding_dim, embedding_dim * 2),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(embedding_dim * 2, embedding_dim)
                )
                
                # 제품 간 상호작용 분석
                self.interaction = nn.Sequential(
                    nn.Linear(embedding_dim * 2, embedding_dim),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(embedding_dim, embedding_dim // 2),
                    nn.ReLU(),
                    nn.Dropout(0.2)
                )
                
                # 분류 및 점수 출력
                self.danger_prediction = nn.Sequential(
                    nn.Linear(embedding_dim // 2, 1),
                    nn.Sigmoid()
                )
                self.synergy_prediction = nn.Sequential(
                    nn.Linear(embedding_dim // 2, 1),
                    nn.Sigmoid()
                )
            
            def forward(self, product_a_ids, product_b_ids, 
                       a_mask=None, b_mask=None):
                # 성분 임베딩
                a_emb = self.embedding(product_a_ids)  # (batch, seq_len_a, embed_dim)
                b_emb = self.embedding(product_b_ids)  # (batch, seq_len_b, embed_dim)
                
                # 마스크 적용 (패딩 제외)
                if a_mask is not None:
                    a_emb = a_emb * a_mask.unsqueeze(-1)
                if b_mask is not None:
                    b_emb = b_emb * b_mask.unsqueeze(-1)
                
                # 제품별 성분 집계 (평균 또는 합)
                a_aggregated = a_emb.mean(dim=1)  # (batch, embed_dim)
                b_aggregated = b_emb.mean(dim=1)  # (batch, embed_dim)
                
                # 제품 표현 정제
                a_rep = self.product_aggregator(a_aggregated)
                b_rep = self.product_aggregator(b_aggregated)
                
                # 제품 간 상호작용
                combined = torch.cat([a_rep, b_rep], dim=-1)
                interaction = self.interaction(combined)
                
                # 위험도와 시너지 예측
                danger = self.danger_prediction(interaction)
                synergy = self.synergy_prediction(interaction)
                
                return {
                    'danger': danger,
                    'synergy': synergy,
                    'product_a_rep': a_rep,
                    'product_b_rep': b_rep,
                    'interaction': interaction
                }
        
        self.model = ProductInteractionModel(self.vocab_size, self.embedding_dim)
        return self.model


if __name__ == "__main__":
    # 테스트
    analyzer = ProductIngredientMatcher()
    
    product_a = ['비타민C', '히알루론산', '토코페롤']
    product_b = ['레티놀', '세라마이드', '나이아신아마이드']
    
    result = analyzer.analyze_product_pair(product_a, product_b)
    
    print(analyzer.format_analysis_result(result))
