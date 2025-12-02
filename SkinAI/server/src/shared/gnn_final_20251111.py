"""
GNN (Graph Neural Network) 기반 성분 조합 분석 모델 - 발표용 최종 버전
날짜: 2025-11-11
기능:
- Early Stopping
- Dropout
- 체크포인트 저장/로드 (중간 저장 및 재시작 지원)
- 학습 곡선 PNG 저장 (loss/accuracy)
- 아키텍처 다이어그램 PNG 저장
- 상세한 진행 상황 표시
- 준지도학습 성능 평가 지표
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
from typing import List, Dict, Tuple, Optional
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
import os
import json
import warnings
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'AppleGothic'  # macOS
# plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows
# plt.rcParams['font.family'] = 'NanumGothic'  # Linux
plt.rcParams['axes.unicode_minus'] = False

# PyTorch Geometric이 없을 경우를 대비한 대체 구현
try:
    from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
    from torch_geometric.data import Data, Batch
    PYG_AVAILABLE = True
    GAT_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False
    GAT_AVAILABLE = False
    print("⚠️ PyTorch Geometric이 설치되지 않았습니다. 대체 구현을 사용합니다.")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_SEQ_LEN = 50  # 최대 성분 수

if not PYG_AVAILABLE:
    # 대체 구현
    class GCNConv(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.linear = nn.Linear(in_channels, out_channels)
        
        def forward(self, x, edge_index):
            return self.linear(x)
    
    class GATConv(nn.Module):
        def __init__(self, in_channels, out_channels, heads=4, dropout=0.0):
            super().__init__()
            self.linear = nn.Linear(in_channels, out_channels)
            self.heads = heads
        
        def forward(self, x, edge_index):
            return self.linear(x)
    
    def global_mean_pool(x, batch):
        return x.mean(dim=0, keepdim=True)
    
    class Data:
        def __init__(self, x=None, edge_index=None):
            self.x = x
            self.edge_index = edge_index
    
    class Batch:
        @staticmethod
        def from_data_list(data_list):
            return data_list


class IngredientFormulaDataset(Dataset):
    """성분 포뮬러 데이터셋 (GNN용) - 라벨 여부 포함"""
    
    def __init__(self, formulas: List[Tuple[List[str], float, float, bool]], vocab_to_idx: Dict[str, int]):
        self.formulas = formulas
        self.vocab_to_idx = vocab_to_idx
        
    def __len__(self):
        return len(self.formulas)
    
    def __getitem__(self, idx):
        ingredients, danger, synergy, has_label = self.formulas[idx]
        
        # 노드 특징 (성분 ID)
        node_ids = [self.vocab_to_idx.get(ing, 0) for ing in ingredients]
        
        # 엣지 인덱스 (완전 연결 그래프)
        num_nodes = len(node_ids)
        edge_index = []
        for i in range(num_nodes):
            for j in range(num_nodes):
                edge_index.append([i, j])
        
        if len(edge_index) > 0:
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.tensor([[], []], dtype=torch.long)
        
        # 라벨이 없는 경우 (-1)는 0으로 변환하되, has_label로 구분
        danger_value = max(0.0, danger)  # -1 -> 0
        synergy_value = max(0.0, synergy)  # -1 -> 0
        
        return {
            'node_ids': torch.tensor(node_ids, dtype=torch.long),
            'edge_index': edge_index,
            'danger': torch.tensor(danger_value, dtype=torch.float),
            'synergy': torch.tensor(synergy_value, dtype=torch.float),
            'has_label': torch.tensor(1.0 if has_label else 0.0, dtype=torch.float),
            'num_nodes': num_nodes
        }


class GNNCollate:
    """GNN 배치 처리"""
    
    def __call__(self, batch):
        if PYG_AVAILABLE:
            data_list = []
            for item in batch:
                data = Data(
                    x=item['node_ids'].unsqueeze(1).float(),
                    edge_index=item['edge_index']
                )
                data.danger = item['danger']
                data.synergy = item['synergy']
                data.has_label = item['has_label']
                data_list.append(data)
            
            batch_data = Batch.from_data_list(data_list)
            return batch_data
        else:
            return batch


class GNNAnalyzerModel(nn.Module):
    """GNN 기반 성분 조합 분석 모델 (GAT 사용, 임베딩 차원 512, Hidden 512)"""
    
    def __init__(self, vocab_size: int, embedding_dim: int = 512, hidden_dim: int = 512, 
                 num_layers: int = 3, dropout: float = 0.4, use_gat: bool = True, num_heads: int = 4):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.use_gat = use_gat and GAT_AVAILABLE
        
        # 임베딩 레이어
        self.ingredient_embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # GNN 레이어 (GAT 또는 GCN)
        self.gnn_layers = nn.ModuleList()
        
        if self.use_gat:
            # GAT 첫 레이어: embedding_dim -> hidden_dim (heads 개의 attention)
            self.gnn_layers.append(GATConv(embedding_dim, hidden_dim // num_heads, heads=num_heads, 
                                         dropout=dropout, concat=True))
            # GAT 중간 레이어들
            for _ in range(num_layers - 2):
                self.gnn_layers.append(GATConv(hidden_dim, hidden_dim // num_heads, heads=num_heads, 
                                               dropout=dropout, concat=True))
            # GAT 마지막 레이어: concat=False로 차원 유지
            if num_layers > 1:
                self.gnn_layers.append(GATConv(hidden_dim, hidden_dim, heads=1, 
                                              dropout=dropout, concat=False))
        else:
            # GCN 사용
            self.gnn_layers.append(GCNConv(embedding_dim, hidden_dim))
            for _ in range(num_layers - 1):
                self.gnn_layers.append(GCNConv(hidden_dim, hidden_dim))
        
        # 출력 헤드 개선 (더 깊은 MLP)
        self.danger_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        self.synergy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, batch_data):
        if PYG_AVAILABLE:
            x = self.ingredient_embedding(batch_data.x.squeeze().long())
            edge_index = batch_data.edge_index
            
            # GNN 레이어 (Residual connection 고려)
            for i, gnn_layer in enumerate(self.gnn_layers):
                x_new = gnn_layer(x, edge_index)
                x_new = F.relu(x_new)
                x_new = F.dropout(x_new, p=self.dropout, training=self.training)
                
                # Residual connection (차원이 같을 때만)
                if i > 0 and x.shape == x_new.shape:
                    x = x + x_new  # Residual connection
                else:
                    x = x_new
            
            # 그래프 풀링
            batch = batch_data.batch if hasattr(batch_data, 'batch') else None
            if batch is not None:
                pooled = global_mean_pool(x, batch)
            else:
                pooled = x.mean(dim=0, keepdim=True)
        else:
            node_ids = batch_data[0]['node_ids']
            x = self.ingredient_embedding(node_ids)
            
            for i, gnn_layer in enumerate(self.gnn_layers):
                x_new = gnn_layer(x, batch_data[0]['edge_index'])
                x_new = F.relu(x_new)
                x_new = F.dropout(x_new, p=self.dropout, training=self.training)
                
                # Residual connection
                if i > 0 and x.shape == x_new.shape:
                    x = x + x_new
                else:
                    x = x_new
            
            pooled = x.mean(dim=0, keepdim=True)
        
        # 출력 헤드 (BatchNorm은 배치 크기가 1일 때 문제가 될 수 있으므로 처리)
        # BatchNorm 대신 LayerNorm 사용하거나, 배치 크기가 1일 때는 BatchNorm을 건너뛰기
        if pooled.shape[0] == 1:
            # 배치 크기가 1이면 BatchNorm을 건너뛰고 직접 계산
            # Sequential 구조: Linear(0) -> BatchNorm(1) -> ReLU(2) -> Dropout(3) -> Linear(4) -> BatchNorm(5) -> ReLU(6) -> Dropout(7) -> Linear(8) -> Sigmoid(9)
            danger_score = self.danger_head[0](pooled)  # Linear
            danger_score = F.relu(danger_score)
            danger_score = F.dropout(danger_score, p=self.dropout, training=self.training)
            danger_score = self.danger_head[4](danger_score)  # Linear
            danger_score = F.relu(danger_score)
            danger_score = F.dropout(danger_score, p=self.dropout, training=self.training)
            danger_score = self.danger_head[8](danger_score)  # Linear
            danger_score = torch.sigmoid(danger_score)
            
            synergy_score = self.synergy_head[0](pooled)
            synergy_score = F.relu(synergy_score)
            synergy_score = F.dropout(synergy_score, p=self.dropout, training=self.training)
            synergy_score = self.synergy_head[4](synergy_score)
            synergy_score = F.relu(synergy_score)
            synergy_score = F.dropout(synergy_score, p=self.dropout, training=self.training)
            synergy_score = self.synergy_head[8](synergy_score)
            synergy_score = torch.sigmoid(synergy_score)
        else:
            danger_score = self.danger_head(pooled)
            synergy_score = self.synergy_head(pooled)
        
        return {
            'danger_score': danger_score,
            'synergy_score': synergy_score
        }


class GNNCosmeticAnalyzer:
    """GNN 기반 화장품 성분 분석기"""
    
    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.vocab = None
        self.vocab_to_idx = None
        self.idx_to_vocab = None
        self.model_path = model_path
        self.device = DEVICE

    def create_formulas_from_pairs(self, pairs: List[Tuple[str, str, float, float]]) -> List[Tuple[List[str], float, float, bool]]:
        """성분 쌍을 포뮬러로 변환 (라벨 여부 포함)"""
        formulas = []
        for ing1, ing2, danger, synergy in pairs:
            # 라벨이 있는지 확인 (danger 또는 synergy가 0보다 크면 라벨 있음)
            has_label = (danger > 0.0) or (synergy > 0.0)
            formulas.append(([ing1, ing2], danger, synergy, has_label))
        return formulas
    
    def _save_architecture_diagram(self, save_dir: str):
        """GNN 모델의 아키텍처 다이어그램을 PNG로 저장 (학습 전에 호출)"""
        os.makedirs(save_dir, exist_ok=True)
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.axis('off')
        y = 0.95
        
        def box(text, y, color='#E3F2FD', height=0.08):
            ax.add_patch(plt.Rectangle((0.1, y-height), 0.8, height, color=color, ec='black', lw=2))
            ax.text(0.5, y-height/2, text, ha='center', va='center', fontsize=12, fontweight='bold')
            return y - height - 0.02
        
        def arrow(y):
            ax.annotate('', xy=(0.5, y-0.02), xytext=(0.5, y+0.02),
                       arrowprops=dict(arrowstyle='->', lw=2, color='black'))
            return y - 0.04
        
        y = box('입력: 성분 목록 (노드), 완전 연결 엣지', y, '#E3F2FD')
        y = arrow(y)
        y = box('임베딩 레이어\n(Embedding, dim=512)', y, '#FFF3E0')
        y = arrow(y)
        y = box('GCN Layers\n(layers=2, dim=256, dropout=0.4)', y, '#E8F5E9')
        y = arrow(y)
        y = box('글로벌 평균 풀링\n(Global Mean Pooling)', y, '#F3E5F5')
        y = arrow(y)
        y = box('Danger Head\nLinear→ReLU→Dropout→Sigmoid', y, '#FFEBEE')
        y = box('Synergy Head\nLinear→ReLU→Dropout→Sigmoid', y-0.1, '#FFEBEE')
        
        ax.text(0.5, 0.02, 'GNN 기반 성분 조합 분석 모델 아키텍처', 
               ha='center', fontsize=14, fontweight='bold')
        
        out = os.path.join(save_dir, 'gnn_architecture.png')
        plt.savefig(out, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"🧩 아키텍처 다이어그램 저장: {out}")
        
    def save_checkpoint(self, save_dir: str, epoch: int, optimizer, scheduler, 
                       train_losses: List, val_losses: List, 
                       train_accuracies: List, val_accuracies: List,
                       best_val_loss: float, best_epoch: int, patience_counter: int,
                       use_weighted_loss: bool = False):
        """체크포인트 저장"""
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        checkpoint_path = os.path.join(save_dir, f"checkpoint_epoch{epoch+1}_{timestamp}.pth")
        
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'vocab': self.vocab,
            'vocab_to_idx': self.vocab_to_idx,
            'idx_to_vocab': self.idx_to_vocab,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_accuracies': train_accuracies,
            'val_accuracies': val_accuracies,
            'best_val_loss': best_val_loss,
            'best_epoch': best_epoch,
            'patience_counter': patience_counter,
            'use_weighted_loss': use_weighted_loss,
            'timestamp': timestamp
        }
        
        torch.save(checkpoint, checkpoint_path)
        return checkpoint_path
    
    def load_checkpoint(self, checkpoint_path: str, optimizer=None, scheduler=None):
        """체크포인트 로드"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.vocab = checkpoint['vocab']
        self.vocab_to_idx = checkpoint['vocab_to_idx']
        self.idx_to_vocab = checkpoint['idx_to_vocab']
        
        vocab_size = len(self.vocab)
        self.model = GNNAnalyzerModel(
            vocab_size=vocab_size,
            embedding_dim=512,
            hidden_dim=512,  # 256 → 512로 증가
            num_layers=3,  # 2 → 3으로 증가
            dropout=0.4,
            use_gat=True,
            num_heads=4
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler is not None:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        print(f"✅ 체크포인트 로드 완료: {checkpoint_path}")
        print(f"   - 에폭: {checkpoint['epoch']}")
        print(f"   - Best Val Loss: {checkpoint['best_val_loss']:.4f}")
        print(f"   - Best Epoch: {checkpoint['best_epoch']}")
        
        return {
            'epoch': checkpoint['epoch'],
            'train_losses': checkpoint.get('train_losses', []),
            'val_losses': checkpoint.get('val_losses', []),
            'train_accuracies': checkpoint.get('train_accuracies', []),
            'val_accuracies': checkpoint.get('val_accuracies', []),
            'best_val_loss': checkpoint['best_val_loss'],
            'best_epoch': checkpoint['best_epoch'],
            'patience_counter': checkpoint.get('patience_counter', 0)
        }
    
    def train(self,
              train_data: List[Tuple[str, str, float, float]],
              val_data: List[Tuple[str, str, float, float]],
              vocab: List[str],
              vocab_to_idx: Dict[str, int],
              num_epochs: int = 30,
              batch_size: int = 64,
              learning_rate: float = 0.001,
              save_dir: str = "models/trained/gnn",
              save_plots: bool = True,
              early_stopping_patience: int = 10,
              early_stopping_min_delta: float = 0.001,
              checkpoint_interval: int = 5,
              resume_from_checkpoint: Optional[str] = None):
        """모델 학습"""
        print(f"🚀 GNN 모델 학습 시작 (Device: {self.device})...")
        
        # 아키텍처 다이어그램 저장 (학습 전)
        try:
            self._save_architecture_diagram(save_dir)
        except Exception as e:
            print(f"⚠️ 아키텍처 다이어그램 저장 실패: {e}")
        
        self.vocab = vocab
        self.vocab_to_idx = vocab_to_idx
        self.idx_to_vocab = {idx: ing for ing, idx in vocab_to_idx.items()}
        
        # 포뮬러로 변환
        train_formulas = self.create_formulas_from_pairs(train_data)
        val_formulas = self.create_formulas_from_pairs(val_data)
        
        # 데이터셋 생성
        train_dataset = IngredientFormulaDataset(train_formulas, vocab_to_idx)
        val_dataset = IngredientFormulaDataset(val_formulas, vocab_to_idx)
        
        collate_fn = GNNCollate()
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        
        # 모델 초기화 (GAT 사용, 임베딩 512, Hidden 512, 레이어 3)
        vocab_size = len(vocab)
        self.model = GNNAnalyzerModel(
            vocab_size=vocab_size,
            embedding_dim=512,
            hidden_dim=512,  # 임베딩과 동일하게 증가
            num_layers=3,  # 2 → 3으로 증가
            dropout=0.4,
            use_gat=True,  # GAT 사용
            num_heads=4  # Attention heads
        ).to(self.device)
        
        print(f"   ✅ GAT (Graph Attention Network) 사용: {self.model.use_gat}")
        print(f"   ✅ 모델 구조: Embedding(512) → GAT Layers(3) → Hidden(512) → Output")
        
        # 데이터 분포 확인 (라벨이 있는 데이터만 분석)
        labeled_data = [(d, s) for _, _, d, s in train_data if d > 0.0 or s > 0.0]
        unlabeled_count = len(train_data) - len(labeled_data)
        
        danger_count = sum(1 for d, _ in labeled_data if d > 0.0)
        synergy_count = sum(1 for _, s in labeled_data if s > 0.0)
        both_count = sum(1 for d, s in labeled_data if d > 0.0 and s > 0.0)
        
        print(f"\n📊 데이터 분포 확인:")
        print(f"   - 라벨이 있는 조합: {len(labeled_data)}개 (정답 레이블)")
        print(f"     * 위험한 조합: {danger_count}개")
        print(f"     * 시너지 조합: {synergy_count}개")
        print(f"     * 위험+시너지 동시: {both_count}개")
        print(f"   - 라벨이 없는 조합: {unlabeled_count}개 (미확인 상태)")
        print(f"   💡 우선순위: 위험(1순위) > 시너지(2순위) > 안전(3순위)")
        print(f"   💡 라벨이 없는 조합은 손실 계산에서 제외됩니다.")
        
        # 옵티마이저 및 손실 함수 (학습률 조정 - nan 방지하면서도 학습 속도 확보)
        # Loss가 안정화되었으므로 학습률을 소폭 상향 (0.0001 → 0.0005)
        safe_learning_rate = min(learning_rate, 0.0005)  # 최대 0.0005로 제한 (안정화 후 상향)
        if learning_rate > 0.0005:
            print(f"   ⚠️ 학습률이 너무 높습니다. {learning_rate} → {safe_learning_rate}로 조정합니다.")
        optimizer = torch.optim.Adam(self.model.parameters(), lr=safe_learning_rate, weight_decay=1e-5)
        
        # 위험한 조합에 더 높은 가중치 부여 (우선순위 1순위) - 극단적으로 높게 설정
        if len(labeled_data) > 0 and danger_count > 0:
            # 위험한 조합에 극단적으로 높은 가중치 (10-20 이상)
            danger_ratio = danger_count / len(labeled_data)
            base_pos_weight = (1.0 - danger_ratio) / danger_ratio
            # 최소 15.0 이상으로 설정 (위험 클래스를 강제로 학습)
            pos_weight = torch.tensor([max(15.0, base_pos_weight * 2.0)]).to(self.device)
            bce_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            print(f"   ✅ 위험도 분류에 클래스 가중치 적용: {pos_weight.item():.2f} (극단적으로 높게 설정)")
        else:
            bce_criterion = nn.BCELoss()
        
        mse_criterion = nn.MSELoss()
        
        # 학습률 스케줄러
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.5, 
            patience=3, 
            min_lr=1e-6
        )
        
        # 체크포인트에서 재시작
        start_epoch = 0
        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []
        best_val_loss = float('inf')
        best_epoch = 0
        patience_counter = 0
        
        if resume_from_checkpoint:
            print(f"📂 체크포인트에서 재시작: {resume_from_checkpoint}")
            checkpoint_data = self.load_checkpoint(resume_from_checkpoint, optimizer, scheduler)
            start_epoch = checkpoint_data['epoch']
            train_losses = checkpoint_data['train_losses']
            val_losses = checkpoint_data['val_losses']
            train_accuracies = checkpoint_data['train_accuracies']
            val_accuracies = checkpoint_data['val_accuracies']
            best_val_loss = checkpoint_data['best_val_loss']
            best_epoch = checkpoint_data['best_epoch']
            patience_counter = checkpoint_data['patience_counter']
            print(f"   재시작 에폭: {start_epoch}/{num_epochs}")
        
        print(f"   - 총 에폭 수: {num_epochs}")
        print(f"   - 시작 에폭: {start_epoch}")
        print(f"   - 배치 크기: {batch_size}")
        print(f"   - 학습률: {safe_learning_rate} (원래: {learning_rate})")
        print(f"   - Early Stopping Patience: {early_stopping_patience}")
        print(f"   - Early Stopping Min Delta: {early_stopping_min_delta}")
        print(f"   - 체크포인트 간격: {checkpoint_interval} 에폭")
        print(f"   - 위험도 손실 가중치: 100.0 (극단적으로 높게 설정)")
        print(f"   - Dropout: 0.4 (과적합 방지 강화)")
        print(f"   - Gradient Clipping: 1.0 (nan 방지)")
        
        total_start_time = time.time()
        
        # 학습 루프
        for epoch in range(start_epoch, num_epochs):
            epoch_start_time = time.time()
            
            # 훈련
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", 
                            leave=False, ncols=100)
            for batch_idx, batch in enumerate(train_pbar):
                if PYG_AVAILABLE:
                    danger_target = batch.danger.to(self.device)
                    synergy_target = batch.synergy.to(self.device)
                    has_label = batch.has_label.to(self.device)
                else:
                    danger_target = torch.stack([item['danger'] for item in batch]).to(self.device)
                    synergy_target = torch.stack([item['synergy'] for item in batch]).to(self.device)
                    has_label = torch.stack([item['has_label'] for item in batch]).to(self.device)
                
                # 순전파
                outputs = self.model(batch)
                
                # 라벨이 있는 데이터만 손실 계산 (우선순위: 위험 > 시너지)
                label_mask = has_label > 0.5
                
                if label_mask.sum() > 0:
                    # 위험도 손실 (1순위) - 라벨이 있는 데이터만
                    danger_pred = outputs['danger_score'].squeeze()[label_mask]
                    danger_tgt = danger_target[label_mask]
                    
                    if isinstance(bce_criterion, nn.BCEWithLogitsLoss):
                        # Sigmoid 출력을 logit으로 안전하게 변환 (nan 방지)
                        # 클리핑하여 0과 1에 너무 가까운 값 방지
                        danger_pred_clipped = torch.clamp(danger_pred, min=1e-7, max=1-1e-7)
                        danger_logits = torch.log(danger_pred_clipped / (1 - danger_pred_clipped))
                        danger_loss = bce_criterion(danger_logits, danger_tgt)
                    else:
                        danger_loss = bce_criterion(danger_pred, danger_tgt)
                    
                    # Loss가 nan인지 확인
                    if torch.isnan(danger_loss) or torch.isinf(danger_loss):
                        print(f"  ⚠️ 경고: Danger Loss가 nan/inf입니다. 이 배치를 건너뜁니다.")
                        continue
                    
                    # 시너지 손실 (2순위) - 위험이 아닌 경우만, 라벨이 있는 데이터만
                    synergy_mask = (danger_tgt == 0.0)  # 위험이 아닌 경우
                    if synergy_mask.sum() > 0:
                        synergy_pred = outputs['synergy_score'].squeeze()[label_mask][synergy_mask]
                        synergy_tgt = synergy_target[label_mask][synergy_mask]
                        synergy_loss = mse_criterion(synergy_pred, synergy_tgt)
                        
                        # 시너지 Loss가 nan인지 확인
                        if torch.isnan(synergy_loss) or torch.isinf(synergy_loss):
                            synergy_loss = torch.tensor(0.0, device=self.device)
                    else:
                        synergy_loss = torch.tensor(0.0, device=self.device)
                    
                    # 우선순위 반영: 위험(1순위) > 시너지(2순위) - 가중치 극단적으로 증가
                    loss = 100.0 * danger_loss + 1.0 * synergy_loss
                    
                    # Loss가 nan인지 확인
                    if torch.isnan(loss) or torch.isinf(loss):
                        print(f"  ⚠️ 경고: Loss가 nan/inf입니다. 이 배치를 건너뜁니다.")
                        continue
                    
                    # 역전파
                    optimizer.zero_grad()
                    loss.backward()
                    
                    # Gradient Clipping 적용 (nan 방지)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    
                    optimizer.step()
                else:
                    # 라벨이 없는 배치의 경우 학습하지 않음 (역전파 건너뛰기)
                    loss = torch.tensor(0.0, device=self.device)
                    danger_loss = torch.tensor(0.0, device=self.device)
                    synergy_loss = torch.tensor(0.0, device=self.device)
                
                train_loss += loss.item()
                
                # 정확도 계산 (라벨이 있는 데이터만, 배치별 표시 제거)
                if label_mask.sum() > 0:
                    # 임계값 0.5 사용 (학습 중에는 고정)
                    danger_pred = (outputs['danger_score'].squeeze()[label_mask] > 0.5).float()
                    train_correct += (danger_pred == danger_target[label_mask]).sum().item()
                    train_total += label_mask.sum().item()
                
                # 진행 상황 표시 (정확도 제거 - 라벨이 희소하여 배치별 정확도는 의미 없음)
                progress = (batch_idx + 1) / len(train_loader) * 100
                elapsed = time.time() - epoch_start_time
                avg_time_per_batch = elapsed / (batch_idx + 1)
                remaining_batches = len(train_loader) - (batch_idx + 1)
                remaining_time = avg_time_per_batch * remaining_batches
                
                train_pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'progress': f'{progress:.1f}%',
                    'remaining': f'{remaining_time:.0f}s'
                })
            
            # 검증
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]", 
                          leave=False, ncols=100)
            with torch.no_grad():
                for batch_idx, batch in enumerate(val_pbar):
                    if PYG_AVAILABLE:
                        danger_target = batch.danger.to(self.device)
                        synergy_target = batch.synergy.to(self.device)
                        has_label = batch.has_label.to(self.device)
                    else:
                        danger_target = torch.stack([item['danger'] for item in batch]).to(self.device)
                        synergy_target = torch.stack([item['synergy'] for item in batch]).to(self.device)
                        has_label = torch.stack([item['has_label'] for item in batch]).to(self.device)
                    
                    outputs = self.model(batch)
                    
                    # 라벨이 있는 데이터만 손실 계산
                    label_mask = has_label > 0.5
                    
                    if label_mask.sum() > 0:
                        # 위험도 손실
                        danger_pred = outputs['danger_score'].squeeze()[label_mask]
                        danger_tgt = danger_target[label_mask]
                        
                        if isinstance(bce_criterion, nn.BCEWithLogitsLoss):
                            # Sigmoid 출력을 logit으로 안전하게 변환 (nan 방지)
                            danger_pred_clipped = torch.clamp(danger_pred, min=1e-7, max=1-1e-7)
                            danger_logits = torch.log(danger_pred_clipped / (1 - danger_pred_clipped))
                            danger_loss = bce_criterion(danger_logits, danger_tgt)
                        else:
                            danger_loss = bce_criterion(danger_pred, danger_tgt)
                        
                        # Loss가 nan인지 확인
                        if torch.isnan(danger_loss) or torch.isinf(danger_loss):
                            danger_loss = torch.tensor(0.0, device=self.device)
                        
                        # 시너지 손실 (위험이 아닌 경우만)
                        synergy_mask = (danger_tgt == 0.0)  # 위험이 아닌 경우
                        if synergy_mask.sum() > 0:
                            synergy_pred = outputs['synergy_score'].squeeze()[label_mask][synergy_mask]
                            synergy_tgt = synergy_target[label_mask][synergy_mask]
                            synergy_loss = mse_criterion(synergy_pred, synergy_tgt)
                            
                            # 시너지 Loss가 nan인지 확인
                            if torch.isnan(synergy_loss) or torch.isinf(synergy_loss):
                                synergy_loss = torch.tensor(0.0, device=self.device)
                        else:
                            synergy_loss = torch.tensor(0.0, device=self.device)
                        
                        loss = 100.0 * danger_loss + 1.0 * synergy_loss
                        
                        # Loss가 nan인지 확인
                        if torch.isnan(loss) or torch.isinf(loss):
                            loss = torch.tensor(0.0, device=self.device)
                    else:
                        loss = torch.tensor(0.0, device=self.device)
                        danger_loss = torch.tensor(0.0, device=self.device)
                        synergy_loss = torch.tensor(0.0, device=self.device)
                    
                    val_loss += loss.item()
                    
                    # 정확도 계산 (라벨이 있는 데이터만)
                    if label_mask.sum() > 0:
                        danger_pred_binary = (outputs['danger_score'].squeeze()[label_mask] > 0.5).float()
                        val_correct += (danger_pred_binary == danger_target[label_mask]).sum().item()
                        val_total += label_mask.sum().item()
                    
                    progress = (batch_idx + 1) / len(val_loader) * 100
                    # 검증 정확도는 에폭 종료 시에만 표시 (배치별 표시 제거)
                    val_pbar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'progress': f'{progress:.1f}%'
                    })
            
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            # 정확도 계산 (ZeroDivisionError 방지)
            train_acc = train_correct / train_total if train_total > 0 else 0.0
            val_acc = val_correct / val_total if val_total > 0 else 0.0
            
            # 라벨이 있는 데이터가 없는 경우 경고
            if train_total == 0:
                print(f"  ⚠️ 경고: 이번 에폭에서 라벨이 있는 훈련 데이터가 없습니다.")
            if val_total == 0:
                print(f"  ⚠️ 경고: 이번 에폭에서 라벨이 있는 검증 데이터가 없습니다.")
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accuracies.append(train_acc)
            val_accuracies.append(val_acc)
            
            epoch_time = time.time() - epoch_start_time
            elapsed_time = time.time() - total_start_time
            avg_time_per_epoch = elapsed_time / (epoch + 1 - start_epoch)
            remaining_epochs = num_epochs - (epoch + 1)
            estimated_remaining = avg_time_per_epoch * remaining_epochs
            
            # 학습률 스케줄러 업데이트 (Loss가 nan이 아닌 경우만)
            if not (torch.isnan(torch.tensor(val_loss)) or torch.isinf(torch.tensor(val_loss))):
                scheduler.step(val_loss)
            
            # Early Stopping 체크 (Loss가 nan이 아닌 경우만)
            improved = False
            if not (torch.isnan(torch.tensor(val_loss)) or torch.isinf(torch.tensor(val_loss))):
                if val_loss < best_val_loss - early_stopping_min_delta:
                    best_val_loss = val_loss
                    best_epoch = epoch + 1
                    patience_counter = 0
                    improved = True
                else:
                    patience_counter += 1
            else:
                # Loss가 nan이면 개선 없음으로 처리
                patience_counter += 1
                print(f"  ⚠️ 경고: Val Loss가 nan/inf입니다. 개선 없음으로 처리합니다.")
            
            # 에폭 종료 시 간단한 요약만 출력 (val_acc만 표시 - 라벨이 희소하여 train_acc는 의미 없음)
            status = "✅ 개선" if improved else f"⏳ 대기 ({patience_counter}/{early_stopping_patience})"
            if val_total > 0:
                print(f"\nEpoch {epoch+1}/{num_epochs} | Loss: {train_loss:.4f}/{val_loss:.4f} | "
                      f"Val Acc: {val_acc*100:.1f}% | {status} | "
                      f"남은 시간: {estimated_remaining/60:.1f}분")
            else:
                print(f"\nEpoch {epoch+1}/{num_epochs} | Loss: {train_loss:.4f}/{val_loss:.4f} | "
                      f"Val Acc: N/A (라벨 없음) | {status} | "
                      f"남은 시간: {estimated_remaining/60:.1f}분")
            
            # 중간 결과 시각화 (매 5 에폭마다)
            if save_plots and (epoch + 1) % 5 == 0:
                self._plot_training_progress(train_losses, val_losses, train_accuracies, val_accuracies, 
                                            epoch + 1, save_dir)
            
            # 체크포인트 저장 (지정된 간격마다)
            if (epoch + 1) % checkpoint_interval == 0:
                checkpoint_path = self.save_checkpoint(
                    save_dir, epoch, optimizer, scheduler,
                    train_losses, val_losses, train_accuracies, val_accuracies,
                    best_val_loss, best_epoch, patience_counter,
                    use_weighted_loss=isinstance(bce_criterion, nn.BCEWithLogitsLoss)
                )
                print(f"  💾 체크포인트 저장: {checkpoint_path}")
            
            # Early Stopping 체크
            if patience_counter >= early_stopping_patience:
                print(f"\n⏹️  Early Stopping! {early_stopping_patience} 에폭 동안 개선이 없어 학습을 중단합니다.")
                print(f"   최고 성능: Val Loss {best_val_loss:.4f} (Epoch {best_epoch})")
                break
        
        # 최종 모델 저장
        total_time = time.time() - total_start_time
        print(f"\n✅ GNN 모델 학습 완료! (총 소요 시간: {total_time/60:.1f}분)")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_model_path = os.path.join(save_dir, f"gnn_model_final_{timestamp}.pth")
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'vocab': self.vocab,
            'vocab_to_idx': self.vocab_to_idx,
            'idx_to_vocab': self.idx_to_vocab,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_accuracies': train_accuracies,
            'val_accuracies': val_accuracies,
            'best_val_loss': best_val_loss,
            'best_epoch': best_epoch,
            'timestamp': timestamp
        }, final_model_path)
        
        print(f"✅ 최종 모델 저장: {final_model_path}")
        
        # 최종 학습 곡선 저장
        if save_plots:
            self._plot_training_progress(train_losses, val_losses, train_accuracies, val_accuracies, 
                                        len(train_losses), save_dir, final=True)
        
        return train_losses, val_losses, train_accuracies, val_accuracies
    
    def _plot_training_progress(self, train_losses, val_losses, train_accs, val_accs, 
                               current_epoch, save_dir, final=False):
        """학습 진행 상황 시각화"""
        os.makedirs(save_dir, exist_ok=True)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        epochs = range(1, len(train_losses) + 1)
        
        # 손실 그래프
        axes[0].plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2, marker='o', markersize=4)
        axes[0].plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2, marker='s', markersize=4)
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Loss', fontsize=12)
        axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 정확도 그래프
        axes[1].plot(epochs, [acc*100 for acc in train_accs], 'b-', label='Train Accuracy', 
                    linewidth=2, marker='o', markersize=4)
        axes[1].plot(epochs, [acc*100 for acc in val_accs], 'r-', label='Val Accuracy', 
                    linewidth=2, marker='s', markersize=4)
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Accuracy (%)', fontsize=12)
        axes[1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim([0, 100])
        
        plt.tight_layout()
        
        if final:
            filename = os.path.join(save_dir, "gnn_training_final.png")
        else:
            filename = os.path.join(save_dir, f"gnn_training_epoch{current_epoch}.png")
        
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        
        if final:
            print(f"📊 최종 학습 곡선 저장: {filename}")
    
    def load_model(self, model_path: str):
        """모델 로드"""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        self.vocab = checkpoint['vocab']
        self.vocab_to_idx = checkpoint['vocab_to_idx']
        self.idx_to_vocab = checkpoint['idx_to_vocab']
        
        vocab_size = len(self.vocab)
        self.model = GNNAnalyzerModel(
            vocab_size=vocab_size,
            embedding_dim=512,
            hidden_dim=512,  # 256 → 512로 증가
            num_layers=3,  # 2 → 3으로 증가
            dropout=0.4,
            use_gat=True,
            num_heads=4
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"✅ 모델 로드 완료: {model_path}")
    
    def evaluate(self, test_data: List[Tuple[str, str, float, float]], batch_size: int = 32, 
                 danger_threshold: float = 0.5):
        """모델 평가 - 준지도학습에 맞는 성능 지표 (라벨이 있는 데이터만 평가)
        
        Args:
            test_data: 테스트 데이터
            batch_size: 배치 크기
            danger_threshold: 위험도 분류 임계값 (기본 0.5, 낮추면 Recall 증가, Precision 감소)
        """
        if self.model is None:
            raise ValueError("모델이 로드되지 않았습니다.")
        
        print("📊 모델 성능 평가 중...")
        print("   ⚠️  라벨이 있는 데이터만 평가합니다.")
        
        test_formulas = self.create_formulas_from_pairs(test_data)
        test_dataset = IngredientFormulaDataset(test_formulas, self.vocab_to_idx)
        collate_fn = GNNCollate()
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        
        self.model.eval()
        all_danger_preds = []
        all_synergy_preds = []
        all_danger_targets = []
        all_synergy_targets = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="평가 중", leave=False):
                if PYG_AVAILABLE:
                    danger_target = batch.danger.to(self.device)
                    synergy_target = batch.synergy.to(self.device)
                    has_label = batch.has_label.to(self.device)
                else:
                    danger_target = torch.stack([item['danger'] for item in batch]).to(self.device)
                    synergy_target = torch.stack([item['synergy'] for item in batch]).to(self.device)
                    has_label = torch.stack([item['has_label'] for item in batch]).to(self.device)
                
                outputs = self.model(batch)
                
                # 라벨이 있는 데이터만 평가
                label_mask = has_label > 0.5
                if label_mask.sum() > 0:
                    all_danger_preds.extend(outputs['danger_score'].cpu().numpy().flatten()[label_mask.cpu().numpy()])
                    all_synergy_preds.extend(outputs['synergy_score'].cpu().numpy().flatten()[label_mask.cpu().numpy()])
                    all_danger_targets.extend(danger_target[label_mask].cpu().numpy())
                    all_synergy_targets.extend(synergy_target[label_mask].cpu().numpy())
        
        all_danger_preds = np.array(all_danger_preds)
        all_synergy_preds = np.array(all_synergy_preds)
        all_danger_targets = np.array(all_danger_targets)
        all_synergy_targets = np.array(all_synergy_targets)
        
        # 준지도학습 성능 지표 계산
        # 1. 위험도 분류 정확도 (0-100%) - 임계값 조정 가능
        print(f"   📌 위험도 분류 임계값: {danger_threshold} (기본 0.5, 낮추면 Recall 증가)")
        danger_preds_binary = (all_danger_preds > danger_threshold).astype(int)
        danger_targets_binary = (all_danger_targets > 0.5).astype(int)
        danger_accuracy = (danger_preds_binary == danger_targets_binary).mean() * 100
        
        # 2. 위험도 Precision, Recall, F1
        true_positives = ((danger_preds_binary == 1) & (danger_targets_binary == 1)).sum()
        false_positives = ((danger_preds_binary == 1) & (danger_targets_binary == 0)).sum()
        false_negatives = ((danger_preds_binary == 0) & (danger_targets_binary == 1)).sum()
        
        danger_precision = (true_positives / (true_positives + false_positives)) * 100 if (true_positives + false_positives) > 0 else 0.0
        danger_recall = (true_positives / (true_positives + false_negatives)) * 100 if (true_positives + false_negatives) > 0 else 0.0
        danger_f1 = (2 * danger_precision * danger_recall / (danger_precision + danger_recall)) if (danger_precision + danger_recall) > 0 else 0.0
        
        # 3. 위험도 MSE
        danger_mse = np.mean((all_danger_preds - all_danger_targets) ** 2)
        
        # 4. 시너지 MSE (시너지 라벨이 있는 경우만, 위험이 아닌 경우)
        synergy_labeled_mask = (all_synergy_targets > 0) & (all_danger_targets == 0)
        if synergy_labeled_mask.sum() > 0:
            synergy_mse = np.mean((all_synergy_preds[synergy_labeled_mask] - all_synergy_targets[synergy_labeled_mask]) ** 2)
        else:
            synergy_mse = 0.0
        
        # 5. 시너지 상관계수 (시너지 라벨이 있는 경우만, 위험이 아닌 경우)
        if synergy_labeled_mask.sum() > 0:
            synergy_correlation = np.corrcoef(all_synergy_preds[synergy_labeled_mask], 
                                             all_synergy_targets[synergy_labeled_mask])[0, 1]
        else:
            synergy_correlation = 0.0
        
        results = {
            'danger_accuracy': danger_accuracy,
            'danger_precision': danger_precision,
            'danger_recall': danger_recall,
            'danger_f1': danger_f1,
            'danger_mse': danger_mse,
            'synergy_mse': synergy_mse,
            'synergy_correlation': synergy_correlation,
            'danger_predictions': all_danger_preds,
            'synergy_predictions': all_synergy_preds,
            'danger_targets': all_danger_targets,
            'synergy_targets': all_synergy_targets,
            'labeled_samples': len(all_danger_targets),
            'total_samples': len(test_data),
            'synergy_labeled_samples': int(synergy_labeled_mask.sum()) if synergy_labeled_mask.sum() > 0 else 0
        }
        
        # 결과 출력
        print(f"\n{'='*80}")
        print("📊 모델 성능 평가 결과 (라벨이 있는 데이터만)")
        print(f"{'='*80}")
        print(f"평가된 샘플 수: {len(all_danger_targets)}개 (전체 {len(test_data)}개 중)")
        print(f"위험도 분류 정확도: {danger_accuracy:.2f}%")
        print(f"위험도 Precision: {danger_precision:.2f}%")
        print(f"위험도 Recall: {danger_recall:.2f}%")
        print(f"위험도 F1 Score: {danger_f1:.2f}%")
        print(f"위험도 MSE: {danger_mse:.4f}")
        if synergy_labeled_mask.sum() > 0:
            print(f"시너지 MSE: {synergy_mse:.4f} (시너지 라벨 {int(synergy_labeled_mask.sum())}개)")
            print(f"시너지 상관계수: {synergy_correlation:.4f}")
        else:
            print(f"시너지 평가: 시너지 라벨이 있는 데이터가 없습니다.")
        print(f"{'='*80}")
        
        return results

