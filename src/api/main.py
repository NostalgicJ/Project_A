"""
화장품 성분 조합 분석 API 서버
"""
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import pandas as pd
import numpy as np
import sys
import os

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.ingredient_analyzer import CosmeticIngredientAnalyzer
from data.data_processor import CosmeticDataProcessor

app = FastAPI(
    title="화장품 성분 조합 분석 API",
    description="화장품 성분 조합의 안전성과 시너지를 분석하는 API",
    version="1.0.0"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 프로덕션에서는 특정 도메인만 허용
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 전역 변수
analyzer = None
products_df = None

# Pydantic 모델
class ProductSearchRequest(BaseModel):
    query: str
    limit: int = 10

class IngredientAnalysisRequest(BaseModel):
    ingredients: List[str]

class ProductAnalysisRequest(BaseModel):
    product_ids: List[str]

class AnalysisResponse(BaseModel):
    predicted_class: str
    confidence: float
    safety_score: float
    synergy_score: float
    analysis: str
    safety_issues: Optional[List[str]] = None
    synergy_benefits: Optional[List[str]] = None

class ProductInfo(BaseModel):
    product_id: str
    brand: str
    name: str
    category: str
    ingredients: List[str]

class RecommendationResponse(BaseModel):
    recommendations: List[Dict]
    analysis: str

@app.on_event("startup")
async def startup_event():
    """서버 시작 시 초기화"""
    global analyzer, products_df
    
    try:
        print("🚀 API 서버 초기화 중...")
        
        # 데이터 로드
        processor = CosmeticDataProcessor()
        processor.load_data()
        products_df = processor.products_df
        
        # 분석기 초기화
        analyzer = CosmeticIngredientAnalyzer()
        
        # 어휘 사전 로드 (있는 경우)
        vocab_path = "data/ingredient_vocab.pkl"
        if os.path.exists(vocab_path):
            analyzer.load_vocabulary(vocab_path)
        
        # 모델 로드 (있는 경우)
        model_path = "models/ingredient_analyzer.pth"
        if os.path.exists(model_path):
            analyzer.load_model(model_path)
        
        print("✅ API 서버 초기화 완료")
        
    except Exception as e:
        print(f"❌ 초기화 실패: {e}")
        raise

@app.get("/")
async def root():
    """루트 엔드포인트"""
    return {
        "message": "화장품 성분 조합 분석 API",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health")
async def health_check():
    """헬스 체크"""
    return {"status": "healthy", "message": "API is running"}

@app.post("/search/products", response_model=List[ProductInfo])
async def search_products(request: ProductSearchRequest):
    """화장품 제품 검색"""
    try:
        if products_df is None:
            raise HTTPException(status_code=500, detail="제품 데이터가 로드되지 않았습니다")
        
        # 제품명으로 검색
        query = request.query.lower()
        mask = (
            products_df['브랜드명_정리'].str.lower().str.contains(query, na=False) |
            products_df['제품명_정리'].str.lower().str.contains(query, na=False)
        )
        
        results = products_df[mask].head(request.limit)
        
        # 결과 변환
        products = []
        for idx, row in results.iterrows():
            ingredients = []
            if pd.notna(row['성분_문자열']):
                ingredients = [ing.strip() for ing in str(row['성분_문자열']).split(',')]
            
            products.append(ProductInfo(
                product_id=f"{row['브랜드명_정리']}_{row['제품명_정리']}",
                brand=row['브랜드명_정리'],
                name=row['제품명_정리'],
                category=row['카테고리'],
                ingredients=ingredients
            ))
        
        return products
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"검색 실패: {str(e)}")

@app.post("/analyze/ingredients", response_model=AnalysisResponse)
async def analyze_ingredients(request: IngredientAnalysisRequest):
    """성분 조합 분석"""
    try:
        if analyzer is None:
            raise HTTPException(status_code=500, detail="분석기가 초기화되지 않았습니다")
        
        if not request.ingredients:
            raise HTTPException(status_code=400, detail="성분 리스트가 비어있습니다")
        
        # 성분 조합 분석
        result = analyzer.analyze_combination(request.ingredients)
        
        return AnalysisResponse(**result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"분석 실패: {str(e)}")

@app.post("/analyze/products", response_model=AnalysisResponse)
async def analyze_products(request: ProductAnalysisRequest):
    """제품 조합 분석"""
    try:
        if analyzer is None or products_df is None:
            raise HTTPException(status_code=500, detail="분석기 또는 제품 데이터가 초기화되지 않았습니다")
        
        # 제품들의 성분 수집
        all_ingredients = []
        for product_id in request.product_ids:
            # 제품 정보 찾기
            brand, name = product_id.split('_', 1)
            product = products_df[
                (products_df['브랜드명_정리'] == brand) & 
                (products_df['제품명_정리'] == name)
            ]
            
            if not product.empty:
                ingredients_str = product.iloc[0]['성분_문자열']
                if pd.notna(ingredients_str):
                    ingredients = [ing.strip() for ing in str(ingredients_str).split(',')]
                    all_ingredients.extend(ingredients)
        
        if not all_ingredients:
            raise HTTPException(status_code=400, detail="제품의 성분 정보를 찾을 수 없습니다")
        
        # 성분 조합 분석
        result = analyzer.analyze_combination(all_ingredients)
        
        return AnalysisResponse(**result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"분석 실패: {str(e)}")

@app.post("/recommend/ingredients", response_model=RecommendationResponse)
async def recommend_ingredients(request: IngredientAnalysisRequest):
    """성분 추천"""
    try:
        if analyzer is None:
            raise HTTPException(status_code=500, detail="분석기가 초기화되지 않았습니다")
        
        if not request.ingredients:
            raise HTTPException(status_code=400, detail="성분 리스트가 비어있습니다")
        
        # 성분 추천
        recommendations = analyzer.get_ingredient_recommendations(request.ingredients)
        
        # 분석 결과
        analysis_result = analyzer.analyze_combination(request.ingredients)
        
        return RecommendationResponse(
            recommendations=recommendations,
            analysis=analysis_result['analysis']
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"추천 실패: {str(e)}")

@app.get("/ingredients/popular")
async def get_popular_ingredients(limit: int = Query(20, ge=1, le=100)):
    """인기 성분 조회"""
    try:
        if analyzer is None or analyzer.vocab is None:
            raise HTTPException(status_code=500, detail="분석기가 초기화되지 않았습니다")
        
        # 상위 인기 성분 반환 (실제로는 빈도 기반으로 계산해야 함)
        popular_ingredients = analyzer.vocab[:limit]
        
        return {
            "ingredients": popular_ingredients,
            "total": len(analyzer.vocab)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"조회 실패: {str(e)}")

@app.get("/stats")
async def get_statistics():
    """통계 정보 조회"""
    try:
        if products_df is None:
            raise HTTPException(status_code=500, detail="제품 데이터가 로드되지 않았습니다")
        
        stats = {
            "total_products": len(products_df),
            "categories": products_df['카테고리'].value_counts().to_dict(),
            "brands": products_df['브랜드명_정리'].nunique(),
            "total_ingredients": len(analyzer.vocab) if analyzer and analyzer.vocab else 0
        }
        
        return stats
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"통계 조회 실패: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

