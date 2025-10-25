#!/usr/bin/env python3
"""
간단한 테스트 서버
"""
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="화장품 성분 조합 분석 API - 테스트")

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "화장품 성분 조합 분석 API", "status": "running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "API is running"}

@app.get("/test")
async def test():
    return {"message": "테스트 성공!", "data": "화장품 성분 조합 분석 시스템이 정상 작동합니다."}

if __name__ == "__main__":
    print("🚀 테스트 서버 시작...")
    print("🌐 웹 인터페이스: http://localhost:8000")
    print("📚 API 문서: http://localhost:8000/docs")
    print("🛑 서버 중지: Ctrl+C")
    uvicorn.run(app, host="0.0.0.0", port=8000)



