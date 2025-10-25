#!/usr/bin/env python3
"""
화장품 성분 조합 분석 서버 실행 스크립트
"""
import os
import sys
import subprocess
import uvicorn
from pathlib import Path

def check_requirements():
    """필요한 패키지 설치 확인"""
    try:
        import fastapi
        import pandas
        import torch
        print("✅ 필요한 패키지들이 설치되어 있습니다.")
        return True
    except ImportError as e:
        print(f"❌ 필요한 패키지가 설치되지 않았습니다: {e}")
        print("다음 명령어로 패키지를 설치해주세요:")
        print("pip install -r requirements.txt")
        return False

def setup_directories():
    """필요한 디렉토리 생성"""
    directories = [
        "data/raw",
        "data/processed",
        "models", 
        "logs",
        "static"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ 디렉토리 생성/확인: {directory}")

def run_data_processing():
    """데이터 전처리 실행"""
    print("🔄 데이터 전처리 실행 중...")
    try:
        from src.data.data_processor import CosmeticDataProcessor
        processor = CosmeticDataProcessor()
        processor.process_all()
        print("✅ 데이터 전처리 완료")
        return True
    except Exception as e:
        print(f"❌ 데이터 전처리 실패: {e}")
        return False

def run_server():
    """서버 실행"""
    print("🚀 서버 시작 중...")
    try:
        # FastAPI 서버 실행
        uvicorn.run(
            "src.api.main:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n🛑 서버가 중지되었습니다.")
    except Exception as e:
        print(f"❌ 서버 실행 실패: {e}")

def main():
    """메인 실행 함수"""
    print("🎯 화장품 성분 조합 분석 서버 시작")
    print("=" * 50)
    
    # 1. 디렉토리 설정
    setup_directories()
    
    # 2. 패키지 확인
    if not check_requirements():
        return
    
    # 3. 데이터 전처리 (선택사항)
    if not os.path.exists("data/processed/ingredient_vocab.pkl"):
        print("📊 데이터 전처리가 필요합니다.")
        if not run_data_processing():
            print("⚠️ 데이터 전처리 실패, 규칙 기반 분석으로 진행합니다.")
    
    # 4. 서버 실행
    print("\n🌐 웹 인터페이스: http://localhost:8000")
    print("📚 API 문서: http://localhost:8000/docs")
    print("🛑 서버 중지: Ctrl+C")
    print("=" * 50)
    
    run_server()

if __name__ == "__main__":
    main()

