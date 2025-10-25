#!/bin/bash

# AWS EC2 배포 스크립트
# 화장품 성분 조합 분석 딥러닝 프로젝트

echo "🚀 AWS EC2 배포 시작"
echo "=================================="

# 1. 시스템 업데이트
echo "📦 시스템 업데이트 중..."
sudo apt update && sudo apt upgrade -y

# 2. 필요한 패키지 설치
echo "🔧 필요한 패키지 설치 중..."
sudo apt install -y python3.9 python3.9-pip python3.9-venv git nginx htop

# 3. 프로젝트 디렉토리 생성
echo "📁 프로젝트 디렉토리 설정 중..."
sudo mkdir -p /var/www/cosmetic-analyzer
sudo chown -R $USER:$USER /var/www/cosmetic-analyzer
cd /var/www/cosmetic-analyzer

# 4. 프로젝트 클론 (실제로는 Git 저장소에서 클론)
echo "📥 프로젝트 파일 복사 중..."
# git clone <your-repository-url> .
# 또는 파일을 직접 업로드

# 5. 가상환경 생성 및 활성화
echo "🐍 Python 가상환경 설정 중..."
python3.9 -m venv venv
source venv/bin/activate

# 6. 패키지 설치
echo "📚 Python 패키지 설치 중..."
pip install --upgrade pip
pip install -r requirements.txt

# 7. 데이터 전처리 실행
echo "🔄 데이터 전처리 실행 중..."
python src/data/data_processor.py

# 8. 모델 훈련 (선택사항)
echo "🧠 모델 훈련 실행 중..."
# python notebooks/model_training/train_ingredient_model.ipynb

# 9. Nginx 설정
echo "🌐 Nginx 설정 중..."
sudo tee /etc/nginx/sites-available/cosmetic-analyzer > /dev/null <<EOF
server {
    listen 80;
    server_name your-domain.com;  # 실제 도메인으로 변경
    
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }
    
    location /static/ {
        alias /var/www/cosmetic-analyzer/static/;
    }
}
EOF

# Nginx 사이트 활성화
sudo ln -s /etc/nginx/sites-available/cosmetic-analyzer /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx

# 10. 시스템 서비스 생성
echo "⚙️ 시스템 서비스 설정 중..."
sudo tee /etc/systemd/system/cosmetic-analyzer.service > /dev/null <<EOF
[Unit]
Description=Cosmetic Ingredient Analyzer API
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=/var/www/cosmetic-analyzer
Environment=PATH=/var/www/cosmetic-analyzer/venv/bin
ExecStart=/var/www/cosmetic-analyzer/venv/bin/python run_server.py
Restart=always

[Install]
WantedBy=multi-user.target
EOF

# 서비스 활성화
sudo systemctl daemon-reload
sudo systemctl enable cosmetic-analyzer
sudo systemctl start cosmetic-analyzer

# 11. 방화벽 설정
echo "🔥 방화벽 설정 중..."
sudo ufw allow 22
sudo ufw allow 80
sudo ufw allow 443
sudo ufw --force enable

# 12. 서비스 상태 확인
echo "✅ 서비스 상태 확인 중..."
sudo systemctl status cosmetic-analyzer
sudo systemctl status nginx

echo ""
echo "🎉 배포 완료!"
echo "=================================="
echo "웹 애플리케이션: http://your-domain.com"
echo "API 문서: http://your-domain.com/docs"
echo "헬스 체크: http://your-domain.com/health"
echo ""
echo "서비스 관리 명령어:"
echo "- 서비스 시작: sudo systemctl start cosmetic-analyzer"
echo "- 서비스 중지: sudo systemctl stop cosmetic-analyzer"
echo "- 서비스 재시작: sudo systemctl restart cosmetic-analyzer"
echo "- 로그 확인: sudo journalctl -u cosmetic-analyzer -f"
echo ""
echo "모니터링:"
echo "- 시스템 상태: htop"
echo "- 디스크 사용량: df -h"
echo "- 메모리 사용량: free -h"



