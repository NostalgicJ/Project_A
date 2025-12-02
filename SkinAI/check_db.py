import sqlite3
import os

# DB 파일 위치
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DB_PATH = os.path.join(BASE_DIR, 'skinai.db')

print(f"📂 DB 파일 경로: {DB_PATH}")

if not os.path.exists(DB_PATH):
    print("❌ DB 파일이 아예 없습니다! 'python3 init_db.py'를 먼저 실행하세요.")
    exit()

# DB 연결 및 확인
try:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # 1. 전체 개수 세기
    count = cursor.execute("SELECT COUNT(*) FROM products_db").fetchone()[0]
    print(f"\n📊 DB 저장 상태: 총 {count}개의 제품이 있습니다.")

    if count > 0:
        print("\n�� [샘플 데이터 5개 확인]")
        rows = cursor.execute("SELECT name FROM products_db LIMIT 5").fetchall()
        for row in rows:
            print(f"- {row[0]}")
            
        # 특정 제품 검색 테스트
        print("\n🔍 ['구달'가 들어간 제품 검색 테스트]")
        search_rows = cursor.execute("SELECT name FROM products_db WHERE name LIKE '%구달%' LIMIT 3").fetchall()
        if search_rows:
            for row in search_rows:
                print(f"찾음: {row[0]}")
        else:
            print("검색 결과 없음.")
    else:
        print("⚠️ DB 파일은 있는데 내용이 텅 비어있습니다! CSV 파일을 확인하거나 init_db.py를 다시 실행하세요.")

except Exception as e:
    print(f"❌ 에러 발생: {e}")
    print("테이블이 아직 안 만들어졌을 수도 있습니다.")

finally:
    if 'conn' in locals():
        conn.close()
