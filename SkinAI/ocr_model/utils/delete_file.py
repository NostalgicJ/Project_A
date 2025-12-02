import argparse
import sys
from pathlib import Path
from itertools import chain
from typing import List, Tuple  # 타입 힌트를 위해 추가

def delete_files_in_directory(target_dir_path: str, extensions: List[str]) -> Tuple[int, int]:
    """
    특정 디렉토리에서 지정된 확장자(.jpg, .txt 등)의 파일을 찾아 삭제합니다.
    (하위 디렉토리 제외)

    Args:
        target_dir_path (str): 삭제할 파일들이 있는 대상 디렉토리 경로
        extensions (List[str]): 삭제할 파일 확장자 패턴 리스트 (예: ["*.jpg", "*.txt"])

    Returns:
        Tuple[int, int]: (성공적으로 삭제한 파일 수, 삭제 실패한 파일 수)
    """
    
    # 1. 경로 유효성 검사
    target_dir = Path(target_dir_path)
    if not target_dir.is_dir():
        # 오류를 sys.stderr로 출력하고, 호출한 쪽에 실패를 알림
        print(f"오류: 디렉토리를 찾을 수 없습니다: {target_dir}", file=sys.stderr)
        return (0, 0)  # (성공 0, 실패 0)

    # 2. 파일 검색 및 삭제 로직
    file_count = 0
    error_count = 0
    
    # 여러 확장자 패턴에 대한 검색 결과를 하나로 합침
    all_files_to_delete = chain.from_iterable(
        target_dir.glob(pattern) for pattern in extensions
    )

    for file_path in all_files_to_delete:
        try:
            if file_path.is_file():
                file_path.unlink()
                file_count += 1
        except OSError as e:
            # 권한 문제 등으로 삭제가 실패할 경우 오류를 출력
            print(f"파일 삭제 오류 ({file_path.name}): {e}", file=sys.stderr)
            error_count += 1
            
    # 3. 결과 반환
    return (file_count, error_count)

def main():
    """
    이 스크립트가 커맨드 라인에서 직접 실행될 때 호출되는 메인 함수입니다.
    """
    
    # --- 1. 커맨드 라인 인수 설정 ---
    parser = argparse.ArgumentParser(
        description='특정 디렉토리에서 .jpg와 .txt 파일을 삭제합니다. (하위 디렉토리 제외)'
    )
    parser.add_argument('target_dir', type=str,
                        help='삭제할 파일들이 있는 대상 디렉토리 경로')
    
    args = parser.parse_args()
    
    # --- 2. 핵심 로직 함수(delete_files_in_directory) 호출 ---
    # 이 스크립트의 기본 동작은 .jpg와 .txt를 삭제하는 것입니다.
    extensions_to_delete = ["*.jpg", "*.txt"]
    
    print(f"'{args.target_dir}'에서 {', '.join(extensions_to_delete)} 파일을 검색 및 삭제합니다...")
    
    success_count, fail_count = delete_files_in_directory(args.target_dir, extensions_to_delete)

    # --- 3. 실행 결과 출력 ---
    if success_count == 0 and fail_count == 0:
        # 디렉토리 자체가 없었던 경우는 delete_files_in_directory 함수가 이미 오류를 출력함
        if Path(args.target_dir).is_dir():
             print("삭제할 파일을 찾지 못했습니다.")
    else:
        print(f"총 {success_count}개의 파일 삭제 완료.")
        if fail_count > 0:
            print(f"{fail_count}개의 파일 삭제에 실패했습니다.")

# --- 4. 스크립트 실행 지점 ---
if __name__ == "__main__":
    # 이 파일이 'python delete_files.py ...'로 직접 실행될 때만 main()이 호출됩니다.
    # 'import delete_files'로 import 될 때는 이 코드가 실행되지 않습니다.
    main()
