export default function Footer() {
  return (
    <footer className="border-t bg-card mt-auto">
      <div className="container max-w-7xl mx-auto py-8 px-6">
        <div className="text-center space-y-2">
          <p className="text-sm text-muted-foreground" data-testid="text-footer-university">
            고려대학교 세종캠퍼스 컴퓨터융합소프트웨어학과
          </p>
          <p className="text-sm text-muted-foreground" data-testid="text-footer-course">
            2025 캡스톤 디자인
          </p>
          <p className="text-sm text-muted-foreground" data-testid="text-footer-team">
            지수민 조여정 유시온
          </p>
        </div>
      </div>
    </footer>
  );
}
