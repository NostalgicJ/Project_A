import { Link } from "wouter";
import { Button } from "@/components/ui/button";
import { Sparkles } from "lucide-react";

export default function Header() {
  return (
    <header className="sticky top-0 z-50 w-full border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
      <div className="container flex h-16 items-center justify-between px-6">
        <Link href="/" data-testid="link-home">
          <span className="flex items-center space-x-2 hover-elevate rounded-md px-3 py-2 cursor-pointer">
            <span className="text-2xl font-bold tracking-tight">COSME</span>
          </span>
        </Link>
        
        <nav className="hidden md:flex items-center space-x-1">
          <Link href="/" data-testid="link-nav-home">
            <Button variant="ghost" className="text-base" data-testid="button-nav-home">
              홈
            </Button>
          </Link>
          <Link href="/analyze" data-testid="link-nav-analyze">
            <Button variant="ghost" className="text-base" data-testid="button-nav-analyze">
              성분 분석
            </Button>
          </Link>
          <Link href="/collection" data-testid="link-nav-collection">
            <Button variant="ghost" className="text-base" data-testid="button-nav-collection">
              내 화장대
            </Button>
          </Link>
          <Link href="/recommendations" data-testid="link-nav-recommendations">
            <Button variant="ghost" className="text-base" data-testid="button-nav-recommendations">
              추천 제품
            </Button>
          </Link>
        </nav>

        <div className="flex items-center space-x-2">
          <Link href="/analyze" data-testid="link-cta-analyze">
            <Button variant="default" className="gap-2" data-testid="button-start-analysis">
              <Sparkles className="h-4 w-4" />
              분석 시작
            </Button>
          </Link>
        </div>
      </div>
    </header>
  );
}
