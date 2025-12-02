import { Button } from "@/components/ui/button";
import { Sparkles } from "lucide-react";
import { Link } from "wouter";
import heroImage from "@assets/generated_images/Hero_cosmetic_product_arrangement_3e1c233e.png";

export default function HeroSection() {
  return (
    <section className="relative min-h-[600px] flex items-center justify-center overflow-hidden">
      <div 
        className="absolute inset-0 bg-cover bg-center"
        style={{
          backgroundImage: `url(${heroImage})`,
        }}
      />
      <div className="absolute inset-0 bg-gradient-to-b from-black/40 via-black/30 to-black/50" />
      
      <div className="relative z-10 container px-6 py-20 text-center">
        <div className="max-w-3xl mx-auto space-y-6">
          <h1 className="text-4xl md:text-6xl font-bold text-white tracking-tight" data-testid="text-hero-title">
            건강한 피부를 위한
            <br />
            스마트한 성분 분석
          </h1>
          
          <p className="text-lg md:text-xl text-white/90 max-w-2xl mx-auto" data-testid="text-hero-subtitle">
            AI 기술로 화장품 성분을 분석하고, 나에게 맞는 제품 조합을 추천받으세요
          </p>

          <div className="flex flex-col sm:flex-row items-center justify-center gap-4 pt-4">
            <Link href="/analyze">
              <a data-testid="link-hero-cta">
                <Button 
                  size="lg" 
                  variant="default"
                  className="text-lg gap-2 min-w-[200px]" 
                  data-testid="button-hero-start"
                >
                  <Sparkles className="h-5 w-5" />
                  분석 시작하기
                </Button>
              </a>
            </Link>
            <Link href="/collection">
              <a data-testid="link-hero-collection">
                <Button 
                  size="lg" 
                  variant="outline"
                  className="text-lg min-w-[200px] bg-background/20 backdrop-blur-sm border-white/30 text-white hover:bg-background/30" 
                  data-testid="button-hero-collection"
                >
                  내 화장대 보기
                </Button>
              </a>
            </Link>
          </div>
        </div>
      </div>
    </section>
  );
}
