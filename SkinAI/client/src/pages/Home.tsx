import HeroSection from "@/components/HeroSection";
import FeatureCard from "@/components/FeatureCard";
import { Sparkles, Shield, Heart, TrendingUp } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Link } from "wouter";

export default function Home() {
  const features = [
    {
      icon: Sparkles,
      title: "AI 성분 분석",
      description: "최신 AI 기술로 화장품 성분을 정밀 분석하고 안전성을 평가합니다.",
    },
    {
      icon: Shield,
      title: "안전성 검증",
      description: "전문가가 검증한 데이터베이스를 기반으로 신뢰할 수 있는 분석 결과를 제공합니다.",
    },
    {
      icon: Heart,
      title: "맞춤 추천",
      description: "사용 중인 제품과 궁합이 좋은 화장품을 추천하여 최적의 스킨케어 루틴을 만들어드립니다.",
    },
    {
      icon: TrendingUp,
      title: "트렌드 분석",
      description: "최신 뷰티 트렌드와 효과적인 성분 조합을 분석하여 정보를 제공합니다.",
    },
  ];

  const steps = [
    { number: 1, title: "제품 등록", description: "사용 중인 화장품의 성분을 입력하세요" },
    { number: 2, title: "AI 분석", description: "AI가 성분을 분석하고 안전성을 평가합니다" },
    { number: 3, title: "결과 확인", description: "상세한 분석 결과와 주의사항을 확인하세요" },
    { number: 4, title: "제품 추천", description: "궁합이 좋은 제품을 추천받으세요" },
  ];

  return (
    <div className="min-h-screen">
      <HeroSection />
      
      <section className="py-20 px-6">
        <div className="container max-w-6xl mx-auto">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold mb-4" data-testid="text-features-title">
              왜 COSME을 선택해야 할까요?
            </h2>
            <p className="text-muted-foreground max-w-2xl mx-auto" data-testid="text-features-subtitle">
              전문적인 성분 분석으로 더 건강하고 효과적인 스킨케어 루틴을 만들어보세요
            </p>
          </div>

          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
            {features.map((feature) => (
              <FeatureCard
                key={feature.title}
                icon={feature.icon}
                title={feature.title}
                description={feature.description}
              />
            ))}
          </div>
        </div>
      </section>

      <section className="py-20 px-6 bg-muted/30">
        <div className="container max-w-6xl mx-auto">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold mb-4" data-testid="text-howto-title">
              이용 방법
            </h2>
            <p className="text-muted-foreground" data-testid="text-howto-subtitle">
              간단한 4단계로 성분 분석을 시작하세요
            </p>
          </div>

          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-8">
            {steps.map((step) => (
              <div key={step.number} className="text-center space-y-4" data-testid={`card-step-${step.number}`}>
                <div className="w-16 h-16 rounded-full bg-primary text-primary-foreground flex items-center justify-center text-2xl font-bold mx-auto">
                  {step.number}
                </div>
                <h3 className="font-semibold text-lg" data-testid={`text-step-title-${step.number}`}>
                  {step.title}
                </h3>
                <p className="text-sm text-muted-foreground" data-testid={`text-step-desc-${step.number}`}>
                  {step.description}
                </p>
              </div>
            ))}
          </div>
        </div>
      </section>

      <section className="py-20 px-6">
        <div className="container max-w-4xl mx-auto text-center space-y-6">
          <h2 className="text-3xl font-bold" data-testid="text-cta-title">
            지금 바로 시작하세요
          </h2>
          <p className="text-muted-foreground text-lg" data-testid="text-cta-subtitle">
            무료로 화장품 성분을 분석하고 더 건강한 피부를 가꿔보세요
          </p>
          <div className="flex flex-col sm:flex-row items-center justify-center gap-4 pt-4">
            <Link href="/analyze">
              <a data-testid="link-cta-analyze">
                <Button size="lg" className="gap-2 min-w-[200px]" data-testid="button-cta-start">
                  <Sparkles className="h-5 w-5" />
                  분석 시작하기
                </Button>
              </a>
            </Link>
            <Link href="/collection">
              <a data-testid="link-cta-collection">
                <Button size="lg" variant="outline" className="min-w-[200px]" data-testid="button-cta-collection">
                  내 화장대 보기
                </Button>
              </a>
            </Link>
          </div>
        </div>
      </section>
    </div>
  );
}
