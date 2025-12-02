import CompatibilityCard from "@/components/CompatibilityCard";
import ProductCard from "@/components/ProductCard";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";

export default function Recommendations() {
  const compatibilityData = [
    {
      products: [
        { id: "1", productName: "청클 비타C 잡티 케어 세럼", ingredientName: "아스코르빈산" },
        { id: "2", productName: "본쎈 2500IU 밀봉샷 퍼펙터", ingredientName: "히알루론산" },
      ],
      compatibility: "good" as const,
      message: "비타민 C(아스코르빈산)와 히알루론산은 함께 사용하면 시너지 효과를 발휘하여 피부 미백과 보습에 도움을 줍니다.",
    },
    {
      products: [
        { id: "1", productName: "청클 비타C 잡티 케어 세럼", ingredientName: "아스코르빈산" },
        { id: "3", productName: "레티놀 앰플", ingredientName: "레티놀" },
      ],
      compatibility: "caution" as const,
      message: "비타민 C(아스코르빈산)와 레티놀을 같은 시간대에 사용하면 피부 자극이 발생할 수 있습니다. 아침에는 비타민 C를, 저녁에는 레티놀을 사용하는 것을 권장합니다.",
    },
  ];

  const recommendedProducts = [
    {
      id: "rec-1",
      name: "센텔라 진정 크림",
      brand: "아토베리어",
      ingredientCount: 16,
      category: "크림",
    },
    {
      id: "rec-2",
      name: "글로우 픽 세럼",
      brand: "코스알엑스",
      ingredientCount: 19,
      category: "세럼",
    },
    {
      id: "rec-3",
      name: "약산성 토너",
      brand: "아이소이",
      ingredientCount: 14,
      category: "토너",
    },
  ];

  return (
    <div className="min-h-screen py-12 px-6">
      <div className="container max-w-7xl mx-auto space-y-12">
        <div>
          <h1 className="text-4xl font-bold mb-2" data-testid="text-recommendations-title">
            관리하기
          </h1>
          <p className="text-muted-foreground" data-testid="text-recommendations-subtitle">
            내 화장대 제품들의 조합을 분석하고 추천 제품을 확인하세요
          </p>
        </div>

        <section>
          <Card>
            <CardHeader>
              <CardTitle data-testid="text-compatibility-title">내 화장대 분석 결과</CardTitle>
              <CardDescription data-testid="text-compatibility-subtitle">
                현재 사용 중인 제품들의 성분 조합 분석 결과입니다
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid md:grid-cols-2 gap-4">
                {compatibilityData.map((data, index) => (
                  <CompatibilityCard key={index} {...data} />
                ))}
              </div>
              <p className="text-sm text-muted-foreground mt-4" data-testid="text-compatibility-note">
                비타민 C와 레티놀은 같은 시간대에 사용하면 피부에 자극을 줄 수 있으니 
                아침에는 비타민 C를, 저녁에는 레티놀을 사용하는 것을 권장합니다.
              </p>
            </CardContent>
          </Card>
        </section>

        <section>
          <div className="mb-6">
            <h2 className="text-2xl font-bold mb-2" data-testid="text-recommended-title">
              추천 제품
            </h2>
            <p className="text-muted-foreground" data-testid="text-recommended-subtitle">
              화장품을 대신 사용할때 추천하는 제품을 참고해주세요
            </p>
          </div>

          <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-6">
            {recommendedProducts.map((product) => (
              <ProductCard
                key={product.id}
                {...product}
                onViewDetails={(id) => console.log("View details:", id)}
              />
            ))}
          </div>
        </section>
      </div>
    </div>
  );
}
