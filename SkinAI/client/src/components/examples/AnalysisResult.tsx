import AnalysisResult from '../AnalysisResult';

export default function AnalysisResultExample() {
  const mockIngredients = [
    { name: "나이아신아마이드", safetyLevel: "safe" as const },
    { name: "히알루론산", safetyLevel: "safe" as const },
    { name: "판테놀", safetyLevel: "safe" as const },
    { name: "레티놀", safetyLevel: "caution" as const },
    { name: "향료", safetyLevel: "caution" as const },
    { name: "파라벤", safetyLevel: "harmful" as const },
  ];

  return (
    <div className="max-w-3xl p-6">
      <AnalysisResult
        productName="청클 비타C 잡티 케어 세럼"
        brand="구달"
        safetyScore={72}
        ingredients={mockIngredients}
        summary="전반적으로 양호한 성분 구성이지만, 일부 주의가 필요한 성분이 포함되어 있습니다."
        warnings={["파라벤 성분이 포함되어 있어 민감성 피부에는 자극을 줄 수 있습니다."]}
        recommendations={["나이아신아마이드와 히알루론산이 함유되어 미백과 보습에 효과적입니다."]}
        onAddToCollection={() => console.log('Added to collection')}
        onCancel={() => console.log('Cancelled')}
      />
    </div>
  );
}
