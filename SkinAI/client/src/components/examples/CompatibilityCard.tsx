import CompatibilityCard from '../CompatibilityCard';
import product1 from '@assets/stock_images/cosmetic_serum_bottl_031e8a7b.jpg';
import product2 from '@assets/stock_images/skincare_cream_tube__319ed8ff.jpg';

export default function CompatibilityCardExample() {
  const mockProducts = [
    { id: "1", name: "비타민 C 세럼", image: product1 },
    { id: "2", name: "보습 크림", image: product2 },
  ];

  return (
    <div className="max-w-md p-6 space-y-4">
      <CompatibilityCard
        products={mockProducts}
        compatibility="good"
        message="비타민 C와 보습 성분이 시너지 효과를 발휘하여 피부 개선에 도움을 줍니다."
      />
    </div>
  );
}
