import FeatureCard from '../FeatureCard';
import { Sparkles } from 'lucide-react';

export default function FeatureCardExample() {
  return (
    <div className="max-w-sm p-6">
      <FeatureCard
        icon={Sparkles}
        title="AI 성분 분석"
        description="최신 AI 기술로 화장품 성분을 정밀 분석하고 안전성을 평가합니다."
      />
    </div>
  );
}
