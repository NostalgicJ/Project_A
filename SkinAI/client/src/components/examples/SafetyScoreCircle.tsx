import SafetyScoreCircle from '../SafetyScoreCircle';

export default function SafetyScoreCircleExample() {
  return (
    <div className="flex items-center justify-center gap-8 p-6">
      <SafetyScoreCircle score={85} size="sm" />
      <SafetyScoreCircle score={65} size="md" />
      <SafetyScoreCircle score={45} size="lg" />
    </div>
  );
}
