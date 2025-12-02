interface SafetyScoreCircleProps {
  score: number;
  size?: "sm" | "md" | "lg";
}

export default function SafetyScoreCircle({ score, size = "md" }: SafetyScoreCircleProps) {
  const dimensions = {
    sm: { container: 80, radius: 32, stroke: 6 },
    md: { container: 120, radius: 48, stroke: 8 },
    lg: { container: 160, radius: 64, stroke: 10 },
  };

  const { container, radius, stroke } = dimensions[size];
  const circumference = 2 * Math.PI * radius;
  const progress = (score / 100) * circumference;
  
  const getColor = () => {
    if (score >= 80) return "hsl(var(--chart-2))";
    if (score >= 60) return "hsl(var(--chart-3))";
    return "hsl(var(--destructive))";
  };

  const getLabel = () => {
    if (score >= 80) return "안전";
    if (score >= 60) return "주의";
    return "위험";
  };

  const textSize = {
    sm: "text-xl",
    md: "text-3xl",
    lg: "text-4xl",
  };

  return (
    <div className="flex flex-col items-center gap-2">
      <div className="relative" style={{ width: container, height: container }}>
        <svg className="transform -rotate-90" width={container} height={container}>
          <circle
            cx={container / 2}
            cy={container / 2}
            r={radius}
            stroke="hsl(var(--muted))"
            strokeWidth={stroke}
            fill="none"
          />
          <circle
            cx={container / 2}
            cy={container / 2}
            r={radius}
            stroke={getColor()}
            strokeWidth={stroke}
            fill="none"
            strokeDasharray={circumference}
            strokeDashoffset={circumference - progress}
            strokeLinecap="round"
            className="transition-all duration-500"
          />
        </svg>
        <div className="absolute inset-0 flex items-center justify-center">
          <div className="text-center">
            <div className={`${textSize[size]} font-bold`} data-testid="text-safety-score">
              {score}
            </div>
            <div className="text-xs text-muted-foreground" data-testid="text-safety-label">
              {getLabel()}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
