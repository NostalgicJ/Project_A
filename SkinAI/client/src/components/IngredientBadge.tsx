import { Badge } from "@/components/ui/badge";
import { AlertCircle, CheckCircle, AlertTriangle } from "lucide-react";

interface IngredientBadgeProps {
  name: string;
  safetyLevel: "safe" | "caution" | "harmful";
  onClick?: () => void;
}

export default function IngredientBadge({ name, safetyLevel, onClick }: IngredientBadgeProps) {
  const getStyles = () => {
    switch (safetyLevel) {
      case "safe":
        return {
          className: "bg-chart-2/10 text-chart-2 border-chart-2/20",
          Icon: CheckCircle,
        };
      case "caution":
        return {
          className: "bg-chart-3/10 text-chart-3 border-chart-3/20",
          Icon: AlertTriangle,
        };
      case "harmful":
        return {
          className: "bg-destructive/10 text-destructive border-destructive/20",
          Icon: AlertCircle,
        };
    }
  };

  const { className, Icon } = getStyles();

  return (
    <Badge 
      variant="outline" 
      className={`gap-1 cursor-pointer hover-elevate ${className}`}
      onClick={onClick}
      data-testid={`badge-ingredient-${name}`}
    >
      <Icon className="h-3 w-3" />
      <span>{name}</span>
    </Badge>
  );
}
