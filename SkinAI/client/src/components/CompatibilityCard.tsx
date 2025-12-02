import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { CheckCircle, XCircle, AlertTriangle } from "lucide-react";

interface ProductWithIngredient {
  id: string;
  productName: string;
  ingredientName: string;
}

interface CompatibilityCardProps {
  products: ProductWithIngredient[];
  compatibility: "good" | "caution" | "bad";
  message: string;
}

export default function CompatibilityCard({ products, compatibility, message }: CompatibilityCardProps) {
  const getStyles = () => {
    switch (compatibility) {
      case "good":
        return {
          Icon: CheckCircle,
          badgeClass: "bg-chart-2 text-white",
          label: "궁합 좋음",
        };
      case "caution":
        return {
          Icon: AlertTriangle,
          badgeClass: "bg-chart-3 text-white",
          label: "주의 필요",
        };
      case "bad":
        return {
          Icon: XCircle,
          badgeClass: "bg-destructive text-destructive-foreground",
          label: "함께 사용 자제",
        };
    }
  };

  const { Icon, badgeClass, label } = getStyles();

  return (
    <Card className="hover-elevate" data-testid={`card-compatibility-${compatibility}`}>
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="text-base">제품 조합</CardTitle>
          <Badge className={badgeClass} data-testid={`badge-compatibility-${compatibility}`}>
            <Icon className="h-3 w-3 mr-1" />
            {label}
          </Badge>
        </div>
      </CardHeader>
      
      <CardContent className="space-y-4">
        <div className="flex flex-wrap items-center gap-2">
          {products.map((product, index) => (
            <div key={product.id} className="flex items-center gap-2">
              {index > 0 && <span className="text-muted-foreground text-lg">+</span>}
              <div className="flex flex-col gap-1">
                <Badge 
                  variant="outline" 
                  className="text-sm"
                  data-testid={`badge-compatibility-product-${product.id}`}
                >
                  {product.productName}
                </Badge>
                <Badge 
                  variant="secondary" 
                  className="text-xs"
                  data-testid={`badge-compatibility-ingredient-${product.id}`}
                >
                  {product.ingredientName}
                </Badge>
              </div>
            </div>
          ))}
        </div>
        
        <p className="text-sm text-muted-foreground" data-testid="text-compatibility-message">
          {message}
        </p>
      </CardContent>
    </Card>
  );
}
