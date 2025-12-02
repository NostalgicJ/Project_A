import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { AlertTriangle, CheckCircle, Plus, X } from "lucide-react";
import SafetyScoreCircle from "./SafetyScoreCircle";
import IngredientBadge from "./IngredientBadge";
import { Badge } from "@/components/ui/badge";

interface Ingredient {
  name: string;
  safetyLevel: "safe" | "caution" | "harmful";
  description?: string;
}

interface AnalysisResultProps {
  productName: string;
  brand: string;
  safetyScore: number;
  ingredients: Ingredient[];
  summary: string;
  warnings?: string[];
  recommendations?: string[];
  onAddToCollection?: () => void;
  onCancel?: () => void;
}

export default function AnalysisResult({
  productName,
  brand,
  safetyScore,
  ingredients,
  summary,
  warnings = [],
  recommendations = [],
  onAddToCollection,
  onCancel,
}: AnalysisResultProps) {
  const safeIngredients = ingredients.filter(i => i.safetyLevel === "safe");
  const cautionIngredients = ingredients.filter(i => i.safetyLevel === "caution");
  const harmfulIngredients = ingredients.filter(i => i.safetyLevel === "harmful");

  return (
    <div className="space-y-6" data-testid="container-analysis-result">
      {(onAddToCollection || onCancel) && (
        <div className="flex gap-3">
          {onAddToCollection && (
            <Button 
              className="flex-1 gap-2" 
              size="lg"
              onClick={onAddToCollection}
              data-testid="button-add-to-collection"
            >
              <Plus className="h-5 w-5" />
              내 화장대에 추가
            </Button>
          )}
          {onCancel && (
            <Button 
              variant="outline" 
              size="lg"
              onClick={onCancel}
              data-testid="button-cancel"
            >
              <X className="h-5 w-5 mr-2" />
              취소하기
            </Button>
          )}
        </div>
      )}
      
      <Card>
        <CardHeader>
          <div className="flex items-start justify-between gap-4">
            <div>
              <CardTitle data-testid="text-result-product-name">{productName}</CardTitle>
              <CardDescription data-testid="text-result-brand">{brand}</CardDescription>
            </div>
            <Badge variant="secondary" data-testid="badge-ingredient-count">
              성분 {ingredients.length}개
            </Badge>
          </div>
        </CardHeader>
        
        <CardContent className="space-y-6">
          <div className="flex flex-col items-center">
            <SafetyScoreCircle score={safetyScore} size="lg" />
            <p className="text-center text-sm text-muted-foreground mt-4 max-w-md" data-testid="text-summary">
              {summary}
            </p>
          </div>
        </CardContent>
      </Card>

      {warnings.length > 0 && (
        <Alert variant="destructive" data-testid="alert-warnings">
          <AlertTriangle className="h-4 w-4" />
          <AlertDescription>
            <ul className="list-disc list-inside space-y-1">
              {warnings.map((warning, index) => (
                <li key={index} data-testid={`text-warning-${index}`}>{warning}</li>
              ))}
            </ul>
          </AlertDescription>
        </Alert>
      )}

      {recommendations.length > 0 && (
        <Alert data-testid="alert-recommendations">
          <CheckCircle className="h-4 w-4" />
          <AlertDescription>
            <ul className="list-disc list-inside space-y-1">
              {recommendations.map((rec, index) => (
                <li key={index} data-testid={`text-recommendation-${index}`}>{rec}</li>
              ))}
            </ul>
          </AlertDescription>
        </Alert>
      )}

      <Card>
        <CardHeader>
          <CardTitle>성분 분석</CardTitle>
        </CardHeader>
        <CardContent className="space-y-6">
          {harmfulIngredients.length > 0 && (
            <div>
              <h4 className="text-sm font-semibold mb-3 text-destructive flex items-center gap-2">
                <AlertTriangle className="h-4 w-4" />
                주의 필요 성분 ({harmfulIngredients.length})
              </h4>
              <div className="flex flex-wrap gap-2">
                {harmfulIngredients.map((ingredient) => (
                  <IngredientBadge
                    key={ingredient.name}
                    name={ingredient.name}
                    safetyLevel={ingredient.safetyLevel}
                    onClick={() => console.log('Ingredient clicked:', ingredient.name)}
                  />
                ))}
              </div>
            </div>
          )}

          {cautionIngredients.length > 0 && (
            <div>
              <h4 className="text-sm font-semibold mb-3 text-chart-3 flex items-center gap-2">
                <AlertTriangle className="h-4 w-4" />
                주의 성분 ({cautionIngredients.length})
              </h4>
              <div className="flex flex-wrap gap-2">
                {cautionIngredients.map((ingredient) => (
                  <IngredientBadge
                    key={ingredient.name}
                    name={ingredient.name}
                    safetyLevel={ingredient.safetyLevel}
                    onClick={() => console.log('Ingredient clicked:', ingredient.name)}
                  />
                ))}
              </div>
            </div>
          )}

          {safeIngredients.length > 0 && (
            <div>
              <h4 className="text-sm font-semibold mb-3 text-chart-2 flex items-center gap-2">
                <CheckCircle className="h-4 w-4" />
                안전한 성분 ({safeIngredients.length})
              </h4>
              <div className="flex flex-wrap gap-2">
                {safeIngredients.map((ingredient) => (
                  <IngredientBadge
                    key={ingredient.name}
                    name={ingredient.name}
                    safetyLevel={ingredient.safetyLevel}
                    onClick={() => console.log('Ingredient clicked:', ingredient.name)}
                  />
                ))}
              </div>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
