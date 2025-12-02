import { Card, CardContent, CardFooter } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Trash2, ChevronRight } from "lucide-react";

interface ProductCardProps {
  id: string;
  name: string;
  brand: string;
  ingredientCount: number;
  category?: string;
  onDelete?: (id: string) => void;
  onViewDetails?: (id: string) => void;
}

export default function ProductCard({
  id,
  name,
  brand,
  ingredientCount,
  category,
  onDelete,
  onViewDetails,
}: ProductCardProps) {
  return (
    <Card className="overflow-hidden hover-elevate" data-testid={`card-product-${id}`}>
      <CardContent className="p-4 space-y-2">
        <div className="flex items-start justify-between gap-2">
          <div className="flex-1 min-w-0">
            <h3 
              className="font-semibold text-base line-clamp-2 cursor-pointer hover:text-primary" 
              onClick={() => onViewDetails?.(id)}
              data-testid={`text-product-name-${id}`}
            >
              {name}
            </h3>
            <p className="text-sm text-muted-foreground mt-1" data-testid={`text-brand-${id}`}>
              {brand}
            </p>
          </div>
        </div>

        {category && (
          <Badge variant="secondary" className="text-xs" data-testid={`badge-category-${id}`}>
            {category}
          </Badge>
        )}
        
        <p className="text-xs text-muted-foreground" data-testid={`text-ingredient-count-${id}`}>
          성분 {ingredientCount}개
        </p>
      </CardContent>

      <CardFooter className="p-4 pt-0 flex gap-2">
        <Button 
          variant="outline" 
          size="sm" 
          className="flex-1 gap-2"
          onClick={() => onViewDetails?.(id)}
          data-testid={`button-view-${id}`}
        >
          상세보기
          <ChevronRight className="h-4 w-4" />
        </Button>
        <Button 
          variant="ghost" 
          size="icon"
          onClick={() => onDelete?.(id)}
          data-testid={`button-delete-${id}`}
        >
          <Trash2 className="h-4 w-4" />
        </Button>
      </CardFooter>
    </Card>
  );
}
