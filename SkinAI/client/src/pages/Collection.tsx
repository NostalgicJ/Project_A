import { useState } from "react";
import ProductCard from "@/components/ProductCard";
import { Button } from "@/components/ui/button";
import { Plus, Search } from "lucide-react";
import { Input } from "@/components/ui/input";
import { Link } from "wouter";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";

export default function Collection() {
  const [searchQuery, setSearchQuery] = useState("");
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [productToDelete, setProductToDelete] = useState<string | null>(null);

  const mockProducts = [
    {
      id: "1",
      name: "청클 비타C 잡티 케어 세럼",
      brand: "구달",
      ingredientCount: 24,
      category: "세럼",
    },
    {
      id: "2",
      name: "본쎈 2500IU 밀봉샷 퍼펙터",
      brand: "디아이즈",
      ingredientCount: 18,
      category: "앰플",
    },
    {
      id: "3",
      name: "로즈워터 토너",
      brand: "마몽드",
      ingredientCount: 15,
      category: "토너",
    },
    {
      id: "4",
      name: "다이발 저분자 히알루론산 세럼",
      brand: "두엔드",
      ingredientCount: 20,
      category: "세럼",
    },
    {
      id: "5",
      name: "시카리카 토너패드",
      brand: "다이소",
      ingredientCount: 22,
      category: "패드",
    },
  ];

  const handleDeleteClick = (id: string) => {
    setProductToDelete(id);
    setDeleteDialogOpen(true);
  };

  const handleConfirmDelete = () => {
    if (productToDelete) {
      console.log("Delete product:", productToDelete);
      // TODO: Implement actual deletion
    }
    setDeleteDialogOpen(false);
    setProductToDelete(null);
  };

  const handleViewDetails = (id: string) => {
    console.log("View details:", id);
  };

  return (
    <div className="min-h-screen py-12 px-6">
      <div className="container max-w-7xl mx-auto">
        <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4 mb-8">
          <div>
            <h1 className="text-4xl font-bold mb-2" data-testid="text-collection-title">
              내 화장대
            </h1>
            <p className="text-muted-foreground" data-testid="text-collection-count">
              총 {mockProducts.length}개의 제품
            </p>
          </div>

          <div className="flex gap-2">
            <div className="relative flex-1 md:w-64">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
              <Input
                placeholder="제품 검색..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="pl-9"
                data-testid="input-search-products"
              />
            </div>
            <Link href="/analyze">
              <a data-testid="link-add-product">
                <Button className="gap-2" data-testid="button-add-product">
                  <Plus className="h-4 w-4" />
                  제품 추가
                </Button>
              </a>
            </Link>
          </div>
        </div>

        <div className="grid sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
          {mockProducts.map((product) => (
            <ProductCard
              key={product.id}
              {...product}
              onDelete={handleDeleteClick}
              onViewDetails={handleViewDetails}
            />
          ))}
        </div>

        <AlertDialog open={deleteDialogOpen} onOpenChange={setDeleteDialogOpen}>
          <AlertDialogContent data-testid="dialog-delete-confirm">
            <AlertDialogHeader>
              <AlertDialogTitle data-testid="text-dialog-title">
                제품을 삭제하시겠습니까?
              </AlertDialogTitle>
              <AlertDialogDescription data-testid="text-dialog-description">
                이 작업은 되돌릴 수 없습니다. 제품이 내 화장대에서 영구적으로 삭제됩니다.
              </AlertDialogDescription>
            </AlertDialogHeader>
            <AlertDialogFooter>
              <AlertDialogCancel data-testid="button-dialog-cancel">취소</AlertDialogCancel>
              <AlertDialogAction 
                onClick={handleConfirmDelete}
                data-testid="button-dialog-confirm"
              >
                삭제
              </AlertDialogAction>
            </AlertDialogFooter>
          </AlertDialogContent>
        </AlertDialog>
      </div>
    </div>
  );
}
