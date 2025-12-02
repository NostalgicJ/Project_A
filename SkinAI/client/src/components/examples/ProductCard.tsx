import ProductCard from '../ProductCard';
import productImage from '@assets/stock_images/cosmetic_serum_bottl_031e8a7b.jpg';

export default function ProductCardExample() {
  return (
    <div className="max-w-sm p-6">
      <ProductCard
        id="1"
        name="청클 비타C 잡티 케어 세럼"
        brand="구달"
        image={productImage}
        safetyScore={85}
        ingredientCount={24}
        category="세럼"
        onDelete={(id) => console.log('Delete:', id)}
        onViewDetails={(id) => console.log('View details:', id)}
      />
    </div>
  );
}
