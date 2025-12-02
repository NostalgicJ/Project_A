import IngredientBadge from '../IngredientBadge';

export default function IngredientBadgeExample() {
  return (
    <div className="flex flex-wrap gap-2 p-6">
      <IngredientBadge 
        name="나이아신아마이드" 
        safetyLevel="safe"
        onClick={() => console.log('Clicked safe ingredient')}
      />
      <IngredientBadge 
        name="레티놀" 
        safetyLevel="caution"
        onClick={() => console.log('Clicked caution ingredient')}
      />
      <IngredientBadge 
        name="파라벤" 
        safetyLevel="harmful"
        onClick={() => console.log('Clicked harmful ingredient')}
      />
    </div>
  );
}
