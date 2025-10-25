import os
import pandas as pd
from typing import List, Dict

from src.pipeline.ingredient_parser import IngredientParser
from src.pipeline.build_master_index import build_master_index, build_synonyms, build_bilingual_maps

PRODUCTS_CSV = "data/processed_cosmetics_final_2.csv"
OUT_PARSED_ING = "data/parsed_ingredients.csv"
OUT_PRODUCTS_PARSED = "data/products_parsed.csv"

SKIP_TOKENS = {"정제수", "향료"}


def main():
	assert os.path.exists(PRODUCTS_CSV), f"Missing file: {PRODUCTS_CSV}"
	os.makedirs("data", exist_ok=True)

	master_by_std = build_master_index()
	synonyms = build_synonyms(master_by_std)
	ko2en, en2ko = build_bilingual_maps(master_by_std)

	parser = IngredientParser(
		master_by_std=master_by_std,
		synonyms=synonyms,
		ko_to_en=ko2en,
		en_to_ko=en2ko,
		skip_tokens=list(SKIP_TOKENS),
	)

	prod_df = pd.read_csv(PRODUCTS_CSV)
	# Heuristic columns
	brand_col = next((c for c in prod_df.columns if '브랜드' in c), None)
	name_col = next((c for c in prod_df.columns if '제품명' in c), None)
	ing_col = next((c for c in prod_df.columns if '성분' in c), None)
	assert ing_col, "성분 리스트 컬럼을 찾지 못했습니다. (예: '성분_문자열')"

	rows_ingredients: List[Dict] = []
	rows_products: List[Dict] = []

	for _, row in prod_df.iterrows():
		brand = str(row.get(brand_col, '')).strip() if brand_col else ''
		name = str(row.get(name_col, '')).strip() if name_col else ''
		full = f"{brand} {name}".strip()
		text = str(row.get(ing_col, '') or '')

		parsed = parser.parse(text)
		for item in parsed:
			rows_ingredients.append({
				"brand": brand,
				"product_name": name,
				"product_full": full,
				**item,
			})

		std_list = [x['standard_key'] for x in parsed]
		rows_products.append({
			"brand": brand,
			"product_name": name,
			"product_full": full,
			"standard_ingredients": ','.join(std_list),
			"num_ingredients": len(std_list),
		})

	pd.DataFrame(rows_ingredients).to_csv(OUT_PARSED_ING, index=False)
	pd.DataFrame(rows_products).to_csv(OUT_PRODUCTS_PARSED, index=False)
	print(f"✅ Wrote {OUT_PARSED_ING} and {OUT_PRODUCTS_PARSED}")


if __name__ == "__main__":
	main()



