import pandas as pd
from typing import Dict

# Input CSV
PRODUCTS_CSV = "data/processed_cosmetics_final_2.csv"
MASTER_CSV = "data/coos_master_ingredients_cleaned.csv"

# Output pickle (optional future)
# MASTER_PICKLE = "data/master_index.pkl"

def build_master_index() -> Dict[str, dict]:
	"""Build master_by_std dict: standard_key -> meta (ko/en/family/pH)."""
	master_df = pd.read_csv(MASTER_CSV)
	master_by_std: Dict[str, dict] = {}
	
	# Heuristic column names (adjust if schema differs)
	ko_col = next((c for c in master_df.columns if '원료명' in c or '한글' in c or 'korean' in c.lower()), None)
	en_col = next((c for c in master_df.columns if '영문' in c or 'english' in c.lower()), None)
	family_col = next((c for c in master_df.columns if '계열' in c or 'family' in c.lower()), None)
	ph_col = next((c for c in master_df.columns if c.lower() == 'ph' or 'ph' in c.lower()), None)
	
	for _, row in master_df.iterrows():
		ko = str(row.get(ko_col, '')).strip() if ko_col else ''
		en = str(row.get(en_col, '')).strip() if en_col else ''
		std = en or ko  # prefer English as standard key
		if not std:
			continue
		master_by_std[std] = {
			"korean_name": ko or None,
			"english_name": en or None,
			"family": (str(row.get(family_col)).strip() if family_col else None) or None,
			"pH": row.get(ph_col) if ph_col else None,
		}
	return master_by_std


def build_synonyms(master_by_std: Dict[str, dict]) -> Dict[str, str]:
	"""Build simple synonyms table (identity and lowercased). Extend later with curated list."""
	syn: Dict[str, str] = {}
	for std in master_by_std.keys():
		syn[std] = std
		syn[std.lower()] = std
		meta = master_by_std[std]
		if meta.get("korean_name"):
			ko = meta["korean_name"]
			syn[ko] = std
			syn[ko.lower()] = std
	return syn


def build_bilingual_maps(master_by_std: Dict[str, dict]) -> tuple[Dict[str, str], Dict[str, str]]:
	ko2en: Dict[str, str] = {}
	en2ko: Dict[str, str] = {}
	for std, meta in master_by_std.items():
		ko = meta.get("korean_name")
		en = meta.get("english_name") or std
		if ko:
			ko2en[ko] = en
			en2ko[en] = ko
	return ko2en, en2ko

if __name__ == "__main__":
	master_by_std = build_master_index()
	synonyms = build_synonyms(master_by_std)
	ko2en, en2ko = build_bilingual_maps(master_by_std)
	print(f"✅ master entries: {len(master_by_std)} | synonyms: {len(synonyms)} | ko2en: {len(ko2en)}")



