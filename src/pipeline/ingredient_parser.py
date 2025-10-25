import re
from typing import List, Dict, Optional, Tuple

try:
	from rapidfuzz import process, fuzz
	HAS_FUZZ = True
except Exception:
	HAS_FUZZ = False


def normalize_text(text: str) -> str:
	text = re.sub(r"\(.*?\)", " ", text)
	text = re.sub(r"\d+(\.\d+)?\s*%", " ", text)
	text = re.sub(r"[·|/;]", ",", text)
	text = re.sub(r"\s+", " ", text).strip()
	return text


def safe_split(s: str) -> List[str]:
	tokens: List[str] = []
	buf: List[str] = []
	depth = 0
	for ch in s:
		if ch == "(":
			depth += 1
		elif ch == ")":
			depth = max(0, depth - 1)
		if ch == "," and depth == 0:
			tok = "".join(buf).strip()
			if tok:
				tokens.append(tok)
			buf = []
		else:
			buf.append(ch)
	last = "".join(buf).strip()
	if last:
		tokens.append(last)
	return tokens


class IngredientParser:
	def __init__(
		self,
		master_by_std: Dict[str, Dict],
		synonyms: Dict[str, str],
		ko_to_en: Dict[str, str],
		en_to_ko: Dict[str, str],
		skip_tokens: Optional[List[str]] = None,
	):
		self.master_by_std = master_by_std
		self.synonyms = synonyms
		self.ko_to_en = ko_to_en
		self.en_to_ko = en_to_ko
		self.skip_tokens = set(skip_tokens or [])
		self.std_keys = list(master_by_std.keys())

	def _to_standard(self, token: str) -> Optional[Tuple[str, Dict, str, int]]:
		t = token.strip()
		if not t or t in self.skip_tokens:
			return None
		# exact by synonyms
		if t in self.synonyms:
			std = self.synonyms[t]
			return std, self.master_by_std.get(std, {}), "exact", 100
		# ko<->en mapping + synonyms
		mapped = self.ko_to_en.get(t) or self.en_to_ko.get(t)
		if mapped and mapped in self.synonyms:
			std = self.synonyms[mapped]
			return std, self.master_by_std.get(std, {}), "mapped", 100
		# fuzzy
		if HAS_FUZZ and self.std_keys:
			res = process.extractOne(t, self.std_keys, scorer=fuzz.WRatio)
			if res and res[1] >= 90:
				std = res[0]
				return std, self.master_by_std.get(std, {}), "fuzzy", int(res[1])
		return None

	def parse(self, full_text: str) -> List[Dict]:
		text = normalize_text(full_text or "")
		raw = safe_split(text)
		out: List[Dict] = []
		seen = set()
		for tok in raw:
			mapped = self._to_standard(tok)
			if not mapped:
				continue
			std, meta, mtype, score = mapped
			if std in seen:
				continue
			seen.add(std)
			out.append({
				"standard_key": std,
				"korean_name": meta.get("korean_name"),
				"english_name": meta.get("english_name"),
				"family": meta.get("family"),
				"pH": meta.get("pH"),
				"source_text": tok,
				"match_type": mtype,
				"match_score": score,
			})
		return out



