# matcher_simple.py — pipeline de matching ENTRADA.xlsx ↔ CATALOGO.txt com reranking.
# Melhorias:
# 1) idx_entrada agora é o NÚMERO DA LINHA original da planilha (linha do Excel).
# 2) Removidos "entrada_unidade" e "unidade_catalogo" das saídas CSV.
# 3) Adicionados "entrada_diametro" e "entrada_comprimento_m".
# 4) Melhor match de diâmetro: extração do texto do catálogo (mm/DN/pol) + filtro/bonus.

import argparse
import json
import math
import re
import unicodedata
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, CrossEncoder

# ------------------- Config -------------------

BI_ENCODER_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # pode trocar por "BAAI/bge-reranker-base"
RERANK_TOP_K = 15  # 10–20 é comum
W_BI = 0.6         # peso do cos_sim (bi-encoder)
W_CE = 0.4         # peso do cross-encoder normalizado
TH_ALTO = 0.90
TH_BAIXO = 0.70
DIAM_TOL_PCT = 0.05     # 5% de tolerância relativa
DIAM_TOL_ABS_MM = 2.0   # e/ou 2mm de tolerância absoluta (o maior prevalece)
DIAM_MATCH_BONUS = 0.05 # bônus se diâmetro bate (clamp no final)

# Defaults de colunas
DEFAULT_DESC_COL = "Descrição do Material"
DEFAULT_EXTRA_NUM_COLS = ["Diâmetro", "Comprimento (m)"]

# Diretório de saída (garante que existe)
ROOT_DIR = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT_DIR / "data" / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CACHE_DIR = ROOT_DIR / "data" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ------------------- Normalização ---------------------

def normalize_text(s: Any) -> str:
    if s is None:
        return ""
    s = str(s)
    s = s.replace("\n", " ").replace("\r", " ").strip()
    s = s.lower()
    s = "".join(ch for ch in unicodedata.normalize("NFD", s) if unicodedata.category(ch) != "Mn")
    s = re.sub(r"[^a-z0-9%/.,:;°\"' \-\(\)\[\]_|]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

UNITS_MAP = {
    "m2": "m2", "m²": "m2",
    "m3": "m3", "m³": "m3",
    "m": "m",
    "mm": "mm",
    "cm": "cm",
    "kg": "kg", "quilograma": "kg",
    "g": "g",
    "un": "un", "und": "un", "unidade": "un", "peca": "un", "peça": "un", "pc": "un",
    "cx": "cx", "caixa": "cx",
    "rl": "rl", "rolo": "rl",
    "pol": "pol", '"': 'pol'
}

def canonical_unit(u: Any) -> str:
    u_norm = normalize_text(u)
    return UNITS_MAP.get(u_norm, u_norm)

# ------------------- Diâmetro helpers -------------------

_INCH_TO_MM = 25.4

DIAM_PATTERNS = [
    re.compile(r"(?:\b|[^0-9])(\d+(?:[.,]\d+)?)\s*mm\b"),
    re.compile(r"(?:ø|diam(?:etro)?\.?\s*)?(\d+(?:[.,]\d+)?)\s*mm\b"),
    re.compile(r"(?:\b|[^0-9])(\d+(?:[.,]\d+)?)\s*(?:\"|''|pol)\b"),
    re.compile(r"\bdn\s*([0-9]+)\b"),
]

def _to_float(x: Any) -> Optional[float]:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    s = str(x).strip()
    if s == "":
        return None
    s = s.replace(" ", "").replace("\u00A0", "")
    s = s.replace(",", ".")
    try:
        return float(s)
    except Exception:
        return None

def parse_diametro_from_text_mm(text: str) -> Optional[float]:
    t = normalize_text(text)
    for pat in DIAM_PATTERNS:
        m = pat.search(t)
        if not m:
            continue
        val = _to_float(m.group(1))
        if val is None:
            continue
        if 'pol' in pat.pattern or '"' in pat.pattern or "''" in pat.pattern:
            return val * _INCH_TO_MM
        if 'dn' in pat.pattern:
            return val
        return val
    return None

def parse_diametro_cell_to_mm(cell: Any) -> Optional[float]:
    if cell is None:
        return None
    s = str(cell).strip()
    if s == "":
        return None
    t = normalize_text(s)
    m = re.search(r"(\d+(?:[.,]\d+)?)\s*mm\b", t)
    if m:
        return _to_float(m.group(1))
    m = re.search(r"(\d+(?:[.,]\d+)?)\s*(?:\"|''|pol)\b", t)
    if m:
        v = _to_float(m.group(1))
        return v * _INCH_TO_MM if v is not None else None
    v = _to_float(s)
    return v

def diameters_compatible(d_in_mm: Optional[float], d_cat_mm: Optional[float]) -> Optional[bool]:
    if d_in_mm is None or d_cat_mm is None:
        return None
    tol = max(DIAM_TOL_ABS_MM, d_in_mm * DIAM_TOL_PCT)
    return abs(d_in_mm - d_cat_mm) <= tol

# ------------------- Parser do catálogo .txt ---------------------

KEY_NORMALIZER = {
    "codigo": "codigo",
    "código": "codigo",
    "cod": "codigo",
    "descricao": "descricao",
    "descrição": "descricao",
    "titulo": "descricao",
    "título": "descricao",
    "unidade": "unidade",
    "preco unitario": "preco_unitario",
    "preço unitario": "preco_unitario",
    "preco unitário": "preco_unitario",
    "preço unitário": "preco_unitario",
    "data do preco": "data_preco",
    "data do preço": "data_preco",
    "ativo": "ativo",
    "diametro": "diametro",
    "diâmetro": "diametro",
    "comprimento (m)": "comprimento_m",
    "comprimento": "comprimento_m",
}

def parse_kv_line(line: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for part in line.split("|"):
        kv = part.strip().split(":", 1)
        if len(kv) != 2:
            continue
        k_raw, v_raw = kv[0].strip(), kv[1].strip()
        k_norm = normalize_text(k_raw)
        k_norm = KEY_NORMALIZER.get(k_norm, k_norm)
        out[k_norm] = v_raw
    return out

def read_catalog_txt(p: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    with p.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            kv = parse_kv_line(line)
            if kv:
                rows.append(kv)
    df = pd.DataFrame(rows).fillna("")
    for c in ["codigo", "descricao", "unidade"]:
        if c not in df.columns:
            df[c] = ""
    if "diametro" not in df.columns:
        df["diametro"] = ""
    df["diametro_mm_txt"] = df.apply(
        lambda r: parse_diametro_from_text_mm(f"{r.get('descricao','')} {r.get('diametro','')}"),
        axis=1
    )
    return df

def canonical_text(descricao: str, unidade: str = "", extras: Dict[str, Any] = None) -> str:
    extras = extras or {}
    desc = normalize_text(descricao)
    u = canonical_unit(unidade)
    parts = [desc]
    if u:
        parts.append(f"unidade {u}")
    for k, v in extras.items():
        v_str = str(v).strip()
        if v_str != "" and v_str.lower() != "nan":
            parts.append(f"{normalize_text(k)} {normalize_text(v_str)}")
    return " | ".join([p for p in parts if p])

# ------------------- Index do catálogo ----------------

def build_or_load_catalog_index(catalog_path: Path, cache_path: Path, model: SentenceTransformer):
    meta_path = cache_path.with_suffix(".meta")
    rebuild = True
    if cache_path.exists() and meta_path.exists():
        try:
            prev_mtime = meta_path.read_text().strip()
            curr_mtime = str(catalog_path.stat().st_mtime_ns)
            rebuild = (prev_mtime != curr_mtime)
        except Exception:
            rebuild = True

    if not rebuild:
        try:
            npz = np.load(cache_path, allow_pickle=True)
            embeddings = npz["embeddings"]
            df_cat = pd.read_json(npz["df_json"].item())
            texts = npz["texts"].tolist()
            return df_cat, texts, embeddings
        except Exception:
            pass

    df_cat = read_catalog_txt(catalog_path)
    texts = []
    for _, r in df_cat.iterrows():
        extras = {}
        if pd.notna(r.get("diametro_mm_txt")) and r.get("diametro_mm_txt") != "" and r.get("diametro_mm_txt") is not None:
            extras["diametro_mm"] = f"{float(r['diametro_mm_txt']):.2f} mm"
        texts.append(canonical_text(r.get("descricao",""), r.get("unidade",""), extras))
    emb = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    np.savez_compressed(cache_path, embeddings=emb, texts=np.array(texts, dtype=object), df_json=df_cat.to_json())
    meta_path.write_text(str(catalog_path.stat().st_mtime_ns))
    return df_cat, texts, emb

# ------------------- Similaridade ----------------

def topk_cosine(sim_vec: np.ndarray, k: int) -> List[int]:
    if k >= len(sim_vec):
        return list(np.argsort(-sim_vec))
    idx = np.argpartition(-sim_vec, k)[:k]
    return idx[np.argsort(-sim_vec[idx])].tolist()

# ------------------- Pipeline entrada ----------------

def process_entrada(entrada_path: Path, df_cat: pd.DataFrame, cat_texts: List[str], cat_emb: np.ndarray,
                    bi_encoder: SentenceTransformer,
                    desc_col: Optional[str], unit_col: Optional[str], extra_num_cols: List[str],
                    cross_encoder: Optional[CrossEncoder], rerank_top_k: int):
    df_in_raw = pd.read_excel(entrada_path).fillna("")
    df_in_raw["__linha_excel__"] = df_in_raw.index + 2  # cabeçalho na linha 1

    lower_map = {c.lower().strip(): c for c in df_in_raw.columns}

    if not desc_col:
        desc_col = DEFAULT_DESC_COL
    desc_col = lower_map.get(desc_col.lower().strip(), desc_col)
    if desc_col not in df_in_raw.columns:
        raise ValueError(f"Não encontrei a coluna de descrição '{desc_col}' na ENTRADA.")

    if unit_col:
        unit_col = lower_map.get(unit_col.lower().strip(), unit_col)

    if not extra_num_cols:
        extra_num_cols = DEFAULT_EXTRA_NUM_COLS
    extra_cols_real = []
    for c in extra_num_cols:
        col = lower_map.get(c.lower().strip())
        if col:
            extra_cols_real.append(col)

    df_in = df_in_raw.copy()

    col_diam = None
    col_comp = None
    for c in extra_cols_real:
        if normalize_text(c).startswith("diametro") or "diam" in normalize_text(c):
            col_diam = c
        if "comprimento" in normalize_text(c):
            col_comp = c

    df_in["__diametro_mm__"] = df_in[col_diam].apply(parse_diametro_cell_to_mm) if col_diam else None
    if col_comp:
        df_in["__comprimento_m__"] = df_in[col_comp].apply(lambda x: _to_float(x))
    else:
        df_in["__comprimento_m__"] = None

    mask_keep = ~((df_in["__diametro_mm__"].isna() | (df_in["__diametro_mm__"] == "")) &
                  (df_in["__comprimento_m__"].isna() | (df_in["__comprimento_m__"] == "")))
    df_in = df_in[mask_keep].reset_index(drop=True)

    matches, revisar, sem_corr = [], [], []

    diam_cat_mm = df_cat.get("diametro_mm_txt", pd.Series([None]*len(df_cat)))

    for i, row in df_in.iterrows():
        descricao = str(row.get(desc_col, "")).strip()
        unidade = str(row.get(unit_col, "")).strip() if unit_col else ""

        diam_in = row.get("__diametro_mm__")
        comp_in = row.get("__comprimento_m__")

        extras_in = {}
        if diam_in is not None and not (isinstance(diam_in, float) and math.isnan(diam_in)):
            extras_in["diametro_mm"] = f"{float(diam_in):.2f} mm"
        if comp_in is not None and not (isinstance(comp_in, float) and math.isnan(comp_in)):
            extras_in["comprimento_m"] = f"{float(comp_in):.3f} m"

        in_text = canonical_text(descricao, unidade, extras_in)
        in_emb = bi_encoder.encode([in_text], convert_to_numpy=True, normalize_embeddings=True)[0]

        candidate_idx = np.arange(len(df_cat))
        if diam_in is not None and not (isinstance(diam_in, float) and math.isnan(diam_in)):
            compat = []
            for j in range(len(df_cat)):
                dcm = diam_cat_mm.iloc[j]
                comp = diameters_compatible(diam_in, float(dcm)) if dcm is not None and dcm == dcm else None
                compat.append(comp)
            compat = np.array(compat, dtype=object)

            pref_true = np.where(compat == True)[0]
            if len(pref_true) >= 1:
                candidate_idx = pref_true
            else:
                pref_none = np.where(compat == None)[0]
                if len(pref_none) >= 1:
                    candidate_idx = pref_none

        sims = (cat_emb[candidate_idx] @ in_emb.T)
        if len(candidate_idx) == 0:
            best_idx_global = []
        else:
            local_top = topk_cosine(sims, RERANK_TOP_K)
            best_idx_global = candidate_idx[local_top]

        ce_score = None
        if cross_encoder is not None and len(best_idx_global) > 0:
            pairs = [(in_text, cat_texts[j]) for j in best_idx_global]
            ce = np.array(cross_encoder.predict(pairs), dtype=float)
            if ce.size > 1 and (ce.max() - ce.min()) > 1e-8:
                ce_norm = (ce - ce.min()) / (ce.max() - ce.min())
            else:
                ce_norm = np.ones_like(ce) * 0.5
            sims_local = (cat_emb[best_idx_global] @ in_emb.T)
            combined = W_BI * sims_local + W_CE * ce_norm
        else:
            if len(best_idx_global) == 0:
                combined = np.array([])
            else:
                combined = (cat_emb[best_idx_global] @ in_emb.T)

        if combined.size == 0:
            sem_corr.append({
                "idx_entrada": int(row["__linha_excel__"]),
                "entrada_descricao": descricao,
                "entrada_diametro": row.get(col_diam, ""),
                "entrada_comprimento_m": row.get(col_comp, ""),
                "motivo": "sem candidatos compatíveis"
            })
            continue

        if diam_in is not None and len(best_idx_global) > 0:
            bonus = np.zeros_like(combined)
            for k, j in enumerate(best_idx_global):
                dcm = diam_cat_mm.iloc[int(j)]
                if dcm is not None and dcm == dcm and diameters_compatible(diam_in, float(dcm)):
                    bonus[k] += DIAM_MATCH_BONUS
            combined = np.clip(combined + bonus, 0.0, 1.0)

        kbest = int(np.argmax(combined))
        top_idx = int(best_idx_global[kbest])
        best_score = float(combined[kbest])

        result_row = {
            "idx_entrada": int(row["__linha_excel__"]),
            "entrada_descricao": descricao,
            "entrada_diametro": row.get(col_diam, ""),
            "entrada_comprimento_m": row.get(col_comp, ""),
            "codigo_sugerido": str(df_cat.loc[top_idx, "codigo"]) if "codigo" in df_cat.columns else "",
            "descricao_catalogo": str(df_cat.loc[top_idx, "descricao"]),
            "score": round(best_score, 4)
        }

        if best_score >= TH_ALTO:
            matches.append(result_row)
        elif best_score >= TH_BAIXO:
            revisar.append(result_row)
        else:
            sem_corr.append({
                "idx_entrada": int(row["__linha_excel__"]),
                "entrada_descricao": descricao,
                "entrada_diametro": row.get(col_diam, ""),
                "entrada_comprimento_m": row.get(col_comp, ""),
                "motivo": f"score {round(best_score,4)} < {TH_BAIXO}"
            })

    return pd.DataFrame(matches), pd.DataFrame(revisar), pd.DataFrame(sem_corr)

# ------------------- CLI ---------------------

def main():
    ap = argparse.ArgumentParser(description="Matcher simplificado: ENTRADA.xlsx vs CATALOGO.txt (+ RERANKING cross-encoder)")
    ap.add_argument("--catalogo", required=True, help="Caminho para o catálogo .txt (fixo)")
    ap.add_argument("--entrada", required=True, help="Caminho para a planilha de entrada .xlsx")
    ap.add_argument("--saida_base", default="resultado", help="Prefixo de saída")
    ap.add_argument("--desc_col", default=None, help="Nome da coluna de descrição na ENTRADA (default: 'Descrição do Material')")
    ap.add_argument("--unit_col", default=None, help="Nome da coluna de unidade (se houver)")
    ap.add_argument("--extra_num_cols", default=None, help="Colunas extras numéricas (ex.: 'Diâmetro,Comprimento (m)')")
    ap.add_argument("--cross_encoder_model", default=CROSS_ENCODER_MODEL, help="Modelo do cross-encoder para reranking")
    ap.add_argument("--rerank_top_k", type=int, default=RERANK_TOP_K, help="Top-K para reranking")
    args = ap.parse_args()

    catalog_path = Path(args.catalogo).resolve()
    entrada_path = Path(args.entrada).resolve()

    bi_encoder = SentenceTransformer(BI_ENCODER_MODEL)
    cache_path = CACHE_DIR / (catalog_path.stem + ".index.npz")
    df_cat, cat_texts, cat_emb = build_or_load_catalog_index(catalog_path, cache_path, bi_encoder)

    ce_model = args.cross_encoder_model.strip() if args.cross_encoder_model else None
    cross_encoder = CrossEncoder(ce_model) if ce_model else None

    if args.extra_num_cols:
        extra_cols = [c.strip() for c in args.extra_num_cols.split(",") if c.strip()]
    else:
        extra_cols = DEFAULT_EXTRA_NUM_COLS

    df_match, df_rev, df_no = process_entrada(
        entrada_path, df_cat, cat_texts, cat_emb,
        bi_encoder,
        desc_col=args.desc_col, unit_col=args.unit_col, extra_num_cols=extra_cols,
        cross_encoder=cross_encoder, rerank_top_k=args.rerank_top_k
    )

    # Mantém nomes de arquivo do seu fluxo atual
    out_match = OUT_DIR / f"{args.saida_base}_matches.csv"
    out_rev   = OUT_DIR / f"{args.saida_base}_revisar.csv"
    out_no    = OUT_DIR / f"{args.saida_base}_sem_negativos.csv"

    df_match.to_csv(out_match, index=False, sep=";")
    df_rev.to_csv(out_rev, index=False, sep=";")
    df_no.to_csv(out_no, index=False, sep=";")

    print(f"[OK] Salvo: {out_match.name}, {out_rev.name}, {out_no.name}")

if __name__ == "__main__":
    main()
