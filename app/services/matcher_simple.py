
"""
matcher_simple.py  —  pipeline simplificado com catálogo .txt fixo
Versão com padrões para colunas:
  - Descrição do Material
  - Diâmetro
  - Comprimento (m)
E regra: ignorar linhas em que **Diâmetro** e **Comprimento (m)** estão vazios.

Uso típico:
  python matcher_simple.py \
    --catalogo catalogo.txt \
    --entrada entrada.xlsx \
    --saida_base resultado

(Parâmetros opcionais ainda existem para sobrescrever se necessário)
"""

import argparse
import re
import unicodedata
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# ------------------- Config -------------------

BI_ENCODER_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
TH_ALTO = 0.80
TH_BAIXO = 0.60
UNIT_BONUS = 0.05  # bônus se unidades forem iguais (canônicas)

# Defaults solicitados
DEFAULT_DESC_COL = "Descrição do Material"
DEFAULT_EXTRA_NUM_COLS = ["Diâmetro", "Comprimento (m)"]

#Diretorio de saída (garante que existe)
ROOT_DIR = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT_DIR / "data" / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ------------------- Normalização ---------------------

def normalize_text(s: Any) -> str:
    s = str(s or "").lower().strip()
    s = unicodedata.normalize('NFKD', s)
    s = ''.join(ch for ch in s if not unicodedata.combining(ch))
    s = s.replace("/", " ").replace("\\", " ").replace("-", " ").replace("_", " ")
    s = re.sub(r"[^\w\s%°²³]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

UNITS_MAP = {
    "m2": "m2", "m²": "m2", "metro quadrado": "m2", "m^2": "m2",
    "m": "m", "metro": "m",
    "kg": "kg", "quilo": "kg", "kgs": "kg",
    "g": "g", "grama": "g", "gramas": "g",
    "l": "l", "litro": "l", "litros": "l",
    "ml": "ml",
    "un": "un", "und": "un", "uni": "un", "pc": "un", "peça": "un", "peca": "un", "pz": "un",
    "cx": "cx", "caixa": "cx",
    "rl": "rl", "rolo": "rl",
    "mm": "mm", "pol": "pol",
}

def canonical_unit(u: Any) -> str:
    u_norm = normalize_text(u)
    return UNITS_MAP.get(u_norm, u_norm)

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
    "comprimento m": "comprimento_m",
}

def normalize_key(k: str) -> str:
    k0 = normalize_text(k)
    return KEY_NORMALIZER.get(k0, k0)

def parse_catalog_txt(txt_path: str) -> pd.DataFrame:
    rows = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = re.split(r"\s*\|\s*", line)
            data: Dict[str, str] = {}
            for p in parts:
                if ":" in p:
                    k, v = p.split(":", 1)
                    k = k.strip()
                    v = v.strip()
                    data[normalize_key(k)] = v
            rows.append({
                "codigo": data.get("codigo", ""),
                "descricao": data.get("descricao", ""),
                "unidade": data.get("unidade", ""),
                "diametro": data.get("diametro", ""),
                "comprimento_m": data.get("comprimento_m", ""),
                "preco_unitario": data.get("preco_unitario", ""),
                "data_preco": data.get("data_preco", ""),
                "ativo": data.get("ativo", ""),
            })
    return pd.DataFrame(rows).fillna("")

# ------------------- Texto canônico -------------------

def canonical_text_row(descricao: str, unidade: str, extras: Dict[str, Any]) -> str:
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
            rebuild = True

    df_cat = parse_catalog_txt(str(catalog_path))
    if "descricao" not in df_cat.columns:
        raise ValueError("Catálogo sem coluna 'descricao'. Verifique o .txt.")

    df_cat = df_cat.reset_index(drop=True).fillna("")
    texts = []
    for _, r in df_cat.iterrows():
        extras = {}
        if "diametro" in df_cat.columns:
            extras["Diâmetro"] = r.get("diametro", "")
        if "comprimento_m" in df_cat.columns:
            extras["Comprimento (m)"] = r.get("comprimento_m", "")
        txt = canonical_text_row(r["descricao"], r.get("unidade", ""), extras)
        texts.append(txt)

    model = SentenceTransformer(BI_ENCODER_MODEL)
    embeddings = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        np.savez_compressed(cache_path, embeddings=embeddings, texts=np.array(texts, dtype=object), df_json=df_cat.to_json())
        meta_path.write_text(str(catalog_path.stat().st_mtime_ns))
    except Exception:
        pass

    return df_cat, texts, embeddings

# ------------------- Similaridade ---------------------

def cosine_sim_matrix(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    sims = A @ b
    return np.clip(sims, 0.0, 1.0)

# ------------------- Helpers ---------------------

def is_blank(val) -> bool:
    s = str(val).strip()
    return s == "" or s.lower() == "nan"

# ------------------- Pipeline entrada ----------------

def process_entrada(entrada_path: Path, df_cat: pd.DataFrame, cat_texts: List[str], cat_emb: np.ndarray,
                    desc_col: Optional[str], unit_col: Optional[str], extra_num_cols: List[str]):
    df_in_raw = pd.read_excel(entrada_path).fillna("")
    lower_map = {c.lower().strip(): c for c in df_in_raw.columns}

    # Descrição (default: "Descrição do Material")
    if not desc_col:
        desc_col = DEFAULT_DESC_COL
    desc_col = lower_map.get(desc_col.lower().strip(), desc_col)
    if desc_col not in df_in_raw.columns:
        raise ValueError(f"Não encontrei a coluna de descrição '{desc_col}' na ENTRADA.")

    # Unidade opcional (não temos por padrão na sua planilha)
    if unit_col:
        unit_col = lower_map.get(unit_col.lower().strip(), unit_col)

    # Extras numéricos (defaults: Diâmetro, Comprimento (m))
    if not extra_num_cols:
        extra_num_cols = DEFAULT_EXTRA_NUM_COLS
    extra_cols_real = []
    for c in extra_num_cols:
        col = lower_map.get(c.lower().strip())
        if col:
            extra_cols_real.append(col)

    df_in = df_in_raw.copy()

    # Remover linhas onde **Diâmetro** e **Comprimento (m)** estão vazios
    if {"Diâmetro", "Comprimento (m)"} <= set(df_in_raw.columns):
        mask_diam = df_in_raw["Diâmetro"].apply(lambda x: is_blank(x))
        mask_comp = df_in_raw["Comprimento (m)"].apply(lambda x: is_blank(x))
        both_empty = mask_diam & mask_comp
        df_in = df_in.loc[~both_empty].reset_index(drop=True)

    model = SentenceTransformer(BI_ENCODER_MODEL)

    matches, revisar, sem_corr = [], [], []

    for i, row in df_in.iterrows():
        descricao = str(row.get(desc_col, ""))
        unidade = str(row.get(unit_col, "")) if unit_col else ""
        # montar extras
        extras_vals = {}
        for c in extra_cols_real:
            extras_vals[c] = row.get(c, "")

        # se ainda assim ambos extras estão vazios, também ignorar (robustez)
        if ("Diâmetro" in extras_vals and "Comprimento (m)" in extras_vals and
            is_blank(extras_vals["Diâmetro"]) and is_blank(extras_vals["Comprimento (m)"])):
            continue

        cano = canonical_text_row(descricao, unidade, extras_vals)
        emb = model.encode([cano], normalize_embeddings=True, convert_to_numpy=True)[0]
        sims = cosine_sim_matrix(cat_emb, emb)

        top_idx = int(np.argmax(sims))
        top_score = float(sims[top_idx])

        # bônus de unidade se existir
        unit_in = canonical_unit(unidade)
        unit_cat = canonical_unit(df_cat.loc[top_idx, "unidade"]) if "unidade" in df_cat.columns else ""
        if unit_in and unit_cat and unit_in == unit_cat:
            top_score = min(1.0, top_score + UNIT_BONUS)

        result_row = {
            "idx_entrada": i,
            "entrada_descricao": descricao,
            "entrada_unidade": unidade,
            "codigo_sugerido": str(df_cat.loc[top_idx, "codigo"]) if "codigo" in df_cat.columns else "",
            "descricao_catalogo": str(df_cat.loc[top_idx, "descricao"]),
            "unidade_catalogo": str(df_cat.loc[top_idx, "unidade"]) if "unidade" in df_cat.columns else "",
            "score": round(top_score, 4)
        }

        if top_score >= TH_ALTO:
            matches.append(result_row)
        elif top_score >= TH_BAIXO:
            revisar.append(result_row)
        else:
            sem_corr.append({
                "idx_entrada": i,
                "entrada_descricao": descricao,
                "entrada_unidade": unidade,
                "motivo": f"score {round(top_score,4)} < {TH_BAIXO}"
            })

    return pd.DataFrame(matches), pd.DataFrame(revisar), pd.DataFrame(sem_corr)

# ------------------- CLI ---------------------

def main():
    ap = argparse.ArgumentParser(description="Matcher simplificado: ENTRADA.xlsx vs CATALOGO.txt (colunas padrão Diâmetro/Comprimento (m))")
    ap.add_argument("--catalogo", required=True, help="Caminho para o catálogo .txt (fixo)")
    ap.add_argument("--entrada", required=True, help="Caminho para a planilha de entrada .xlsx")
    ap.add_argument("--saida_base", default="resultado", help="Prefixo de saída")
    ap.add_argument("--desc_col", default=None, help="Nome da coluna de descrição na ENTRADA (default: 'Descrição do Material')")
    ap.add_argument("--unit_col", default=None, help="Nome da coluna de unidade (se houver)")
    ap.add_argument("--extra_num_cols", default="", help="Colunas numéricas extras (csv). Default: 'Diâmetro,Comprimento (m)'")
    args = ap.parse_args()

    catalog_path = Path(args.catalogo)
    entrada_path = Path(args.entrada)
    cache_path = catalog_path.with_suffix(".index.npz")

    # catálogo
    model = SentenceTransformer(BI_ENCODER_MODEL)
    df_cat, cat_texts, cat_emb = build_or_load_catalog_index(catalog_path, cache_path, model)

    # parse extras informados (ou usa defaults)
    extra_cols = [c.strip() for c in args.extra_num_cols.split(",") if c.strip()]

    df_match, df_rev, df_no = process_entrada(
        entrada_path, df_cat, cat_texts, cat_emb,
        desc_col=args.desc_col, unit_col=args.unit_col, extra_num_cols=extra_cols
    )

    out_match = OUT_DIR / f"{args.saida_base}_matches.csv"
    out_rev   = OUT_DIR / f"{args.saida_base}_revisar.csv"
    out_no    = OUT_DIR / f"{args.saida_base}_sem_correspondente.csv"

    df_match.to_csv(out_match, index=False, sep=";")
    df_rev.to_csv(out_rev, index=False, sep=";")
    df_no.to_csv(out_no, index=False, sep=";")

    print(f"[OK] Salvo: {out_match.name}, {out_rev.name}, {out_no.name}")

if __name__ == "__main__":
    main()
