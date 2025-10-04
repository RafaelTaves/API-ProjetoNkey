# -- Fluxo de RAG com LangChain e OpenAI --

"""
matcher_pipeline.py

Pipeline de correspondência entre planilha de ENTRADA e planilha PADRÃO usando:
- Normalização de textos e unidades
- Blocking (TF-IDF + BM25 + Fuzzy) para gerar candidatos
- Embeddings (bi-encoder) para similaridade semântica
- Reranking (cross-encoder) opcional para refinar top-N
- Combinação de scores e decisão por limiar

Como usar (linha de comando):
    python matcher_pipeline.py --entrada entrada.xlsx --padrao catalogo_padrao.xlsx --saida resultado.xlsx

Colunas esperadas (flexível, mas recomendado):
    - Descrição (ou Titulo / Nome): texto principal do item
    - Unidade: unidade de medida (ex: "m2", "kg", "un", "ml")
    - (Opcional) Categoria, Marca, Código, etc.

O script tenta adivinhar colunas por nomes comuns. Ajuste em CONFIG se necessário.
"""

import argparse
import unicodedata
import re
import sys
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd

from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder


# =========================
# ======= CONFIG ==========
# =========================

CONFIG = {
    # nomes possíveis para a coluna de texto principal
    "text_cols_candidates": ["descricao", "descrição", "titulo", "título", "nome", "produto", "item"],
    # nomes possíveis de unidade
    "unit_cols_candidates": ["unidade", "und", "uni", "unit", "u.m.", "u.m"],
    # colunas extras que entram no texto canônico (se existirem)
    "extra_text_cols": ["categoria", "subcategoria", "marca", "codigo", "código"],

    "bi_encoder_model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "cross_encoder_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",  # defina None para desativar
    "blocking_top_k": 120,
    "rerank_top_n": 15,

    "weights": {
        "tfidf": 0.20,
        "bm25": 0.15,
        "fuzzy": 0.15,
        "bi_encoder": 0.35,
        "cross_encoder": 0.15
    },

    "unit_bonus": 0.05,
    "unit_penalty": 0.00,

    "THRESH_HIGH": 0.80,
    "THRESH_LOW": 0.60,
}

# =========================
# ====== UTILIDADES =======
# =========================

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
}

def canonical_unit(u: Any) -> str:
    u_norm = normalize_text(u)
    return UNITS_MAP.get(u_norm, u_norm)

def pick_first_existing(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols_lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c in cols_lower:
            return cols_lower[c]
    return None

def build_canonical_text(row: pd.Series, text_col: str, unit_col: Optional[str], extra_cols: List[str]) -> str:
    parts = [normalize_text(row.get(text_col, ""))]
    if unit_col:
        parts.append(f"unidade {canonical_unit(row.get(unit_col, ''))}")
    for c in extra_cols:
        if c in row.index:
            parts.append(normalize_text(row.get(c, "")))
    return " | ".join([p for p in parts if p])

def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    return df.reset_index(drop=True).fillna("")

def bm25_matrix_fit(texts: List[str]) -> Tuple[BM25Okapi, List[List[str]]]:
    tokenized = [t.split() for t in texts]
    return BM25Okapi(tokenized), tokenized

def bm25_scores(bm25: BM25Okapi, tokenized_corpus: List[List[str]], query: str) -> np.ndarray:
    tokens = query.split()
    scores = bm25.get_scores(tokens)
    if len(scores) == 0:
        return np.zeros(0)
    s = np.array(scores, dtype=np.float32)
    mn, mx = float(s.min()), float(s.max())
    if mx <= mn:
        return np.zeros_like(s)
    return (s - mn) / (mx - mn)

# =========================
# === PARSER DO CATÁLOGO .TXT
# =========================

# Mapeia chaves diversas para nomes canônicos de coluna
KEY_NORMALIZER = {
    "codigo": "codigo",
    "código": "codigo",
    "cod": "codigo",
    "descricao": "descrição",
    "descrição": "descrição",
    "titulo": "descrição",
    "título": "descrição",
    "unidade": "unidade",
    "preco unitario": "preco_unitario",
    "preço unitario": "preco_unitario",
    "preco unitário": "preco_unitario",
    "preço unitário": "preco_unitario",
    "data do preco": "data_preco",
    "data do preço": "data_preco",
    "ativo": "ativo",
}

def normalize_key(k: str) -> str:
    k0 = normalize_text(k)
    return KEY_NORMALIZER.get(k0, k0)

def parse_catalog_txt(txt_path: str) -> pd.DataFrame:
    """
    Lê um catálogo .txt onde cada linha é "Chave: valor | Chave: valor | ..."
    e devolve um DataFrame com colunas canônicas: codigo, descrição, unidade, preco_unitario, data_preco, ativo
    """
    rows = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # divide por " | " (aceitando variações de espaço)
            parts = re.split(r"\s*\|\s*", line)
            data: Dict[str, str] = {}
            for p in parts:
                if ":" in p:
                    k, v = p.split(":", 1)
                    k = k.strip()
                    v = v.strip()
                    k_norm = normalize_key(k)
                    data[k_norm] = v
            # garante colunas esperadas
            rows.append({
                "codigo": data.get("codigo", ""),
                "descrição": data.get("descrição", data.get("descricao", "")),
                "unidade": data.get("unidade", ""),
                "preco_unitario": data.get("preco_unitario", ""),
                "data_preco": data.get("data_preco", ""),
                "ativo": data.get("ativo", ""),
                # extras opcionais
            })

    df = pd.DataFrame(rows)
    # tipagem leve (quando possível)
    if "preco_unitario" in df.columns:
        with pd.option_context('mode.chained_assignment', None):
            df["preco_unitario"] = pd.to_numeric(df["preco_unitario"].str.replace(",", ".", regex=False), errors="ignore")
    return df

# =========================
# ======= PIPELINE ========
# =========================

class MatcherPipeline:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.bi_encoder = SentenceTransformer(cfg["bi_encoder_model"])
        self.cross_encoder = None
        if cfg.get("cross_encoder_model"):
            try:
                self.cross_encoder = CrossEncoder(cfg["cross_encoder_model"])
            except Exception as e:
                print(f"[AVISO] Não foi possível carregar cross-encoder ({e}). Seguindo sem reranking).")

        self.padrao_df: Optional[pd.DataFrame] = None
        self.padrao_text_col: Optional[str] = None
        self.padrao_unit_col: Optional[str] = None
        self.padrao_extra_cols: List[str] = []

        self.padrao_texts: List[str] = []
        self.vec: Optional[TfidfVectorizer] = None
        self.nn: Optional[NearestNeighbors] = None
        self.X_tfidf = None

        self.bm25: Optional[BM25Okapi] = None
        self.bm25_tokens: Optional[List[List[str]]] = None

        self.padrao_embeddings: Optional[np.ndarray] = None

    def _detect_columns(self, df: pd.DataFrame) -> Tuple[str, Optional[str], List[str]]:
        text_col_lower = pick_first_existing(df, [c for c in self.cfg["text_cols_candidates"]])
        if not text_col_lower:
            raise ValueError(
                f"Não encontrei coluna de texto. Renomeie para um dos nomes: {self.cfg['text_cols_candidates']} ou ajuste CONFIG."
            )
        unit_col = pick_first_existing(df, [c for c in self.cfg["unit_cols_candidates"]])

        # extras que existirem
        extra_cols_real = []
        df_cols_lower = {c.lower(): c for c in df.columns}
        for c in self.cfg["extra_text_cols"]:
            if c in df_cols_lower:
                extra_cols_real.append(df_cols_lower[c])

        return text_col_lower, unit_col, extra_cols_real

    def fit_padroes(self, df_padroes: pd.DataFrame):
        df = prepare_dataframe(df_padroes)
        self.padrao_text_col, self.padrao_unit_col, self.padrao_extra_cols = self._detect_columns(df)
        self.padrao_df = df

        # constrói texto canônico por linha
        self.padrao_texts = [
            build_canonical_text(row, self.padrao_text_col, self.padrao_unit_col, self.padrao_extra_cols)
            for _, row in df.iterrows()
        ]

        # TF-IDF + KNN
        self.vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        self.X_tfidf = self.vec.fit_transform(self.padrao_texts)
        self.nn = NearestNeighbors(metric="cosine", n_neighbors=min(self.cfg["blocking_top_k"], len(self.padrao_texts)))
        self.nn.fit(self.X_tfidf)

        # BM25
        self.bm25, self.bm25_tokens = bm25_matrix_fit(self.padrao_texts)

        # Embeddings
        self.padrao_embeddings = self.bi_encoder.encode(
            self.padrao_texts, normalize_embeddings=True, convert_to_numpy=True
        )

    def _blocking_candidates(self, entrada_text: str) -> np.ndarray:
        y = self.vec.transform([entrada_text])
        dist, idxs = self.nn.kneighbors(y)
        return idxs[0]

    def _scores_for_candidates(self, entrada_text: str, entrada_unit: str, cand_idxs: np.ndarray) -> pd.DataFrame:
        weights = self.cfg["weights"]

        # TF-IDF cos-sim
        y = self.vec.transform([entrada_text])
        Xc = self.X_tfidf[cand_idxs]
        y_norm = np.sqrt((y.multiply(y)).sum())
        Xc_norms = np.sqrt(Xc.multiply(Xc).sum(axis=1)).A1 + 1e-8
        tfidf_sims = (Xc.dot(y.T).A1) / (Xc_norms * (float(y_norm) + 1e-8))
        tfidf_sims = np.clip(tfidf_sims, 0.0, 1.0)

        # BM25
        bm25_all = bm25_scores(self.bm25, self.bm25_tokens, entrada_text)
        bm25_sel = bm25_all[cand_idxs] if len(bm25_all) else np.zeros_like(tfidf_sims)

        # Fuzzy
        fuzzy_scores = np.array(
            [fuzz.token_set_ratio(entrada_text, self.padrao_texts[j]) / 100.0 for j in cand_idxs],
            dtype=np.float32
        )

        # Bi-encoder
        e_emb = self.bi_encoder.encode([entrada_text], normalize_embeddings=True, convert_to_numpy=True)[0]
        cand_emb = self.padrao_embeddings[cand_idxs]
        bi_sims = np.clip(cand_emb @ e_emb, 0.0, 1.0)

        # Cross-encoder (opcional)
        cross_scores = np.zeros_like(bi_sims)
        if self.cross_encoder is not None and self.cfg["rerank_top_n"] > 0 and len(cand_idxs) > 0:
            top_n = min(self.cfg["rerank_top_n"], len(cand_idxs))
            ord_idx = np.argsort(-bi_sims)[:top_n]
            pairs = [(entrada_text, self.padrao_texts[cand_idxs[k]]) for k in ord_idx]
            try:
                ce_raw = self.cross_encoder.predict(pairs)
                ce_raw = np.array(ce_raw, dtype=np.float32)
                if len(ce_raw) > 0:
                    mn, mx = float(ce_raw.min()), float(ce_raw.max())
                    ce_norm = (ce_raw - mn) / (mx - mn) if mx > mn else np.zeros_like(ce_raw)
                else:
                    ce_norm = np.zeros_like(ce_raw)
                for r, k in enumerate(ord_idx):
                    cross_scores[k] = ce_norm[r]
            except Exception as e:
                print(f"[AVISO] Cross-encoder falhou ({e}). Seguindo sem reranking).")
                cross_scores = np.zeros_like(cross_scores)

        # Unidade: bônus/penalidade
        entrada_unit_canon = canonical_unit(entrada_unit)
        unit_bonus = np.zeros_like(bi_sims)
        if entrada_unit_canon and self.padrao_unit_col:
            for ii, j in enumerate(cand_idxs):
                cand_unit = canonical_unit(self.padrao_df.loc[j, self.padrao_unit_col])
                if cand_unit and cand_unit == entrada_unit_canon:
                    unit_bonus[ii] = self.cfg["unit_bonus"]
                elif cand_unit and cand_unit != entrada_unit_canon and self.cfg["unit_penalty"] != 0:
                    unit_bonus[ii] = self.cfg["unit_penalty"]

        final_score = (
            weights["tfidf"] * tfidf_sims
            + weights["bm25"] * bm25_sel
            + weights["fuzzy"] * fuzzy_scores
            + weights["bi_encoder"] * bi_sims
            + weights["cross_encoder"] * cross_scores
            + unit_bonus
        )
        final_score = np.clip(final_score, 0.0, 1.0)

        out = pd.DataFrame({
            "idx_catalogo": cand_idxs,
            "score_tfidf": tfidf_sims,
            "score_bm25": bm25_sel,
            "score_fuzzy": fuzzy_scores,
            "score_bi": bi_sims,
            "score_cross": cross_scores,
            "unit_bonus": unit_bonus,
            "score_final": final_score
        }).sort_values("score_final", ascending=False).reset_index(drop=True)

        return out

    def match_one(self, entrada_row: pd.Series, entrada_text_col: str, entrada_unit_col: Optional[str]) -> Dict[str, Any]:
        entrada_text = build_canonical_text(
            entrada_row, entrada_text_col, entrada_unit_col, self.padrao_extra_cols
        )
        entrada_unit = entrada_row.get(entrada_unit_col, "") if entrada_unit_col else ""

        cand_idxs = self._blocking_candidates(entrada_text)
        cand_df = self._scores_for_candidates(entrada_text, entrada_unit, cand_idxs)

        TH, TL = self.cfg["THRESH_HIGH"], self.cfg["THRESH_LOW"]
        best = cand_df.iloc[0] if len(cand_df) else None

        if best is None:
            status = "sem_correspondente"
            top3 = []
        else:
            s = float(best["score_final"])
            status = "match" if s >= TH else ("revisar" if s >= TL else "sem_correspondente")
            top3 = cand_df.head(3).to_dict(orient="records")

        out = {
            "entrada_texto": entrada_text,
            "entrada_unidade": entrada_unit,
            "status": status,
            "top3": []
        }
        for rec in top3:
            j = int(rec["idx_catalogo"])
            out["top3"].append({
                "idx_catalogo": j,
                "score_final": float(rec["score_final"]),
                "descricao_catalogo": str(self.padrao_df.loc[j, self.padrao_text_col]),
                "unidade_catalogo": str(self.padrao_df.loc[j, self.padrao_unit_col]) if self.padrao_unit_col else "",
                "scores": {
                    "tfidf": float(rec["score_tfidf"]),
                    "bm25": float(rec["score_bm25"]),
                    "fuzzy": float(rec["score_fuzzy"]),
                    "bi": float(rec["score_bi"]),
                    "cross": float(rec["score_cross"]),
                    "unit_bonus": float(rec["unit_bonus"]),
                }
            })

        if out["top3"]:
            out["melhor_idx_catalogo"] = out["top3"][0]["idx_catalogo"]
            out["melhor_score"] = out["top3"][0]["score_final"]
        else:
            out["melhor_idx_catalogo"] = None
            out["melhor_score"] = 0.0
        return out

    def match_dataframe(self, df_entrada: pd.DataFrame) -> pd.DataFrame:
        df = prepare_dataframe(df_entrada)
        entrada_text_col, entrada_unit_col, _ = self._detect_columns(df)

        results = []
        for i, row in df.iterrows():
            res = self.match_one(row, entrada_text_col, entrada_unit_col)
            base = {
                "idx_entrada": i,
                "entrada_descricao": str(df.loc[i, entrada_text_col]),
                "entrada_unidade": str(df.loc[i, entrada_unit_col]) if entrada_unit_col else "",
                "status": res["status"],
                "melhor_idx_catalogo": res["melhor_idx_catalogo"],
                "melhor_score": res["melhor_score"],
            }
            for k in range(3):
                if k < len(res["top3"]):
                    t = res["top3"][k]
                    base[f"top{k+1}_idx"] = t["idx_catalogo"]
                    base[f"top{k+1}_score"] = t["score_final"]
                    base[f"top{k+1}_descricao_catalogo"] = t["descricao_catalogo"]
                    base[f"top{k+1}_unidade_catalogo"] = t["unidade_catalogo"]
                else:
                    base[f"top{k+1}_idx"] = None
                    base[f"top{k+1}_score"] = None
                    base[f"top{k+1}_descricao_catalogo"] = None
                    base[f"top{k+1}_unidade_catalogo"] = None
            results.append(base)

        return pd.DataFrame(results)

# =========================
# ========= MAIN ==========
# =========================

def main():
    parser = argparse.ArgumentParser(description="Matcher Entrada vs Catálogo Padrão (.xlsx ou .txt).")
    parser.add_argument("--entrada", required=True, help="Caminho para XLSX de entrada.")
    parser.add_argument("--padrao", required=True, help="Caminho para catálogo padrão (.xlsx OU .txt).")
    parser.add_argument("--saida", default="resultado.xlsx", help="Arquivo de saída (xlsx/csv).")
    args = parser.parse_args()

    # ENTRADA sempre .xlsx (ajuste se quiser também .csv/.txt)
    try:
        df_entrada = pd.read_excel(args.entrada)
    except Exception as e:
        print(f"Erro ao ler ENTRADA: {e}")
        sys.exit(1)

    # PADRÃO: .txt (parse) ou .xlsx
    try:
        if args.padrao.lower().endswith(".txt"):
            df_padrao = parse_catalog_txt(args.padrao)
            # para o pipeline detectar as colunas corretamente, forçamos nomes em minúsculas
            df_padrao.columns = [c.lower().strip() for c in df_padrao.columns]
            # garante que exista uma coluna de texto principal com os nomes esperados
            if "descrição" not in df_padrao.columns and "descricao" not in df_padrao.columns:
                # cria "descrição" a partir de algum campo — aqui, usamos o que veio do txt
                if "descricao" in df_padrao.columns:
                    df_padrao.rename(columns={"descricao": "descrição"}, inplace=True)
                else:
                    raise ValueError("O .txt não trouxe a coluna 'Descrição'. Verifique o formato das linhas.")
        else:
            df_padrao = pd.read_excel(args.padrao)
    except Exception as e:
        print(f"Erro ao ler CATÁLOGO PADRÃO: {e}")
        sys.exit(1)

    # normaliza nomes de colunas (minúsculas) na ENTRADA também
    df_entrada.columns = [c.lower().strip() for c in df_entrada.columns]
    df_padrao.columns = [c.lower().strip() for c in df_padrao.columns]

    pipe = MatcherPipeline(CONFIG)
    print(">> Preparando catálogo padrão (TF-IDF, BM25, Embeddings)...")
    pipe.fit_padroes(df_padrao)

    print(">> Comparando itens da entrada...")
    resultado = pipe.match_dataframe(df_entrada)

    try:
        if args.saida.lower().endswith(".csv"):
            resultado.to_csv(args.saida, index=False, sep=";")
        else:
            with pd.ExcelWriter(args.saida, engine="openpyxl") as w:
                resultado.to_excel(w, index=False, sheet_name="resultado")
        print(f">> OK! Resultado salvo em: {args.saida}")
    except Exception as e:
        print(f"Erro ao salvar resultado: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
