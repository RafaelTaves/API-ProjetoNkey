import re
import json
import unicodedata
import pandas as pd
from pathlib import Path

INPUT_XLSX = "data/input/padrao_tratada.xlsx"
SHEET_NAME = 0

def norm(s: str) -> str:
    if s is None or str(s).lower() == "nan":
        return ""
    s = str(s).strip()
    s = s.replace("\r", " ").replace("\n", " ")
    s = " ".join(s.split())
    s = "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")
    return s

def is_empty_row(values):
    return all(not str(v).strip() or pd.isna(v) for v in values)

EXPECTED_HEADERS = {
    "codigo","descricao","unidade","preco unitario","data do preco","ativo"
}
MIN_HEADER_HITS = 3

def detect_header(cells):
    hits = 0
    header_names = []
    for v in cells:
        nv = norm(v).lower()
        header_names.append(nv)
        if nv in EXPECTED_HEADERS:
            hits += 1
    return (hits >= MIN_HEADER_HITS, header_names)

def is_family_row(cells):
    non_empty = [norm(c) for c in cells if str(c).strip()]
    if len(non_empty) == 1:
        s = non_empty[0].lower()
        return bool(re.match(r"^familia\b", s))
    return False

def extract_family_text(cells):
    non_empty = [norm(c) for c in cells if str(c).strip()]
    return non_empty[0] if non_empty else ""

raw = pd.read_excel(INPUT_XLSX, sheet_name=SHEET_NAME, header=None, dtype=object)

records = []
current_header = None
current_header_raw = None
header_idx_map = None
current_family = ""

for _, row in raw.iterrows():
    values = list(row.values)

    if is_empty_row(values):
        continue

    if is_family_row(values):
        current_family = extract_family_text(values)
        continue

    is_hdr, header_names = detect_header(values)
    if is_hdr:
        current_header = header_names
        current_header_raw = [str(v).strip() if v is not None else "" for v in values]

        header_idx_map = {}
        for i, v in enumerate(values):
            pretty = str(v).strip() if v is not None else ""
            npretty = norm(pretty).lower()
            if not npretty or npretty == "nan":
                continue  # <-- ignora colunas sem nome
            header_idx_map[i] = pretty
        continue

    if not current_header or header_idx_map is None:
        continue

    same_as_header = True
    for a, b in zip([str(v).strip() if v is not None else "" for v in values], current_header_raw):
        if a != b:
            same_as_header = False
            break
    if same_as_header:
        continue

    rec = {"Familia": current_family} if current_family else {}

    for i, colname in header_idx_map.items():
        if i < len(values):
            val = norm(values[i])
            if val:  # só adiciona se não for vazio
                rec[colname] = val

    if any(v for k, v in rec.items() if k != "Familia"):
        records.append(rec)

def flatten_record(rec: dict, sep=" | "):
    parts = []
    fam = rec.get("Familia", "")
    if fam:
        parts.append(f"Família: {fam}")
    for k, v in rec.items():
        if k == "Familia":
            continue
        if v:
            parts.append(f"{k}: {v}")
    return sep.join(parts)

flatten_texts = [flatten_record(r) for r in records]

Path("data/output/catalogo.txt").write_text("\n".join(flatten_texts), encoding="utf-8")

print(f"Registros válidos: {len(records)}")
