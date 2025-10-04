from fastapi import FastAPI, HTTPException, Depends, status, UploadFile, File, Form
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from datetime import timedelta
import io, zipfile, tempfile
from pathlib import Path

# Imports do seu app
import app.models.models as models
import app.schemas.schemas as schemas
import app.db.crud as crud
import app.core.config as config
import app.core.auth as auth
from app.core.auth import verify_token
from app.db.database import engine
from app.api.dependencies import get_db, get_current_user
from sentence_transformers import SentenceTransformer, CrossEncoder
from app.services.matcher_simple import (
    build_or_load_catalog_index,
    process_entrada,
)

models.Base.metadata.create_all(bind=engine)

app = FastAPI(title="PDF Data Extraction API", version="1.0", description="Extrai dados de PDFs e imagens usando FastAPI e .")

# Lista de origens que podem acessar
origins = [
    "http://localhost:3000", 
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,       # Ou ["*"] para liberar geral
    allow_credentials=True,
    allow_methods=["*"],         # Permite GET, POST, PUT, DELETE, OPTIONS
    allow_headers=["*"],         # Permite qualquer cabeçalho
)

#Diretorios
PROJECT_ROOT = Path(__file__).resolve().parents[1]   # .../API
DATA_DIR = PROJECT_ROOT / "data"                     # .../API/data
CACHE_DIR = DATA_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

CATALOG_PATH = (DATA_DIR / "output" / "catalogo.txt").resolve()

# Nome base interno para os CSVs dentro do ZIP (pode ser fixo ou timestampado)
DEFAULT_SAIDA_BASE = "resultado"

# Defaults matcher
BI_ENCODER_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANK_TOP_K = 15

@app.post("/register", response_model=schemas.UserRead, tags=["Authentication"])
def register(user: schemas.UserCreate, db: Session = Depends(get_db), current_user: schemas.UserRead = Depends(get_current_user)):
    db_user = crud.get_user(db, username=user.username)
    if db_user:
        raise HTTPException(status_code=400, detail="Username already registered")
    return crud.create_user(db=db, user=user)

@app.post("/login", response_model=schemas.Token, tags=["Authentication"])
def login_for_access_token(db: Session = Depends(get_db), form_data: OAuth2PasswordRequestForm = Depends()):
    user = auth.authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=config.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = auth.create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer", "api_key": user.api_key}

@app.get("/me", response_model=schemas.UserRead, tags=["Authentication"])
def read_users_me(current_user: schemas.UserRead = Depends(get_current_user)):
    return current_user

@app.get("/verify-token/{token}", tags=["Authentication"])
async def verify_user_token(token: str):
    verify_token(token=token)
    return {"message": "Token is valid"}

@app.post("/match", tags=["Match"])
async def match_upload(
    file: UploadFile = File(..., description="Planilha entrada.xlsx"),
    desc_col: str = Form("Descrição do Material"),
    unit_col: str = Form(None),
    extra_num_cols: str = Form("Diâmetro,Comprimento (m)"),
    rerank_top_k: int = Form(RERANK_TOP_K),
    cross_encoder_model: str = Form(CROSS_ENCODER_MODEL),
    current_user: schemas.UserRead = Depends(get_current_user)
):
    # 1) Salva o XLSX enviado em arquivo temporário
    if file.content_type not in ("application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "application/octet-stream"):
        raise HTTPException(status_code=400, detail="Envie um arquivo .xlsx válido")

    try:
        catalog_path = CATALOG_PATH
        if not catalog_path.exists():
            raise HTTPException(status_code=500, detail=f"Catálogo interno não encontrado: {catalog_path}")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_xlsx = Path(tmpdir) / "entrada.xlsx"
            tmp_xlsx.write_bytes(await file.read())

            # 2) Carrega/gera índice do catálogo (usa cache em data/cache)
            bi_encoder = SentenceTransformer(BI_ENCODER_MODEL)
            cache_path = CACHE_DIR / (catalog_path.stem + ".index.npz")
            df_cat, cat_texts, cat_emb = build_or_load_catalog_index(catalog_path, cache_path, bi_encoder)

            cross_encoder = CrossEncoder(cross_encoder_model) if cross_encoder_model else None
            extra_cols = [c.strip() for c in extra_num_cols.split(",") if c.strip()]

            # 3) Executa o pipeline do matcher (mesmo fluxo do script)
            df_match, df_revisar, df_sem = process_entrada(
                entrada_path=tmp_xlsx,
                df_cat=df_cat,
                cat_texts=cat_texts,
                cat_emb=cat_emb,
                bi_encoder=bi_encoder,
                desc_col=desc_col or "Descrição do Material",
                unit_col=unit_col,
                extra_num_cols=extra_cols,
                cross_encoder=cross_encoder,
                rerank_top_k=rerank_top_k
            )

            # 4) Gera ZIP em memória com os 3 CSVs
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                zf.writestr(f"{DEFAULT_SAIDA_BASE}_matches.csv", df_match.to_csv(index=False, sep=";").encode("utf-8-sig"))
                zf.writestr(f"{DEFAULT_SAIDA_BASE}_revisar.csv", df_revisar.to_csv(index=False, sep=";").encode("utf-8-sig"))
                zf.writestr(f"{DEFAULT_SAIDA_BASE}_negativo.csv", df_sem.to_csv(index=False, sep=";").encode("utf-8-sig"))

            zip_buffer.seek(0)

            return StreamingResponse(
                zip_buffer,
                media_type="application/zip",
                headers={"Content-Disposition": f'attachment; filename="{DEFAULT_SAIDA_BASE}.zip"'}
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao processar: {e}")
