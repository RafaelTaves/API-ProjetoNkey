from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from datetime import timedelta

# Imports do seu app
import app.models.models as models
import app.schemas.schemas as schemas
import app.db.crud as crud
import app.core.config as config
import app.core.auth as auth
from app.core.auth import verify_token
from app.db.database import engine
from app.api.dependencies import get_db, get_current_user

PROMPTS_FILE = "prompts.json"

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
