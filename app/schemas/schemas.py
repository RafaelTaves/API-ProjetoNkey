from pydantic import BaseModel

class UserBase(BaseModel):
    username: str
    api_key: str

class UserCreate(UserBase):
    password: str

class UserRead(UserBase):
    id_user: int

    class Config:
        orm_mode = True

class Token(BaseModel):
    access_token: str
    token_type: str
    api_key: str

class TokenData(BaseModel):
    username: str | None = None
    

    
    