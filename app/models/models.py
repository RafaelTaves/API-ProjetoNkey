from sqlalchemy import Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class User(Base):
    __tablename__ = "user"

    id_user = Column(Integer, primary_key=True, index=True)
    username = Column(String(30), unique=True, index=True)
    hashed_password = Column(String(60))
    api_key = Column(String(100), unique=True, index=True)
    

