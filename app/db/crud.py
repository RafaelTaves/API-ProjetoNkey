from sqlalchemy.orm import Session
import app.models.models as models
import app.schemas.schemas as schemas
from app.core.auth import get_password_hash


def get_user(db: Session, username: str):
    return db.query(models.User).filter(models.User.username == username).first()

def create_user(db: Session, user: schemas.UserCreate):
    hashed_password = get_password_hash(user.password)
    db_user = models.User(username=user.username, hashed_password=hashed_password, api_key=user.api_key)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user


