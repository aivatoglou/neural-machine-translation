from datetime import datetime, timedelta

from core.config import settings
from db.database import MongoDatabase
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from models.document import Document, inputDocument
from models.user import NewUser, User
from mt.easy_nmt import Translator
from seq2seq.inference import SeqTranslator

from routes.helpers import (
    authenticate_user,
    create_access_token,
    get_current_active_user,
    get_password_hash,
    get_time,
)

mongodb = MongoDatabase(
    settings.mongo_usr,
    settings.mongo_pwd,
    settings.mongo_clstr,
    settings.mongo_db,
    settings.trans_col,
    settings.users_col,
    settings.mongo_url,
)
translator = Translator()
seq_translator = SeqTranslator()
router = APIRouter()

# !------------------- Rest APIs -------------------! #
@router.post("/services/translate", tags=["services"])
async def translate(
    text: inputDocument, current_user: User = Depends(get_current_active_user)
):

    ##### EasyMMT translator
    translator.translation(text.text)
    output_dict = {
        "Original": text.text,
        "Translation": translator.sent,
        "Language": translator.lang,
        "Timestamp": get_time(),
    }

    ##### Custom trained translator
    # seq_translator.translate(text.text)
    # output_dict = {
    #    "Original": text.text,
    #    "Translation": seq_translator.sent,
    #    "Timestamp": get_time(),
    # }

    mongodb.insert_one(data=output_dict, collection=mongodb.collection_documents)
    output_dict["_id"] = str(output_dict["_id"])
    return output_dict


@router.get("/services/history", tags=["services"])
async def history(current_user: User = Depends(get_current_active_user)):

    translations = mongodb.find_all(collection=mongodb.collection_documents)
    translations = [Document(**document) for document in translations]
    output_list = [
        {
            str(document.id): {
                "Original": document.Original,
                "Translation": document.Translation,
                "Language": document.Language,
                "Timestamp": document.Timestamp,
            }
        }
        for document in translations
    ]
    output_dict = dict((key, val) for k in output_list for key, val in k.items())
    return output_dict


@router.get("/services/delete_all", tags=["services"])
async def delete_all(current_user: User = Depends(get_current_active_user)):

    recycle_bin = mongodb.delete_all(collection=mongodb.collection_documents)
    return {f"{recycle_bin.deleted_count} documents deleted."}


# !------------------- User Authentication -------------------! #
@router.post("/token", tags=["authentication"])
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):

    response = mongodb.find_user(
        data={"username": form_data.username}, collection=mongodb.collection_users
    )

    if not list(response):

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token_expires = timedelta(minutes=30)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}


@router.get("/users/me/", response_model=User, tags=["users"])
async def read_users_me(current_user: User = Depends(get_current_active_user)):
    return current_user


# !------------------- User Registration -------------------! #
@router.post("/users/register", tags=["users"])
async def register_user(newuser: NewUser):

    response = mongodb.find_user(
        data={"email": newuser.email}, collection=mongodb.collection_users
    )
    if list(response):
        return {"status": "Email already registered!"}
    else:
        response = mongodb.find_user(
            data={"username": newuser.username}, collection=mongodb.collection_users
        )
        if list(response):
            return {"status": "Username already registered!"}
        else:
            hashed_pwd = get_password_hash(newuser.password)
            mongodb.register_user(
                data={
                    "username": newuser.username,
                    "email": newuser.email,
                    "full_name": newuser.full_name,
                    "pwd": hashed_pwd,
                    "Timestamp": get_time(),
                },
                collection=mongodb.collection_users,
            )

            return {"status": "User registered successfully!"}
