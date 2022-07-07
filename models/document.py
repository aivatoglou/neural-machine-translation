from datetime import datetime
from typing import Union

from bson.objectid import ObjectId
from pydantic import BaseModel, Field


class PyObjectId(ObjectId):

    """Custom Type for reading MongoDB IDs"""

    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid object_id")
        return ObjectId(v)

    @classmethod
    def __modify_schema__(cls, field_schema):
        field_schema.update(type="string")


class Document(BaseModel):

    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    Original: str
    Translation: str
    Language: Union[str, None] = None  # optional field
    Timestamp: datetime


class inputDocument(BaseModel):

    text: str
