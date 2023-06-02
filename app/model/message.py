from pydantic import BaseModel
from enum import Enum


class Role(Enum):
    human = "human"
    ai = "ai"


class Chat(BaseModel):
    role: Role
    content: str


class Message(BaseModel):
    index: str
    chat: list[Chat]
