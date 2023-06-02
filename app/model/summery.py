from pydantic import BaseModel


class Summery_Text(BaseModel):
    text: str


class Summery_URL(BaseModel):
    url: str
