from typing import Annotated
import aiofiles
from fastapi import APIRouter
from app.llm.chat import chat_with_gpt
from app.llm.generate_script import generate_script
from app.model.message import Message
router = APIRouter(
    prefix='/generate-script',
    tags=['generate-script']
)


@router.get("/")
async def script_from_pdf(topic: str, uuid: str):
    return generate_script(topic, uuid)
