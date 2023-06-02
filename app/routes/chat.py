from typing import Annotated
import aiofiles
from fastapi import APIRouter
from app.llm.chat import chat_with_gpt
from app.model.message import Message
router = APIRouter(
    prefix='/chat',
    tags=['chat']
)


@router.post("/")
async def chat(message: Message):
    """
    Chat with the ai.

    """
    replay = chat_with_gpt(message)
    return {
        'role': 'ai',
        'content': replay.content
    }
