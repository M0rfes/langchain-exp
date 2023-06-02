
import uuid
from typing import Annotated
import aiofiles
from fastapi import APIRouter, UploadFile

from app.llm.open_ai import save_pdf_vec


OUT_FILE_DIR = "app/static"

router = APIRouter(
    prefix='/upload',
    tags=['upload']
)


@router.post("/pdf")
async def upload_pdf(file: UploadFile):
    """
    Uploads a pdf file to the server and converts it to a vector.
    deletes the pdf file after conversion.
    Returns the uuid of the file.

    """
    uuid_srt = str(uuid.uuid4())
    async with aiofiles.open(f"{OUT_FILE_DIR}/{uuid_srt}.pdf", 'wb+') as out_file:
        content = await file.read()
        await out_file.write(content)
        save_pdf_vec(f"{OUT_FILE_DIR}/{uuid_srt}.pdf", uuid_srt)
    return {"uuid": f"{uuid_srt}"}
