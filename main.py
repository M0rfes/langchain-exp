from fastapi import FastAPI
from app.routes import upload, chat, summery, generate_script
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins="*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload.router)
app.include_router(chat.router)
app.include_router(summery.router)
app.include_router(generate_script.router)


@app.get("/")
async def root():
    return {"message": "Hello World"}
