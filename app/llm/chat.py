from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    HumanMessage,
    SystemMessage,
    AIMessage,
    BaseMessage
)
import tiktoken
from app.model.message import Message, Role

from .open_ai import embedding, OUT_FILE_DIR, load_vec


chat = ChatOpenAI(temperature=0.5, frequency_penalty=0.7, presence_penalty=0.3)


human_template = """
pdf: {context}
answer the given question: {question}
"""
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

chat_prompt = ChatPromptTemplate.from_messages(
    [human_message_prompt])


def chat_with_gpt(message: Message):
    index = message.index
    chats = message.chat
    db = load_vec(index)
    content = chats[-1].content
    req: list[BaseMessage] = [
        SystemMessage(
            content="You are a helpfully AI. That remembers the chat history and evaluates the given pdf to answer the question. use the pdf and the chat history to answer the question. Answer the question from your knowledge base is not found in the pdf and chat history."
        )
    ]

    for c in chats:
        if c.role == Role.human:
            req.append(HumanMessage(content=c.content))
        elif c.role == Role.ai:
            req.append(AIMessage(content=c.content))
    ds = db.similarity_search(content, k=10)
    contexts = ''.join([doc.page_content for doc in ds])
    req += chat_prompt.format_prompt(
        context=contexts, question=content).to_messages()
    for r in req:
        print("---------------")
        print(r.content)
        print("---------------")

    encoding = tiktoken.encoding_for_model(chat.model_name)
    tokens = encoding.encode(''.join([r.content for r in req]))
    print(len(tokens))
    while len(tokens) >= 4000:
        req.pop(1)
        tokens = encoding.encode(''.join([r.content for r in req]))
    print(len(tokens))
    res = chat(req)
    return res
