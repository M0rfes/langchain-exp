from langchain.text_splitter import RecursiveCharacterTextSplitter

import json
res = []
with open("./next13.txt") as f:
    text = f.read()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=0,
    )
    texts = text_splitter.split_text(text)
    res = [{"role": "user", "content": "context:"+t} for t in texts]

print(json.dumps(res))
