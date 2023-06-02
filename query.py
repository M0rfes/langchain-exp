import os
from langchain.vectorstores import Pinecone
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone

open_api_key = "<open-ai key>"

os.environ['OPENAI_API_KEY'] = open_api_key

pinecone.init(
    api_key="<pincone-key>",
    environment="<pincone env>",
)

llm = OpenAI(openai_api_key=open_api_key, temperature=0.5,)
embedding = OpenAIEmbeddings(openai_api_key=open_api_key)


chain = load_qa_chain(llm=llm, chain_type="stuff")

docsearch = Pinecone.from_existing_index("chemistry-2e", embedding=embedding)

query = "what book are we using?"
docs = docsearch.similarity_search(query)


print(chain.run(input_documents=docs, question=query))
