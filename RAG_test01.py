# pip install langchain
# pip install openai
# pip install openai chromadb
# pip install tiktoken

import os
from openAI_api_key import openAI_api 
from langchain.llms import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader


os.environ['OPENAI_API_KEY'] = openAI_api

llm = OpenAI(temperature=0.9)


# loading the document (tested with txt documents, and noticed that with some of them there were issues 
# (in the case where there was an issue the text had been copied from a website))
loader = TextLoader("./testData.txt")
document = loader.load()


# defining a text splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)


# splitting the text in smaller chunks with the text splitter defined above
texts = text_splitter.split_documents(document)


# defining the embeddings
embeddings = OpenAIEmbeddings(openai_api_key=os.environ['OPENAI_API_KEY'])


# creating a vector store with texts data embedded with the embeddings defined above
vectorstore = Chroma.from_documents(texts, embeddings)


qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever())


query = "Summarize in a short simple paragraph what this document deals with."

print(qa_chain.run(query))