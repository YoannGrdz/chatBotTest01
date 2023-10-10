# pip install langchain, openai, tiktoken, pinecone-client, unstructured[all-docs]

import os
import sys
import pinecone
import copy
from langchain.vectorstores import Pinecone
import time
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings

from openAI_api_key import openAI_api
from pinecone_api_key import pinecone_api
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)

from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter




os.environ["OPENAI_API_KEY"] = openAI_api



# get API key from app.pinecone.io and environment from console
pinecone.init(
    api_key=pinecone_api,
    environment="gcp-starter"
)


# Index creation or connection  ---------------------------------------------------------------------------------
index_name = 'rag-test-index01'

if index_name not in pinecone.list_indexes():
    print("creating new index")
    pinecone.create_index(
        index_name,
        dimension=1536,
        metric='cosine'
    )
    # wait for index to finish initialization
    while not pinecone.describe_index(index_name).status['ready']:
        time.sleep(1)
else:
    print("using existing index")

index = pinecone.Index(index_name)



# Loading data  ---------------------------------------------------------------------------------
loader = DirectoryLoader("testData/", show_progress=True)
docs = loader.load()
print("number of documents loaded: ", len(docs))


# Text splitting  ---------------------------------------------------------------------------------
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(docs)


# Building the knowledge base  ---------------------------------------------------------------------------------
# storing the text content, the metadata, and an id for each chunk in 3 separated lists which will be zipped together later
# the goal is to obtain from "texts" (documents list) a list of dictionaries each with the following structure: 
# {
#     id: "a unique id for each dict"
#     embeddings: "the vectors created from the text"
#     metadata : {
        
#         text : "the content of the text"
#         source : "the source document"
#         title : "the title (if it is included in the metadata)"
#         otherKeys : "if there are any"
#     }
# }
chunks = []
metadataList = []
chunk_ids = []
for i in range(len(texts)):
    
    # creating a unique id for each vector using the name of the source and i
    chunk_ids.append(texts[i].metadata['source']+str(i))
    chunks.append(texts[i].page_content)

    # each newMetadata dictionary will contain whatever the metadata dictionary of the texts[i] contained, as well as a text key with the content of the text
    newMetadata = copy.deepcopy(texts[i].metadata)
    newMetadata["text"] = texts[i].page_content
    metadataList.append(newMetadata)


# embedding the text content chunks
embed_model = OpenAIEmbeddings(
    model="text-embedding-ada-002", openai_api_key=openAI_api
)
embeddings = embed_model.embed_documents(texts=chunks)
print("number of embeddings: ", len(embeddings))

# zipping the 3 lists together
vectors = zip(chunk_ids, embeddings, metadataList)
print("type of vectors", type(vectors))

# upsert to pinecone (the vector database)
index.upsert(vectors=vectors)


# to get information about the index
print("index info: ", index.describe_index_stats())


# vector store
# the text_field refers to the text key in the metadata dictionary inside each dictionary of the list of vectors
text_field = "text"
vectorstore = Pinecone(
    index, embed_model.embed_query, text_field
)

# testing that we can fetch relevant chunks with similarity_search
# question = "What is langchain?"
# result = vectorstore.similarity_search(
#     question,  # our search query
#     k=1  # return n most relevant docs
# )
# print("result type:", type(result))
# print(len(result))


# conversation flow ---------------------------------------------------------------------------------
chat = ChatOpenAI(
    openai_api_key=os.environ["OPENAI_API_KEY"],
    model='gpt-3.5-turbo'
)

messages = [
    # the system prompt
    SystemMessage(content="You are a helpful assistant. If you do not know the answer to a question or do not have access to information that would be required to answer, just tell the user that you are sorry and that you don't know. It is possible that you be provided with context or documentation that helps you answer the user's query, so if that is the case, you should try your best to use this context to base your answer on, but you are not limited to this context and you can also use your base training and your logic to answer. However, if you are not sure, do not invent stuff and just say that you are sorry but that you don't have the kowledge necessary to answer, or ask the user whether they can provide you with more information.")
]

#function for prompt augmentation
def augment_prompt(query: str):
    # get top 3 results from knowledge base
    results = vectorstore.similarity_search(query, k=3)
    # get the text from the results
    source_knowledge = "\n".join([x.page_content for x in results])
    # feed into an augmented prompt
    augmented_prompt = f"""Using the contexts below, answer the query.

    Contexts:
    {source_knowledge}

    Query: {query}"""
    return augmented_prompt

query = None

print("Hello! How can I help you today?")

while True:

    # Asking user for input
    if not query:
        query = input("Prompt: ")
        augmentedPrompt = augment_prompt(query)

    # Exiting the program if the user inputs one of the following
    if query in ['quit', 'q', 'exit']:
        sys.exit()

    
    # the prompt wil be a human message built from the user input and augmented by RAG
    RAGprompt = HumanMessage(
        content = augmentedPrompt
    )

    # the prompt is added to the messages object
    messages.append(RAGprompt)
    answer = chat(messages)
    print(answer.content)

    query = None
