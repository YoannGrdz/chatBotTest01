#pip install langchain openai chromadb tiktoken unstructured unstructured[all-docs]
import os
import sys

import openai
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import DirectoryLoader, TextLoader, UnstructuredFileLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.llms import OpenAI
from langchain.vectorstores import Chroma

from openAI_api_key import openAI_api

os.environ["OPENAI_API_KEY"] = openAI_api

# Enable to save to disk & reuse the model (for repeated queries on the same data)
PERSIST = False

query = None
# checks for a command line argument and if there is one, assign it to the query variable
# Basically, checks whether the user has typed something, and if so, stores the content as a question. 
if len(sys.argv) > 1:
  query = sys.argv[1]

# If PERSIST is set to true and that a persist index directory already exists, fetch the index inside it and assign it as the current index
if PERSIST and os.path.exists("persist"):
  print("Reusing index...\n")
  vectorstore = Chroma(persist_directory="persist", embedding_function=OpenAIEmbeddings())
  index = VectorStoreIndexWrapper(vectorstore=vectorstore)


else:
  # single text file loader
  # loader = TextLoader("./data.txt")

  # directory loader
  loader = DirectoryLoader("testData/")

  # single unstructured file loader
  # loader = UnstructuredFileLoader("./data/YoannGrudzienCoverLetter.pdf")

  if PERSIST:

    # creates an index in a persisting directory for future usage
    index = VectorstoreIndexCreator(vectorstore_kwargs={"persist_directory":"persist"}).from_loaders([loader])


  else:

    # creates a new index based on the files provided
    index = VectorstoreIndexCreator().from_loaders([loader])

# langchain chain used to combine a question from the user with the content of the chat history to formulate a new question for the LLM, 
# while taking the custom data into consideration for RAG.
chain = ConversationalRetrievalChain.from_llm(
  llm=ChatOpenAI(model="gpt-3.5-turbo"),

  # The "k" search parameter defines how many top results should be returned for a given search
  retriever=index.vectorstore.as_retriever(search_kwargs={"k": 3}),
)

# list storing the content of the chat
chat_history = []
while True:

  # Asking user for input
  if not query:
    query = input("Prompt: ")

  # Exiting the program if the user inputs one of the following
  if query in ['quit', 'q', 'exit']:
    sys.exit()

  # Generating an answer based on the user's prompt, and the chat history
  result = chain({"question": query, "chat_history": chat_history})
  print(result['answer'])

  # Adding the last user prompt and the last answer to the chat history
  chat_history.append((query, result['answer']))
  query = None
