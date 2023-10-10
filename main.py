import os
import copy
from langchain.chat_models import ChatOpenAI

from openAI_api_key import openAI_api
from langchain.schema import (
    SystemMessage,
    HumanMessage,
)

from langchain.document_loaders import DirectoryLoader




os.environ["OPENAI_API_KEY"] = openAI_api


# Loading data  ---------------------------------------------------------------------------------
loader = DirectoryLoader("testData02/", show_progress=True)
docs = loader.load()
print("number of documents loaded: ", len(docs))

# Keep only the text part and stores each text in a list
texts = []
for i in range(len(docs)):
    texts.append(docs[i].page_content)


# Initialize chat model
chat = ChatOpenAI(
    openai_api_key=os.environ["OPENAI_API_KEY"],
    # model='gpt-3.5-turbo',
    model="gpt-4",
    temperature=0.3
)

systemPrompt = """

## Overview of your task:
You will be given a piece of documentation, some menu or text from a website, or any other type of written document, and return a detailed table of content (possibly with paragraphs of text as extra detail if the original document contains paragraphs) for the document using markdown syntax.

## Documents containing tables of content or lists
If the document you are given is already a table of content or a list of sections or a list of bullet points without paragraphs of text, or if the document contains one of these things, return the table of content as it is without adding anything, and use the markdown syntax.
This is very important, do not create extra information for lists of items.
Your job is to make things shorter or equal, not longer.

### example with a list / table of content:

Original document: “””
History of automobile
Brands of cars
Famous cars
Car racing
Luxury cars
Cars in movies

”””

What you will return: “””

History of automobile
Brands of cars
Famous cars
Car racing
Luxury cars
Cars in movies
“””



## Documents with paragraphs of text
For documents with paragraphs of text, create abbreviated paragraphs for them and place them under the relevant title / bullet point / number in the table of content.


## Returning only the table of content.
Your answer will only contain your short version of the document, you will not greet the user, you will not add an intro sentence like “Here is the summary of your text:” or “Sure, I can help you summarize this document.”, and instead, you will directly write the table of content as asked, and nothing else. You will not mention the document in a sentence like “The document deals with the different types of car engines”, instead you will simply rewrite a concise version of each part, with headers and subheaders when possible.

## In case the document is unsuitable for the request:
If you are unavailable to provide a shortened version of the document, just return “Unable to provide a summary of the document.”

"""

messages = [
    # the system prompt
    SystemMessage(content=systemPrompt),
    HumanMessage(content="")
]


# initialize an empty list to store the summary of each document
summaries = []

for i, text in enumerate(texts):

    # The query will store the current text in a human message object
    query = HumanMessage(content=text)

    # In the messages object, we modify replace the previous human message if there was one by the one containing the current text
    # We are not appending because we don't need a chat history
    messages[1] = query

    # We add the returned summary to our list of summaries
    summaries.append(chat(messages))
    print("summary number", (i+1), "created")

print(len(texts), "summmaries have been created")

# Now each summary string of the summaries list will be saved as a txt file in the summaries folder
# Specify to save the text files
directory = "summaries/"

# Iterate through the list of summaries
for i, summary in enumerate(summaries):
    # Create a file name based on the index (or any other logic you prefer)
    file_name = f"summary_{i + 1}.txt"
    
    # Combine the directory and file name to get the full file path
    file_path = directory + file_name
    
    # Open the file in write mode and write the summary to it
    with open(file_path, "w") as file:
        file.write(summary.content)
