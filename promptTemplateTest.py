# import os
# from openAiAPIKey import api_key 
# from langchain.prompts import PromptTemplate
# from langchain.llms import OpenAI

# os.environ['OPENAI_API_KEY'] = api_key

# llm = OpenAI(temperature=0.9)


# prompt template example
# cool_city_locations_template = PromptTemplate(
#     input_variables=["city", "number"],
#     template="What are {number} places to see in {city} for someone who likes nice, not too known, cool city locations?"
# )

# text = cool_city_locations_template.format(city="Tokyo", number=7)

# answer = llm(text)

# print(answer)