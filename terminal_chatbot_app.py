import configparser
from dotenv import load_dotenv
import os
from langchain_cohere import *
from helper import *

config = configparser.ConfigParser()
config.read('config.ini')

#   Get Parameter Variables
chain_parameters, ui_parameters = initialize_parameters()

# Get Cohere API Key from Environment Variable
load_dotenv()
api_key=os.getenv("COHERE_API_KEY")
if api_key == None:
    raise Exception('Require a COHERE_API_KEY to be setup as an Environment variable or the .env file') 

chain = setup_chain(api_key, **chain_parameters)

question = ""
print(ui_parameters['title'])
print(ui_parameters['init_assistant_message'])
print("Type 'Stop' to end the program")
while question != 'Stop':
    question = input("Prompt: ")
    if question != 'Stop':
        response = chain.run({"query": question})
        print('Response:',response)




