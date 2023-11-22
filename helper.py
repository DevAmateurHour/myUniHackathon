import configparser

def initialize_parameters():
    config = configparser.ConfigParser()
    config.read('config.ini')
    
    chain_parameters = {}
    chain_parameters['temperature'] = float(config['LANGCHAIN_COHERE']['TEMPERATURE'])
    chain_parameters['search_k'] = int(config['LANGCHAIN_COHERE']['SEARCH_K'])
    chain_parameters['chunk_size'] = int(config['LANGCHAIN_COHERE']['CHUNK_SIZE'])
    chain_parameters['max_tokens'] = int(config['LANGCHAIN_COHERE']['MAX_TOKENS'])
    chain_parameters['chunk_overlap'] = float(config['LANGCHAIN_COHERE']['CHUNK_OVERLAP'])
    chain_parameters['verbose'] = config.getboolean('LANGCHAIN_COHERE','VERBOSE')
    
    ui_parameters = {}
    ui_parameters['chat_input'] = config['UI']['CHAT_INPUT'] 
    ui_parameters['init_assistant_message'] = config['UI']['INIT_ASSISTANCE_MESSAGE'] 
    ui_parameters['title'] = config['UI']['TITLE']

    return chain_parameters, ui_parameters