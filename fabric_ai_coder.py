import gradio as gr
import random
import time
import response as response
import configparser
import logging
import os
from datetime import datetime
from pathlib import Path

config = configparser.ConfigParser()
config.read('fabric_ai.conf')

# Logging Set-up
log_loc = config['Logging']['log_loc']
if os.path.exists(log_loc):
    now = datetime.now()
    date_str = now.strftime("%Y%m%d%H%M")
    filename = date_str + "_custom.log"
    log_loc = Path(os.path.dirname(log_loc), filename)
    log_loc = str(log_loc)

# ONLY configure custom logger
logging.getLogger().handlers.clear()

custom_logger = logging.getLogger('custom')
custom_logger.setLevel(logging.INFO)

custom_handler = logging.FileHandler(log_loc)
custom_formatter = logging.Formatter('%(asctime)s - %(message)s')
custom_handler.setFormatter(custom_formatter)
custom_logger.addHandler(custom_handler)

def chat_response(question, model, tool_type):
    """
    Gathers the database location and call generate_response 

    Arguments:
        - Question: the user question, collected by gradio textbox
        - model: the model user picked, collected by gradio dropdown (this will be taken out once we decide what model is best)
        - tool_type: 'Code Generation' or 'Q&A Tool' as selected by the user
    Returns:
        - ai_response: the response generated to answer the user's question
    """
    rag = "With RAG"
    temperature = 0.5
    doc_count = 4
    
    # Select the database path and tool context
    if tool_type == "Code Generation":
        db = "Code Generation"
        db_loc = config['VectorDB']['code_db_loc']
    elif tool_type == "Q&A Tool":
        db = "KB and Forum"
        db_loc = config['VectorDB']['kb_forum_db_loc']
    else:
        db = None
        db_loc = None

    ai_response = response.generate_response(question, db_loc, db, model, temperature, doc_count)
 

    return ai_response
    

def authenticate(username, password):
    """
    Authenticates the user with given username and password

    Arguments:
        - username: username of the user 
        - password: password for the user
    Returns: 
        - true if user is authenticated successfully, false otherwise 
    """
    print(username, password)
    print(config['USERS'][username])

    return True if (password == config['USERS'][username]) else False


# ------------------------------------- Define the UI portion -----------------------------------------
with gr.Blocks(theme=gr.themes.Default(text_size=gr.themes.sizes.text_lg)) as demo:

    # Specify the model choices available
    model_choices = ["phi4",
                     "llama3.3",
                     "gemma3:1b", 
                     "deepseek-r1", 
                     "mistral-large",
                     "gpt-4o-mini"]

    gr.Markdown("# 🤖 Code Generator and Knowledge Base Bot for FABRIC")

    # Display the options
    with gr.Row():
        question = gr.Textbox(label="Ask a question", placeholder="Type your question here...")
        with gr.Column():
            model = gr.Dropdown(choices=model_choices, 
                                label="Choose Model", 
                                value="phi4")
            tool_type = gr.Radio(choices=["Code Generation", "Q&A Tool"],
                     label="Choose Tool Type",
                     value="Q&A Tool")
            

    # Define submit button 
    submit_btn = gr.Button("Generate Response")
    # Define output format
    output = gr.Markdown(label="LLM Response")

    # Define submit button's action
    submit_btn.click(chat_response, 
                 inputs=[question, model, tool_type], 
                 outputs=output)

# Launch the UI
demo.launch(server_name=config['SERVER']['host_url'], server_port=7862,  auth=authenticate)
