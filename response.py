import getpass
import os
import re
import configparser
        
from langchain_chroma import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from langchain.chat_models import init_chat_model

from typing_extensions import List, TypedDict
from langchain_core.documents import Document
from langgraph.graph import START, StateGraph


def generate_response(query, db_loc, tool_type, model, temp, num_docs) -> str:
    """
    Generates response to print on GUI after user asks a question.
    Arguments:
        - query: the user question, collected by gradio textbox
        - db_loc: location of vectorstore to use 
        - tool_type: the type of tool, QA or code generation, chosen by user
        - model: the model user picked, collected by gradio dropdown
        - temp: the temperature user picked, collected by gradio slider
        - num_docs: no. of docs user wants to retreive from vectorstore for RAG, 
                    collected by gradio slider 
    Returns: 
        - res: a string that consists the output to print for user query (including sources and URLs)
    """

    # -------------------------------- Define Vectorstore and Model Parameters ----------------------------------

    # Take a look at the arguments passed in
    print(f"Query: {query}")
    print(f"Database:  {db_loc}")
    print(f"Model: {model}")
    print(f"Temperature: {temp}")
    print(f"N of Documents: {num_docs}")

    # Load config globally for this function
    config = configparser.ConfigParser()
    config.read('fabric_ai.conf')

    # Specify Vetorstore and create a retriever
    embedding_model = HuggingFaceEmbeddings(model_name='all-mpnet-base-v2')
    vectorstore = Chroma(persist_directory=db_loc,
          embedding_function=embedding_model)
    #retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": k})

    # Specify the LLM model
    # If it's gpt, handle separately using keys 
    if model == "gpt-4o-mini":
        openai_secret = config['API_KEYS']['openai_key']
        llm = init_chat_model("gpt-4o-mini", model_provider="openai", 
                               openai_api_key=openai_secret)
    
    else:
        # Set context window
        context_nums = {
            #"codestral": 32768,
            "codestral": 16384,
            "codellama:13b": 16384,
            #"codellama:34b": 16384,
            #"mistral-large": 131072,
            "mistral-small": 16384,
            "codegemma:7b": 8192,
            "phi4": 16384,
            #"deepseek-coder-v2": 163840 
            "deepseek-coder-v2": 16384
            } 
        
        #llm = OllamaLLM(model=model, num_ctx=context_nums[model],
        llm = OllamaLLM(model=model, num_ctx=context_nums[model],
                    temperature = temp) # higher more creative, lower coherent
       
    # ---------------------------------- Generate Response --------------------------------------------
   
    # Define the system prompt template
    if tool_type == "Code Generation":
        template = config['Template']['code_template']
    else:
        template = config['Template']['qa_template']
    
    # Build prompt from template
    prompt = PromptTemplate.from_template(template)


    # Define state for application
    class State(TypedDict):
        question: str
        context: List[Document]
        answer: str
    

    # Define application steps
    def retrieve(state: State, k) -> dict:
        """
        Retrieves relevant documents from vectorstore
        Arguments:
            - state: this is the current state of the LLM application
            - k: this is the no.of documents to retreive, given by the user
        Returns:
            - context: documents retrieved, a piece of the state that will be merged to application state
        """
        retrieved_docs = vectorstore.similarity_search(state["question"], k)
        return {"context": retrieved_docs}
    
    def generate(state: State):
        """
        Generates the response to user query 
        Arguments: 
            - state: this is the current state of the LLM application
        Returns:
            - answer: answer generated, a piece of the state that will be merged to application state
        """
        # Combine content from each doc retrieved to pass in as context
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        messages = prompt.invoke({"question": state["question"], "context": docs_content})
        response = llm.invoke(messages)
        return {"answer": response}
    
    
    # Create the state graph
    graph_builder = StateGraph(State)
    
    # Register the nodes with names
    graph_builder.add_node("retrieve", lambda state: retrieve(state, k=num_docs))
    graph_builder.add_node("generate", generate)
    
    # Define the execution flow
    graph_builder.add_edge(START, "retrieve")  # Start at retrieve
    graph_builder.add_edge("retrieve", "generate")  # Pass retrieved docs to generate

    graph = graph_builder.compile()
    
    # Invoke the graph, which will get us the response 
    response = graph.invoke({"question": query})
    
    # for debugging
    #print(response)
  
    # ---------------------------------- Helper functions for print output --------------------------------
    def remove_first_line(text):
        """
        Removed the first line of the text
        Arguments: 
            - text: text to process
        Returns: 
            - lines: text, with first line removed
        """
        lines = text.splitlines(True)
        if lines:
            lines.pop(0)
        return "".join(lines)

    def print_context_list(contexts):
        """
        Returns the source of the documents retreieved
        Arguments: 
            - Contexts: the documents retrevied 
        Returns: 
            - Sources: the source of the documents retrieved
        """
        sources_with_urls = []

        for document in contexts:
            # If code generation, get physical file loc
            if tool_type == "Code Generation":
                source = os.path.basename(document.metadata['source']).replace("py", "ipynb")
            # If QA, get title of article/post
            else:
                source = document.metadata['title']

            url = document.metadata['url']
            sources_with_urls.append(f"[{source}]({url})")
        
        return  "\n\n ----\n\n" + "## Sources\n\n" + str(sources_with_urls)

    
    # If model is gpt, clean up using helper funcions
    if model == "gpt-4o-mini":
        if (response["answer"].content)[3:11] == "markdown":
            res = remove_first_line(response["answer"].content) + print_context_list(response["context"]) 
        else:
            res = response["answer"].content + print_context_list(response["context"])
    # Clean up for deepseek
    elif model=="deepseek-coder-v2":
        answer = response["answer"]
        res = answer.replace('</think>', ' ').replace('<think>', ' ') + print_context_list(response["context"])
    # For all other functions
    else:
        res = response["answer"] + print_context_list(response["context"])

    # For debugging
    # print(res)

    return res

