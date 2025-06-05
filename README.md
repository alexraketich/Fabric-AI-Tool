# test-for-AI-tool-integration

## To run

```
source .venv/bin/activate
python fabric_ai_helper.py
```


## Config file (fabric_ai.conf) format


```
[SERVER]
host_url=<server.edu>

[VectorDB]
code_db_loc = <path/to/db>
kb_forum_db_loc = <path/to/db>
kb_db_loc = <path/to/db>
forum_db_loc = <path/to/db>

[Template]
code_template = """You are an AI Code Assstant. Use the following pieces of context (examples) 
    to generate python code to implement the user's question specifically for FABRIC testbed. 
    Use FablibManager whenver possible. Make sure to include correct import statements.
    Generate the answer in Markdown.
    If the question is very different from the context provided, simply say you cannot help.
    {context}
    
    Question: On FABRIC Testbed, {question} Use FablibManager as much as possible. Include 
    import statements.
    
    Here is how you will implement that (in markdown):"""
qa_template = """You are an AI Help Desk assistant. Use the following information to answer
    the question below.
    
    {context}
    
    Question: On FABRIC Testbed, {question} 
    
    Here is the answer based on the given information: """


[API_KEYS]
openai_key = <api key>

[USERS]
<username> = <password>
```


## For notebooks

```
jupyter lab [--ip 0.0.0.0]
```


## Python Version (Tested)
3.12
