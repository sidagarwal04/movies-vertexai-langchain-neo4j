import json
import gradio as gr
import typing_extensions
import os
import vertexai
import tempfile

from langchain_google_vertexai import VertexAI
from langchain.prompts.prompt import PromptTemplate
from langchain_community.graphs import Neo4jGraph
from langchain.chains import GraphCypherQAChain
from langchain.memory import ConversationBufferMemory

# process of getting credentials
def get_credentials():
    creds_json_str = os.getenv("BOB") # get json credentials stored as a string
    if creds_json_str is None:
        raise ValueError("GOOGLE_APPLICATION_CREDENTIALS_JSON not found in environment")

    # create a temporary file
    with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".json") as temp:
        temp.write(creds_json_str) # write in json format
        temp_filename = temp.name 

    return temp_filename

# pass
os.environ["GOOGLE_APPLICATION_CREDENTIALS"]= get_credentials()

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
project_id = os.getenv("PROJECT_ID")
location = os.getenv("LOCATION")

vertexai.init(project=project_id, location=location)
  
CYPHER_GENERATION_TEMPLATE = """You are an expert Neo4j Cypher translator who understands the question in english and convert to Cypher strictly based on the Neo4j Schema provided and following the instructions below:
1. Generate Cypher query compatible ONLY for Neo4j Version 5
2. Do not use EXISTS, SIZE keywords in the cypher. Use alias when using the WITH keyword
3. Please do not use same variable names for different nodes and relationships in the query.
4. Use only Nodes and relationships mentioned in the schema
5. Always enclose the Cypher output inside 3 backticks
6. Always do a case-insensitive and fuzzy search for any properties related search. Eg: to search for a Company name use `toLower(c.name) contains 'neo4j'`
7. Candidate node is synonymous to Manager
8. Always use aliases to refer the node in the query
9. 'Answer' is NOT a Cypher keyword. Answer should never be used in a query.
10. Please generate only one Cypher query per question. 
11. Cypher is NOT SQL. So, do not mix and match the syntaxes.
12. Every Cypher query always starts with a MATCH keyword.
13. Always do fuzzy search for any properties related search. Eg: when the user asks for "matrix" instead of "the matrix", make sure to search for a Movie name using use `toLower(c.name) contains 'matrix'` 
Schema:
{schema}
Samples:
Question: List down 5 movies that released after the year 2000
Answer: MATCH (m:Movie)  WHERE m.released > 2000  RETURN m LIMIT 5
Question: Get all the people who acted in a movie that was released after 2010
Answer: MATCH (p:Person)-[r:ACTED_IN]->(m:Movie)  WHERE m.released > 2010  RETURN p,r,m
Question: Name the Director of the movie Apollo 13
Answer: MATCH (m:Movie)<-[:DIRECTED]-(p:Person)  WHERE toLower(m.title) contains "apollo 13"  RETURN p.name
Question: {question}
Answer: 
"""

CYPHER_GENERATION_PROMPT = PromptTemplate(
    input_variables=["schema","question"], validate_template=True, template=CYPHER_GENERATION_TEMPLATE
)

graph = Neo4jGraph(
    url=NEO4J_URI, 
    username=NEO4J_USERNAME, 
    password=NEO4J_PASSWORD
)


chain = GraphCypherQAChain.from_llm(
    VertexAI(model_name='gemini-1.5-pro-preview-0409', max_output_tokens=8192, temperature=0.0),
    graph=graph,
    cypher_prompt=CYPHER_GENERATION_PROMPT,
    verbose=True,
    return_direct=True
)

def chat(que):
    r = chain(que)
    print(r)
    llm=VertexAI(model_name='gemini-1.5-pro-preview-0409', max_output_tokens=8192, temperature=0.0)
    summary_prompt_tpl = f"""Human: 
    Fact: {json.dumps(r['result'])}
    * Summarise the above fact as if you are answering this question "{r['query']}"
    * When the fact is not empty, assume the question is valid and the answer is true
    * Do not return helpful or extra text or apologies
    * Just return summary to the user. DO NOT start with Here is a summary
    * List the results in rich text format if there are more than one results
    Assistant:
    """
    return llm(summary_prompt_tpl)  


memory = ConversationBufferMemory(memory_key = "chat_history", return_messages = True)

def chat_response(input_text,history):

    try:
        return chat(input_text)
    except:
        # a bit of protection against exposed error messages
        # we could log these situations in the backend to revisit later in development
        return "I'm sorry, there was an error retrieving the information you requested."


# Define your custom CSS
custom_css = """
/* Custom CSS for the chat interface */
.gradio-container {
    # background: #f0f0f0; /* Change background color */
    border: 0
    border-radius: 15px; /* Add border radius */
}
.primary.svelte-cmf5ev{
    background: linear-gradient(90deg, #9848FC 0%, #DC8855 100%);
    # background-clip: text;
    # -webkit-background-clip: text;
    # -webkit-text-fill-color: transparent;
}
.v-application .secondary{
    background-color: #EEEEEE !important
}
# /* Custom CSS for the chat input */
# .gradio-chat-input input[type="text"] {
#     background-color: #ffffff; /* Change input background color */
#     border-radius: 5px; /* Add border radius */
#     border: 1px solid #cccccc; /* Change border color */
# }
# /* Custom CSS for the chat button */
# .gradio-chat-input button {
#     # background-color: #ff0000; /* Change button background color */
#     # border-radius: 5px; /* Add border radius */
#     # color: #ffffff; /* Change text color */
#     background: linear-gradient(90deg, #9848FC 0%, #DC8855 100%);
#     background-clip: text;
#     -webkit-background-clip: text;
#     -webkit-text-fill-color: transparent;
# }
"""

interface = gr.ChatInterface(fn = chat_response,
                             title = "Movies Chatbot",
                             theme = "soft",
                             chatbot = gr.Chatbot(height=430),
                             undo_btn = None,
                             clear_btn = "\U0001F5D1 Clear Chat",
                             css=custom_css,
                             examples = ["List down 5 movies that released after the year 2000",
                                         "Get all the people who acted in a movie that was released after 2010",
                                         "Name the Director of the movie Apollo 13",
                                         "Name all actors of the movie V for Vendetta"])

# Launch the interface
interface.launch(share=True)
