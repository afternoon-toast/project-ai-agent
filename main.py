from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from langchain.agents import create_tool_calling_agent,AgentExecutor
from langchain_core.utils.json import parse_json_markdown

from tools import search_tool,wiki_tool

import json

load_dotenv() 

class researchresponse(BaseModel):
    topic : str 
    summary : str
    sources: list[str]
    tools_used: list[str]

#llm2 = ChatAnthropic(model="claude-3-5-sonnet-20241022")

llm = ChatOpenAI(model="gpt-4o-mini")
parser = PydanticOutputParser(pydantic_object=researchresponse)  #calling the class

#prompt tempelate
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a reserach assistant that will help generate a research paper.
            Answer the user query and use necessary tools.
            Wrap the output in this format and provide no other text\n{format_instructions}
            """,
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions = parser.get_format_instructions)


tools = [wiki_tool]
agent = create_tool_calling_agent(
    llm = llm,
    prompt = prompt,
    tools = tools
)

agent_executor = AgentExecutor( agent = agent, tools = tools, verbose= True)

query = input("what can i help with in the research?\n")
raw_response = agent_executor.invoke({"query" : query})

#print(raw_response)

json_obj = parse_json_markdown(raw_response.get("output"))
try:
    structured_response = parser.parse(json.dumps(json_obj))  # if `parser` expects JSON string
    print(structured_response)
except Exception as e:
    print("Better luck next time!")

# response = llm.invoke("what does surya means?")
# print(response)
