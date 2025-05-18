from langchain_community.tools import WikipediaQueryRun,DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.tools import Tool
from datetime import datetime


search = DuckDuckGoSearchRun()
search_tool = Tool(
    name = "Search",
    func = search.run,
    description = "this is a search tool.",
)

api_wrapper = WikipediaAPIWrapper(top_k_results= 2,doc_content_chars_max= 1000,lang = "en")
wiki_tool = WikipediaQueryRun(api_wrapper = api_wrapper)