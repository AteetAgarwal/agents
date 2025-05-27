from pydantic import BaseModel
from agents import Agent, OpenAIChatCompletionsModel
from openai import AsyncAzureOpenAI
from dotenv import load_dotenv
import os

HOW_MANY_SEARCHES = 2

INSTRUCTIONS = f"You are a helpful research assistant. Given a query, come up with a set of web searches \
to perform to best answer the query. Output {HOW_MANY_SEARCHES} terms to query for."

load_dotenv(override=True)
azure_openai_client=AsyncAzureOpenAI(
    api_key=os.getenv('AZURE_OPENAI_API_KEY'),  
    api_version=os.getenv("AZURE_OPENAI_API_VERSION_LATEST"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)
azure_openai_model= OpenAIChatCompletionsModel(model=os.getenv('AZURE_OPENAI_MODEL'), openai_client=azure_openai_client)


class WebSearchItem(BaseModel):
    reason: str
    "Your reasoning for why this search is important to the query."

    query: str
    "The search term to use for the web search."


class WebSearchPlan(BaseModel):
    searches: list[WebSearchItem]
    """A list of web searches to perform to best answer the query."""


planner_agent = Agent(
    name="PlannerAgent",
    instructions=INSTRUCTIONS,
    model=azure_openai_model,
    output_type=WebSearchPlan,
)