from agents import Agent, WebSearchTool, ModelSettings, OpenAIChatCompletionsModel, function_tool
from openai import AsyncAzureOpenAI
from dotenv import load_dotenv
import os

import requests

load_dotenv(override=True)
azure_openai_client=AsyncAzureOpenAI(
    api_key=os.getenv('AZURE_OPENAI_API_KEY'),  
    api_version=os.getenv("AZURE_OPENAI_API_VERSION_LATEST"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)
azure_openai_model= OpenAIChatCompletionsModel(model=os.getenv('AZURE_OPENAI_MODEL'), openai_client=azure_openai_client)

@function_tool
def web_search_tool(self, input_text):
    query = input_text.strip()
    serpapi_key = "42ed6911d74a87a946c4c509ebfeb8d15a4018e049772af825a3954a2b35c656"
    response = requests.get(
        "https://serpapi.com/search",
        params={"q": query, "api_key": serpapi_key, "engine": "google"},
    )
    data = response.json()
    results = data.get("organic_results", [])
    return "\n".join([r["title"] + ": " + r["link"] for r in results[:3]])

INSTRUCTIONS = (
    "You are a research assistant. Given a search term, you search the web for that term and "
    "produce a concise summary of the results. The summary must 2-3 paragraphs and less than 300 "
    "words. Capture the main points. Write succintly, no need to have complete sentences or good "
    "grammar. This will be consumed by someone synthesizing a report, so its vital you capture the "
    "essence and ignore any fluff. Do not include any additional commentary other than the summary itself."
)

search_agent = Agent(
    name="Search agent",
    instructions=INSTRUCTIONS,
    tools=[WebSearchTool(search_context_size="low")],
    model=azure_openai_model,
    model_settings=ModelSettings(tool_choice="required"),
)