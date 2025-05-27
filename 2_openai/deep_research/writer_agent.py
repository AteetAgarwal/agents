from pydantic import BaseModel
from agents import Agent, OpenAIChatCompletionsModel
from openai import AsyncAzureOpenAI
from dotenv import load_dotenv
import os

load_dotenv(override=True)
azure_openai_client=AsyncAzureOpenAI(
    api_key=os.getenv('AZURE_OPENAI_API_KEY'),  
    api_version=os.getenv("AZURE_OPENAI_API_VERSION_LATEST"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)
azure_openai_model= OpenAIChatCompletionsModel(model=os.getenv('AZURE_OPENAI_MODEL'), openai_client=azure_openai_client)

INSTRUCTIONS = (
    "You are a senior researcher tasked with writing a cohesive report for a research query. "
    "You will be provided with the original query, and some initial research done by a research assistant.\n"
    "You should first come up with an outline for the report that describes the structure and "
    "flow of the report. Then, generate the report and return that as your final output.\n"
    "The final output should be in markdown format, and it should be lengthy and detailed. Aim "
    "for 5-10 pages of content, at least 1000 words."
)


class ReportData(BaseModel):
    short_summary: str
    """A short 2-3 sentence summary of the findings."""

    markdown_report: str
    """The final report"""

    follow_up_questions: list[str]
    """Suggested topics to research further"""


writer_agent = Agent(
    name="WriterAgent",
    instructions=INSTRUCTIONS,
    model=azure_openai_model,
    output_type=ReportData,
)