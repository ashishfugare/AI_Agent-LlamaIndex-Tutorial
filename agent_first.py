from dotenv import load_dotenv

load_dotenv()



import os
import asyncio
from llama_index.llms.google_genai import GoogleGenAI
# Import both AgentRunner and ReActAgent

from llama_index.core.agent.workflow import FunctionAgent

# 1. Load your Hugging Face API key
try:
    api_token = os.getenv("GOOGLE_API_KEY")
    print(f"DEBUG: Read GOOGLE_API_KEY = {api_token}")
    HF_TOKEN = os.environ["GOOGLE_API_KEY"]
except KeyError:
    raise ValueError("Please set the GOOGLE_API_KEY environment variable.")

# 2. Set up the LLM from Hugging Face
llm = GoogleGenAI(
    model="gemini-2.5-flash",
    token=HF_TOKEN
)

# --- Your Tools ---
def multiply(a: float, b: float) -> float:
    """Multiplies two numbers, a and b."""
    print(f"Calling multiply with: a={a}, b={b}")
    return a * b

def add(a: float, b: float) -> float:
    """Adds two numbers, a and b."""
    print(f"Calling add with: a={a}, b={b}")
    return a + b

#initilaizie the agent

workflow = FunctionAgent(
    tools=[multiply ,add ],
    llm=llm,
    system_prompt="YOu are an agnet that can perform basic arithmatic operation using tools"
)


async def main():
    response = await workflow.run(user_msg="what is 20+(2*4)?")
    print(response)



if __name__== "__main__":
   
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"an erro occured:{e}")    