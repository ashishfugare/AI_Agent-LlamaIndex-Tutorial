from llama_index.core.agent.workflow  import FunctionAgent
#tools for LLM

from llama_index.llms.ollama import Ollama


llm = Ollama(model="phi3", request_timeout=120.0)


def multiply(a:float ,b:float ) -> float:
    """
    Multiplies two numbers, a and b.
    Use this tool for multiplication.
    """

    return a*b

def add(a:float ,b:float) -> float :
    """
    Adds two numbers, a and b.
    Use this tool for addition.
    """
    """
    THis type of Docstring is needed for agent to understand this
    """
    return a+b


#agent woll use tools nmae , parameters and docstring to deterimine what too to 

#create our agetn and system promot for what type of agent to be 

workflow = FunctionAgent(
    tools = [multiply,add],
    llm=llm,
    system_prompt = "You are an agent that can perform basic mathematical operations"

)

async def main():
    response = await workflow.run(user_msg="What is 20*(4+2)")
    print(response)

if __name__ == "__main__":
    import asyncio
    
    asyncio.run(main())


