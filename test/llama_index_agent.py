from typing import List, Callable, Any, Dict
import asyncio
from mlux_reactly import Tracer, ZeroTracer, Tool, LLM
from llama_index.llms.ollama import Ollama
from llama_index.core.agent import FunctionAgent, ReActAgent
from llama_index.core.llms import LLM as LLamaLLM
from llama_index.core.tools import FunctionTool


default_llm = Ollama(
    model="qwen2.5:7b-instruct-q8_0"
)

def make_ollama_llm(l: LLM|LLamaLLM) -> LLamaLLM:
    if isinstance(l, LLamaLLM):
        return l
    else:
        return Ollama(model=l.model)

def convert_tools(tools: List[Tool | Callable[..., Any]]) -> List[Dict[str, Any]]:
    llama_tools = []

    for t in tools:
        if isinstance(t, Tool):
            func = t.run
        elif callable(t):
            func = t
        else:
            raise TypeError(f"Unsupported tool type: {type(t)}")
        llama_tools.append(FunctionTool.from_defaults(func))
    return llama_tools

class LlamaFunctionAgentWrapper:
    agent: FunctionAgent

    def __init__(
            self, 
            tools: List[Tool | Callable], *, 
            tracer: Tracer = ZeroTracer(),
            llm: LLM|LLamaLLM = default_llm):
        self.agent = FunctionAgent(
            tools=convert_tools(tools),
            llm=make_ollama_llm(llm),
            #system_prompt="You are a helpful assistant that can multiply two numbers.",
        )

    async def query(self, user_question: str):
        handler = self.agent.run(user_question)
        answer = await handler
        return str(answer)


class LlamaReActAgentWrapper:
    agent: ReActAgent

    def __init__(
            self, 
            tools: List[Tool | Callable], *, 
            tracer: Tracer = ZeroTracer(),
            llm: LLM|LLamaLLM = default_llm):
        self.agent = ReActAgent(
            tools=convert_tools(tools),
            llm=make_ollama_llm(llm),
            #system_prompt="You are a helpful assistant that can multiply two numbers.",
        )


    async def query(self, user_question: str):
        handler = self.agent.run(user_question)
        answer = await handler
        return str(answer)
