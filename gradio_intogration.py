import gradio as gr
import os
import boto3
import uuid
from langchain_aws import ChatBedrockConverse
from langchain.messages import AIMessage, HumanMessage 
from langchain.agents import create_agent
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from dotenv import load_dotenv

# 1. Load the variables from .env
load_dotenv()

@tool
def calculate_sum(a: int, b: int) -> int:
    """Add two numbers together. Use for arithmetic operations."""
    return a + b

agent = create_agent(
    model='bedrock:us.amazon.nova-2-lite-v1:0',
    tools=[calculate_sum],
    checkpointer=InMemorySaver(),
    system_prompt="You are a helpful math assistant.",
    name="math_agent"  # Always provide a name!
)

THREAD_ID = str(uuid.uuid4())

def predict(message, history):
    history_langchain_format = []
    for msg in history:
        if msg['role'] == "user":
            history_langchain_format.append(HumanMessage(content=msg['content']))
        elif msg['role'] == "assistant":
            history_langchain_format.append(AIMessage(content=msg['content']))
    config = {"configurable": {"thread_id": THREAD_ID}}
    history_langchain_format.append(HumanMessage(content=message))
    input_data = {"messages": [HumanMessage(content=message)]}
    gpt_response = agent.invoke(input_data, config=config)
    return gpt_response.content

demo = gr.ChatInterface(
    predict,
    api_name="chat",
    placeholder="Hello! I am MathBot. Ask me to add numbers!"
)

demo.launch()