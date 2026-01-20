import gradio as gr
import os
import boto3
import uuid
from langchain_aws import ChatBedrock
from langchain.messages import AIMessage, HumanMessage 
from langchain.agents import create_agent
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from dotenv import load_dotenv

# 1. Load the variables from .env
load_dotenv()
region=os.getenv("BEDROCK_AWS_REGION")
llm = ChatBedrock(model_id="amazon.nova-lite-v1:0", region_name=region)

@tool
def calculate_sum(a: int, b: int) -> int:
    """Add two numbers together. Use for arithmetic operations."""
    return a + b

@tool
def user_credential_request() -> str:
    """When the user request PII(Personally Identifiable Information) or financial transactions request credentials"""
    return "Please provide your username,email and billing zip to contiue."

@tool
def verify_user_credential(username: str,email: str,zip: str) -> str:
    """When the user provides his information ie his username,email and billing zip"""
    return "request sent"

agent = create_agent(
    model=llm,
    tools=[calculate_sum],
    checkpointer=InMemorySaver(),
    system_prompt="You are a helpful math assistant.",
    name="math_agent"  # Always provide a name!

)

THREAD_ID = str(uuid.uuid4())

def predict(message, history):
    # 4. Config for Memory
    config = {"configurable": {"thread_id": THREAD_ID}}
    
    # 5. Invoke the Agent
    input_data = {"messages": [HumanMessage(content=message)]}
    
    result = agent.invoke(input_data, config=config)
    
    # 6. Extract Response
    return result['messages'][-1].content

demo = gr.ChatInterface(
    predict,
    api_name="chat"
)

demo.launch()