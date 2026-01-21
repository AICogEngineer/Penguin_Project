from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool, ToolRuntime
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents.structured_output import ToolStrategy
from langchain.messages import AIMessage, HumanMessage 
from langchain_ollama import ChatOllama
import gradio as gr
import os
import uuid
import getpass
from dotenv import load_dotenv

SYSTEM_PROMPT = """ You are a customer support agent for the company, PenguinZ.

You have the ability to answer questions about company policy regarding refunds, the AI, and all sorts of stuff.
However, your tools are still under development, so if the user or customer asks about anything regarding that,
please tell them that it is still under development.

End all your sentences with -penguin, connecting -penguin to the last word with no spaces inbetween such as "Hello there-penguin!" or "How may I help you today-penguin?" or "I'm sorry I can't do that-penguin.". 
Put the punctuation mark after the -penguin.
"""


model = ChatOllama(
    model = "gpt-oss",
    validate_model_on_init=True,
    temperature=0.7
)

checkpointer = InMemorySaver()

agent = create_agent(
    model = model,
    system_prompt=SYSTEM_PROMPT,
    checkpointer=checkpointer
)

THREAD_ID = str(uuid.uuid4())

def predict(message, history):
    config = {"configurable": {"thread_id": THREAD_ID}}
    input_data = {"messages": [HumanMessage(content=message)]}
    result = agent.invoke(input_data, config=config)
    return result['messages'][-1].content

demo = gr.ChatInterface(
    predict,
    api_name="chat"
)

demo.launch()