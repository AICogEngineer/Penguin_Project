import gradio as gr
import os
import uuid
import getpass
from langchain_google_genai import ChatGoogleGenerativeAI # <--- NEW IMPORT
from langchain_core.messages import AIMessage, HumanMessage
from langchain.agents import create_agent
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from dotenv import load_dotenv

load_dotenv()

# 1. Setup Google API Key
if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google AI API key: ")

# 2. Define the LLM (Google Gemini)
# Common models: "gemini-1.5-flash" (Fast/Cheap), "gemini-1.5-pro" (Smart)
llm = ChatGoogleGenerativeAI(
    model="gemini-3-flash-preview", 
    temperature=0,
    max_retries=2,
    # google_api_key=... # Optional if you set the env variable above
)

@tool
def calculate_sum(a: int, b: int) -> int:
    """Add two numbers together. Use for arithmetic operations."""
    return a + b

@tool
def secrect_password() -> str:
    """When the user ask you what the password is you provide them with this response"""
    return "Penguin's are awsome and ai is cool"

# 3. Create the Agent
# We pass the 'llm' object, just like we did for Bedrock
agent = create_agent(
    model=llm, 
    tools=[calculate_sum,secrect_password],
    checkpointer=InMemorySaver(),
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
    fn=predict,
    api_name="chat"
)

demo.launch()