from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool, ToolRuntime
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents.structured_output import ToolStrategy
from langchain_aws import ChatBedrockConverse
from langchain.messages import AIMessage, HumanMessage 
from langgraph.graph import StateGraph, START, END
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import TypedDict, Annotated, Literal, List, Dict
import gradio as gr
import os
import operator
import uuid
import getpass
from dotenv import load_dotenv
import snowflake.connector
from IPython.display import Image, display
from policy_engine.rag.retriever import get_policy_retriever
from langgraph.types import Send

#TODO: Organize all nodes, states, and tools into utils

SYSTEM_PROMPT = """ You are a customer support agent for the company, PenguinZ.

You have the ability to answer questions about company policy regarding refunds, the AI, and all sorts of stuff.
However, your tools are still under development, so if the user or customer asks about anything regarding that,
please tell them that it is still under development.

End all your sentences with -penguin, connecting -penguin to the last word with no spaces inbetween such as "Hello there-penguin!" or "How may I help you today-penguin?" or "I'm sorry I can't do that-penguin.". 
Put the punctuation mark after the -penguin.
"""


load_dotenv()

USER = os.getenv('SNOWFLAKE_USER')
PASSWORD = os.getenv('SNOWFLAKE_PASSWORD')
ACCOUNT = os.getenv('SNOWFLAKE_ACCOUNT')
WAREHOUSE = os.getenv('SNOWFLAKE_WAREHOUSE')
DATABASE = os.getenv('SNOWFLAKE_DATABASE')
SCHEMA = os.getenv('SNOWFLAKE_SCHEMA')

# conn = snowflake.connector.connect(
#     user=USER,
#     password=PASSWORD,
#     account=ACCOUNT,
#     warehouse=WAREHOUSE,
#     database=DATABASE,
#     schema=SCHEMA
#     )

# cur = conn.cursor()
# try:
#     cur.execute('select * from FCT_TRANSACTIONS')
#     ret = cur.fetchmany(3)
#     print(ret)
# finally:
#     cur.close()

#Define Model
llm = ChatBedrockConverse(
    model="us.amazon.nova-lite-v1:0",
    temperature=0.7,
    aws_access_key_id=os.getenv("BEDROCK_AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("BEDROCK_AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("BEDROCK_AWS_REGION", "us-east-1")
)

class AgentState(TypedDict):
    query: str                      # customer query
    order_details: Dict             # order details (placeholder for snowflake)
    documents: Annotated[List[Document], operator.add]       # where chunks are stored
    questions: List[str]         # list of sub-questions (if any)
    curr_answer: str                # final answer by agent (legacy)
    answers: Annotated[List[str], operator.add] # collected answers

# retrieval helper
def retrieve_docs(query: str):
    print(f"--- RETRIEVING POLICY FOR: {query} ---")
    retriever = get_policy_retriever(k=3)
    docs = retriever.invoke(query)
    return docs

# generation helper
def generate_response(query: str, docs: List[Document]):
    print("--- GENERATING ANSWER ---")
    
    context = "\n\n".join([d.page_content for d in docs])
    
    prompt_text = """You are a helpful customer support assistant for Penguin Inc. 
    Use the following pieces of retrieved context to answer the user's question. 
    If you don't know the answer, just say that you don't know. 
    
    Context:
    {context}
    
    Question:
    {query}
    
    Answer:"""
    
    try:
        
        llm = ChatBedrockConverse(
            model="us.amazon.nova-lite-v1:0",
            temperature=0.1,
            aws_access_key_id=os.getenv("BEDROCK_AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("BEDROCK_AWS_SECRET_ACCESS_KEY"),
            region_name=os.getenv("BEDROCK_AWS_REGION", "us-east-1")
        )
        
        prompt = ChatPromptTemplate.from_template(prompt_text)
        chain = prompt | llm
        
        response = chain.invoke({"context": context, "query": query})
        response_text = response.content
        
    except Exception as e:
        print(f"Error calling AWS Bedrock: {e}")
        print("Falling back to mocked response...")
        
        # mocked response (fallback)
        response_text = "I have checked the policy. Based on the retrieved documents:\n"
        for i, doc in enumerate(docs):
            # citing the chunks
            preview = doc.page_content[:100].replace('\n', ' ')
            response_text += f"- Policy Chunk {i+1}: {preview}...\n"
            
        response_text += "\n(Note: This is a fallback mocked response because AWS Bedrock call failed.)"
    
    return response_text

# Node to split query
def split_query(state: AgentState):
    original_query = state['query']
    print(f"--- SPLITTING QUERY: {original_query} ---")
    
    # Check if multiple questions exist
    prompt_text = """Split the following user query into individual, standalone questions. 
    Return the questions as a JSON list of strings. Do not include any other text.
    
    Example input: "can I get a refund and what is the shipping time"
    Example output: ["can I get a refund", "what is the shipping time"]
    
    Query: {query}
    """
    
    try:
        model_id = os.getenv("LLM_MODEL", "us.amazon.nova-lite-v1:0")
        llm = ChatBedrockConverse(
            model="us.amazon.nova-lite-v1:0",
            temperature=0.0,
            aws_access_key_id=os.getenv("BEDROCK_AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("BEDROCK_AWS_SECRET_ACCESS_KEY"),
            region_name=os.getenv("BEDROCK_AWS_REGION", "us-east-1")
        )

        prompt = ChatPromptTemplate.from_template(prompt_text)
        chain = prompt | llm
        response = chain.invoke({"query": original_query})
        
        import json
        content = response.content.replace('```json', '').replace('```', '').strip()
        try:
            questions = json.loads(content)
        except json.JSONDecodeError:
            # Fallback to newline splitting if JSON fails
            questions = [q.strip() for q in content.split('\n') if q.strip()]
            
    except Exception as e:
        print(f"Error splitting query: {e}")
        # Very basic fallback
        questions = [original_query]

    return {"questions": questions}

# Node to process a single question
def process_question(state: AgentState):
    # state['query'] here will be the sub-question due to Send mapping
    question = state['query']
    docs = retrieve_docs(question)
    answer = generate_response(question, docs)
    return {"answers": [f"Q: {question}\nA: {answer}"]}

# Conditional edge logic
def continue_to_verification(state: AgentState):
    return [Send("process_question", {"query": q}) for q in state['questions']]

# construct graph
def build_policy_checker_graph():
    workflow = StateGraph(AgentState)
    
    workflow.add_node("split_query", split_query)
    workflow.add_node("process_question", process_question)
    
    workflow.set_entry_point("split_query")
    workflow.add_conditional_edges("split_query", continue_to_verification, ["process_question"])
    workflow.add_edge("process_question", END)
    
    return workflow.compile()


graph = build_policy_checker_graph()
# test_input = {"query": "what is the refund policy also what is the ai policy and tell me about the latte method", "order_details": {}}
# result = graph.invoke(test_input)

def predict(message, history):
    input_data = {"query": [HumanMessage(content=message)], "order_details": {}}
    result = graph.invoke(input_data)
    print(result)

    return result['answers'][-1]

demo = gr.ChatInterface(
    predict,
    api_name="chat"
)

demo.launch()





# #Define Tools
# @tool
# def multiply(a: int, b: int) -> int:
#     """Multiply `a` and `b`.

#     Args:
#         a: First int
#         b: Second int
#     """
#     return a * b


# @tool
# def add(a: int, b: int) -> int:
#     """Adds `a` and `b`.

#     Args:
#         a: First int
#         b: Second int
#     """
#     return a + b


# @tool
# def divide(a: int, b: int) -> float:
#     """Divide `a` and `b`.

#     Args:
#         a: First int
#         b: Second int
#     """
#     return a / b


# # Adds tools to LLM
# tools = [add, multiply, divide]
# tools_by_name = {tool.name: tool for tool in tools}
# model_with_tools = llm.bind_tools(tools)


# #Define State

# from langchain.messages import AnyMessage
# from typing_extensions import TypedDict, Annotated
# import operator


# class MessagesState(TypedDict):
#     messages: Annotated[list[AnyMessage], operator.add]
#     llm_calls: int


# #Define Model Node used to call the LLM
# from langchain.messages import SystemMessage

# def llm_call(state: dict):
#     """LLM decides whether to call a tool or not"""

#     return {
#         "messages": [
#             model_with_tools.invoke(
#                 [
#                     SystemMessage(
#                         content="You are a helpful assistant tasked with performing arithmetic on a set of inputs."
#                     )
#                 ]
#                 + state["messages"]
#             )
#         ],
#         "llm_calls": state.get('llm_calls', 0) + 1
#     }

# #Define tool node
# from langchain.messages import ToolMessage


# def tool_node(state: dict):
#     """Performs the tool call"""

#     result = []
#     for tool_call in state["messages"][-1].tool_calls:
#         tool = tools_by_name[tool_call["name"]]
#         observation = tool.invoke(tool_call["args"])
#         result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
#     return {"messages": result}

# # Define END logic
# from typing import Literal
# from langgraph.graph import StateGraph, START, END


# def should_continue(state: MessagesState) -> Literal["tool_node", END]:
#     """Decide if we should continue the loop or stop based upon whether the LLM made a tool call"""

#     messages = state["messages"]
#     last_message = messages[-1]

#     # If the LLM makes a tool call, then perform an action
#     if last_message.tool_calls:
#         return "tool_node"

#     # Otherwise, we stop (reply to the user)
#     return END



# # Build it all

# # Build workflow
# agent_builder = StateGraph(MessagesState)

# # Add nodes
# agent_builder.add_node("llm_call", llm_call)
# agent_builder.add_node("tool_node", tool_node)

# # Add edges to connect nodes
# agent_builder.add_edge(START, "llm_call")
# agent_builder.add_conditional_edges(
#     "llm_call",
#     should_continue,
#     ["tool_node", END]
# )
# agent_builder.add_edge("tool_node", "llm_call")

# # Compile the agent
# agent = agent_builder.compile()

# # Show the agent
# from IPython.display import Image, display
# display(Image(agent.get_graph(xray=True).draw_mermaid_png()))

# # Invoke
# from langchain.messages import HumanMessage
# messages = [HumanMessage(content="Add 3 and 4.")]
# messages = agent.invoke({"messages": messages})
# for m in messages["messages"]:
#     m.pretty_print()


# from typing_extensions import Literal
# from langchain.messages import HumanMessage, SystemMessage


# # Schema for structured output to use as routing logic
# class Route(BaseModel):
#     step: Literal["poem", "story", "joke"] = Field(
#         None, description="The next step in the routing process"
#     )


# # Augment the LLM with schema for structured output
# router = llm.with_structured_output(Route)


# # State
# class State(TypedDict):
#     input: str
#     decision: str
#     output: str


# # Nodes
# def llm_call_1(state: State):
#     """Write a story"""

#     result = llm.invoke(state["input"])
#     return {"output": result.content}


# def llm_call_2(state: State):
#     """Write a joke"""

#     result = llm.invoke(state["input"])
#     return {"output": result.content}


# def llm_call_3(state: State):
#     """Write a poem"""

#     result = llm.invoke(state["input"])
#     return {"output": result.content}


# def llm_call_router(state: State):
#     """Route the input to the appropriate node"""

#     # Run the augmented LLM with structured output to serve as routing logic
#     decision = router.invoke(
#         [
#             SystemMessage(
#                 content="Route the input to story, joke, or poem based on the user's request."
#             ),
#             HumanMessage(content=state["input"]),
#         ]
#     )

#     return {"decision": decision.step}


# # Conditional edge function to route to the appropriate node
# def route_decision(state: State):
#     # Return the node name you want to visit next
#     if state["decision"] == "story":
#         return "llm_call_1"
#     elif state["decision"] == "joke":
#         return "llm_call_2"
#     elif state["decision"] == "poem":
#         return "llm_call_3"


# # Build workflow
# router_builder = StateGraph(State)

# # Add nodes
# router_builder.add_node("llm_call_1", llm_call_1)
# router_builder.add_node("llm_call_2", llm_call_2)
# router_builder.add_node("llm_call_3", llm_call_3)
# router_builder.add_node("llm_call_router", llm_call_router)

# # Add edges to connect nodes
# router_builder.add_edge(START, "llm_call_router")
# router_builder.add_conditional_edges(
#     "llm_call_router",
#     route_decision,
#     {  # Name returned by route_decision : Name of next node to visit
#         "llm_call_1": "llm_call_1",
#         "llm_call_2": "llm_call_2",
#         "llm_call_3": "llm_call_3",
#     },
# )
# router_builder.add_edge("llm_call_1", END)
# router_builder.add_edge("llm_call_2", END)
# router_builder.add_edge("llm_call_3", END)

# # Compile workflow
# router_workflow = router_builder.compile()

# # Show the workflow
# display(Image(router_workflow.get_graph().draw_mermaid_png()))

# # Invoke
# state = router_workflow.invoke({"input": "Write me a joke about cats"})
# print(state["output"])





# model = ChatOllama(
#     model = "gpt-oss",
#     validate_model_on_init=True,
#     temperature=0.7
# )

# checkpointer = InMemorySaver()

# agent = create_agent(
#     model = model,
#     system_prompt=SYSTEM_PROMPT,
#     checkpointer=checkpointer
# )

# THREAD_ID = str(uuid.uuid4())

# def predict(message, history):
#     config = {"configurable": {"thread_id": THREAD_ID}}
#     input_data = {"messages": [HumanMessage(content=message)]}
#     result = agent.invoke(input_data, config=config)
#     return result['messages'][-1].content

# demo = gr.ChatInterface(
#     predict,
#     api_name="chat"
# )

# demo.launch()





# # Define tools
# @tool
# def access_policy_document() -> int:
#     """Multiply `a` and `b`.

#     Args:
#         a: First int
#         b: Second int
#     """
#     return a * b



# # Augment the LLM with tools
# tools = [add, multiply, divide]
# tools_by_name = {tool.name: tool for tool in tools}
# llm_with_tools = llm.bind_tools(tools)


# from langgraph.graph import MessagesState
# from langchain.messages import SystemMessage, HumanMessage, ToolMessage


# # Nodes
# def llm_call(state: MessagesState):
#     """LLM decides whether to call a tool or not"""

#     return {
#         "messages": [
#             llm_with_tools.invoke(
#                 [
#                     SystemMessage(
#                         content="You are a helpful assistant tasked with performing arithmetic on a set of inputs."
#                     )
#                 ]
#                 + state["messages"]
#             )
#         ]
#     }


# def tool_node(state: dict):
#     """Performs the tool call"""

#     result = []
#     for tool_call in state["messages"][-1].tool_calls:
#         tool = tools_by_name[tool_call["name"]]
#         observation = tool.invoke(tool_call["args"])
#         result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
#     return {"messages": result}


# # Conditional edge function to route to the tool node or end based upon whether the LLM made a tool call
# def should_continue(state: MessagesState) -> Literal["tool_node", END]:
#     """Decide if we should continue the loop or stop based upon whether the LLM made a tool call"""

#     messages = state["messages"]
#     last_message = messages[-1]

#     # If the LLM makes a tool call, then perform an action
#     if last_message.tool_calls:
#         return "tool_node"

#     # Otherwise, we stop (reply to the user)
#     return END


# # Build workflow
# agent_builder = StateGraph(MessagesState)

# # Add nodes
# agent_builder.add_node("llm_call", llm_call)
# agent_builder.add_node("tool_node", tool_node)

# # Add edges to connect nodes
# agent_builder.add_edge(START, "llm_call")
# agent_builder.add_conditional_edges(
#     "llm_call",
#     should_continue,
#     ["tool_node", END]
# )
# agent_builder.add_edge("tool_node", "llm_call")

# # Compile the agent
# agent = agent_builder.compile()

# # Invoke
# messages = [HumanMessage(content="Add 3 and 4.")]
# messages = agent.invoke({"messages": messages})
# for m in messages["messages"]:
#     m.pretty_print()