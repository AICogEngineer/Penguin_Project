
from typing import TypedDict, Annotated, List, Dict
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage
from langchain_core.documents import Document
from langchain_aws import ChatBedrock
from langchain_core.prompts import ChatPromptTemplate
from langchain_aws import ChatBedrockConverse
from dotenv import load_dotenv
from pydantic import BaseModel, Field

import operator
import sys
import os

from typing import Annotated, List
import operator

load_dotenv()

llm = ChatBedrockConverse(
    model="us.amazon.nova-lite-v1:0",
    temperature=0.7,
    aws_access_key_id=os.getenv("BEDROCK_AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("BEDROCK_AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("BEDROCK_AWS_REGION", "us-east-1")
)

from typing_extensions import TypedDict
from langgraph.graph.state import StateGraph, START


# Define subgraph
class SubgraphState(TypedDict):
    # note that none of these keys are shared with the parent graph state
    bar: str
    baz: str

def subgraph_node_1(state: SubgraphState):
    return {"baz": "baz"}

def subgraph_node_2(state: SubgraphState):
    return {"bar": state["bar"] + state["baz"]}

subgraph_builder = StateGraph(SubgraphState)
subgraph_builder.add_node(subgraph_node_1)
subgraph_builder.add_node(subgraph_node_2)
subgraph_builder.add_edge(START, "subgraph_node_1")
subgraph_builder.add_edge("subgraph_node_1", "subgraph_node_2")
subgraph = subgraph_builder.compile()

# Define parent graph
class ParentState(TypedDict):
    foo: str

def node_1(state: ParentState):
    return {"foo": "hi! " + state["foo"]}

def node_2(state: ParentState):
    # Transform the state to the subgraph state
    response = subgraph.invoke({"bar": state["foo"]})
    # Transform response back to the parent state
    return {"foo": response["bar"]}


builder = StateGraph(ParentState)
builder.add_node("node_1", node_1)
builder.add_node("node_2", node_2)
builder.add_edge(START, "node_1")
builder.add_edge("node_1", "node_2")
graph = builder.compile()

for chunk in graph.stream({"foo": "foo"}, subgraphs=True):
    print(chunk)


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

# # Invoke
# #state = router_workflow.invoke({"input": "Write me a joke about cats"})
# #print(state["output"])