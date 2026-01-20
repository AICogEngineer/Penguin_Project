
from typing import TypedDict, Annotated, List, Dict
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.documents import Document

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from policy_engine.rag.retriever import get_policy_retriever

class AgentState(TypedDict):
    query: str                      # customer query
    order_details: Dict             # order details (placeholder for snowflake)
    documents: List[Document]       # where chunks are stored
    curr_answer: str                # final answer by agent

# retrieval node
def retrieve_policy(state: AgentState):
    """
    Node to retrieve relevant policy documents based on the query.
    Step:
    1. Look at state['query'].
    2. Search ChromaDB for the top 3 most similar text chunks.
    3. Update state['documents'] with these chunks.
    """
    print(f"--- RETRIEVING POLICY FOR: {state['query']} ---")
    retriever = get_policy_retriever(k=3)
    docs = retriever.invoke(state['query'])
    return {"documents": docs}

# generation node (MOCKED)
#  LLM will be here
# it will take the documents from the state and the query and generate answer.
def generate_answer(state: AgentState):
    print("--- GENERATING ANSWER (MOCKED) ---")
    docs = state['documents']
    
    # TODO:when ready, format this context string and send it to Bedrock:
    # prompt = f"Answer the user query based on this context: {context}..."
    context = "\n\n".join([d.page_content for d in docs])
    
    # mocked response construction
    response_text = "I have checked the policy. Based on the retrieved documents:\n"
    for i, doc in enumerate(docs):
        # citing the chunks
        preview = doc.page_content[:100].replace('\n', ' ')
        response_text += f"- Policy Chunk {i+1}: {preview}...\n"
    
    # TODO: when ready, send this to Bedrock and get the final answer.
    response_text += "\n(Note: This is a placeholder response. In production, bedrock llm would read these chunks and answer naturally.)"
    
    return {"curr_answer": response_text}

# construct graph
def build_policy_checker_graph():
    workflow = StateGraph(AgentState)
    
    # add nodes
    workflow.add_node("retrieve", retrieve_policy)
    workflow.add_node("generate", generate_answer)
    
    # define flow edges
    # start -> retrieve -> generate -> end
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)
    
    return workflow.compile()

if __name__ == "__main__":
    # test graph locally
    graph = build_policy_checker_graph()
    test_input = {"query": "What is the return policy for electronics?", "order_details": {}}
    result = graph.invoke(test_input)
    print("\nFINAL RESULT:")
    print(result['curr_answer'])
