
from typing import TypedDict, Annotated, List, Dict
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.documents import Document
from langchain_aws import ChatBedrock
from langchain_core.prompts import ChatPromptTemplate

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
    2. Search Pinecone for the top 3 most similar text chunks.
    3. Update state['documents'] with these chunks.
    """
    print(f"--- RETRIEVING POLICY FOR: {state['query']} ---")
    retriever = get_policy_retriever(k=3)
    docs = retriever.invoke(state['query'])
    return {"documents": docs}

# generation node
def generate_answer(state: AgentState):
    print("--- GENERATING ANSWER ---")
    docs = state['documents']
    query = state['query']
    
    
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
        llm = ChatBedrock(
            model_id=os.getenv("LLM_MODEL", "anthropic.claude-3-sonnet-20240229-v1:0"),
            model_kwargs={"temperature": 0.1},
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
    
    return {"curr_answer": response_text}

# construct graph
def build_policy_checker_graph():
    workflow = StateGraph(AgentState)
    
    workflow.add_node("retrieve", retrieve_policy)
    workflow.add_node("generate", generate_answer)
    
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)
    
    return workflow.compile()

if __name__ == "__main__":
    graph = build_policy_checker_graph()
    test_input = {"query": "What is the return policy?", "order_details": {}}
    result = graph.invoke(test_input)
    print("\nFINAL RESULT:")
    print(result['curr_answer'])
