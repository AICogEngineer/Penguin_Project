
from typing import TypedDict, Annotated, List, Dict
from langgraph.graph import StateGraph, END
from langgraph.types import Send
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage
from langchain_core.documents import Document
from langchain_aws import ChatBedrock
from langchain_core.prompts import ChatPromptTemplate

from dotenv import load_dotenv
import operator
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

# Explicitly load .env from project root
dotenv_path = os.path.join(os.path.dirname(__file__), '../../.env')
load_dotenv(dotenv_path)

from policy_engine.rag.retriever import get_policy_retriever

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
        model_id = os.getenv("LLM_MODEL", "us.amazon.nova-lite-v1:0")
        
        llm = ChatBedrock(
            model_id=model_id,
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
        llm = ChatBedrock(
            model_id=model_id,
            model_kwargs={"temperature": 0.0},
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

if __name__ == "__main__":
    graph = build_policy_checker_graph()
    test_input = {"query": "what is the refund policy also what is the ai policy and tell me about the latte method", "order_details": {}}
    result = graph.invoke(test_input)
    print("\nFINAL RESULT:")
    for ans in result.get('answers', []):
        print(ans)
        print("-" * 20)
