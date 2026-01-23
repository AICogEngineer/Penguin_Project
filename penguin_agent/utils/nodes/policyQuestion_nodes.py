from utils.states import PolicyQuestionsState
from utils.tools import generate_response, retrieve_docs
from langgraph.types import Send
from langchain_aws import ChatBedrockConverse
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv

load_dotenv()

# Policy Retriever Nodes 

# Node to split query
def split_query(state: PolicyQuestionsState):
    original_query = state['query']
    print(f"--- SPLITTING QUERY: {original_query} ---")
    
    # Check if multiple questions exist
    prompt_text = """Split the following user query into individual, standalone questions. 
    Return the questions as a JSON list of strings. Do not include any other text.
    
    Example input: "can I get a refund and what is the shipping time"
    Example output: ["can I get a refund", "what is the shipping time"]

    Example input: "What is the AI and refund policy?"
    Example output: ["What is the AI policy?", "What is the refund policy?"]
    
    Query: {query}
    """
    
    try:
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
def process_question(state: PolicyQuestionsState):
    # state['query'] here will be the sub-question due to Send mapping
    question = state['query']
    docs = retrieve_docs(question)
    answer = generate_response(question, docs)
    return {"answers": [f"Q: {question}\nA: {answer}\n"]}

# Conditional edge logic
def continue_to_verification(state: PolicyQuestionsState):
    return [Send("process_question", {"query": q}) for q in state['questions']]
