import re
import os
from typing import List
from langchain_core.documents import Document
from langchain_aws import ChatBedrockConverse
from langchain_core.prompts import ChatPromptTemplate
from utils.retriever import get_policy_retriever

# Tools

# For HITL verifying user
def censor_sensitive_data(text):
    if not text: return text
    email_pattern = r'\b([a-zA-Z0-9]{1,2})[a-zA-Z0-9._%+-]*@([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})\b'
    text = re.sub(email_pattern, r'\1***@\2', text)
    # Censor Zip Codes (5 digits) but avoid years (19xx, 20xx) to spare dates
    # Very basic negative lookahead or just assume years are 19/20
    # Or just don't censor 5-digit numbers? 
    # Let's just censor 5 digit numbers that DO NOT start with 19 or 20
    zipcode_pattern = r'\b(?!19|20)(\d)\d{4}\b'
    text = re.sub(zipcode_pattern, r'\1****', text)
    return text


# For Policy Retreival
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


