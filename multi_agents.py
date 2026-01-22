from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool, ToolRuntime
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents.structured_output import ToolStrategy
from langchain_aws import ChatBedrockConverse
from langchain.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.types import Send, Command, interrupt
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import TypedDict, Annotated, Literal, List, Dict
import gradio as gr
import os
import re
import operator
import hashlib
import uuid
import getpass
from dotenv import load_dotenv
import snowflake.connector
from IPython.display import Image, display
from policy_engine.rag.retriever import get_policy_retriever


#TODO: Organize all nodes, states, and tools into utils

#TODO: Add guardrails, quotas, and limits

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


# Router
class Route(BaseModel):
    step: Literal["userInfoInquiry", "policyQuestion", "misc"] = Field(
        None, description="The next step in the routing process."
    )

router = llm.with_structured_output(Route)



#TODO: Convert both to MessagesStates
# Formatted Outputs : States

class PolicyQuestionsState(TypedDict):
    query: str                      # customer query
    order_details: Dict             # order details (placeholder for snowflake)
    documents: Annotated[List[Document], operator.add]       # where chunks are stored
    questions: List[str]         # list of sub-questions (if any)
    curr_answer: str                # final answer by agent (legacy)
    answers: Annotated[List[str], operator.add] # collected answers


class VerifyUserInfoState(MessagesState):
    """Extended state for our agent workflow."""
    credentials: dict = {}
    approval_status: str = "pending"
    thread_id: str = ""
    dialogue_state: Literal[
        "idle", 
        "awaiting_username", 
        "awaiting_email", 
        "awaiting_zipcode"
    ] = "idle"

PENDING_APPROVALS = {}


class ParentRouterState(MessagesState):
    input: str
    thread_id: str
    decision: str
    output: str
    credentials: Dict = {}
    active_mode: Literal["userInfoInquiry", "policyQuestion", None] = None


# Tools

# For HITL verifying user

def censor_sensitive_data(text):
    if not text: return text
    email_pattern = r'\b([a-zA-Z0-9]{1,2})[a-zA-Z0-9._%+-]*@([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})\b'
    text = re.sub(email_pattern, r'\1***@\2', text)
    zipcode_pattern = r'\b(\d)\d{4}\b'
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


# Nodes

# HITL User Verify Nodes

def classify_request(state: VerifyUserInfoState) -> Command[Literal["request_credentials", "collect_username", "collect_email", "collect_zipcode", "handle_normal"]]:
    current_stage = state.get("dialogue_state", "idle")
    
    if current_stage == "awaiting_username": return Command(goto="collect_username")
    elif current_stage == "awaiting_email": return Command(goto="collect_email")
    elif current_stage == "awaiting_zipcode": return Command(goto="collect_zipcode")
        
    last_message = state["messages"][-1].content.lower()
    pii_keywords = ["pii", "personal", "credentials", "financial", "transaction", "verify", "account", "balance", "information"]
    
    if any(keyword in last_message for keyword in pii_keywords):
        return Command(goto="request_credentials")
    return Command(goto="handle_normal")

def request_credentials(state: VerifyUserInfoState) -> Command[Literal[END]]:
    response1 = AIMessage(content="🔐 To proceed with your PII/financial transaction request, I need to verify your identity.")
    response2 = AIMessage(content="Let's verify your identity. First, please provide your **username**:")
    return Command(
        update={"messages": state["messages"] + [response1, response2], "dialogue_state": "awaiting_username"},
        goto=END 
    )

def collect_username(state: VerifyUserInfoState) -> Command[Literal[END]]:
    last_message = state["messages"][-1].content
    credentials = state.get("credentials", {}).copy()
    credentials["username"] = last_message.strip()
    response = AIMessage(content="Great! Now, please provide your **email address**:")
    return Command(
        update={"credentials": credentials, "messages": state["messages"] + [response], "dialogue_state": "awaiting_email"},
        goto=END 
    )

def collect_email(state: VerifyUserInfoState) -> Command[Literal[END]]:
    last_message = state["messages"][-1].content
    credentials = state.get("credentials", {}).copy()
    email = last_message.strip()
    if "@" not in email:
        response = AIMessage(content="⚠️ That doesn't look like a valid email. Please provide a valid email address:")
        return Command(update={"messages": state["messages"] + [response]}, goto=END)
    
    credentials["email"] = email
    response = AIMessage(content="Almost done! Please provide your **5-digit zip code**:")
    return Command(
        update={"credentials": credentials, "messages": state["messages"] + [response], "dialogue_state": "awaiting_zipcode"},
        goto=END
    )

def collect_zipcode(state: VerifyUserInfoState) -> Command[Literal[END, "submit_for_review"]]:
    last_message = state["messages"][-1].content
    credentials = state.get("credentials", {}).copy()
    zipcode = last_message.strip()
    if not zipcode.isdigit() or len(zipcode) != 5:
        response = AIMessage(content="⚠️ Please provide a valid 5-digit zip code:")
        return Command(update={"messages": state["messages"] + [response]}, goto=END)
    
    credentials["zip_code"] = zipcode
    return Command(update={"credentials": credentials, "dialogue_state": "idle"}, goto="submit_for_review")

def submit_for_review(state: VerifyUserInfoState) -> Command[Literal["human_review"]]:
    credentials = state["credentials"]
    response = AIMessage(
        content=f"📝 Credentials received:\n- Username: {credentials['username']}\n- Email: {credentials['email']}\n- Zip Code: {credentials['zip_code']}\n\n✋ Your request has been submitted for admin approval. Please type 'check' to see the status."
    )
    return Command(update={"messages": state["messages"] + [response]}, goto="human_review")

def human_review(state: VerifyUserInfoState) -> Command[Literal["process_approval"]]:
    credentials = state["credentials"]
    thread_id = state.get("thread_id", "unknown")
    PENDING_APPROVALS[thread_id] = {
        "credentials": credentials,
        "status": "pending_review"
    }
    
    # --- INTERRUPT ---
    approval_decision = interrupt({
        "message": "Waiting for admin approval",
        "credentials": credentials,
        "thread_id": thread_id
    })
    
    return Command(update={"approval_status": approval_decision}, goto="process_approval")

def process_approval(state: VerifyUserInfoState) -> Command[Literal["access_database", "handle_rejection"]]:
    approval_status = state.get("approval_status", "pending")
    if approval_status == "approved": return Command(goto="access_database")
    else: return Command(goto="handle_rejection")

def access_database(state: VerifyUserInfoState) -> Command[Literal[END]]:
    credentials = state["credentials"]
    response = AIMessage(
        content=f"✅ **Access Granted!**\n\nWelcome back, {credentials['username']}! 🎉\n\n📊 Your account information has been retrieved:\n- Balance: $10,543.21\n- Security Level: Verified ✓\n\nHow can I assist you with your account today?"
    )
    return Command(update={"messages": state["messages"] + [response]}, goto=END)

def handle_rejection(state: VerifyUserInfoState) -> Command[Literal[END]]:
    credentials = state["credentials"]
    response = AIMessage(content=f"❌ **Access Denied**\n\nSorry, {credentials['username']}. Your credentials could not be verified.")
    return Command(update={"messages": state["messages"] + [response]}, goto=END)

def handle_normal(state: VerifyUserInfoState) -> Command[Literal[END]]:
    messages = state["messages"]
    system_message = SystemMessage(content="You are a helpful assistant. Respond naturally.")
    response = llm.invoke([system_message] + messages)
    return Command(update={"messages": state["messages"] + [response]}, goto=END)


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
    return {"answers": [f"Q: {question}\nA: {answer}"]}

# Conditional edge logic
def continue_to_verification(state: PolicyQuestionsState):
    return [Send("process_question", {"query": q}) for q in state['questions']]


"""
TODO: Set up subgraphs for each component state.
Router will route to the needed subgraph.
"""

# Shared state if graph put in directly
# Different state if graph put in node indirectly


#Graph


def build_hitl_graph():
    workflow = StateGraph(VerifyUserInfoState)
    workflow.add_node("classify_request", classify_request)
    workflow.add_node("request_credentials", request_credentials)
    workflow.add_node("collect_username", collect_username)
    workflow.add_node("collect_email", collect_email)
    workflow.add_node("collect_zipcode", collect_zipcode)
    workflow.add_node("submit_for_review", submit_for_review)
    workflow.add_node("human_review", human_review)
    workflow.add_node("process_approval", process_approval)
    workflow.add_node("access_database", access_database)
    workflow.add_node("handle_rejection", handle_rejection)
    workflow.add_node("handle_normal", handle_normal)
    workflow.add_edge(START, "classify_request")
    
    return workflow.compile(checkpointer=True)

verifyUserInfo_subgraph = build_hitl_graph()


# construct graph
def build_policy_checker_graph():
    workflow = StateGraph(PolicyQuestionsState)
    
    # Add Nodes
    workflow.add_node("split_query", split_query)
    workflow.add_node("process_question", process_question)
    
    # Set Edges
    workflow.set_entry_point("split_query")
    workflow.add_conditional_edges("split_query", continue_to_verification, ["process_question"])
    workflow.add_edge("process_question", END)
    

    # Compile graph
    return workflow.compile()

checkCompanyPolicy_subgraph = build_policy_checker_graph()





# Router Nodes

def llm_call_router(state: ParentRouterState):
    """Route the input to the appropriate node."""

    if state.get("active_mode") == "userInfoInquiry":
        return {"decision": "userInfoInquiry"}

    decision = router.invoke(
        [
            SystemMessage(
                content="Route the user's input to userInfoInquiry, policyQuestion, or misc based on the user's request."
            ),
            HumanMessage(content=state["input"]),
        ]
    )

    return {"decision": decision.step}

# Condition edge for router
def route_decision(state: ParentRouterState):
    if state["decision"] == "userInfoInquiry":
        return "userInfoInquiry"
    elif state["decision"] == "policyQuestion":
        return "policyQuestion"
    elif state["decision"] == "misc":
        return "misc_call"

def misc_call(state: ParentRouterState):
    """
    Do NOT answer any questions NOT related to company policy or personal user information. 
    Kindly remind the user that we cannot answer any unrelated questions, and ask if they would like help with anything else related to company policy, refunds, or user information such as orders.
    """

    system_message = SystemMessage(content=SYSTEM_PROMPT)
    
    # WRAP state["input"] in HumanMessage
    user_message = HumanMessage(content=state["input"])
    
    # Pass the list of message objects
    result = llm.invoke([system_message, user_message])
    
    return {"output": result.content, "messages": [result]}

def userInfoInquiry(state: ParentRouterState):
    """
    The state the user is routed to if their inquiry relates to personal user information 
    such as orders made, refunds made, requesting a refund, requesting to submit a complaint, 
    or anything related to personal stored user information.
    
    :param state: Description
    :type state: State
    """

    initial_msg = HumanMessage(content=state["input"])

    response = verifyUserInfo_subgraph.invoke({
        "messages": [initial_msg],
        "thread_id": state["thread_id"],
        "credentials": state.get("credentials", {})
    })

    sub_dialogue_state = response.get("dialogue_state", "idle")
    if sub_dialogue_state != "idle":
        new_active_mode = "userInfoInquiry"
    else:
        new_active_mode = None

    final_credentials = response.get("credentials", {})

    return {
        "output": response["messages"][-1].content,
        "messages": response["messages"][-1],
        "active_mode": new_active_mode,
        "credentials": final_credentials
    }

def policyQuestion(state: ParentRouterState):
    """
    The state the user is routed to if their inquiry relates to anything company policy without requiring
    any personal user information stored, such as specific orders they made, refunds they made, submitting any complains or refunds, 
    or anything specifically about the user.
    
    :param state: Description
    :type state: ParentRouterState
    """

    response = checkCompanyPolicy_subgraph.invoke({
        "query": state["input"]
    })

    return {"output": response["answers"], "messages": response["answers"]}



def build_router_graph():
    checkpointer = InMemorySaver()

    router_builder = StateGraph(ParentRouterState)

    router_builder.add_node("llm_call_router", llm_call_router)
    router_builder.add_node("misc_call", misc_call)
    router_builder.add_node("policyQuestion", policyQuestion) # User Info Inquiry (Refunds/Complaints/Orders)
    router_builder.add_node("userInfoInquiry", userInfoInquiry) # General Policy Questions

    router_builder.add_edge(START, "llm_call_router")
    router_builder.add_conditional_edges(
        "llm_call_router",
        route_decision,
        {
            "userInfoInquiry": "userInfoInquiry",
            "policyQuestion": "policyQuestion",
            "misc_call": "misc_call"
        },
    )

    router_builder.add_edge("misc_call", END)
    router_builder.add_edge("policyQuestion", END)
    router_builder.add_edge("userInfoInquiry", END)

    return router_builder.compile(checkpointer=checkpointer)

router_graph = build_router_graph()



# User Side

# Pass `session_id` as an argument (handled by Gradio State)
def user_predict(message, history, session_id):
    # 1. Generate Thread ID if new session
    if not session_id:
        session_id = f"session_{hashlib.md5(str(os.urandom(16)).encode()).hexdigest()[:8]}"
    
    thread_id = session_id
    config = {"configurable": {"thread_id": thread_id}}
    censored_user_message = censor_sensitive_data(message)
    bot_response = ""
    
    # Fetch current state
    state = router_graph.get_state(config)
    
    # --- LOGIC FOR "CHECK" / STATUS UPDATES ---
    if message.lower().strip() in ["check", "status", "update", "done?"]:
        # Scenario A: Graph is running/paused (Waiting for Admin)
        if state.next:
            snapshot = router_graph.get_state(config, subgraphs=True)
            if snapshot.tasks:
                sub_state = snapshot.tasks[0].state
                if sub_state.next and "human_review" in sub_state.next:
                    bot_response = "⏳ **Still Waiting...** \n\nThe admin has not approved the request yet. Please wait a moment and type 'check' again."
        
        # Scenario B: Graph successfully FINISHED (Admin approved, graph ran to END)
        # If no next steps, look at the history
        if not bot_response and not state.next:
            # Check the very last message in the parent history
            messages = state.values.get("messages", [])
            if messages:
                last_msg = messages[-1]
                if isinstance(last_msg, AIMessage):
                    bot_response = censor_sensitive_data(last_msg.content)
            
            if not bot_response:
                 bot_response = "It looks like the request is finished, but I have no new updates."

        # Scenario C: Fallback
        if not bot_response:
             bot_response = "No active requests found."
             
        history.append({"role": "user", "content": censored_user_message})
        history.append({"role": "assistant", "content": bot_response})
        return history, "", session_id

    # --- NORMAL CHAT FLOW ---
    
    # Prevent chatting if stuck in approval
    if state.values and state.values.get("decision") == "userInfoInquiry":
        snapshot = router_graph.get_state(config, subgraphs=True)
        if snapshot.tasks:
            sub_state = snapshot.tasks[0].state
            if sub_state.next and "human_review" in sub_state.next:
                 bot_response = "⚠️ **Blocked**: You have a pending request waiting for approval. Type 'check' to see if it's been processed."
                 history.append({"role": "user", "content": censored_user_message})
                 history.append({"role": "assistant", "content": bot_response})
                 return history, "", session_id

    input_state = {
        "input": message,
        "thread_id": thread_id
    }
    
    # Run the graph
    result = router_graph.invoke(input_state, config=config)

    # Output Handling
    if not bot_response:
        # Check if we just hit the interrupt
        snapshot = router_graph.get_state(config, subgraphs=True)
        if snapshot.tasks:
            sub_state = snapshot.tasks[0].state
            if sub_state.next and "human_review" in sub_state.next:
                 bot_response = (f"✋ **Approval Needed**\n\n"
                            f"Your credentials have been submitted for review.\n"
                            f"Our admin will review your request. You can check your progress by typing **'check'**")
    
    if not bot_response and "output" in result:
        output_data = result["output"]
        if isinstance(output_data, list):
            bot_response = "\n\n".join(output_data)
        elif hasattr(output_data, 'content'):
            clean_content = re.sub(r'<thinking>.*?</thinking>', '', output_data.content, flags=re.DOTALL).strip()
            bot_response = censor_sensitive_data(clean_content)
        else:
            bot_response = str(output_data)

    history.append({"role": "user", "content": censored_user_message})
    history.append({"role": "assistant", "content": bot_response})
    
    # Return session_id back to Gradio State
    return history, "", session_id

# Admin Side

def refresh_admin_view():
    """Returns updated text for the dashboard AND updated choices for the dropdown."""
    if not PENDING_APPROVALS:
        # Return status text and an empty list of choices
        return "No pending requests.", gr.update(choices=[], value=None)
    
    # Build text display
    display_text = ""
    # Build dropdown choices list (Thread IDs)
    thread_choices = []
    
    for tid, data in PENDING_APPROVALS.items():
        creds = data['credentials']
        thread_choices.append(tid)
        display_text += (
            f"🔹 **Thread ID:** `{tid}`\n"
            f"   **Username:** {creds.get('username', 'N/A')}\n"
            f"   **Email:** {creds.get('email', 'N/A')}\n"
            f"   **Zip Code:** {creds.get('zip_code', 'N/A')}\n"
            f"   **Status:** {data['status']}\n\n"
        )
    
    # Update dropdown with new choices and auto-select the first one
    return display_text, gr.update(choices=thread_choices, value=thread_choices[0] if thread_choices else None)

def admin_approve(target_tid, decision):
    if not target_tid or target_tid not in PENDING_APPROVALS:
        return f"❌ Error: ID '{target_tid}' not found or invalid."
    
    config = {"configurable": {"thread_id": target_tid}}
    approval_status = "approved" if decision == "Approve" else "rejected"
    
    try:
        # Resume graph
        resume_command = Command(resume=approval_status)
        router_graph.invoke(resume_command, config=config)
        
        del PENDING_APPROVALS[target_tid]
        
        return f"✅ Request {decision}d for Thread {target_tid}.\nThe user can now type 'check' to see the result."
    except Exception as e:
        return f"❌ Error processing decision: {str(e)}"


# UI

with gr.Blocks(title="AI Agent System") as demo:
    gr.Markdown("# 🤖 Corporate AI Assistant with LangGraph")
    
    session_state = gr.State(value=None)

    with gr.Tabs():
        # TAB 1: USER
        with gr.TabItem("💬 Chat"):
            chatbot = gr.Chatbot(label="Chat", 
                                 height=500,
                                 value=[{"role": "assistant", "content": "Hello! I am the PenguinZ customer support chat bot. Ask me anything about company policies, orders, refunds, or any other information related."}]
            )
            msg = gr.Textbox(label="Type your message here...", placeholder="Ask me anything or say 'check' to see approval status")
            clear = gr.Button("Clear Chat")
            
            gr.Examples(examples=["I need to verify my credentials", "What's the refund policy?"], inputs=msg)
            
            def user_message_handler(message, history, session_id):
                return user_predict(message, history, session_id)
            
            def clear_chat():
                if hasattr(user_predict, 'thread_id'): delattr(user_predict, 'thread_id')
                return [{"role": "assistant", "content": "Hello! I am the PenguinZ customer support chat bot. Ask me anything about company policies, orders, refunds, or any other information related."}], ""
            
            msg.submit(user_message_handler, [msg, chatbot, session_state], [chatbot, msg, session_state])
            clear.click(clear_chat, None, [chatbot, msg, session_state])
        
        # TAB 2: ADMIN
        with gr.TabItem("🔒 Admin Dashboard"):
            gr.Markdown("### 🛡️ Security Approval Queue")
            with gr.Row():
                refresh_btn = gr.Button("🔄 Refresh List")
                queue_display = gr.Markdown("No pending requests.")
            gr.Markdown("---")
            with gr.Row():
                # NEW: Dropdown instead of Textbox
                tid_dropdown = gr.Dropdown(label="Select Thread ID", choices=[], interactive=True)
                decision_radio = gr.Radio(["Approve", "Reject"], label="Action", value="Approve")
                process_btn = gr.Button("Submit Decision", variant="primary")
            admin_output = gr.Markdown()

            # Wiring Admin Events
            # Clicking Refresh updates BOTH the text display AND the dropdown choices
            refresh_btn.click(refresh_admin_view, outputs=[queue_display, tid_dropdown])
            
            # Clicking Process uses the selected value from the dropdown
            process_btn.click(admin_approve, inputs=[tid_dropdown, decision_radio], outputs=admin_output)

demo.launch()