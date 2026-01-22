import gradio as gr
import os
import re
import hashlib
from langchain_aws import ChatBedrock
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command, interrupt
from typing import Literal
from dotenv import load_dotenv

# --- 1. SETUP & STATE ---

load_dotenv()
region = os.getenv("BEDROCK_AWS_REGION")
llm = ChatBedrock(model_id="amazon.nova-lite-v1:0", region_name=region)

class AgentState(MessagesState):
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

# --- 2. HELPER FUNCTIONS ---

def censor_sensitive_data(text):
    if not text: return text
    email_pattern = r'\b([a-zA-Z0-9]{1,2})[a-zA-Z0-9._%+-]*@([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})\b'
    text = re.sub(email_pattern, r'\1***@\2', text)
    zipcode_pattern = r'\b(\d)\d{4}\b'
    text = re.sub(zipcode_pattern, r'\1****', text)
    return text

# --- 3. GRAPH NODES ---

def classify_request(state: AgentState) -> Command[Literal["request_credentials", "collect_username", "collect_email", "collect_zipcode", "handle_normal"]]:
    current_stage = state.get("dialogue_state", "idle")
    
    if current_stage == "awaiting_username": return Command(goto="collect_username")
    elif current_stage == "awaiting_email": return Command(goto="collect_email")
    elif current_stage == "awaiting_zipcode": return Command(goto="collect_zipcode")
        
    last_message = state["messages"][-1].content.lower()
    pii_keywords = ["pii", "personal", "credentials", "financial", "transaction", "verify", "account", "balance", "information"]
    
    if any(keyword in last_message for keyword in pii_keywords):
        return Command(goto="request_credentials")
    return Command(goto="handle_normal")

def request_credentials(state: AgentState) -> Command[Literal[END]]:
    response1 = AIMessage(content="🔐 To proceed with your PII/financial transaction request, I need to verify your identity.")
    response2 = AIMessage(content="Let's verify your identity. First, please provide your **username**:")
    return Command(
        update={"messages": state["messages"] + [response1, response2], "dialogue_state": "awaiting_username"},
        goto=END 
    )

def collect_username(state: AgentState) -> Command[Literal[END]]:
    last_message = state["messages"][-1].content
    credentials = state.get("credentials", {}).copy()
    credentials["username"] = last_message.strip()
    response = AIMessage(content="Great! Now, please provide your **email address**:")
    return Command(
        update={"credentials": credentials, "messages": state["messages"] + [response], "dialogue_state": "awaiting_email"},
        goto=END 
    )

def collect_email(state: AgentState) -> Command[Literal[END]]:
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

def collect_zipcode(state: AgentState) -> Command[Literal[END, "submit_for_review"]]:
    last_message = state["messages"][-1].content
    credentials = state.get("credentials", {}).copy()
    zipcode = last_message.strip()
    if not zipcode.isdigit() or len(zipcode) != 5:
        response = AIMessage(content="⚠️ Please provide a valid 5-digit zip code:")
        return Command(update={"messages": state["messages"] + [response]}, goto=END)
    
    credentials["zip_code"] = zipcode
    return Command(update={"credentials": credentials, "dialogue_state": "idle"}, goto="submit_for_review")

def submit_for_review(state: AgentState) -> Command[Literal["human_review"]]:
    credentials = state["credentials"]
    response = AIMessage(
        content=f"📝 Credentials received:\n- Username: {credentials['username']}\n- Email: {credentials['email']}\n- Zip Code: {credentials['zip_code']}\n\n✋ Your request has been submitted for admin approval. Please type 'check' to see the status."
    )
    return Command(update={"messages": state["messages"] + [response]}, goto="human_review")

def human_review(state: AgentState) -> Command[Literal["process_approval"]]:
    credentials = state["credentials"]
    thread_id = state.get("thread_id", "unknown")
    PENDING_APPROVALS[thread_id] = {
        "credentials": credentials,
        "status": "pending_review"
    }
    
    # --- INTERRUPT ---
    # The value passed to resume=... in admin_approve will be returned here
    approval_decision = interrupt({
        "message": "Waiting for admin approval",
        "credentials": credentials,
        "thread_id": thread_id
    })
    
    return Command(update={"approval_status": approval_decision}, goto="process_approval")

def process_approval(state: AgentState) -> Command[Literal["access_database", "handle_rejection"]]:
    approval_status = state.get("approval_status", "pending")
    if approval_status == "approved": return Command(goto="access_database")
    else: return Command(goto="handle_rejection")

def access_database(state: AgentState) -> Command[Literal[END]]:
    credentials = state["credentials"]
    response = AIMessage(
        content=f"✅ **Access Granted!**\n\nWelcome back, {credentials['username']}! 🎉\n\n📊 Your account information has been retrieved:\n- Balance: $10,543.21\n- Security Level: Verified ✓\n\nHow can I assist you with your account today?"
    )
    return Command(update={"messages": state["messages"] + [response]}, goto=END)

def handle_rejection(state: AgentState) -> Command[Literal[END]]:
    credentials = state["credentials"]
    response = AIMessage(content=f"❌ **Access Denied**\n\nSorry, {credentials['username']}. Your credentials could not be verified.")
    return Command(update={"messages": state["messages"] + [response]}, goto=END)

def handle_normal(state: AgentState) -> Command[Literal[END]]:
    messages = state["messages"]
    system_message = SystemMessage(content="You are a helpful assistant. Respond naturally.")
    response = llm.invoke([system_message] + messages)
    return Command(update={"messages": state["messages"] + [response]}, goto=END)

# --- 4. BUILD GRAPH ---

def build_graph():
    workflow = StateGraph(AgentState)
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
    
    checkpointer = MemorySaver()
    return workflow.compile(checkpointer=checkpointer)

graph = build_graph() 

# --- 5. USER SIDE LOGIC ---

def user_predict(message, history):
    if not hasattr(user_predict, 'thread_id'):
        user_predict.thread_id = f"session_{hashlib.md5(str(os.urandom(16)).encode()).hexdigest()[:8]}"
    
    thread_id = user_predict.thread_id
    config = {"configurable": {"thread_id": thread_id}}
    censored_user_message = censor_sensitive_data(message)
    state = graph.get_state(config)
    bot_response = ""
    
    # --- STATUS CHECK LOGIC ---
    if message.lower().strip() in ["check", "status", "update", "done?"]:
        # 1. Check if we are still waiting at human_review
        if state.next and "human_review" in state.next:
            bot_response = "⏳ **Still Waiting...** \n\nThe admin has not approved the request yet. Please wait a moment and type 'check' again."
            history.append({"role": "user", "content": censored_user_message})
            history.append({"role": "assistant", "content": bot_response})
            return history, ""
        
        # 2. If NOT waiting, the graph might be finished. Check the history.
        # When admin resumes, the graph runs to END. So state.next will be empty.
        messages = state.values.get("messages", [])
        if messages:
            last_msg = messages[-1]
            # Check if this is the "final" message (Access Granted or Denied)
            # We want to display it if the user hasn't seen it yet.
            if isinstance(last_msg, AIMessage):
                clean_content = re.sub(r'<thinking>.*?</thinking>', '', last_msg.content, flags=re.DOTALL).strip()
                bot_response = censor_sensitive_data(clean_content)
                history.append({"role": "user", "content": censored_user_message})
                history.append({"role": "assistant", "content": bot_response})
                return history, ""
        
        bot_response = "No updates yet."
        history.append({"role": "user", "content": censored_user_message})
        history.append({"role": "assistant", "content": bot_response})
        return history, ""
    
    # --- NORMAL CHAT LOGIC ---
    if state.next and "human_review" in state.next:
        bot_response = "⚠️ **Blocked**: You have a pending request waiting for approval. Type 'check' to see if it's been processed."
        history.append({"role": "user", "content": censored_user_message})
        history.append({"role": "assistant", "content": bot_response})
        return history, ""

    input_state = {
        "messages": [HumanMessage(content=message)],
        "thread_id": thread_id
    }
    
    try:
        result = graph.invoke(input_state, config=config)
        updated_state = graph.get_state(config)
        
        if updated_state.next and "human_review" in updated_state.next:
            bot_response = (f"✋ **Approval Needed**\n\n"
                           f"Your credentials have been submitted for review.\n"
                           f"**Request ID:** `{thread_id}`\n\n"
                           f"_Please ask the admin to approve this ID, then type **'check'** here._")
        
        elif result and "messages" in result:
            last_msg = result["messages"][-1]
            if isinstance(last_msg, AIMessage):
                clean_content = re.sub(r'<thinking>.*?</thinking>', '', last_msg.content, flags=re.DOTALL).strip()
                bot_response = censor_sensitive_data(clean_content)
            
    except Exception as e:
        bot_response = f"❌ Error: {str(e)}"
    
    history.append({"role": "user", "content": censored_user_message})
    history.append({"role": "assistant", "content": bot_response})
    return history, ""

# --- 6. ADMIN LOGIC (FIXED) ---

def get_pending_requests():
    if not PENDING_APPROVALS: return "No pending requests."
    display_text = ""
    for tid, data in PENDING_APPROVALS.items():
        creds = data['credentials']
        display_text += (
            f"🔹 **Thread ID:** `{tid}`\n"
            f"   **Username:** {creds.get('username', 'N/A')}\n"
            f"   **Email:** {creds.get('email', 'N/A')}\n"
            f"   **Zip Code:** {creds.get('zip_code', 'N/A')}\n"
            f"   **Status:** {data['status']}\n\n"
        )
    return display_text

def admin_approve(thread_id_input, decision):
    target_tid = thread_id_input.strip()
    if target_tid not in PENDING_APPROVALS: return "❌ Error: ID not found."
    
    config = {"configurable": {"thread_id": target_tid}}
    
    # Determine value to pass back to interrupt()
    approval_status = "approved" if decision == "Approve" else "rejected"
    
    try:
        # CRITICAL FIX: Use Command(resume=...) to satisfy the interrupt!
        resume_command = Command(resume=approval_status)
        
        # This will un-pause 'human_review', pass 'approval_status' to the variable,
        # and execute the rest of the graph to completion (END).
        graph.invoke(resume_command, config=config)
        
        del PENDING_APPROVALS[target_tid]
        return f"✅ Request {decision}d for Thread {target_tid}.\nThe user can now type 'check' to see the result."
    except Exception as e:
        return f"❌ Error processing decision: {str(e)}"

# --- 7. UI LAYOUT ---

with gr.Blocks(title="AI Agent System") as demo:
    gr.Markdown("# 🤖 Corporate AI Assistant with LangGraph")
    
    with gr.Tabs():
        # TAB 1: USER
        with gr.TabItem("💬 Chat"):
            chatbot = gr.Chatbot(label="Chat", height=500) 
            msg = gr.Textbox(label="Type your message here...", placeholder="Ask me anything or say 'check' to see approval status")
            clear = gr.Button("Clear Chat")
            
            gr.Examples(examples=["I need to verify my credentials", "What's the weather like?"], inputs=msg)
            
            def user_message_handler(message, history):
                return user_predict(message, history)
            
            def clear_chat():
                if hasattr(user_predict, 'thread_id'): delattr(user_predict, 'thread_id')
                return [], ""
            
            msg.submit(user_message_handler, [msg, chatbot], [chatbot, msg])
            clear.click(clear_chat, None, [chatbot, msg])
        
        # TAB 2: ADMIN
        with gr.TabItem("🔒 Admin Dashboard"):
            gr.Markdown("### 🛡️ Security Approval Queue")
            with gr.Row():
                refresh_btn = gr.Button("🔄 Refresh List")
                queue_display = gr.Markdown("No pending requests.")
            gr.Markdown("---")
            with gr.Row():
                tid_input = gr.Textbox(label="Paste Thread ID here")
                decision_radio = gr.Radio(["Approve", "Reject"], label="Action", value="Approve")
                process_btn = gr.Button("Submit Decision", variant="primary")
            admin_output = gr.Markdown()

            refresh_btn.click(get_pending_requests, outputs=queue_display)
            process_btn.click(admin_approve, inputs=[tid_input, decision_radio], outputs=admin_output)

demo.launch()