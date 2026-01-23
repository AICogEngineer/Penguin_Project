import gradio as gr
import os
import uuid
import re
from langchain_aws import ChatBedrock
from langchain_core.tools import tool
from langchain.messages import AIMessage, ToolMessage
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware 
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from dotenv import load_dotenv

# 1. Setup
load_dotenv()
region = os.getenv("BEDROCK_AWS_REGION")
llm = ChatBedrock(model_id="amazon.nova-lite-v1:0", region_name=region)

# Global store to simulate a database of pending approvals
PENDING_APPROVALS = {}

# 2. Define Tools
@tool
def calculate_sum(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

@tool
def user_request_pii_or_ft() -> str:
    """When the user requests PII (Personally Identifiable Information) or financial transactions, use this tool to request credentials."""
    return "To proceed with your PII/financial transaction request, please provide: username, email, and billing zip code."

@tool
def verify_user_credential(username: str, email: str, zip_code: str) -> str:
    """Submit user credentials for verification. This will be sent to a human reviewer for approval.
    After calling this tool, YOU MUST wait for admin approval, then call request_approved() or request_rejected()."""
    return f"Credentials submitted for review:\nUsername: {username}\nEmail: {email}\nZip: {zip_code}"

@tool
def request_approved(username: str) -> str:
    """Call this tool AFTER admin approves the credentials to confirm approval to the user."""
    return f"✅ Great news, {username}! Your credentials have been verified and approved. You now have access to your PII/financial information. How can I assist you further?"

@tool
def request_rejected(username: str) -> str:
    """Call this tool AFTER admin rejects the credentials to inform the user."""
    return f"❌ Sorry, {username}. Your credentials could not be verified. Please check your information and try again."

# 3. Create Agent
checkpointer = InMemorySaver()

hitl_config = HumanInTheLoopMiddleware(
    interrupt_on={
        "verify_user_credential": {"allowed_decisions": ["approve", "reject"]}, 
        "calculate_sum": False, 
    },
    description_prefix="Approval Required",
)

agent = create_agent(
    model=llm,
    tools=[calculate_sum, user_request_pii_or_ft, verify_user_credential, request_approved, request_rejected],
    middleware=[hitl_config],
    checkpointer=checkpointer,
    system_prompt="""You are a helpful assistant. When a user requests PII or financial transactions:
1. First use user_request_pii_or_ft() to ask for credentials
2. When they provide credentials, use verify_user_credential() to submit them
3. After verify_user_credential() returns, the system will pause for admin approval
4. Once resumed, you MUST call either request_approved() or request_rejected() based on the admin's decision
5. Then provide the final response to the user"""
)

# --- HELPER FUNCTIONS ---

def censor_sensitive_data(text):
    """Censor email addresses and zip codes in the text for display purposes."""
    if not text:
        return text
    
    # Censor email addresses (keep first 2 chars and domain)
    email_pattern = r'\b([a-zA-Z0-9]{1,2})[a-zA-Z0-9._%+-]*@([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})\b'
    text = re.sub(email_pattern, r'\1***@\2', text)
    
    # Censor zip codes (show only first digit)
    zipcode_pattern = r'\b(\d)\d{4}\b'
    text = re.sub(zipcode_pattern, r'\1****', text)
    
    return text

# --- USER SIDE LOGIC ---

# --- USER SIDE LOGIC ---

def user_predict(message, history):
    # Extract thread_id from a global session store
    # For simplicity, we'll use a consistent thread per gradio session
    if not hasattr(user_predict, 'thread_id'):
        import hashlib
        user_predict.thread_id = f"session_{hashlib.md5(str(os.urandom(16)).encode()).hexdigest()[:8]}"
    
    thread_id = user_predict.thread_id
    config = {"configurable": {"thread_id": thread_id}}
    
    # Censor the user's message for display
    censored_user_message = censor_sensitive_data(message)
    
    state = agent.get_state(config)
    
    # --- STATUS CHECK LOGIC ---
    if message.lower().strip() in ["check", "status", "update", "done?"]:
        
        # Scenario A: Still Paused
        if state.next and state.next[0] == "tools":
            bot_response = "⏳ **Still Waiting...** \n\nThe admin has not approved the request yet. Please wait a moment and type 'check' again."
            history.append({"role": "user", "content": censored_user_message})
            history.append({"role": "assistant", "content": bot_response})
            return history, ""
        
        # Scenario B: Admin Approved/Rejected - Continue execution
        messages = state.values.get("messages", [])
        if messages:
            last_msg = messages[-1]
            
            # If it's an AI message, return it (without the generic "Request Processed" wrapper)
            if isinstance(last_msg, AIMessage) and last_msg.content:
                clean_content = re.sub(r'<thinking>.*?</thinking>', '', last_msg.content, flags=re.DOTALL).strip()
                bot_response = censor_sensitive_data(clean_content)
                history.append({"role": "user", "content": censored_user_message})
                history.append({"role": "assistant", "content": bot_response})
                return history, ""
            
            # If it's a tool message, continue the agent to get the AI response
            if isinstance(last_msg, ToolMessage):
                result = agent.invoke(None, config=config)
                clean_content = re.sub(r'<thinking>.*?</thinking>', '', result["messages"][-1].content, flags=re.DOTALL).strip()
                bot_response = censor_sensitive_data(clean_content)
                history.append({"role": "user", "content": censored_user_message})
                history.append({"role": "assistant", "content": bot_response})
                return history, ""

        bot_response = "No updates yet. Please wait for admin approval."
        history.append({"role": "user", "content": censored_user_message})
        history.append({"role": "assistant", "content": bot_response})
        return history, ""

    # --- NORMAL CHAT LOGIC ---
    
    # Check if blocked
    if state.next and state.next[0] == "tools":
        bot_response = "⚠️ **Blocked**: You have a pending request waiting for approval. You cannot send new messages until the Admin approves or rejects it. Type 'check' to see if it's been processed."
        history.append({"role": "user", "content": censored_user_message})
        history.append({"role": "assistant", "content": bot_response})
        return history, ""

    input_data = {"messages": [{"role": "user", "content": message}]}  # Send UNCENSORED to agent
    result = agent.invoke(input_data, config=config)
    
    # Check for Interrupt
    if "__interrupt__" in result:
        interrupt_data = result["__interrupt__"][0]
        action = interrupt_data.value["action_requests"][0]
        tool_args = action.get("args", action.get("arguments", {}))
        
        PENDING_APPROVALS[thread_id] = {
            "action": action,
            "tool_name": action["name"],
            "args": tool_args,
            "status": "pending_review"
        }
        
        bot_response = (f"✋ **Approval Needed**\n\n"
                f"I need admin permission to verify your credentials.\n"
                f"**Request ID:** `{thread_id}`\n\n"
                f"_Your information has been submitted securely. Please ask the admin to approve this ID, then type **'check'** here._")
        
        history.append({"role": "user", "content": censored_user_message})
        history.append({"role": "assistant", "content": bot_response})
        return history, ""

    # Return final response (CENSORED)
    raw_content = result["messages"][-1].content
    clean_content = re.sub(r'<thinking>.*?</thinking>', '', raw_content, flags=re.DOTALL).strip()
    bot_response = censor_sensitive_data(clean_content)
    
    history.append({"role": "user", "content": censored_user_message})
    history.append({"role": "assistant", "content": bot_response})
    return history, ""


# --- ADMIN LOGIC ---

def get_pending_requests():
    if not PENDING_APPROVALS:
        return "No pending requests."
    
    display_text = ""
    for tid, data in PENDING_APPROVALS.items():
        display_text += (
            f"🔹 **Thread ID:** `{tid}`\n"
            f"   **Tool:** {data['tool_name']}\n"
            f"   **Args:** {data['args']}\n"
            f"   **Status:** {data['status']}\n\n"
        )
    return display_text

def admin_approve(thread_id_input, decision):
    target_tid = thread_id_input.strip()
    
    if target_tid not in PENDING_APPROVALS:
        return "❌ Error: ID not found."

    config = {"configurable": {"thread_id": target_tid}}
    decision_type = "approve" if decision == "Approve" else "reject"
    
    # Get the username from the pending approval
    username = PENDING_APPROVALS[target_tid]["args"].get("username", "User")
    
    # Resume the agent with the decision
    resume_command = Command(resume={"decisions": [{"type": decision_type}]})
    agent.invoke(resume_command, config=config)
    
    # Now explicitly tell the agent to call the appropriate tool
    if decision_type == "approve":
        follow_up_msg = f"The admin has approved the credentials. You must now call the request_approved tool with username '{username}' to inform the user."
    else:
        follow_up_msg = f"The admin has rejected the credentials. You must now call the request_rejected tool with username '{username}' to inform the user."
    
    follow_up = {"messages": [{"role": "user", "content": follow_up_msg}]}
    agent.invoke(follow_up, config=config)
    
    # Clean up
    del PENDING_APPROVALS[target_tid]
    
    return f"✅ Request {decision}d for Thread {target_tid}.\nThe user can now type 'check' to see the result."

# --- UI LAYOUT ---

with gr.Blocks(title="AI Agent System") as demo:
    gr.Markdown("# 🤖 Corporate AI Assistant")
    
    with gr.Tabs():
        
        # TAB 1: USER (Custom Chat Interface with censoring)
        with gr.TabItem("💬 Chat"):
            chatbot = gr.Chatbot(label="Chat", height=500, type="messages")
            msg = gr.Textbox(label="Type your message here...", placeholder="Ask me anything or say 'check' to see approval status")
            clear = gr.Button("Clear Chat")
            
            gr.Examples(
                examples=["I need to verify my credentials", "Calculate 50 + 100"],
                inputs=msg
            )
            
            def user_message_handler(message, history):
                return user_predict(message, history)
            
            def clear_chat():
                return [], ""
            
            msg.submit(user_message_handler, [msg, chatbot], [chatbot, msg])
            clear.click(clear_chat, None, [chatbot, msg])
        
        # TAB 2: ADMIN (Dashboard)
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

            # Wiring Admin Events
            refresh_btn.click(get_pending_requests, outputs=queue_display)
            process_btn.click(admin_approve, inputs=[tid_input, decision_radio], outputs=admin_output)

demo.launch()