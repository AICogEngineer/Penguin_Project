import gradio as gr
import os
import re
import hashlib
import snowflake.connector
from langchain_aws import ChatBedrock
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, MessagesState, START, END
#from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command, interrupt
from typing import Literal
from dotenv import load_dotenv


# --- 1. SETUP & STATE ---

load_dotenv()

USER = os.getenv('SNOWFLAKE_USER')
PASSWORD = os.getenv('SNOWFLAKE_PASSWORD')
ACCOUNT = os.getenv('SNOWFLAKE_ACCOUNT')
WAREHOUSE = os.getenv('SNOWFLAKE_WAREHOUSE')
DATABASE = os.getenv('SNOWFLAKE_DATABASE')
SCHEMA = os.getenv('SNOWFLAKE_SCHEMA')

region = os.getenv("BEDROCK_AWS_REGION")
llm = ChatBedrock(model_id="amazon.nova-lite-v1:0", region_name=region)

# class AgentState(MessagesState):
#     """Extended state for our agent workflow."""
#     credentials: dict = {}
#     approval_status: str = "pending"
#     thread_id: str = ""
#     dialogue_state: Literal[
#         "idle", 
#         "awaiting_username", 
#         "awaiting_email", 
#         "awaiting_zipcode"
#     ] = "idle"

# PENDING_APPROVALS = {}

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
    data_request_type: Literal["pii", "transactions", "both", None] = None
    snowflake_results: dict = {}

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

def query_snowflake(credentials: dict, query_type: str):
    """
    Helper function to query Snowflake for either PII or transaction data.
    
    Args:
        credentials: Dict containing user credentials (email, zip_code, username)
        query_type: Either "pii" or "transactions"
    
    Returns:
        Dict with 'success' boolean and either 'data' or 'error'
    """
    try:
        # Connect to Snowflake using your environment variables
        conn = snowflake.connector.connect(
            user=USER,
            password=PASSWORD,
            account=ACCOUNT,
            warehouse=WAREHOUSE,
            database=DATABASE,
            schema=SCHEMA
        )
        cursor = conn.cursor()
        
        # # Execute appropriate query based on type
        # if query_type == "both":
        #     # Query DIM_CUSTOMERS using EMAIL
        #     query = """
        #         SELECT 
        #             *
        #         FROM DIM_CUSTOMERS
        #         WHERE EMAIL = %s
        #     """
        #     cursor.execute(query, (credentials['email'],))

        if query_type == "pii":
            # Query DIM_CUSTOMERS using EMAIL
            query = """
                SELECT 
                    *
                FROM DIM_CUSTOMERS
                WHERE EMAIL = %s
            """
            cursor.execute(query, (credentials['email'],))
            
        elif query_type == "transactions":
            
            user_query = """
                SELECT * FROM FCT_TRANSACTIONS
                WHERE BILLING_ZIP_CODE = %s
            """
            cursor.execute(user_query, (credentials['zip_code'],))
            user_result = cursor.fetchone()
            
            if not user_result:
                cursor.close()
                conn.close()
                return {"success": True, "data": []}  # No user found
            
            user_id = user_result[0]
            
        
        else:
            cursor.close()
            conn.close()
            return {"success": False, "error": f"Invalid query_type: {query_type}"}
        
        # Fetch results and convert to list of dictionaries
        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()
        results = [dict(zip(columns, row)) for row in rows]
        
        # Clean up
        cursor.close()
        conn.close()
        
        return {"success": True, "data": results}
        
    except snowflake.connector.errors.ProgrammingError as e:
        # SQL syntax errors, table not found, etc.
        print(f"Snowflake ProgrammingError: {e}")
        return {"success": False, "error": f"Database query error: {str(e)}"}
    
    except snowflake.connector.errors.DatabaseError as e:
        # Connection issues, authentication failures, etc.
        print(f"Snowflake DatabaseError: {e}")
        return {"success": False, "error": f"Database connection error: {str(e)}"}
    
    except Exception as e:
        # Catch-all for any other errors
        print(f"Unexpected error in query_snowflake: {e}")
        return {"success": False, "error": f"Unexpected error: {str(e)}"}

def determine_data_request(user_message: str) -> Literal["pii", "transactions", "both"]:
    """Determine if user wants PII, transactions, or both based on their message."""
    message_lower = user_message.lower()
    
    pii_keywords = ["personal", "information", "profile", "account details", "email", "address", "pii"]
    transaction_keywords = ["transaction", "purchase", "order", "payment", "balance", "history"]
    
    has_pii = any(keyword in message_lower for keyword in pii_keywords)
    has_transactions = any(keyword in message_lower for keyword in transaction_keywords)
    
    if has_pii and has_transactions:
        return "both"
    elif has_pii:
        return "pii"
    elif has_transactions:
        return "transactions"
    else:
        return "both"  # Default to both if unclear

def classify_request(state: VerifyUserInfoState) -> Command[Literal["request_credentials", "collect_username", "collect_email", "collect_zipcode", "handle_normal"]]:
    current_stage = state.get("dialogue_state", "idle")
    
    if current_stage == "awaiting_username": return Command(goto="collect_username")
    elif current_stage == "awaiting_email": return Command(goto="collect_email")
    elif current_stage == "awaiting_zipcode": return Command(goto="collect_zipcode")
        
    last_message = state["messages"][-1].content.lower()
    pii_keywords = ["pii", "personal", "credentials", "financial", "transaction", "verify", "account", "balance", "information"]
    
    if any(keyword in last_message for keyword in pii_keywords):
        # Determine what type of data they want
        data_type = determine_data_request(last_message)
        return Command(
            update={"data_request_type": data_type},
            goto="request_credentials"
        )
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

def process_approval(state: VerifyUserInfoState) -> Command[Literal["route_to_data_query", "handle_rejection"]]:
    approval_status = state.get("approval_status", "pending")
    if approval_status == "approved": 
        return Command(goto="route_to_data_query")  # CHANGED: was "access_database"
    else: 
        return Command(goto="handle_rejection")
    
# def access_database(state: AgentState) -> Command[Literal[END]]:
#     credentials = state["credentials"]
#     response = AIMessage(
#         content=f"✅ **Access Granted!**\n\nWelcome back, {credentials['username']}! 🎉\n\n📊 Your account information has been retrieved:\n- Balance: $10,543.21\n- Security Level: Verified ✓\n\nHow can I assist you with your account today?"
#     )
#     return Command(update={"messages": state["messages"] + [response]}, goto=END)

def route_to_data_query(state: VerifyUserInfoState) -> Command[Literal["query_pii", "query_transactions", "query_both"]]:
    """Route to appropriate data query based on request type."""
    data_type = state.get("data_request_type", "both")
    
    if data_type == "pii":
        return Command(goto="query_pii")
    elif data_type == "transactions":
        return Command(goto="query_transactions")
    else:
        return Command(goto="query_both")


def query_pii(state: VerifyUserInfoState) -> Command[Literal["format_response"]]:
    credentials = state["credentials"]
    result = query_snowflake(credentials, "pii")
    return Command(
        update={"snowflake_results": {"pii": result}},
        goto="format_response"
    )


def query_transactions(state: VerifyUserInfoState) -> Command[Literal["format_response"]]:
    credentials = state["credentials"]
    result = query_snowflake(credentials, "transactions")
    return Command(
        update={"snowflake_results": {"transactions": result}},
        goto="format_response"
    )


def query_both(state: VerifyUserInfoState) -> Command[Literal["format_response"]]:
    credentials = state["credentials"]
    pii_result = query_snowflake(credentials, "pii")
    trans_result = query_snowflake(credentials, "transactions")
    return Command(
        update={"snowflake_results": {"pii": pii_result, "transactions": trans_result}},
        goto="format_response"
    )


def format_response(state: VerifyUserInfoState) -> Command[Literal[END]]:
    """Format the Snowflake query results into a user-friendly response."""
    credentials = state["credentials"]
    results = state.get("snowflake_results", {})
    
    # Start building the response
    response_content = f"✅ **Access Granted!**\n\nWelcome back, {credentials['username']}-penguin! 🎉\n\n"
    
    has_data = False
    errors = []
    
    # Format PII information if available
    if "pii" in results:
        if results["pii"]["success"]:
            pii_data = results["pii"]["data"]
            if pii_data and len(pii_data) > 0:
                has_data = True
                customer = pii_data[0]  # Should only be one customer
                
                response_content += "👤 **Personal Information-penguin:**\n"
                
                # ✅ DYNAMICALLY DISPLAY ALL COLUMNS
                for column_name, column_value in customer.items():
                    # Format column name to be more readable (e.g., USER_ID -> User ID)
                    readable_name = column_name.replace('_', ' ').title()
                    response_content += f"- {readable_name}: {column_value}\n"
                
                response_content += "\n"
            else:
                response_content += "👤 **Personal Information-penguin:** No customer record found-penguin.\n\n"
        else:
            errors.append("personal information")
    
    # Format transaction information if available
    if "transactions" in results:
        if results["transactions"]["success"]:
            trans_data = results["transactions"]["data"]
            if trans_data and len(trans_data) > 0:
                has_data = True
                response_content += f"💳 **Recent Transactions-penguin** ({len(trans_data)} shown):\n\n"
                
                # Calculate total if AMOUNT column exists
                total_amount = 0
                for transaction in trans_data:
                    try:
                        amount = float(transaction.get('AMOUNT', 0))
                        total_amount += amount
                    except (ValueError, TypeError):
                        pass
                
                # Display transactions with ALL columns
                for i, transaction in enumerate(trans_data, 1):
                    response_content += f"**Transaction #{i}-penguin**\n"
                    
                    # ✅ DYNAMICALLY DISPLAY ALL COLUMNS
                    for column_name, column_value in transaction.items():
                        # Format column name to be more readable
                        readable_name = column_name.replace('_', ' ').title()
                        
                        # Special formatting for certain columns
                        if column_name == 'AMOUNT' and column_value is not None:
                            response_content += f"   💵 {readable_name}: ${column_value}\n"
                        elif column_name == 'TRANSACTION_DATE' and hasattr(column_value, 'strftime'):
                            formatted_date = column_value.strftime('%Y-%m-%d %H:%M:%S')
                            response_content += f"   📅 {readable_name}: {formatted_date}\n"
                        elif column_name == 'STATUS':
                            response_content += f"   📊 {readable_name}: {column_value}\n"
                        else:
                            response_content += f"   {readable_name}: {column_value}\n"
                    
                    response_content += "\n"
                
                if total_amount > 0:
                    response_content += f"💰 **Total Recent Activity-penguin:** ${total_amount:.2f}\n\n"
            else:
                response_content += "💳 **Recent Transactions-penguin:** No transaction history found for the provided email and zip code-penguin.\n\n"
        else:
            errors.append("transaction history")
    
    # Add error messages if any queries failed
    if errors:
        error_list = " and ".join(errors)
        response_content += f"⚠️ There was an issue retrieving your {error_list}-penguin. "
        response_content += "Please contact support for assistance-penguin.\n\n"
    
    # Add closing message
    if has_data:
        response_content += "📊 **Security Level-penguin:** Verified ✓\n\n"
        response_content += "How can I assist you with your account today-penguin?"
    else:
        response_content += "We couldn't find any records matching your credentials-penguin. "
        response_content += "Please verify your information or contact support-penguin."
    
    response = AIMessage(content=response_content)
    
    return Command(
        update={"messages": state["messages"] + [response]},
        goto=END
    )

def handle_rejection(state: VerifyUserInfoState) -> Command[Literal[END]]:
    credentials = state["credentials"]
    response = AIMessage(content=f"❌ **Access Denied**\n\nSorry, {credentials['username']}. Your credentials could not be verified.")
    return Command(update={"messages": state["messages"] + [response]}, goto=END)

def handle_normal(state: VerifyUserInfoState) -> Command[Literal[END]]:
    messages = state["messages"]
    system_message = SystemMessage(content="You are a helpful assistant. Respond naturally.")
    response = llm.invoke([system_message] + messages)
    return Command(update={"messages": state["messages"] + [response]}, goto=END)

# --- 4. BUILD GRAPH ---

def build_hitl_graph():
    # Use InMemorySaver for checkpointing if not already defined
    checkpointer = InMemorySaver()
    
    workflow = StateGraph(VerifyUserInfoState)
    
    # Original nodes
    workflow.add_node("classify_request", classify_request)
    workflow.add_node("request_credentials", request_credentials)
    workflow.add_node("collect_username", collect_username)
    workflow.add_node("collect_email", collect_email)
    workflow.add_node("collect_zipcode", collect_zipcode)
    workflow.add_node("submit_for_review", submit_for_review)
    workflow.add_node("human_review", human_review)
    workflow.add_node("process_approval", process_approval)
    workflow.add_node("handle_rejection", handle_rejection)
    workflow.add_node("handle_normal", handle_normal)
    
    # NEW: Snowflake data query nodes
    workflow.add_node("route_to_data_query", route_to_data_query)
    workflow.add_node("query_pii", query_pii)
    workflow.add_node("query_transactions", query_transactions)
    workflow.add_node("query_both", query_both)
    workflow.add_node("format_response", format_response)
    
    # Set entry point
    workflow.add_edge(START, "classify_request")
    
    # Compile with checkpointer
    return workflow.compile(checkpointer=checkpointer)

graph = build_hitl_graph()

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
        if state.next and "human_review" in state.next:
            bot_response = "⏳ **Still Waiting...** \n\nThe admin has not approved the request yet. Please wait a moment and type 'check' again."
            history.append({"role": "user", "content": censored_user_message})
            history.append({"role": "assistant", "content": bot_response})
            return history, ""
        
        # Check if completed
        messages = state.values.get("messages", [])
        if messages:
            last_msg = messages[-1]
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
                           f"Our admin will review your request, You can check your progress by typing **'check'**")
        
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

# --- 6. ADMIN LOGIC (WITH DROPDOWN) ---

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
            chatbot = gr.Chatbot(label="Chat", height=500, value=[{"role": "assistant", "content": "Hello! I am the PenguinZ customer support chat bot. Ask me anything about company policies, orders, refunds, or any other information related."}]) 
            msg = gr.Textbox(label="Type your message here...", placeholder="Ask me anything or say 'check' to see approval status")
            clear = gr.Button("Clear Chat")
            
            gr.Examples(examples=["I need to verify my credentials", "What's the weather like?"], inputs=msg)
            
            def user_message_handler(message, history):
                return user_predict(message, history)
            
            def clear_chat():
                if hasattr(user_predict, 'thread_id'): delattr(user_predict, 'thread_id')
                return [{"role": "assistant", "content": "Hello! I am the PenguinZ customer support chat bot. Ask me anything about company policies, orders, refunds, or any other information related."}], ""
            
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