from utils.states import VerifyUserInfoState
from langgraph.types import Command, interrupt
import datetime
from typing import Literal
from utils.tools import generate_response, retrieve_docs
from langchain.messages import AIMessage, SystemMessage
from langchain_aws import ChatBedrockConverse
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END
from dotenv import load_dotenv
import snowflake.connector
import os

load_dotenv()

USER = os.getenv('SNOWFLAKE_USER')
PASSWORD = os.getenv('SNOWFLAKE_PASSWORD')
ACCOUNT = os.getenv('SNOWFLAKE_ACCOUNT')
WAREHOUSE = os.getenv('SNOWFLAKE_WAREHOUSE')
DATABASE = os.getenv('SNOWFLAKE_DATABASE')
SCHEMA = os.getenv('SNOWFLAKE_SCHEMA')

llm = ChatBedrockConverse(
    model="us.amazon.nova-lite-v1:0",
    temperature=0.7,
    aws_access_key_id=os.getenv("BEDROCK_AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("BEDROCK_AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("BEDROCK_AWS_REGION", "us-east-1")
)


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
        
        if query_type == "pii":
            # Query DIM_CUSTOMERS using EMAIL
            query = """
                SELECT DISTINCT
                    EMAIL, FIRST_NAME, LAST_NAME, ACCOUNT_TYPE, LOYALTY_POINTS
                FROM DIM_CUSTOMERS, FCT_TRANSACTIONS
                WHERE DIM_CUSTOMERS.USER_ID = FCT_TRANSACTIONS.USER_ID
                AND EMAIL = %s AND BILLING_ZIP_CODE = %s;
            """
            cursor.execute(query, (credentials['email'], credentials["zip_code"]))
            
        elif query_type == "transactions":
            
            user_query = """
                SELECT 
                CREATED_DATE, TRANSACTION_TYPE, PRODUCT_NAME, QUANTITY, BRAND, MANUFACTURER, COST_PRICE, UNIT_PRICE, TAX, SUBTOTAL, TOTAL, TRANSACTION_ID
                FROM FCT_TRANSACTIONS, DIM_CUSTOMERS, DIM_PRODUCTS
                WHERE DIM_CUSTOMERS.USER_ID = FCT_TRANSACTIONS.USER_ID AND DIM_PRODUCTS.PRODUCT_ID = FCT_TRANSACTIONS.PRODUCT_ID
                AND EMAIL = %s
                ORDER BY CREATED_DATE DESC;
            """
            cursor.execute(user_query, (credentials['email'],))
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

def determine_data_request(user_message: str) -> Literal["refund", "pii", "transactions", "both"]:
    """Determine if user wants a refund, PII, transactions, or both PII and transactions based on their message."""
    message_lower = user_message.lower()
    
    pii_keywords = ["personal", "information", "profile", "account details", "email", "address", "pii"]
    transaction_keywords = ["transaction", "purchase", "order", "payment", "balance", "history", "orders", "transactions", "purchases", "payments", "balances"]
    refund_keywords = ["refund", "return", "refunds", "returns"]
    
    has_pii = any(keyword in message_lower for keyword in pii_keywords)
    has_transactions = any(keyword in message_lower for keyword in transaction_keywords)
    has_refund = any(keyword in message_lower for keyword in refund_keywords)
    
    if has_refund:
        return "refund"
    elif has_pii and has_transactions:
        return "both"
    elif has_pii:
        return "pii"
    elif has_transactions:
        return "transactions"
    else:
        return "both"  # Default to both if unclear

def classify_request(state: VerifyUserInfoState) -> Command[Literal["request_credentials", "collect_username", "collect_email", "collect_zipcode", "handle_normal", "route_to_data_query", "query_refund_eligibility"]]:
    current_stage = state.get("dialogue_state", "idle")
    
    if current_stage == "awaiting_username": return Command(goto="collect_username")
    elif current_stage == "awaiting_email": return Command(goto="collect_email")
    elif current_stage == "awaiting_zipcode": return Command(goto="collect_zipcode")
    elif current_stage == "awaiting_product_selection":
        # Treating input as product name
        user_input = state["messages"][-1].content.strip()
        
        # Explicitly clarify intent for the LLM if needed
        # If user just typed the name (e.g. "microwave"), prepend context.
        # If they typed "refund microwave", leave it be or just normalize.
        if "refund" in user_input.lower() or "return" in user_input.lower():
            clarified_content = user_input
        else:
             clarified_content = f"I want to refund the {user_input}"
        
        clarified_message = HumanMessage(content=clarified_content)
        
        return Command(
            update={"messages": state["messages"] + [clarified_message]},
            goto="query_refund_eligibility"
        )
        
    last_message = state["messages"][-1].content.lower()
    pii_keywords = ["pii", "personal", "credentials", "financial", "transaction", "verify", "account", "balance", "information", "refund", "return", "order", "orders", "transactions"]
    
    #TODO: Change this to use an LLM to decide

    if any(keyword in last_message for keyword in pii_keywords):
        # Determine what type of data they want
        data_type = determine_data_request(last_message)

        if state["credentials"]:
            return Command(
                update={"data_request_type": data_type},
                goto="route_to_data_query"
            )

        return Command(
            update={"data_request_type": data_type},
            goto="request_credentials"
        )
    return Command(goto="handle_normal")

# ... (Previous Code) ...

    if status == "ask_product":
         # Helper to list products
         product_list = ", ".join([t.get('PRODUCT_NAME', 'Unknown') for t in transactions_context[:3]])
         response_msg = AIMessage(content=f"🤔 **Which item?**\n\nI see a few recent purchases: **{product_list}**.\n\nWhich one would you like to refund?")
         return Command(
            update={"messages": state["messages"] + [response_msg], "dialogue_state": "awaiting_product_selection"},
            goto=END
         )

    if status == "eligible":
        # Reset state if successful
        updated_state = {"dialogue_state": "idle"}
        
        # Check Red Flags
        flags = check_red_flags(transactions)
        
        if flags:
            # HITL Required
            refund_details = {
                "product_name": product_name,
                "eligibility_status": status,
                "transaction_id": transaction_id,
                "reason": reason
            }
            updated_state.update({"red_flags": flags, "refund_details": refund_details})
            return Command(
                update=updated_state,
                goto="human_review_refund"
            )
        else:
            # Auto-approve (Fast Track)
            response_msg = AIMessage(content=f"✅ **Refund Processed!**\n\nGreat news! We've verified that your request for **{product_name}** is within our policy.\nreason: {reason}\n\nSince your account is in good standing, we've fast-tracked your refund. You should see the funds shortly.")
            updated_state.update({"messages": state["messages"] + [response_msg]})
            return Command(
                update=updated_state,
                goto=END
            )

def request_credentials(state: VerifyUserInfoState) -> Command[Literal[END]]:
    response1 = AIMessage(content="To proceed with your PII/financial transaction request, I need to verify your identity.")
    response2 = AIMessage(content="Let's verify your identity. First, please provide your **username**:")
    return Command(
        update={"messages": state["messages"] + [response1, response2], "dialogue_state": "awaiting_username", "approval_status": "pending"},
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
        response = AIMessage(content="That doesn't look like a valid email. Please provide a valid email address:")
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
        response = AIMessage(content="Please provide a valid 5-digit zip code:")
        return Command(update={"messages": state["messages"] + [response]}, goto=END)
    
    credentials["zip_code"] = zipcode
    return Command(update={"credentials": credentials, "dialogue_state": "idle"}, goto="submit_for_review")

def validate_credentials_snowflake(credentials: dict) -> bool:
    """
    Validate if the credentials match a record in Snowflake.
    """
    try:
        conn = snowflake.connector.connect(
            user=USER,
            password=PASSWORD,
            account=ACCOUNT,
            warehouse=WAREHOUSE,
            database=DATABASE,
            schema=SCHEMA
        )
        cursor = conn.cursor()
        
        # Check for matching Email and Zip Code (Zip is in FCT_TRANSACTIONS)
        query = """
            SELECT c.FIRST_NAME 
            FROM DIM_CUSTOMERS c
            JOIN FCT_TRANSACTIONS t ON c.USER_ID = t.USER_ID
            WHERE c.EMAIL = %s AND t.BILLING_ZIP_CODE = %s
            LIMIT 1
        """
        cursor.execute(query, (credentials['email'], credentials["zip_code"]))
        result = cursor.fetchone()
        
        cursor.close()
        conn.close()
        
        return bool(result)
        
    except Exception as e:
        print(f"Validation Error: {e}")
        return False

def submit_for_review(state: VerifyUserInfoState) -> Command[Literal["human_review", "process_approval", "handle_rejection"]]:
    credentials = state["credentials"]
    
    # Automate Verification check
    is_valid = validate_credentials_snowflake(credentials)
    
    if is_valid:
        # Auto-Approve
        return Command(
            update={"approval_status": "approved"},
            goto="process_approval"
        )
    else:
        # Auto-Reject (or could route to human if we wanted, but user asked for automation)
        return Command(
            update={"approval_status": "rejected"},
            goto="handle_rejection"
        )

def human_review(state: VerifyUserInfoState) -> Command[Literal["process_approval"]]:
    # This node is potentially bypassed now for Identity, but kept for fallback or other flows if needed.
    credentials = state["credentials"]
    thread_id = state.get("thread_id", "unknown")
    from utils.states import PENDING_APPROVALS

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
    

def route_to_data_query(state: VerifyUserInfoState) -> Command[Literal["query_refund_eligibility", "query_pii", "query_transactions", "query_both"]]:
    """Route to appropriate data query based on request type."""
    data_type = state.get("data_request_type", "both")
    
    if data_type == "pii":
        return Command(goto="query_pii")
    elif data_type == "transactions":
        return Command(goto="query_transactions")
    elif data_type == "refund":
        return Command(goto="query_refund_eligibility")
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

def query_refund_eligibility(state: VerifyUserInfoState) -> Command[Literal[END, "human_review_refund"]]:
    credentials = state["credentials"]
    llm = ChatBedrockConverse(
            model="us.amazon.nova-lite-v1:0",
            temperature=0.7,
            aws_access_key_id=os.getenv("BEDROCK_AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("BEDROCK_AWS_SECRET_ACCESS_KEY"),
            region_name=os.getenv("BEDROCK_AWS_REGION", "us-east-1")
        )
    
    docs_refund = retrieve_docs("What is the refund policy?")
    docs_return = retrieve_docs("What is the return policy?")
    docs = docs_refund + docs_return

    context = "\n\n".join([d.page_content for d in docs])
    query = state["messages"][-1].content
    transactions_result = query_snowflake(credentials, "transactions")
    transactions = transactions_result.get("data", [])
    
    # Filter for Purchases ONLY to prevent LLM confusion
    # (Sometimes 'purchase' or 'purchased' depending on DB value, filtering case-insensitively)
    purchased_transactions = [
        t for t in transactions 
        if t.get('TRANSACTION_TYPE', '').lower() in ['purchased', 'purchase']
    ]
    
    # We pass ONLY purchased items to the LLM for eligibility check
    transactions_context = purchased_transactions
    transactions_context = purchased_transactions
    now = str(datetime.datetime.now())

    prompt_text = """You are a helpful customer support assistant for Penguin Inc. 
    Use the following pieces of retrieved context to determine if the customer's requested transaction is eligible for a refund or return or none.
    
    Using the customer's question, determine which product they want to refund or return from the given transactions.
    
    1. If the user DID NOT specify a product name, return ["None", "ask_product", "N/A", "User did not specify a product"]
    
    2. If the user DID specify a product, match it to the TRANSACTIONS list.
       - Ignore date/time checks (we are waiving the 30-day policy for this specific system test).
       - Mark the product as "eligible" if found.
    
    You should respond as a JSON list of strings [PRODUCT_NAME, STATUS, TRANSACTION_ID, REASON].
    
      Example input: "I want a refund." (No product named)
      Example output: ["None", "ask_product", "N/A", "User needs to specify a product."]

      Example input: "I want a refund on the microwave."
      Example output: ["PowerMicroWave", "eligible", "f051b3cb...", "Eligible for return."]
    
    Context:
    {context}

    Transactions:
    {transactions}
    
    Question:
    {query}
    
    Answer:"""

    prompt = ChatPromptTemplate.from_template(prompt_text)
    chain = prompt | llm
    response = chain.invoke({"context": context, "query": query, "transactions": transactions_context, "now": now})

    import json
    content = response.content.replace('```json', '').replace('```', '').strip()
    try:
        eligibility = json.loads(content)
        # Parse result
        if isinstance(eligibility, list) and len(eligibility) >= 4:
            product_name = eligibility[0]
            status = eligibility[1]
            transaction_id = eligibility[2]
            reason = eligibility[3]
        else:
             # Fallback
            product_name = "Unknown Item"
            status = "not_eligible"
            transaction_id = "N/A"
            reason = "Could not parse eligibility."

    except json.JSONDecodeError:
        # Fallback to newline splitting if JSON fails
        eligibility = [q.strip() for q in content.split('\n') if q.strip()]
        if len(eligibility) >= 4:
            product_name = eligibility[0]
            status = eligibility[1]
            transaction_id = eligibility[2]
            reason = eligibility[3]
        else:
            product_name = "Unknown Item"
            status = "not_eligible"
            transaction_id = "N/A"
            reason = "Could not parse JSON response."

    # Logic Checking
    
    if status == "ask_product":
         # Helper to list products
         product_list = ", ".join([t.get('PRODUCT_NAME', 'Unknown') for t in transactions_context[:3]])
         response_msg = AIMessage(content=f"**Which item?**\n\nI see a few recent purchases: **{product_list}**.\n\nWhich one would you like to refund?")
         return Command(
            update={"messages": state["messages"] + [response_msg]},
            goto=END
         )

    if status == "eligible":
        # Check Red Flags
        flags = check_red_flags(transactions)
        
        if flags:
            # HITL Required
            refund_details = {
                "product_name": product_name,
                "eligibility_status": status,
                "transaction_id": transaction_id,
                "reason": reason
            }
            return Command(
                update={"red_flags": flags, "refund_details": refund_details},
                goto="human_review_refund"
            )
        else:
            # Auto-approve (Fast Track)
            response_msg = AIMessage(content=f"**Refund Processed!**\n\nGreat news! We've verified that your request for **{product_name}** is within our policy.\nreason: {reason}\n\nSince your account is in good standing, we've fast-tracked your refund. You should see the funds shortly.")
            return Command(
                update={"messages": state["messages"] + [response_msg]},
                goto=END
            )

    else:
        # Not eligible
        prompt_text = """You are a helpful customer support assistant.
        The user requested a refund but it was deemed 'not_eligible'.
        Explain nicely why.
        
        Reason provided: {reason}
        """
        response_msg = AIMessage(content=f"**Refund Request Denied**\n\n{reason}")
        return Command(
            update={"messages": state["messages"] + [response_msg]},
            goto=END
        )




def format_response(state: VerifyUserInfoState) -> Command[Literal[END]]:
    """Format the Snowflake query results into a user-friendly response."""
    credentials = state["credentials"]
    results = state.get("snowflake_results", {})
    
    # Start building the response
    response_content = f"**Access Granted!**\n\nWelcome back, {credentials['username']}!\n\n"
    
    has_data = False
    errors = []
    
    # Format PII information if available
    if "pii" in results:
        if results["pii"]["success"]:
            pii_data = results["pii"]["data"]
            if pii_data and len(pii_data) > 0:
                has_data = True
                customer = pii_data[0]  # Should only be one customer
                
                response_content += "**Personal Information:**\n"
                
                #  DYNAMICALLY DISPLAY ALL COLUMNS
                for column_name, column_value in customer.items():
                    # Format column name to be more readable (e.g., USER_ID -> User ID)
                    readable_name = column_name.replace('_', ' ').title()
                    response_content += f"- {readable_name}: {column_value}\n"
                
                response_content += "\n"
            else:
                response_content += "**Personal Information:** No customer record found.\n\n"
        else:
            errors.append("personal information")
    
    # Format transaction information if available
    if "transactions" in results:
        if results["transactions"]["success"]:
            trans_data = results["transactions"]["data"]
            if trans_data and len(trans_data) > 0:
                has_data = True
                response_content += f"**Recent Transactions** ({len(trans_data)} shown):\n\n"
                
                # Calculate total if AMOUNT column exists
                total_amount = 0
                for transaction in trans_data:
                    try:
                        amount = float(transaction.get('AMOUNT', 0))
                        total_amount += amount
                    except (ValueError, TypeError):
                        pass
                
                # Display transactions with ALL columns (Limit to latest 5)
                for i, transaction in enumerate(trans_data[:5], 1):
                    response_content += f"**Transaction #{i}**\n"
                    
                    #  DYNAMICALLY DISPLAY ALL COLUMNS
                    for column_name, column_value in transaction.items():
                        # Format column name to be more readable
                        readable_name = column_name.replace('_', ' ').title()
                        
                        # Special formatting for certain columns
                        if column_name == 'AMOUNT' and column_value is not None:
                            response_content += f"   {readable_name}: ${column_value}\n"
                        elif column_name == 'TRANSACTION_DATE' and hasattr(column_value, 'strftime'):
                            formatted_date = column_value.strftime('%Y-%m-%d %H:%M:%S')
                            response_content += f"   {readable_name}: {formatted_date}\n"
                        elif column_name == 'STATUS':
                            response_content += f"   {readable_name}: {column_value}\n"
                        else:
                            response_content += f"   {readable_name}: {column_value}\n"
                    
                    response_content += "\n"
                
                if total_amount > 0:
                    response_content += f"**Total Recent Activity:** ${total_amount:.2f}\n\n"
            else:
                response_content += "**Recent Transactions:** No transaction history found for the provided email and zip code.\n\n"
        else:
            errors.append("transaction history")
    
    # Add error messages if any queries failed
    if errors:
        error_list = " and ".join(errors)
        response_content += f"There was an issue retrieving your {error_list}. "
        response_content += "Please contact support for assistance.\n\n"
    
    # Add closing message
    if has_data:
        response_content += "**Security Level:** Verified\n\n"
        response_content += "How can I assist you with your account today?"
    else:
        response_content += "We couldn't find any records matching your credentials. "
        response_content += "Please verify your information or contact support."
    
    response = AIMessage(content=response_content)
    
    return Command(
        update={"messages": state["messages"] + [response]},
        goto=END
    )

def handle_rejection(state: VerifyUserInfoState) -> Command[Literal[END]]:
    credentials = state["credentials"]
    response = AIMessage(content=f"❌ **Access Denied**\n\nSorry, {credentials['username']}. Your credentials could not be verified.")
    return Command(update={"messages": state["messages"] + [response], "approval_status": "none"},goto=END)

def handle_normal(state: VerifyUserInfoState) -> Command[Literal[END]]:
    messages = state["messages"]
    system_message = SystemMessage(content="You are a helpful assistant. Respond naturally.")
    response = llm.invoke([system_message] + messages)
    return Command(update={"messages": state["messages"] + [response], "dialogue_state": "idle", "approval_status":"none"}, goto=END)

def check_red_flags(transactions: list) -> list:
    """
    Check for red flags in the user's transaction history.
    
    Red Flags:
    1. Refund Velocity: > 5 refunds total (Adjusted for demo as dates are unavailable).
    2. Chargeback Risk: Any history of chargebacks.
    """
    flags = []
    
    # 1. Refund Velocity
    # Note: FCT_TRANSACTIONS lacks a transaction date, so we cannot check "last 30 days".
    # (CREATED_DATE in the query comes from DIM_PRODUCTS, which refers to product launch, not purchase).
    # We fallback to total all-time refunds with a higher threshold for the demo.
    
    refunds = [t for t in transactions if t.get('TRANSACTION_TYPE', '').lower() == 'refund']
    
    if len(refunds) >= 5:
        flags.append(f"Refund Velocity: {len(refunds)} lifetime refunds found (High Risk)")

    # 2. Chargeback Risk
    chargebacks = [t for t in transactions if t.get('TRANSACTION_TYPE', '').lower() == 'chargeback']
    
    if chargebacks:
        flags.append(f"Chargeback Risk: {len(chargebacks)} previous chargebacks detected.")

    return flags

def human_review_refund(state: VerifyUserInfoState) -> Command[Literal["process_refund_approval"]]:
    """
    Node for human review of refund requests that have red flags.
    """
    credentials = state["credentials"]
    thread_id = state.get("thread_id", "unknown")
    red_flags = state.get("red_flags", [])
    refund_details = state.get("refund_details", {})
    from utils.states import PENDING_APPROVALS

    PENDING_APPROVALS[thread_id] = {
        "credentials": credentials,
        "status": "pending_refund_review",
        "red_flags": red_flags,
        "refund_details": refund_details
    }

    # --- INTERRUPT ---
    # We pass the details so the UI can show them
    approval_decision = interrupt({
        "message": "Waiting for refund approval (Red Flags Detected)",
        "credentials": credentials,
        "thread_id": thread_id,
        "red_flags": red_flags,
        "refund_details": refund_details
    })
    
    return Command(update={"approval_status": approval_decision}, goto="process_refund_approval")

def process_refund_approval(state: VerifyUserInfoState) -> Command[Literal[END]]:
    approval_status = state.get("approval_status", "pending")
    refund_details = state.get("refund_details", {})
    product_name = refund_details.get("product_name", "item")
    
    if approval_status == "approved":
        # Process the refund (Mocking the action)
        response = AIMessage(content=f"✅ **Refund Approved!**\n\nThe refund for **{product_name}** has been processed successfully despite the warnings. The funds should appear in your account within 3-5 business days.")
    else:
        response = AIMessage(content=f"❌ **Refund Rejected**\n\nAfter manual review, we cannot process the refund for **{product_name}** at this time due to policy restrictions.")
        
    return Command(update={"messages": state["messages"] + [response], "approval_status": "none"}, goto=END)
