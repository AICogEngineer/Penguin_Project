from utils.states import VerifyUserInfoState
from langgraph.types import Command, interrupt
from typing import Literal
from langchain.messages import AIMessage, SystemMessage
from langchain_aws import ChatBedrockConverse
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

def classify_request(state: VerifyUserInfoState) -> Command[Literal["request_credentials", "collect_username", "collect_email", "collect_zipcode", "handle_normal", "route_to_data_query"]]:
    current_stage = state.get("dialogue_state", "idle")
    
    if current_stage == "awaiting_username": return Command(goto="collect_username")
    elif current_stage == "awaiting_email": return Command(goto="collect_email")
    elif current_stage == "awaiting_zipcode": return Command(goto="collect_zipcode")
        
    last_message = state["messages"][-1].content.lower()
    pii_keywords = ["pii", "personal", "credentials", "financial", "transaction", "verify", "account", "balance", "information"]
    
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

def request_credentials(state: VerifyUserInfoState) -> Command[Literal[END]]:
    response1 = AIMessage(content="🔐 To proceed with your PII/financial transaction request, I need to verify your identity.")
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
        update={"messages": state["messages"] + [response], "approval_status": "none"},
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
    return Command(update={"messages": state["messages"] + [response]}, goto=END)
