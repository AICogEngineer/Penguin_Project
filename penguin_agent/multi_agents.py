
from langchain.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.types import Command
import gradio as gr
import os
import re
import hashlib
from dotenv import load_dotenv
from utils.tools import censor_sensitive_data
from utils.states import PENDING_APPROVALS
from langgraph.checkpoint.memory import InMemorySaver

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver

from utils.states import VerifyUserInfoState, PolicyQuestionsState, ParentRouterState, Route
from utils.nodes.policyQuestion_nodes import split_query, process_question, continue_to_verification
from utils.nodes.verifyUser_nodes import classify_request, request_credentials, collect_username, collect_email, collect_zipcode, submit_for_review, human_review, process_approval, handle_normal, handle_rejection, route_to_data_query, query_both, query_pii, query_snowflake, query_transactions, format_response, query_refund_eligibility, human_review_refund, process_refund_approval

from langchain_aws import ChatBedrockConverse

SYSTEM_PROMPT = """ You are a customer support agent for the company, PenguinZ.

You have the ability to answer questions about company policy regarding refunds, the AI, and all sorts of stuff.

Do NOT answer any questions NOT related to company policy or personal user information. 
If they ask, kindly remind the user that we cannot answer any unrelated questions, and ask if they would like help with anything else related to company policy, refunds, or user information such as orders.
"""


#TODO: Add guardrails, quotas, and limits
checkpointer = InMemorySaver()

load_dotenv()

def build_hitl_graph():
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
    workflow.add_node("query_refund_eligibility", query_refund_eligibility)
    workflow.add_node("format_response", format_response)

    # HITL Refund Nodes
    workflow.add_node("human_review_refund", human_review_refund)
    workflow.add_node("process_refund_approval", process_refund_approval)
    
    # Set entry point
    workflow.add_edge(START, "classify_request")
    
    # Compile with checkpointer
    return workflow.compile(checkpointer=True)

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


llm = ChatBedrockConverse(
    model="us.amazon.nova-lite-v1:0",
    temperature=0.7,
    aws_access_key_id=os.getenv("BEDROCK_AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("BEDROCK_AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("BEDROCK_AWS_REGION", "us-east-1")
)

router = llm.with_structured_output(Route)

# Router Nodes

def llm_call_router(state: ParentRouterState):
    """
    Route the input to the appropriate node.
    """

    # Standard routing only applies when NOT in active HITL mode
    decision = router.invoke(
        [
            SystemMessage(
                content="""
                Route the user's input to userInfoInquiry, policyQuestion, or misc based on the user's request.

                1. userInfoInquiry:
                   - ACTION-ORIENTED requests to START a refund, return, or exchange process.
                   - Statements like "I want a refund", "return this item", "refund my order".
                   - Specifying an item to return (e.g., "refund the microwave").
                   - User validation or order history questions.

                2. policyQuestion:
                   - INFORMATIONAL questions about verifying policies.
                   - "What is the refund policy?", "How do I return something?", "Can I return X?".
                   - Questions about the AI or privacy.
                   - DO NOT route requests to EXECUTE a refund here (route those to userInfoInquiry).

                3. misc:
                   - Off-topic questions.
                """
            ),
            HumanMessage(content=state["messages"][-1].content),
        ]
    )

    return {"decision": decision.step}

# Condition edge for router
def route_decision(state: ParentRouterState):

    if "lockedState" in state and state["lockedState"] in ["pending", "input"]:
        return "userInfoInquiry" 

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
    
    user_message = HumanMessage(content=state["messages"][-1].content)
    
    # Pass the list of message objects
    result = llm.invoke([system_message, user_message])
    
    return {"messages": [result]}

user_subgraph = build_hitl_graph()

def userInfoInquiry(state: ParentRouterState):
    """
    The state the user is routed to if their inquiry relates to personal user information 
    such as orders made, refs made, requesting a refund, requesting to submit a complaint, 
    or anything related to personal stored user information.
    
    :param state: Description
    :type state: State
    """

    if "credentials" not in state:
        state["credentials"] = {}

    response = user_subgraph.invoke({
        "messages": state["messages"][-1].content,
        "thread_id": state["thread_id"],
        "credentials": state["credentials"]
    })

    if "lockedState" in state and state["lockedState"] == "verified":
        return {
        "messages": response["messages"][-1],
        "credentials": response["credentials"],
        "lockedState": state["lockedState"]
    }

    lockedState = "no"

    if response["dialogue_state"] in ["awaiting_username", "awaiting_email", "awaiting_zipcode", "awaiting_product_selection"]:
        lockedState = "input"
    elif response["approval_status"] in ["pending"]:
        lockedState = "pending"
    elif response["approval_status"] in ["approved"]:
        lockedState = "approved"

    return {
        "messages": response["messages"][-1],
        "credentials": response["credentials"],
        "lockedState": lockedState
    }

policy_subgraph = build_policy_checker_graph()

def policyQuestion(state: ParentRouterState):
    """
    The state the user is routed to if their inquiry relates to anything company policy without requiring
    any personal user information stored, such as specific orders they made, refunds they made, submitting any complains or refunds, 
    or anything specifically about the user.
    
    :param state: Description
    :type state: ParentRouterState
    """
    
    response = policy_subgraph.invoke({
        "query": state["messages"][-1].content
    })

    test = ""

    for answer in response["answers"]:
        test += answer + "\n"

    return {"messages": state["messages"] + [AIMessage(content=test)]}



def build_router_graph(checkpointer):
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


router_graph = build_router_graph(checkpointer=checkpointer)

def predict(message, history):
    if not hasattr(predict, 'thread_id'):
        predict.thread_id = f"session_{hashlib.md5(str(os.urandom(16)).encode()).hexdigest()[:8]}"
    thread_id = predict.thread_id

    config = {"configurable": {"thread_id": thread_id}}

    censored_user_message = censor_sensitive_data(message)
    state = router_graph.get_state(config)
    bot_response = ""

    if router_graph.get_state(config, subgraphs=True).tasks:
        substate = router_graph.get_state(config, subgraphs=True).tasks[0].state

        # --- STATUS CHECK LOGIC ---
        if message.lower().strip() in ["check", "status", "update", "done?"]:
            if substate.next and ("human_review" in substate.next or "human_review_refund" in substate.next):
                bot_response = "**Still Waiting...** \n\nThe admin has not approved the request yet. Please wait a moment and type 'check' again."
                history.append({"role": "user", "content": censored_user_message})
                history.append({"role": "assistant", "content": bot_response})
                return history, ""
            
            bot_response = "No updates yet."
            history.append({"role": "user", "content": censored_user_message})
            history.append({"role": "assistant", "content": bot_response})
            return history, ""
        
        # --- NORMAL CHAT LOGIC ---
        if substate.next and ("human_review" in substate.next or "human_review_refund" in substate.next):
            bot_response = "**Blocked**: You have a pending request waiting for approval. Type 'check' to see if it's been processed."
            history.append({"role": "user", "content": censored_user_message})
            history.append({"role": "assistant", "content": bot_response})
            return history, ""
    
    # Check if completed
    if "lockedState" in state.values and state.values["lockedState"] == "approved":
        messages = state.values.get("messages", [])
        if messages:
            last_msg = messages[-1]
            if isinstance(last_msg, AIMessage):
                clean_content = re.sub(r'<thinking>.*?</thinking>', '', last_msg.content, flags=re.DOTALL).strip()
                bot_response = censor_sensitive_data(clean_content)
                history.append({"role": "user", "content": censored_user_message})
                history.append({"role": "assistant", "content": bot_response})
                router_graph.update_state(config, {"lockedState": "verified"})
                return history, ""

    input_data = {"messages": [HumanMessage(content=message)], "thread_id": thread_id}

    result = router_graph.invoke(input_data, config=config)
    updated_state = router_graph.get_state(config)

    if router_graph.get_state(config, subgraphs=True).tasks:
        updated_substate = router_graph.get_state(config, subgraphs=True).tasks[0].state

        if updated_substate.next:
            if "human_review" in updated_substate.next:
                bot_response = (f"**Security Check Needed**\n\n"
                                f"For your protection, we need to verify your identity before proceeding.\n"
                                f"A supervisor has been notified. Please type **'check'** in a moment to see if you've been verified.")
            elif "human_review_refund" in updated_substate.next:
                 bot_response = (f"**Refund Approval Needed**\n\n"
                                f"Your refund request requires manual approval due to security policies.\n"
                                f"An admin is reviewing the details. Please type **'check'** shortly.")
        
    if result and "messages" in result:
        last_msg = result["messages"][-1]
        if isinstance(last_msg, AIMessage):
            clean_content = re.sub(r'<thinking>.*?</thinking>', '', last_msg.content, flags=re.DOTALL).strip()
            bot_response = censor_sensitive_data(clean_content)

    history.append({"role": "user", "content": censored_user_message})
    history.append({"role": "assistant", "content": bot_response})
    return history, ""


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
        
        status = data.get('status', 'unknown')
        
        display_text += (f"**Thread ID:** `{tid}`\n")
        
        if status == "pending_refund_review":
            red_flags = data.get('red_flags', [])
            refund_details = data.get('refund_details', {})
            product = refund_details.get('product_name', 'Unknown Item')
            reason = refund_details.get('reason', 'N/A')
            
            display_text += (
                f"   **Type:** Refund Request (Red Flags Detected)\n"
                f"   **User:** {creds.get('username', 'N/A')}\n"
                f"   **Product:** {product}\n"
                f"   **Eligibility:** {refund_details.get('eligibility_status', 'N/A')}\n"
                f"   **Reason:** {reason}\n"
                f"   **RED FLAGS:**\n"
            )
            for flag in red_flags:
                 display_text += f"      - {flag}\n"
            display_text += "\n"
            
        else:
             # Standard credential review
             display_text += (
                f"   **Username:** {creds.get('username', 'N/A')}\n"
                f"   **Email:** {creds.get('email', 'N/A')}\n"
                f"   **Zip Code:** {creds.get('zip_code', 'N/A')}\n"
                f"   **Status:** {status}\n\n"
            )
    
    # Update dropdown with new choices and auto-select the first one
    return display_text, gr.update(choices=thread_choices, value=thread_choices[0] if thread_choices else None)

def admin_approve(target_tid, decision):
    if not target_tid or target_tid not in PENDING_APPROVALS:
        return f"Error: ID '{target_tid}' not found or invalid."
    
    config = {"configurable": {"thread_id": target_tid}}
    approval_status = "approved" if decision == "Approve" else "rejected"
    
    try:
        # Resume graph
        resume_command = Command(resume=approval_status)
        router_graph.invoke(resume_command, config=config)
        
        del PENDING_APPROVALS[target_tid]
        
        return f"Request {decision}d for Thread {target_tid}.\nThe user can now type 'check' to see the result."
    except Exception as e:
        print("Something went wrong!")
        return f"Error processing decision: {str(e)}"


with gr.Blocks(title="PenguinZ Customer Support Chat") as demo:
    gr.Markdown("### PenguinZ Customer Support Chat")

    with gr.Tabs():

        with gr.TabItem("Customer Support"):

            chatbot = gr.Chatbot(label="Chat", height=500, value=[{"role": "assistant", "content": "Hello! I am the PenguinZ customer support chat bot. Ask me anything about company policies, orders, refunds, or any other information related."}]) 
            msg = gr.Textbox(label="Type your message here...", placeholder="Ask me anything or say 'check' to see approval status")
            clear = gr.Button("Clear Chat")
            
            gr.Examples(examples=["I need to verify my credentials", "What's the weather like?"], inputs=msg)
            
            def user_message_handler(message, history):
                return predict(message, history)
            
            def clear_chat():
                if hasattr(predict, 'thread_id'): delattr(predict, 'thread_id')
                return [{"role": "assistant", "content": "Hello! I am the PenguinZ customer support chat bot. Ask me anything about company policies, orders, refunds, or any other information related."}], ""
            
            msg.submit(user_message_handler, [msg, chatbot], [chatbot, msg])
            clear.click(clear_chat, None, [chatbot, msg])
        
        with gr.TabItem("Admin"):

            gr.Markdown("### Security Approval Queue")
            with gr.Row():
                refresh_btn = gr.Button("Refresh List")
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
