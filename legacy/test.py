
from typing import TypedDict, Annotated, List, Dict
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage
from langchain_core.documents import Document
from langchain_aws import ChatBedrock
from langchain_core.prompts import ChatPromptTemplate
from langchain_aws import ChatBedrockConverse
from dotenv import load_dotenv
from pydantic import BaseModel, Field

import operator
import sys
import os

from typing import Annotated, List
import operator

load_dotenv()

# USER = os.getenv('SNOWFLAKE_USER')
# PASSWORD = os.getenv('SNOWFLAKE_PASSWORD')
# ACCOUNT = os.getenv('SNOWFLAKE_ACCOUNT')
# WAREHOUSE = os.getenv('SNOWFLAKE_WAREHOUSE')
# DATABASE = os.getenv('SNOWFLAKE_DATABASE')
# SCHEMA = os.getenv('SNOWFLAKE_SCHEMA')


import sqlite3
from typing import TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, interrupt


class FormState(TypedDict):
    age: int | None


def get_age_node(state: FormState):
    prompt = "What is your age?"

    while True:
        answer = interrupt(prompt)  # payload surfaces in result["__interrupt__"]

        if isinstance(answer, int) and answer > 0:
            return {"age": answer}

        prompt = f"'{answer}' is not a valid age. Please enter a positive number."


builder = StateGraph(FormState)
builder.add_node("collect_age", get_age_node)
builder.add_edge(START, "collect_age")
builder.add_edge("collect_age", END)

#checkpointer = MemorySaver()
graph = builder.compile()

# config = {"configurable": {"thread_id": "form-1"}}
# first = graph.invoke({"age": None}, config=config)
# print(first["__interrupt__"])  # -> [Interrupt(value='What is your age?', ...)]

# # Provide invalid data; the node re-prompts
# retry = graph.invoke(Command(resume="thirty"), config=config)
# print(retry["__interrupt__"])  # -> [Interrupt(value="'thirty' is not a valid age...", ...)]

# # Provide valid data; loop exits and state updates
# final = graph.invoke(Command(resume=30), config=config)
# print(final["age"])  # -> 30




# # conn = snowflake.connector.connect(
# #     user=USER,
# #     password=PASSWORD,
# #     account=ACCOUNT,
# #     warehouse=WAREHOUSE,
# #     database=DATABASE,
# #     schema=SCHEMA
# #     )

# # cur = conn.cursor()
# # try:
# #     cur.execute('select * from FCT_TRANSACTIONS')
# #     ret = cur.fetchmany(3)
# #     print(ret)
# # finally:
# #     cur.close()

# llm = ChatBedrockConverse(
#     model="us.amazon.nova-lite-v1:0",
#     temperature=0.7,
#     aws_access_key_id=os.getenv("BEDROCK_AWS_ACCESS_KEY_ID"),
#     aws_secret_access_key=os.getenv("BEDROCK_AWS_SECRET_ACCESS_KEY"),
#     region_name=os.getenv("BEDROCK_AWS_REGION", "us-east-1")
# )

# from typing_extensions import TypedDict
# from langgraph.graph.state import StateGraph, START


# # Define subgraph
# class SubgraphState(TypedDict):
#     # note that none of these keys are shared with the parent graph state
#     bar: str
#     baz: str

# def subgraph_node_1(state: SubgraphState):
#     return {"baz": "baz"}

# def subgraph_node_2(state: SubgraphState):
#     return {"bar": state["bar"] + state["baz"]}

# subgraph_builder = StateGraph(SubgraphState)
# subgraph_builder.add_node(subgraph_node_1)
# subgraph_builder.add_node(subgraph_node_2)
# subgraph_builder.add_edge(START, "subgraph_node_1")
# subgraph_builder.add_edge("subgraph_node_1", "subgraph_node_2")
# subgraph = subgraph_builder.compile()

# # Define parent graph
# class ParentState(TypedDict):
#     foo: str

# def node_1(state: ParentState):
#     return {"foo": "hi! " + state["foo"]}

# def node_2(state: ParentState):
#     # Transform the state to the subgraph state
#     response = subgraph.invoke({"bar": state["foo"]})
#     # Transform response back to the parent state
#     return {"foo": response["bar"]}


# builder = StateGraph(ParentState)
# builder.add_node("node_1", node_1)
# builder.add_node("node_2", node_2)
# builder.add_edge(START, "node_1")
# builder.add_edge("node_1", "node_2")
# graph = builder.compile()

# for chunk in graph.stream({"foo": "foo"}, subgraphs=True):
#     print(chunk)


# from typing_extensions import Literal
# from langchain.messages import HumanMessage, SystemMessage


# # Schema for structured output to use as routing logic
# class Route(BaseModel):
#     step: Literal["poem", "story", "joke"] = Field(
#         None, description="The next step in the routing process"
#     )


# # Augment the LLM with schema for structured output
# router = llm.with_structured_output(Route)


# # State
# class State(TypedDict):
#     input: str
#     decision: str
#     output: str


# # Nodes
# def llm_call_1(state: State):
#     """Write a story"""

#     result = llm.invoke(state["input"])
#     return {"output": result.content}


# def llm_call_2(state: State):
#     """Write a joke"""

#     result = llm.invoke(state["input"])
#     return {"output": result.content}


# def llm_call_3(state: State):
#     """Write a poem"""

#     result = llm.invoke(state["input"])
#     return {"output": result.content}


# def llm_call_router(state: State):
#     """Route the input to the appropriate node"""

#     # Run the augmented LLM with structured output to serve as routing logic
#     decision = router.invoke(
#         [
#             SystemMessage(
#                 content="Route the input to story, joke, or poem based on the user's request."
#             ),
#             HumanMessage(content=state["input"]),
#         ]
#     )

#     return {"decision": decision.step}


# # Conditional edge function to route to the appropriate node
# def route_decision(state: State):
#     # Return the node name you want to visit next
#     if state["decision"] == "story":
#         return "llm_call_1"
#     elif state["decision"] == "joke":
#         return "llm_call_2"
#     elif state["decision"] == "poem":
#         return "llm_call_3"


# # Build workflow
# router_builder = StateGraph(State)

# # Add nodes
# router_builder.add_node("llm_call_1", llm_call_1)
# router_builder.add_node("llm_call_2", llm_call_2)
# router_builder.add_node("llm_call_3", llm_call_3)
# router_builder.add_node("llm_call_router", llm_call_router)

# # Add edges to connect nodes
# router_builder.add_edge(START, "llm_call_router")
# router_builder.add_conditional_edges(
#     "llm_call_router",
#     route_decision,
#     {  # Name returned by route_decision : Name of next node to visit
#         "llm_call_1": "llm_call_1",
#         "llm_call_2": "llm_call_2",
#         "llm_call_3": "llm_call_3",
#     },
# )
# router_builder.add_edge("llm_call_1", END)
# router_builder.add_edge("llm_call_2", END)
# router_builder.add_edge("llm_call_3", END)

# # Compile workflow
# router_workflow = router_builder.compile()

# # Invoke
# #state = router_workflow.invoke({"input": "Write me a joke about cats"})
# #print(state["output"])


# User Side

# Pass `session_id` as an argument (handled by Gradio State)
# def user_predict(message, history, session_id):
#     # 1. Generate Thread ID if new session
#     if not session_id:
#         session_id = f"session_{hashlib.md5(str(os.urandom(16)).encode()).hexdigest()[:8]}"
    
#     thread_id = session_id
#     config = {"configurable": {"thread_id": thread_id}}
#     censored_user_message = censor_sensitive_data(message)
#     bot_response = ""
    
#     # Fetch current state
#     state = router_graph.get_state(config)
    
#     # --- LOGIC FOR "CHECK" / STATUS UPDATES ---
#     if message.lower().strip() in ["check", "status", "update", "done?"]:
#         # Scenario A: Graph is running/paused (Waiting for Admin)
#         if state.next:
#             snapshot = router_graph.get_state(config, subgraphs=True)
#             if snapshot.tasks:
#                 sub_state = snapshot.tasks[0].state
#                 if sub_state and sub_state.next and "human_review" in sub_state.next:
#                     bot_response = "⏳ **Still Waiting...** \n\nThe admin has not approved the request yet. Please wait a moment and type 'check' again."
        
#         # Scenario B: Graph successfully FINISHED (Admin approved, graph ran to END)
#         # If no next steps, look at the history
#         if not bot_response and not state.next:
#             # Check the very last message in the parent history
#             messages = state.values.get("messages", [])
#             if messages:
#                 last_msg = messages[-1]
#                 if isinstance(last_msg, AIMessage):
#                     bot_response = censor_sensitive_data(last_msg.content)
            
#             if not bot_response:
#                  bot_response = "It looks like the request is finished, but I have no new updates."

#         # Scenario C: Fallback
#         if not bot_response:
#              bot_response = "No active requests found."
             
#         history.append({"role": "user", "content": censored_user_message})
#         history.append({"role": "assistant", "content": bot_response})
#         return history, "", session_id

#     # --- NORMAL CHAT FLOW ---
    
#     # Prevent chatting if stuck in approval
#     if state.values and state.values.get("decision") == "userInfoInquiry":
#         snapshot = router_graph.get_state(config, subgraphs=True)
#         if snapshot.tasks:
#             sub_state = snapshot.tasks[0].state
#             if sub_state and sub_state.next and "human_review" in sub_state.next:
#                  bot_response = "⚠️ **Blocked**: You have a pending request waiting for approval. Type 'check' to see if it's been processed."
#                  history.append({"role": "user", "content": censored_user_message})
#                  history.append({"role": "assistant", "content": bot_response})
#                  return history, "", session_id

#     input_state = {
#         "input": message,
#         "thread_id": thread_id
#     }
    
#     # Run the graph
#     result = router_graph.invoke(input_state, config=config)

#     # Output Handling
#     if not bot_response:
#         # Check if we just hit the interrupt
#         snapshot = router_graph.get_state(config, subgraphs=True)
#         if snapshot.tasks:
#             sub_state = snapshot.tasks[0].state
#             if sub_state and sub_state.next and "human_review" in sub_state.next:
#                  bot_response = (f"✋ **Approval Needed**\n\n"
#                             f"Your credentials have been submitted for review.\n"
#                             f"Our admin will review your request. You can check your progress by typing **'check'**")
    
#     if not bot_response and "output" in result:
#         output_data = result["output"]
#         if isinstance(output_data, list):
#             bot_response = "\n\n".join(output_data)
#         elif hasattr(output_data, 'content'):
#             clean_content = re.sub(r'<thinking>.*?</thinking>', '', output_data.content, flags=re.DOTALL).strip()
#             bot_response = censor_sensitive_data(clean_content)
#         else:
#             bot_response = str(output_data)

#     history.append({"role": "user", "content": censored_user_message})
#     history.append({"role": "assistant", "content": bot_response})
    
#     # Return session_id back to Gradio State
#     return history, "", session_id

# # Admin Side

# def refresh_admin_view():
#     """Returns updated text for the dashboard AND updated choices for the dropdown."""
#     if not PENDING_APPROVALS:
#         # Return status text and an empty list of choices
#         return "No pending requests.", gr.update(choices=[], value=None)
    
#     # Build text display
#     display_text = ""
#     # Build dropdown choices list (Thread IDs)
#     thread_choices = []
    
#     for tid, data in PENDING_APPROVALS.items():
#         creds = data['credentials']
#         thread_choices.append(tid)
#         display_text += (
#             f"🔹 **Thread ID:** `{tid}`\n"
#             f"   **Username:** {creds.get('username', 'N/A')}\n"
#             f"   **Email:** {creds.get('email', 'N/A')}\n"
#             f"   **Zip Code:** {creds.get('zip_code', 'N/A')}\n"
#             f"   **Status:** {data['status']}\n\n"
#         )
    
#     # Update dropdown with new choices and auto-select the first one
#     return display_text, gr.update(choices=thread_choices, value=thread_choices[0] if thread_choices else None)

# def admin_approve(target_tid, decision):
#     if not target_tid or target_tid not in PENDING_APPROVALS:
#         return f"❌ Error: ID '{target_tid}' not found or invalid."
    
#     config = {"configurable": {"thread_id": target_tid}}
#     approval_status = "approved" if decision == "Approve" else "rejected"
    
#     try:
#         # Resume graph
#         resume_command = Command(resume=approval_status)
#         router_graph.invoke(resume_command, config=config)
        
#         del PENDING_APPROVALS[target_tid]
        
#         return f"✅ Request {decision}d for Thread {target_tid}.\nThe user can now type 'check' to see the result."
#     except Exception as e:
#         return f"❌ Error processing decision: {str(e)}"


# # UI

# with gr.Blocks(title="AI Agent System") as demo:
#     gr.Markdown("# 🤖 Corporate AI Assistant with LangGraph")
    
#     session_state = gr.State(value=None)

#     with gr.Tabs():
#         # TAB 1: USER
#         with gr.TabItem("💬 Chat"):
#             chatbot = gr.Chatbot(label="Chat", 
#                                  height=500,
#                                  value=[{"role": "assistant", "content": "Hello! I am the PenguinZ customer support chat bot. Ask me anything about company policies, orders, refunds, or any other information related."}]
#             )
#             msg = gr.Textbox(label="Type your message here...", placeholder="Ask me anything or say 'check' to see approval status")
#             clear = gr.Button("Clear Chat")
            
#             gr.Examples(examples=["I need to verify my credentials", "What's the refund policy?"], inputs=msg)
            
#             def user_message_handler(message, history, session_id):
#                 return user_predict(message, history, session_id)
            
#             def clear_chat():
#                 if hasattr(user_predict, 'thread_id'): delattr(user_predict, 'thread_id')
#                 return [{"role": "assistant", "content": "Hello! I am the PenguinZ customer support chat bot. Ask me anything about company policies, orders, refunds, or any other information related."}], "", None
            
#             msg.submit(user_message_handler, [msg, chatbot, session_state], [chatbot, msg, session_state])
#             clear.click(clear_chat, None, [chatbot, msg, session_state])
        
#         # TAB 2: ADMIN
#         with gr.TabItem("🔒 Admin Dashboard"):
#             gr.Markdown("### 🛡️ Security Approval Queue")
#             with gr.Row():
#                 refresh_btn = gr.Button("🔄 Refresh List")
#                 queue_display = gr.Markdown("No pending requests.")
#             gr.Markdown("---")
#             with gr.Row():
#                 # NEW: Dropdown instead of Textbox
#                 tid_dropdown = gr.Dropdown(label="Select Thread ID", choices=[], interactive=True)
#                 decision_radio = gr.Radio(["Approve", "Reject"], label="Action", value="Approve")
#                 process_btn = gr.Button("Submit Decision", variant="primary")
#             admin_output = gr.Markdown()

#             # Wiring Admin Events
#             # Clicking Refresh updates BOTH the text display AND the dropdown choices
#             refresh_btn.click(refresh_admin_view, outputs=[queue_display, tid_dropdown])
            
#             # Clicking Process uses the selected value from the dropdown
#             process_btn.click(admin_approve, inputs=[tid_dropdown, decision_radio], outputs=admin_output)

# demo.launch()