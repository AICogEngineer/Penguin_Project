from pydantic import BaseModel, Field
from typing import TypedDict, Annotated, Literal, List, Dict
from langgraph.graph import MessagesState
from langchain_core.documents import Document
import operator

# Router
class Route(BaseModel):
    step: Literal["userInfoInquiry", "policyQuestion", "misc"] = Field(
        None, description="The next step in the routing process."
    )

class PolicyQuestionsState(TypedDict):
    query: str                      # customer query
    documents: Annotated[List[Document], operator.add]       # where chunks are stored
    questions: List[str]         # list of sub-questions (if any)
    answers: Annotated[List[str], operator.add] # collected answers

class VerifyUserInfoState(MessagesState):
    """Extended state for our agent workflow."""
    credentials: Dict = {}
    approval_status: str = "none"
    thread_id: str = ""
    dialogue_state: Literal[
        "idle", 
        "awaiting_username", 
        "awaiting_email", 
        "awaiting_zipcode"
    ] = "idle"
    data_request_type: Literal["pii", "transactions", "both", None] = None
    snowflake_results: Dict = {}

PENDING_APPROVALS = {}

class ParentRouterState(MessagesState):
    thread_id: str
    decision: str
    credentials: Dict = {}
    lockedState: str = "no"
    

