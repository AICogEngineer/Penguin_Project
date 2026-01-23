from utils.states import ParentRouterState, Route
from langchain.messages import SystemMessage, HumanMessage, AIMessage
from langchain_aws import ChatBedrockConverse
import os
from dotenv import load_dotenv

load_dotenv()



PENDING_APPROVALS = {}




