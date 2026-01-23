# Penguin Project - AI Agent Instructions

## Project Overview
E-commerce customer support AI bot with human-in-the-loop (HITL) approval workflows, built using **LangGraph** for stateful agent orchestration, AWS Bedrock (Nova Lite) for LLM, and Gradio for dual-interface (User/Admin) UI.

## Architecture: Multi-Agent Router System

### Core Graph Structure
The system uses a **parent-child subgraph pattern** with three main workflows:

1. **Router Graph** (`ParentRouterState`) - Parent orchestrator in [utils/graphs.py](utils/graphs.py):
   - Routes user queries to: `userInfoInquiry`, `policyQuestion`, or `misc_call`
   - Uses structured LLM routing with `router = llm.with_structured_output(Route)`
   - Maintains conversation state via `InMemorySaver` checkpointer with `thread_id`

2. **User Verification Subgraph** (`VerifyUserInfoState`) - HITL credential collection:
   - Multi-stage dialogue: username → email → zip_code → admin approval
   - Uses `dialogue_state` field to track: `"idle"`, `"awaiting_username"`, `"awaiting_email"`, `"awaiting_zipcode"`
   - **Critical**: Returns `Command(goto=END)` after each collection step to allow user response
   - Admin approval uses `interrupt()` to pause execution - stores state in `PENDING_APPROVALS` dict
   - Resume via `Command(resume=approval_status)` where `approval_status` = `"approved"` | `"rejected"`

3. **Policy Checker Subgraph** (`PolicyQuestionsState`) - RAG-based Q&A:
   - Splits multi-part questions using LLM, then uses `Send()` for fan-out parallelization
   - Retrieves policy docs from ChromaDB via [utils/retriever.py](utils/retriever.py)
   - Accumulates answers using `Annotated[List[str], operator.add]`

### Critical LangGraph Patterns

#### Command Pattern for Control Flow
```python
# Combine state updates + routing in single node return
return Command(
    update={"credentials": credentials, "dialogue_state": "awaiting_email"},
    goto=END  # or goto="next_node"
)
```

#### Human-in-the-Loop Interrupts
```python
# In node - pauses graph execution
approval_decision = interrupt({
    "message": "Waiting for admin approval",
    "thread_id": thread_id
})

# To resume (in multi_agents.py admin handler):
router_graph.invoke(Command(resume="approved"), config={"configurable": {"thread_id": tid}})
```

#### Subgraph State Isolation
- Parent passes initial data to subgraph via `.invoke({"messages": [...], "thread_id": ...})`
- Subgraph returns final state; parent extracts relevant fields (e.g., `response["messages"][-1]`)
- Use `checkpointer=True` in subgraph compilation for independent memory (see [utils/graphs.py](utils/graphs.py))

## Key Integration Points

### AWS Bedrock LLM Setup
- Model: `us.amazon.nova-lite-v1:0` (low-cost, fast)
- All LLM calls in [utils/nodes.py](utils/nodes.py) and [utils/tools.py](utils/tools.py) use:
  ```python
  llm = ChatBedrockConverse(
      model="us.amazon.nova-lite-v1:0",
      temperature=0.7,
      aws_access_key_id=os.getenv("BEDROCK_AWS_ACCESS_KEY_ID"),
      aws_secret_access_key=os.getenv("BEDROCK_AWS_SECRET_ACCESS_KEY"),
      region_name=os.getenv("BEDROCK_AWS_REGION", "us-east-1")
  )
  ```
- Fallback mocking in [utils/tools.py](utils/tools.py):`generate_response()` if Bedrock fails

### Snowflake Database (Commented Out)
- Connection code in [multi_agents.py](multi_agents.py) lines 25-40 (currently disabled)
- Intended for order/transaction lookup in HITL verification flow
- **TODO**: Integrate with `access_database()` node after approval

### Gradio Dual-Interface Pattern
- **User Tab**: Chat interface with "check" status command for pending approvals
- **Admin Tab**: Approval queue with dropdown selection + Approve/Reject buttons
- Session management via `gr.State(value=None)` → generates `thread_id` = `f"session_{hashlib.md5(...)[:8]}"`
- **Check command logic** (lines 100-130 in [multi_agents.py](multi_agents.py)):
  - Queries `router_graph.get_state(config, subgraphs=True)` to detect paused HITL state
  - Returns "⏳ Still Waiting..." if `sub_state.next` contains `"human_review"`

## Development Workflows

### Running the Application
```bash
source .venv/bin/activate
python multi_agents.py
```
Launches Gradio on `http://127.0.0.1:7860` with live reload.

### LangGraph CLI (langgraph.json)
```bash
langgraph dev  # Serves API + Studio UI for graph debugging
```
- Exposes graphs: `router_graph`, `verifyUserInfo_subgraph`, `checkCompanyPolicy_subgraph`
- Use LangGraph Studio to visualize execution paths and inspect checkpoints

### Testing HITL Flow
1. User: "I need to check my order balance"
2. Router → `userInfoInquiry` → credential collection (3 steps)
3. User types "check" to see status while admin pending
4. Admin approves via dashboard → user types "check" again → sees "✅ Access Granted!" message

## Project-Specific Conventions

### Sensitive Data Handling
- All user messages censored via `censor_sensitive_data()` before logging/display
- Emails: `a***@domain.com`, Zip codes: `1****`

### State Management Best Practices
- **Never mutate state directly** - always return updates via `Command` or dict
- Use `MessagesState` base class for chat-based graphs (inherits message history handling)
- `credentials` dict structure: `{"username": str, "email": str, "zip_code": str}`

### Node Naming Convention
- Router nodes: `llm_call_router`, `route_decision`
- Action nodes: verb phrases (`classify_request`, `collect_email`, `process_approval`)
- Subgraph invocation nodes: match intent category (`userInfoInquiry`, `policyQuestion`)

### Error Handling Pattern
All LLM calls wrapped in try/except with fallback responses (see [utils/tools.py](utils/tools.py):75-90).

## TODO Items (from codebase)
1. Organize all nodes/states/tools into utils (partially done)
2. Add guardrails, quotas, and rate limits for LLM calls
3. Integrate Snowflake for real order/transaction data
4. Convert `PolicyQuestionsState` to `MessagesState` for consistency

## Key Files Reference
- [multi_agents.py](multi_agents.py) - Main entry point, Gradio UI, admin handlers
- [utils/graphs.py](utils/graphs.py) - Graph builders for all three workflows
- [utils/nodes.py](utils/nodes.py) - All node functions (50+ nodes across 3 graphs)
- [utils/states.py](utils/states.py) - TypedDict definitions for graph states
- [utils/tools.py](utils/tools.py) - RAG helpers: `retrieve_docs()`, `generate_response()`, `censor_sensitive_data()`
- [langgraph.json](langgraph.json) - LangGraph CLI configuration

## Common Pitfalls
1. **Forgetting to return `Command(goto=END)`** in HITL nodes → graph hangs waiting for next input
2. **Not checking `dialogue_state` in router** → interrupts multi-turn credential collection
3. **Directly accessing `state.next` without `get_state()`** → stale state during "check" command
4. **Using wrong `thread_id`** → resumes/queries different conversation checkpoint

## MCP Tools
- SearchDocsByLangChain: Use this whenever dealing with LangChain or LangGraph code to get the most up to date information.