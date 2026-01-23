# Penguin Project - AI Agent Instructions

## Project Overview
E-commerce customer support AI bot with human-in-the-loop (HITL) approval workflows, built using **LangGraph** for stateful agent orchestration, AWS Bedrock (Nova Lite) for LLM, and Gradio for dual-interface (User/Admin) UI.

## Architecture Overview

### Three-Graph Design
1. **Router Graph** ([multi_agents.py](multi_agents.py) lines 92-109) - Routes queries to three subgraphs:
   - `userInfoInquiry`: Account data, orders, refunds, complaints (requires credential verification)
   - `policyQuestion`: General company policy Q&A (no auth needed)
   - `misc_call`: Rejections for off-topic questions
   - Uses `llm.with_structured_output(Route)` for structured routing

2. **User Verification Subgraph** ([verifyUser_nodes.py](penguin_agent/utils/nodes/verifyUser_nodes.py)) - Credential collection via `VerifyUserInfoState`:
   - Multi-turn: `idle` → `awaiting_username` → `awaiting_email` → `awaiting_zipcode` → admin review
   - **Critical**: Return `Command(goto=END)` after each step to yield control back to user input
   - Admin approval via `interrupt()` pauses execution; resume with `Command(resume="approved"|"rejected")`

3. **Policy Subgraph** ([policyQuestion_nodes.py](penguin_agent/utils/nodes/policyQuestion_nodes.py)) - RAG pipeline:
   - `split_query`: LLM breaks multi-part questions into standalone questions
   - `continue_to_verification`: Uses `Send()` to fan-out parallel processing
   - `process_question`: Retrieves policy docs (ChromaDB via [retriever.py](penguin_agent/utils/retriever.py)), generates answer
   - Accumulates answers via `Annotated[List[str], operator.add]`

### State Flow
- **ParentRouterState**: Main state with `messages`, `thread_id`, `decision`, `credentials`, `lockedState`
- **VerifyUserInfoState** (extends `MessagesState`): Adds `credentials`, `dialogue_state`, `approval_status`, `data_request_type`, `snowflake_results`
- **PolicyQuestionsState**: `query`, `documents`, `questions`, `answers` (independent from message history)

## Critical Patterns

### Command Pattern for Multi-Turn Interactions
```python
# Collect credential, update state, pause for user input
return Command(
    update={"credentials": creds, "dialogue_state": "awaiting_email"},
    goto=END  # Returns control to user; graph waits for next invoke
)
```

### Human-in-the-Loop with interrupt()
```python
# In [verifyUser_nodes.py](penguin_agent/utils/nodes/verifyUser_nodes.py): human_review node
approval_decision = interrupt({"message": "Waiting for admin approval", "thread_id": thread_id})

# Resume in Gradio admin handler: [multi_agents.py](multi_agents.py):330-340
router_graph.invoke(
    Command(resume=approval_status), 
    config={"configurable": {"thread_id": thread_id}}
)
```

### Subgraph Invocation
Parent passes minimal data; subgraph returns full state:
```python
# Line 161 in [multi_agents.py](multi_agents.py)
response = user_subgraph.invoke({
    "messages": state["messages"][-1].content,  # String, not list
    "thread_id": state["thread_id"],
    "credentials": state["credentials"]
})
# Parent extracts relevant fields: response["messages"][-1], response["credentials"]
```

## Integration Points

### AWS Bedrock (Nova Lite v1)
- Initialized in every node file with env vars: `BEDROCK_AWS_ACCESS_KEY_ID`, `BEDROCK_AWS_SECRET_ACCESS_KEY`, `BEDROCK_AWS_REGION`
- Temperature 0.7 for general chat, 0.0 for query splitting (see [policyQuestion_nodes.py](penguin_agent/utils/nodes/policyQuestion_nodes.py):13)
- Fallback mocking in [tools.py](penguin_agent/utils/tools.py):50-65 if API fails (non-fatal)

### Snowflake Queries
- Configured in [verifyUser_nodes.py](penguin_agent/utils/nodes/verifyUser_nodes.py):17-22 via env vars
- Two query types: `pii` (DIM_CUSTOMERS join FCT_TRANSACTIONS) and `transactions` (ORDER BY CREATED_DATE DESC)
- Called after approval: routes to `query_pii`, `query_transactions`, or `query_both` based on `data_request_type` field

### ChromaDB Retriever
- Policy docs loaded via [retriever.py](penguin_agent/utils/retriever.py): `get_policy_retriever(k=3)` returns top-3 chunks
- Used in [policyQuestion_nodes.py](penguin_agent/utils/nodes/policyQuestion_nodes.py):87 `process_question` node

## Development Workflows

### Run Locally
```bash
source .venv/bin/activate
python penguin_agent/multi_agents.py
```
Opens Gradio at `http://127.0.0.1:7860` with user and admin tabs.

### Test Credential Flow
1. User: "I need to check my account balance"
2. Router → `userInfoInquiry` subgraph → requests username (Command→END)
3. User responds → collects email (Command→END)
4. User responds → collects zipcode (Command→END)
5. Graph pauses at `human_review` node (interrupt)
6. Admin approves via Gradio tab
7. Graph resumes, queries Snowflake, formats response

### Check HITL Status
User can type "check" to poll `router_graph.get_state()` for `lockedState` (see [multi_agents.py](multi_agents.py):300-310):
- `"input"`: Waiting for user to provide next credential
- `"pending"`: Admin review in progress
- `"approved"`: Ready to query database

## Project-Specific Conventions

### Sensitive Data Censoring
- Function: `censor_sensitive_data()` in [tools.py](penguin_agent/utils/tools.py):6-13
- Emails → `a***@domain.com`, Zip codes → `1****`, usernames logged as `***`

### Node Organization
- **Nodes split by workflow** in [nodes/](penguin_agent/utils/nodes/) subdirectory:
  - [router_nodes.py](penguin_agent/utils/nodes/router_nodes.py): Router graph nodes
  - [verifyUser_nodes.py](penguin_agent/utils/nodes/verifyUser_nodes.py): Credential collection (16 nodes)
  - [policyQuestion_nodes.py](penguin_agent/utils/nodes/policyQuestion_nodes.py): RAG pipeline (3 nodes)
- **Naming**: Action verbs for nodes (`collect_email`, `process_approval`), subgraphs match intent (`userInfoInquiry`, `policyQuestion`)

### State Immutability
- Always return dict or `Command` with updates—never mutate state in place
- Parent router uses `lockedState` field to track credential collection progress (not to be confused with LangGraph's internal `next`)

## Known Limitations & TODOs
1. `langgraph.json` points to deleted `/legacy/test.py:graph` — CLI dev currently broken
2. Snowflake queries incomplete (lines 70-90 in [verifyUser_nodes.py](penguin_agent/utils/nodes/verifyUser_nodes.py))
3. `PolicyQuestionsState` should inherit `MessagesState` for message deduplication
4. No rate limiting or guardrails on LLM calls (TODO marker at [multi_agents.py](penguin_agent/multi_agents.py):38)

## Key Files
- [multi_agents.py](penguin_agent/multi_agents.py) — Entry point, Gradio UI, graph builders
- [states.py](penguin_agent/utils/states.py) — TypedDict definitions + `PENDING_APPROVALS` dict
- [nodes/](penguin_agent/utils/nodes/) — All node functions
- [tools.py](penguin_agent/utils/tools.py) — Helpers: RAG, censoring, Bedrock error handling
- [retriever.py](penguin_agent/utils/retriever.py) — ChromaDB policy doc loader