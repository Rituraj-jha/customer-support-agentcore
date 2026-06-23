# Customer Support Workflow Agent (LangGraph + AgentCore Runtime)

Production-ready LangGraph workflow showcasing Amazon Bedrock AgentCore Runtime capabilities:

- Conditional branching
- LLM reasoning with Bedrock-hosted models
- TypedDict state propagation
- PostgreSQL checkpoint persistence
- AgentCore Memory integration
- Human-in-the-loop approval
- Durable pause/resume execution
- Context-aware intent classification
- Validation guardrails before routing

## 1. Project Structure

```text
src/
├── state.py
├── graph.py
├── nodes/
│   ├── memory.py
│   ├── retrieval.py
│   ├── validation.py
│   ├── intent.py
│   ├── refund.py
│   ├── technical.py
│   ├── account.py
│   ├── product.py
│   ├── approval.py
│   └── response.py
├── services/
│   ├── bedrock.py
│   ├── postgres.py
│   ├── memory_service.py
│   └── knowledge_base.py
├── api/
│   └── app.py
└── tests/
```

## 2. Environment Variables

Copy `.env.example` to `.env` and set values:

- `AWS_REGION`
- `BEDROCK_MODEL_ID` (Claude Sonnet / Opus / Nova Pro model IDs)
- `BEDROCK_DEBUG_ERRORS` (set `true` locally to return Bedrock error detail in API response)
- `POSTGRES_CHECKPOINT_DSN`
- `AGENTCORE_MEMORY_ENDPOINT`
- `AGENTCORE_MEMORY_API_KEY` (optional)
- `REFUND_APPROVAL_THRESHOLD`

Note: For newer Claude models (including Claude 4.5), use an inference profile ID
such as `us.anthropic.claude-sonnet-4-5-20250929-v1:0` instead of a raw foundation
model ID when required by Bedrock throughput rules.

## 3. Install and Run

```bash
python -m venv .venv
. .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
uvicorn src.api.app:app --host 0.0.0.0 --port 8080
```

## 4. API Endpoints

- `GET /health`
- `POST /workflow/start`
- `POST /workflow/approval`
- `GET /workflow/state/{thread_id}`

## 4.1 AgentCore Runtime Compatibility

Supported AgentCore contract endpoints:

- `GET /ping`
- `POST /invocations`

The `/invocations` endpoint accepts both payload shapes:

- Workflow shape: `session_id`, `user_id`, `message`, optional `request_metadata`
- AgentCore-style minimal shape: `prompt` (auto-mapped to `message`)

When minimal shape is used, defaults are applied:

- `session_id` defaults to `thread_id` or generated runtime id
- `user_id` defaults to `agentcore-user`

### Example invocation request

```json
{
  "session_id": "demo-1",
  "user_id": "user-1",
  "message": "Refund order ORDER123 for $150",
  "request_metadata": {
    "order_id": "ORDER123",
    "refund_amount": 150
  }
}
```

### Example invocation response (waiting for approval)

```json
{
  "session_id": "demo-1",
  "thread_id": "demo-1",
  "workflow_status": "waiting_for_human_approval",
  "final_response": null,
  "interrupt": {
    "type": "human_approval_required",
    "reason": "Refund exceeds approval threshold",
    "order_id": "ORDER123",
    "refund_amount": 150,
    "instructions": "Resume workflow with decision: approve or reject"
  },
  "metadata": {
    "intent": "refund",
    "approval_required": true,
    "approval_status": "pending",
    "validation_passed": true,
    "validation_errors": []
  }
}
```

### Example resume through `/invocations`

```json
{
  "session_id": "demo-1",
  "user_id": "user-1",
  "message": "",
  "thread_id": "demo-1",
  "action": "resume",
  "decision": "approve"
}
```

### Start workflow request

```json
{
  "session_id": "sess-001",
  "user_id": "user-123",
  "message": "Please refund order ORDER12345 for $150",
  "request_metadata": {
    "order_id": "ORDER12345",
    "refund_amount": 150
  }
}
```

### Approval resume request

```json
{
  "thread_id": "sess-001",
  "decision": "approve"
}
```

## 5. Mandatory Durability Demonstration

This scenario proves checkpoint durability and recovery:

1. Start a refund request above threshold (`$150`) using `POST /workflow/start`.
2. Workflow interrupts at `human_approval`.
3. Stop the application process.
4. Restart the application with same `POSTGRES_CHECKPOINT_DSN`.
5. Submit approval using `POST /workflow/approval` with the same `thread_id`.
6. Workflow resumes from checkpoint and completes.

If the resume succeeds after restart, durable execution is validated.

## 6. AgentCore Memory Integration

- Read memory before intent classification (`load_memory` node).
- Inject memory into classification and response prompts.
- Write memory at completion (`persist_memory` node) using `actor_id = user_id`.

Example retained memory:

- Session 1: "I prefer technical explanations"
- Session 2: response style adapts automatically.

## 7. Docker

Build image:

```bash
docker build -t support-agentcore:latest .
```

Run container:

```bash
docker run --rm -p 8080:8080 \
  -e AWS_REGION=us-east-1 \
  -e BEDROCK_MODEL_ID=us.amazon.nova-micro-v1:0 \\
  -e POSTGRES_CHECKPOINT_DSN=postgresql://postgres:postgres@host.docker.internal:5434/langgraph \\
  -e AGENTCORE_MEMORY_ENDPOINT=https://your-agentcore-memory-endpoint \
  support-agentcore:latest
```

## 7.1 Docker Compose (App + PostgreSQL)

Run the full local stack with one command:

```bash
docker compose up --build -d
```

Stop the stack:

```bash
docker compose down
```

Stop and remove stack plus database volume:

```bash
docker compose down -v
```

Compose provisions:

- `app` on port `8080`
- `postgres` exposed on host port `5434` (container port `5432`)
- persistent PostgreSQL volume `postgres_data`

The compose default DSN is:

```text
postgresql://postgres:postgres@postgres:5432/langgraph
```

You can override values via `.env` in the project root.

## 8. AgentCore Runtime Deployment Notes

1. Build and push this container to an accessible registry (e.g., ECR).
2. Configure runtime environment variables in AgentCore Runtime.
3. Set startup command:

```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8080
```

4. Ensure network access to:
- Bedrock model endpoint in your region
- PostgreSQL checkpoint database
- AgentCore Memory endpoint

## 8.1 Diagnostic Runtime Mode

To verify AgentCore request reachability, you can run a minimal diagnostic runtime
that logs all requests and includes a catch-all route.

When using Docker, switch the app module at runtime:

```bash
docker run --rm -p 8080:8080 \
  -e APP_MODULE=src.api.diagnostic_app:app \
  support-agentcore:latest
```

When using Docker Compose:

```bash
APP_MODULE=src.api.diagnostic_app:app docker compose up --build -d
```

The diagnostic runtime includes:

- `GET /ping`
- `POST /invocations`
- catch-all `/{path:path}` for all common HTTP methods

## 10. Durability Restart Demo Using Docker Compose

1. Start stack with `docker compose up --build -d`.
2. Call `POST /workflow/start` with refund amount above threshold (for example, `$150`).
3. Confirm response contains interrupt payload from `human_approval` node.
4. Simulate restart: `docker compose restart app`.
5. Submit approval to `POST /workflow/approval` with same `thread_id`.
6. Confirm workflow resumes and completes from PostgreSQL checkpoint.

## 9. Test

```bash
pytest -q src/tests
```

Tests include validation behavior and approval pause/resume flow.
