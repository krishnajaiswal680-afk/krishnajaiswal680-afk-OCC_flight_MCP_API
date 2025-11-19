# ag_ui_adapter.py
import os
import json
import asyncio
import uuid
from typing import AsyncGenerator

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

# import your existing MCP client (reuses planner + tool invocation)
from client import FlightOpsMCPClient

# NOTE: The user provided AG-UI SDK docs define event types and recommended payloads.
# We will emit plain JSON event objects as SSE 'data: <json>\n\n' lines using AG-UI event shapes.
# If you install `ag-ui-protocol`, you can import typed models and replace dicts with model instances.

app = FastAPI(title="FlightOps — AG-UI Adapter")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Create shared FlightOpsMCPClient (keeps connection to MCP)
mcp_client = FlightOpsMCPClient()


# Utility to format SSE data
def sse_event(data: dict) -> str:
    payload = json.dumps(data, default=str)
    return f"data: {payload}\n\n"


async def ensure_mcp_connected():
    if not mcp_client.session:
        await mcp_client.connect()


@app.on_event("startup")
async def startup_event():
    # Connect to MCP at startup to reduce latency
    try:
        await ensure_mcp_connected()
    except Exception as e:
        # don't crash server if DB/MCP not ready — clients will get errors when they call
        app.logger = getattr(app, "logger", None)
        if app.logger:
            app.logger.warning(f"Could not preconnect to MCP: {e}")


@app.get("/")
async def root():
    """Root endpoint for health check"""
    return {"message": "FlightOps AG-UI Adapter is running", "status": "ok"}


@app.get("/health")
async def health():
    """Health check endpoint"""
    try:
        await ensure_mcp_connected()
        return {"status": "healthy", "mcp_connected": True}
    except Exception as e:
        return {"status": "unhealthy", "mcp_connected": False, "error": str(e)}


@app.post("/agent", response_class=StreamingResponse)
async def run_agent(request: Request):
    """
    AG-UI-compatible /agent endpoint.
    Accepts a JSON body with at least:
      - thread_id (optional)
      - run_id (optional)
      - messages: list (we expect last message from user)
      - tools/context (optional)

    This endpoint will:
      1) Emit RUN_STARTED
      2) Call your planner (FlightOpsMCPClient.plan_tools)
      3) For each plan step emit TOOL_CALL_START / TOOL_CALL_ARGS / TOOL_CALL_END
         and then execute the MCP tool, emitting TOOL_CALL_RESULT
      4) Emit assistant TEXT_MESSAGE_CONTENT chunks for summary
      5) Emit RUN_FINISHED
    """
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    # derive run/thread ids
    thread_id = body.get("thread_id") or str(uuid.uuid4())
    run_id = body.get("run_id") or str(uuid.uuid4())
    messages = body.get("messages", [])
    tools = body.get("tools", [])

    # determine user query — prefer last user message
    user_query = ""
    if messages:
        # messages are expected as AG-UI Message objects; fallback to string list
        last = messages[-1]
        if isinstance(last, dict) and last.get("role") == "user":
            user_query = last.get("content", "")
        elif isinstance(last, str):
            user_query = last

    if not user_query or not user_query.strip():
        raise HTTPException(status_code=400, detail="No user query in messages payload")

    async def event_stream() -> AsyncGenerator[str, None]:
        # 1) RUN_STARTED
        start_event = {
            "type": "RUN_STARTED",
            "thread_id": thread_id,
            "run_id": run_id,
        }
        yield sse_event(start_event)

        # ensure MCP connected
        try:
            await ensure_mcp_connected()
        except Exception as e:
            yield sse_event({"type": "RUN_ERROR", "error": str(e)})
            return

        # 2) Ask the LLM to plan (synchronous wrapper)
        yield sse_event({"type": "TEXT_MESSAGE_CONTENT", "content": "Generating tool plan..."})

        loop = asyncio.get_event_loop()
        # plan_tools is synchronous in client.py (calls Azure API synchronously) — but in class it is not async
        # In your client, plan_tools() calls Azure synchronously; running it blocking in thread executor
        plan_data = await loop.run_in_executor(None, mcp_client.plan_tools, user_query)
        plan = plan_data.get("plan", [])

        # Emit a snapshot of the planned steps
        yield sse_event({"type": "STATE_SNAPSHOT", "snapshot": {"plan": plan}})

        if not plan:
            yield sse_event({"type": "TEXT_MESSAGE_CONTENT", "content": "LLM did not produce a valid plan."})
            yield sse_event({"type": "RUN_FINISHED"})
            return

        # 3) Execute steps sequentially, emitting TOOL_CALL events
        results = []
        for step_index, step in enumerate(plan):
            tool_name = step.get("tool")
            args = step.get("arguments", {}) or {}
            tool_call_id = f"toolcall-{uuid.uuid4().hex[:8]}"

            # TOOL_CALL_START
            yield sse_event({
                "type": "TOOL_CALL_START",
                "toolCallId": tool_call_id,
                "toolCallName": tool_name,
                "parentMessageId": None
            })

            # Emit TOOL_CALL_ARGS as a single chunk (frontend will accumulate if needed)
            # We stream args as JSON string fragments (delta-style). For large arg streams, break into pieces.
            args_json = json.dumps(args, default=str)
            yield sse_event({
                "type": "TOOL_CALL_ARGS",
                "toolCallId": tool_call_id,
                "delta": args_json
            })

            # TOOL_CALL_END
            yield sse_event({
                "type": "TOOL_CALL_END",
                "toolCallId": tool_call_id
            })

            # Execute the actual MCP tool (async call)
            try:
                # invoke_tool is async — call it directly
                tool_result = await mcp_client.invoke_tool(tool_name, args)
            except Exception as exc:
                tool_result = {"error": str(exc)}

            # TOOL_CALL_RESULT (role: tool message)
            tool_message = {
                "id": f"msg-{uuid.uuid4().hex[:8]}",
                "role": "tool",
                "content": json.dumps(tool_result, default=str),
                "tool_call_id": tool_call_id,
            }
            yield sse_event({
                "type": "TOOL_CALL_RESULT",
                "message": tool_message
            })

            results.append({tool_name: tool_result})

            # Optionally emit step finished event
            yield sse_event({
                "type": "STEP_FINISHED",
                "step_index": step_index,
                "tool": tool_name
            })

        # 4) Summarize results by asking LLM (use existing summarize_results)
        yield sse_event({"type": "TEXT_MESSAGE_CONTENT", "content": "Summarizing results..."})

        try:
            # summarize_results is synchronous on top of _call_azure_openai — run in threadpool
            summary = await loop.run_in_executor(None, mcp_client.summarize_results, user_query, plan, results)
            assistant_text = summary.get("summary", "") if isinstance(summary, dict) else str(summary)
        except Exception as e:
            assistant_text = f"Failed to summarize results: {e}"

        # Stream assistant message content in chunks — (here we send single chunk)
        yield sse_event({
            "type": "TEXT_MESSAGE_CONTENT",
            "message": {
                "id": f"msg-{uuid.uuid4().hex[:8]}",
                "role": "assistant",
                "content": assistant_text
            }
        })

        # 5) Final snapshot with results, and RUN_FINISHED
        yield sse_event({"type": "STATE_SNAPSHOT", "snapshot": {"plan": plan, "results": results}})
        yield sse_event({"type": "RUN_FINISHED", "run_id": run_id})

    return StreamingResponse(event_stream(), media_type="text/event-stream")