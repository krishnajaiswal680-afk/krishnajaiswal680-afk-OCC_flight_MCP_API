# ag_ui_adapter_with_redis.py
"""
FlightOps — AG-UI Adapter with Redis tool-result caching (Option A: only tool result caching)

This file is a drop-in replacement for your original ag_ui_adapter.py with Redis integration
that caches MCP tool results for a configurable TTL. The cache is used in the tool execution
loop: if a cached result exists for (tool_name + args) it will be returned instead of calling MCP.

Design decisions:
- Caching is performed in the adapter layer (not in client.py) — recommended best practice.
- The LLM planning & summarization are left unchanged.
- If Redis is unavailable the adapter continues to work normally.

Configuration via environment variables:
- REDIS_HOST, REDIS_PORT, REDIS_CLIENT_ID, REDIS_CLIENT_SECRET, REDIS_TENANT_ID
- REDIS_CACHE_TTL_SECONDS (default: 300 seconds)

"""
import os
import json
import asyncio
import uuid
import logging
from typing import AsyncGenerator

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

# import your existing MCP client (reuses planner + tool invocation)
from client import FlightOpsMCPClient

# Redis imports
from redis import Redis
try:
    # external helper that your other project used
    from redis_entraid.cred_provider import create_from_service_principal
except Exception:
    create_from_service_principal = None

# Setup logging
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("flightops.agui")

app = FastAPI(title="FlightOps — AG-UI Adapter (with Redis cache)")

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


# ----------------- REDIS CONFIG (ADAPTER LAYER) -----------------
REDIS_HOST = os.getenv("REDIS_HOST")
REDIS_PORT = int(os.getenv("REDIS_PORT", "10000"))
REDIS_CLIENT_ID = os.getenv("REDIS_CLIENT_ID")
REDIS_CLIENT_SECRET = os.getenv("REDIS_CLIENT_SECRET")
REDIS_TENANT_ID = os.getenv("REDIS_TENANT_ID")
REDIS_CACHE_TTL_SECONDS = int(os.getenv("REDIS_CACHE_TTL_SECONDS", str(60 * 5)))  # default 5 minutes

redis_client = None

if REDIS_HOST:
    try:
        if create_from_service_principal and REDIS_CLIENT_ID and REDIS_CLIENT_SECRET and REDIS_TENANT_ID:
            credential_provider = create_from_service_principal(
                REDIS_CLIENT_ID, REDIS_CLIENT_SECRET, REDIS_TENANT_ID
            )
            redis_client = Redis(
                host=REDIS_HOST,
                port=REDIS_PORT,
                ssl=True,
                credential_provider=credential_provider,
                decode_responses=True,
                socket_timeout=5,
                socket_connect_timeout=5,
                # DEV: if you need to disable cert verification set env DISABLE_REDIS_SSL_VERIFY=1
                ssl_cert_reqs=None if os.getenv("DISABLE_REDIS_SSL_VERIFY", "0") == "1" else "required",
                ssl_check_hostname=False if os.getenv("DISABLE_REDIS_SSL_VERIFY", "0") == "1" else True,
            )
        else:
            # Fallback to plain Redis connection (e.g., local redis, or connection string not using credential provider)
            redis_client = Redis(
                host=REDIS_HOST,
                port=REDIS_PORT,
                decode_responses=True,
                socket_timeout=5,
                socket_connect_timeout=5,
            )

        # quick check
        try:
            ok = redis_client.ping()
            logger.info("Redis ping -> %s", ok)
        except Exception as e:
            logger.warning("Redis ping failed (continuing without cache): %s", e)
            # keep redis_client but allow operations to fail gracefully
    except Exception as e:
        logger.error("Failed to initialize Redis client: %s", e)
        redis_client = None
else:
    logger.info("REDIS_HOST not set — running without Redis caching")


# Cache key naming
NAMESPACE = os.getenv("CACHE_NAMESPACE", "nonprod")
PROJECT = os.getenv("CACHE_PROJECT", "flightops")
MODULE = os.getenv("CACHE_MODULE", "mcp_adapter")


def make_cache_key(tool_name: str, args: dict) -> str:
    """Deterministic cache key for tool invocation. Uses sorted JSON for args."""
    try:
        # Normalize args: convert numbers/booleans to native types so string stable
        args_key = json.dumps(args, sort_keys=True, default=str)
    except Exception:
        args_key = json.dumps(str(args))
    # Keep key length reasonable — key may be long, Redis supports long keys but avoid extreme lengths
    return f"{NAMESPACE}:{PROJECT}:{MODULE}:tool:{tool_name}:{args_key}"


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
        logger.warning(f"Could not preconnect to MCP: {e}")


@app.get("/")
async def root():
    """Root endpoint for health check"""
    return {"message": "FlightOps AG-UI Adapter is running", "status": "ok"}


@app.get("/health")
async def health():
    """Health check endpoint"""
    try:
        await ensure_mcp_connected()
        redis_ok = None
        if redis_client:
            try:
                redis_ok = redis_client.ping()
            except Exception:
                redis_ok = False
        return {"status": "healthy", "mcp_connected": True, "redis": redis_ok}
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
        # plan_tools is synchronous in client.py (calls Azure API synchronously) — run in thread executor
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

            # ---- Redis caching wrapper (Option A: only tool result caching) ----
            cache_key = make_cache_key(tool_name, args)
            cached = None
            if redis_client:
                try:
                    cached = redis_client.get(cache_key)
                except Exception as e:
                    logger.warning(f"Redis read error for key={cache_key}: {e}")

            if cached:
                try:
                    tool_result = json.loads(cached)
                    logger.info(f"Cache hit for {tool_name}")
                except Exception:
                    # fallback: ignore cache if corrupted
                    tool_result = {"error": "corrupted cache"}
            else:
                # Execute the actual MCP tool (async call)
                try:
                    tool_result = await mcp_client.invoke_tool(tool_name, args)
                except Exception as exc:
                    tool_result = {"error": str(exc)}

                # Attempt to write to cache (best-effort)
                if redis_client:
                    try:
                        redis_client.set(cache_key, json.dumps(tool_result, default=str), ex=REDIS_CACHE_TTL_SECONDS)
                        logger.info(f"Cached result for {tool_name} (key={cache_key})")
                    except Exception as e:
                        logger.warning(f"Failed to write cache for key={cache_key}: {e}")

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
