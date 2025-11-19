from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport
from openai import AzureOpenAI
from ag_ui.encoder import EventEncoder
from ag_ui.core import (
    TextMessageStartEvent,
    TextMessageContentEvent,
    TextMessageEndEvent,
    ToolCallStartEvent,
    ToolCallArgsEvent,
    ToolCallResultEvent,
    RunStartedEvent,
    RunFinishedEvent,
    RunErrorEvent,
    EventType,
)
import asyncio
import json
import os
import traceback
import sys
import httpx
from dotenv import load_dotenv
import uvicorn
from toon import encode
from redis import Redis
from redis_entraid.cred_provider import create_from_service_principal
load_dotenv()
import ssl
import logging
from variables import weather_schema, toon_payload, msg , build_table_data , build_chart_data


app = FastAPI()
 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
 
# MCP Server configuration - aligned with your test script
MCP_BASE_URL = os.getenv("MCP_BASE_URL", "http://127.0.0.1:8000")
MCP_TOKEN_URL = f"{MCP_BASE_URL}/auth/token"
MCP_SERVER_URL = f"{MCP_BASE_URL}/mcp"
 
# Azure OpenAI configuration
llm = AzureOpenAI(
    api_key=os.getenv("subscription_key"),
    api_version=os.getenv("api_version"),
    azure_endpoint=os.getenv("endpoint"),
)
# ----------------- Redis CONFIG -----------------
# : In production, keep these in .env, yaha concept test ke liye assume env se aa rahe
# ...existing code...
# ----------------- Redis CONFIG -----------------
REDIS_HOST = "occh-uamr01.centralindia.redis.azure.net"
REDIS_PORT = 10000
 
REDIS_CLIENT_ID = os.getenv("REDIS_CLIENT_ID") # = CLIENT_ID
REDIS_CLIENT_SECRET = os.getenv("REDIS_CLIENT_SECRET") # = CLIENT_SECRET
REDIS_TENANT_ID = os.getenv("REDIS_TENANT_ID") # = TENANT_ID
 
# create credential provider only (do not instantiate Redis yet)
try:
    redis_credential_provider = create_from_service_principal(
        REDIS_CLIENT_ID,
        REDIS_CLIENT_SECRET,
        REDIS_TENANT_ID,
    )
except Exception as e:
    logger.error("Failed to create redis credential provider: %s", e)
    redis_credential_provider = None
 
# DEV: create SSL flags that disable verification (ONLY FOR DEV)
_dev_disable_ssl_verify = os.getenv("DISABLE_REDIS_SSL_VERIFY", "1") == "1"
if _dev_disable_ssl_verify:
    logger.warning("Redis SSL verification DISABLED (development)")
 
redis_client = None
 
try:
    if redis_credential_provider:
        # NOTE: redis-py versions differ in accepted SSL params.
        # Many versions do NOT accept ssl_context in Redis(...), so use ssl_cert_reqs and ssl_check_hostname.
        redis_client = Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            ssl=True,
            credential_provider=redis_credential_provider,
            decode_responses=True,
            socket_timeout=5,            # read timeout
            socket_connect_timeout=5,    # connect timeout
            ssl_cert_reqs=None if _dev_disable_ssl_verify else "required",
            ssl_check_hostname=False if _dev_disable_ssl_verify else True,
        )
        # quick non-fatal ping to surface connectivity issues early
        try:
            ok = redis_client.ping()
            logger.info("Redis ping -> %s", ok)
        except Exception as e:
            logger.warning("Redis ping failed (will continue): %s", e)
    else:
        redis_client = None
except TypeError as te:
    logger.error("Redis client init TypeError (unsupported arg): %s", te)
    redis_client = None
except Exception as e:
    logger.error("Failed to create redis_client: %s", e)
    redis_client = None
# ...existing code...
# ...existing code...
# Key namespace structure: <namespace>:<project>:<module>:history:<session_id>
NAMESPACE = "non-prod"
PROJECT = "occhub"
MODULE = "weather_mcp"
 
# Concept test: single dummy session
DUMMY_SESSION_ID = "test-session-1"
 
HISTORY_TTL_SECONDS = 60 * 60 * 24 # 1 day
MAX_HISTORY_MESSAGES = 20 # last 20 messages (user+assistant)
 
def make_history_key(session_id: str) -> str:
    """
    <namespace>:<project>:<module>:chathistory:<session_id>
    """
    return f"{NAMESPACE}:{PROJECT}:{MODULE}:history:{session_id}"
 
 
# ...existing code...
 
def append_turn_to_history(session_id: str, user_msg: str, assistant_msg: str) -> None:
    """
    Store one full turn (user + assistant) in Redis List.
    This function is defensive: it returns immediately if Redis is unavailable
    or if any Redis operation fails (logs the error).
    """
    if not redis_client:
        logger.debug("append_turn_to_history: redis_client is not available, skipping")
        return
 
    key = make_history_key(session_id)
    user_entry = json.dumps({"role": "user", "content": user_msg}, ensure_ascii=False)
    assistant_entry = json.dumps({"role": "assistant", "content": assistant_msg}, ensure_ascii=False)
 
    try:
        pipe = redis_client.pipeline()
        pipe.rpush(key, user_entry, assistant_entry)
        pipe.expire(key, HISTORY_TTL_SECONDS)
        pipe.execute()
    except Exception as e:
        logger.warning("append_turn_to_history failed for key=%s: %s", key, e)
 
 
def load_history_messages(session_id: str, max_messages: int = MAX_HISTORY_MESSAGES):
    """
    Load last N messages from Redis and return as list of dicts.
    Defensive: returns empty list if Redis is unavailable or on error.
    """
    if not redis_client:
        logger.debug("load_history_messages: redis_client is not available")
        return []
 
    key = make_history_key(session_id)
    try:
        length = redis_client.llen(key)
    except Exception as e:
        logger.warning("Redis llen failed for key=%s: %s", key, e)
        return []
 
    if not length:
        return []
 
    start = max(0, length - max_messages)
    try:
        raw_msgs = redis_client.lrange(key, start, -1)
    except Exception as e:
        logger.warning("Redis lrange failed for key=%s: %s", key, e)
        return []
 
    messages = []
    for raw in raw_msgs:
        try:
            messages.append(json.loads(raw))
        except Exception:
            continue
    return messages
 
# ...existing code...
encoder = EventEncoder()
 
async def fetch_mcp_token() -> str:
    """Fetch authentication token for MCP server - aligned with test script."""
    try:
        async with httpx.AsyncClient(timeout=15.0) as http:
            print(f"🔐 Requesting token from: {MCP_TOKEN_URL}")
            response = await http.post(MCP_TOKEN_URL)
            response.raise_for_status()
            data = response.json()
           
            if not data.get("access_token"):
                raise RuntimeError(f"Token API returned error: {data}")
           
            print(f"✅ Successfully obtained MCP authentication token")
            print(f"📊 Token expires in: {data.get('expires_in', 'unknown')} seconds")
            return data["access_token"]
           
    except httpx.RequestError as e:
        print(f"❌ Failed to fetch MCP token - Connection error: {e}")
        # Authentication server not available - this is expected during development
        return None
    except httpx.HTTPStatusError as e:
        print(f"❌ Failed to fetch MCP token - HTTP {e.response.status_code}: {e.response.text}")
        return None
    except Exception as e:
        print(f"❌ Unexpected error fetching MCP token: {e}")
        return None
 
async def create_mcp_client():
    """Create MCP client with authentication - aligned with test script transport."""
    try:
        # Try to get authentication token first
        token = await fetch_mcp_token()
       
        if token:
            # Use HTTP transport with authentication (same as test script)
            print(f"🔐 Connecting to MCP server with authentication: {MCP_SERVER_URL}")
            transport = StreamableHttpTransport(
                url=MCP_SERVER_URL,
                headers={"Authorization": f"Bearer {token}"}
            )
            return Client(transport)
        else:
            # Fall back to stdio transport for development
            print(f"🔓 Authentication not available, falling back to stdio transport")
            print(f"💡 Make sure MCP server is running on {MCP_BASE_URL}")
            return Client("../weather/app.py")
           
    except Exception as e:
        print(f"❌ Failed to create MCP client: {e}")
        traceback.print_exc()
        # Final fallback
        print(f"🔄 Using stdio connection as final fallback")
        return Client("../weather/app.py")
 
 
 
async def test_mcp_connection():
    """Test MCP connection using ping tool (like the test script)."""
    try:
        client = await create_mcp_client()
        async with client:
            # Test with ping tool like your test script
            result = await client.call_tool("ping")
            print(f"🏓 MCP Server ping test: {result.data}")
            return True
    except Exception as e:
        print(f"❌ MCP connection test failed: {e}")
        return False
 
async def interact_with_server(user_prompt: str):
    """Main orchestration generator that yields AG-UI events for streaming."""
    session_id=DUMMY_SESSION_ID
    client = None
    try:
        # Create authenticated MCP client
        client = await create_mcp_client()
       
        async with client:
            # Start the run
            yield encoder.encode(RunStartedEvent(
                type=EventType.RUN_STARTED,
                thread_id="thread_1",
                run_id="run_1"
            ))
           
            # Start assistant message
            yield encoder.encode(TextMessageStartEvent(
                type=EventType.TEXT_MESSAGE_START,
                message_id="msg_1",
                role="assistant"
            ))
 
            # Discover tools from MCP server
            print(f"🔍 Discovering available tools from MCP server...")
           
            # Read schema resource
            # schema = await client.read_resource("resource://metar_json_schema")
 
           
            tool_descriptions = await client.list_tools()
            print(f"📋 Found {len(tool_descriptions)} tools: {[t.name for t in tool_descriptions]}")
           
            openai_tools = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.inputSchema,
                    },
                }
                for tool in tool_descriptions
            ]
 
            toon = toon_payload
            schema = weather_schema          
            schema_toon = encode(schema)
 
            sys_msg =msg
 
            messages = [
                {"role": "system", "content": sys_msg},
                {"role": "system", "content": "key Value pairs of airport:\n" + toon},
                {"role": "system", "content": "Schema (JSON):\n" + schema_toon},
                # {"role": "user", "content": "The User prompt is as follows:\n"+user_prompt},
            ]
 
           
            # 2) Conversation history from Redis (dummy session)
            history_messages = load_history_messages(session_id)
            print("1.5")
            if history_messages:
                print(f"🧠 Loaded {len(history_messages)} history messages from Redis for session {session_id}")
            messages.extend(history_messages)
           
            print("2")
            # 3) Current user prompt
            messages.append(
                {
                    "role": "user",
                    "content": "The User prompt is as follows:\n" + user_prompt,
                }
            )
 
            # messages = [{
            #     "role": "user",
            #     "content": f"""
            #         You are an intelligent agent capable of orchestrating multiple tools to assist users. Below is a list of available tools, each with a name, description of what it does, and the input it requires.
           
            #         Guardrails:
           
            #         - You may only provide answers that are directly related to the database of airports, city details, or weather data.
           
            #         - For Casual greetings or simple pleasantries (e.g., "Hello", "Namaskar","How are you?"), you may respond conversationally(e.g.,"Hi! How can I Assist you today?").
               
            #         - For Casual conversation like (e.g., "ok","Thankyou","amazing") you may respond conversationally(e.g.,"Thank You anything else you want me to assist with you").
               
            #         - Do not provide answers or guesses about anything outside this scope.
           
            #         - If the user's request is outside this scope, respond politely:
 
            #         "I'm sorry, I can only provide information related to weather of any airport. Can I help you with that?"
 
            #         Instructions:
           
            #         1. Identify which tools can be used to fulfill their request.
           
            #         2. Call one or more tools as needed.
           
            #         3. Explain how these tools will be used.
           
            #         4. Ask for any additional details if required.
           
            #         5. Do not give any additional explanation, context, or interpretation. Do not hesitate or ask follow-up questions unless the user explicitly asks for explanation or interpretation of Metar Data.
           
            #         6. If duplicate Mongo DB results are present, return only one. If there are differences, return all the unique values.
           
            #         7. If the user specifically asks for Metar data, just provide the Raw Metar Data Value.
           
            #         8. If asked for Hours Back data and no results come back from query running then specify the latest timestamp that is present in MongoDB
           
            #         The user's request is: "{user_prompt}".
 
            #         Database schema: {schema}
            #     """
            # }]
            print(messages)
            
            while True:
                print(f"🤖 Sending request to Azure OpenAI...")
                response = llm.chat.completions.create(
                    model=os.getenv("deployment"),
                    messages=messages,
                    tool_choice="auto",
                    tools=openai_tools if openai_tools else None,
                    stream=False,
                )
 
                print(response.usage)
 
                message = response.choices[0].message
                finish_reason = response.choices[0].finish_reason
 
                # === TOOL CALLING BRANCH ===
                if message.tool_calls:
                    print(f"🔧 LLM wants to call {len(message.tool_calls)} tool(s)")
                   
                    messages.append({
                        "role": "assistant",
                        "content": message.content,
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments,
                                },
                            }
                            for tc in message.tool_calls
                        ],
                    })
 
                    for tool_call in message.tool_calls:
                        tool_name = tool_call.function.name
                        tool_args = json.loads(tool_call.function.arguments)
 
                        print(f"  ⚙️  Calling tool: {tool_name} with args: {tool_args}")
 
                        yield encoder.encode(
                            ToolCallStartEvent(
                                type=EventType.TOOL_CALL_START,
                                tool_call_id=tool_call.id,
                                tool_call_name=tool_name,
                            )
                        )
                       
                        yield encoder.encode(
                            ToolCallArgsEvent(
                                type=EventType.TOOL_CALL_ARGS,
                                tool_call_id=tool_call.id,
                                delta=json.dumps(tool_args),
                            )
                        )
 
                        # Call the tool with authentication (like test script)
                        try:
                            print(f"  📡 Executing authenticated tool call on MCP server...")
                            result = await client.call_tool(tool_name, tool_args)
                           
                            # Handle result data properly
                            if hasattr(result, 'data'):
                                result_data = result.data
                            else:
                                result_data = result
                           
                            if isinstance(result_data, dict):
                                result_content = result_data.get("content", str(result_data))
                            else:
                                result_content = str(result_data)
 
                            print(f"  ✅ Tool result: {result_content[:200]}{'...' if len(result_content) > 200 else ''}")
                           
                        except Exception as tool_error:
                            print(f"  ❌ Tool call failed: {tool_error}")
                            traceback.print_exc()
                            result_content = f"Tool call failed: {str(tool_error)}"
 
                        yield encoder.encode(
                            ToolCallResultEvent(
                                type=EventType.TOOL_CALL_RESULT,
                                message_id="msg_1",
                                tool_call_id=tool_call.id,
                                content=result_content,
                                role="tool",
                            )
                        )
 
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": result_content,
                        })
 
                    continue
 
                # === TEXT RESPONSE BRANCH ===
                else:
                    print(f"💬 LLM final response (finish_reason: {finish_reason})")
                   
                    if message.content:
                        content = message.content
                        print(f"📝 Starting to stream {len(content)} characters...")
                       
                        # Stream character by character
                        for i, char in enumerate(content):
                            event_data = encoder.encode(
                                TextMessageContentEvent(
                                    type=EventType.TEXT_MESSAGE_CONTENT,
                                    message_id="msg_1",
                                    delta=char,
                                )
                            )
                            yield event_data
                           
                            # Print progress every 50 characters
                            if (i + 1) % 50 == 0:
                                print(f"  📤 Streamed {i + 1}/{len(content)} chars", flush=True)
                           
                            # Delay for typing effect
                            await asyncio.sleep(0.02)
                       
                        print(f"  ✅ Finished streaming all {len(content)} characters")
                   
                        try:
                            append_turn_to_history(session_id, user_prompt, content)
                            print(f"💾 Saved turn to Redis for session={session_id}")
                        except Exception as redis_err:
                            print(f"⚠️ Failed to write chat history to Redis: {redis_err}")
                   
                    table = build_table_data()
                    chart = build_chart_data()
 
                    final_event = {
                        "type": "RUN_FINISHED",
                        "table": table,
                        "chart": {
                            "data": chart["data"],
                            "xKey": chart["xKey"],
                            "yKey": chart["yKey"],
                            "chartType": chart["chartType"],
                        },
                    }
                    # print(final_event)
                    yield "data: " + json.dumps(final_event) + "\n\n"
 
                   
                    yield encoder.encode(
                        TextMessageEndEvent(
                            type=EventType.TEXT_MESSAGE_END,
                            message_id="msg_1"
                        )
                    )
                   
                    yield encoder.encode(
                        RunFinishedEvent(
                            type=EventType.RUN_FINISHED,
                            thread_id="thread_1",
                            run_id="run_1"
                        )
                    )
                   
                    print("✅ Conversation complete!")
                    break
 
    except Exception as e:
        print(f"❌ Error in interact_with_server: {str(e)}")
        traceback.print_exc()
        yield encoder.encode(
            RunErrorEvent(
                type=EventType.RUN_ERROR,
                message=str(e)
            )
        )
    finally:
        if client:
            print("🔚 MCP client interaction complete.")
 
 
@app.post("/get_data")
async def stream_response(userprompt: str = Query(...)):
    print(f"\n{'='*60}")
    print(f"🟡 NEW REQUEST: {userprompt}")
    print(f"{'='*60}\n")
   
    async def event_generator():
        try:
            async for event in interact_with_server(userprompt):
                # event is a string from encoder.encode()
                # Ensure event ends with newline for SSE format
                if not event.endswith('\n'):
                    event = event + '\n'
                yield event
                # Force flush with tiny delay
                await asyncio.sleep(0)
        except Exception as e:
            print(f"❌ Generator error: {e}")
            traceback.print_exc()
 
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "Content-Type": "text/event-stream",
        },
    )
 
@app.get("/health")
async def health_check():
    """Health check endpoint that also tests MCP server connectivity."""
    try:
        # Test MCP connection using ping (like your test script)
        mcp_connected = await test_mcp_connection()
       
        client = await create_mcp_client()
        async with client:
            tools = await client.list_tools()
           
            return {
                "status": "healthy" if mcp_connected else "degraded",
                "mcp_server": "connected" if mcp_connected else "disconnected",
                "available_tools": len(tools),
                "tools": [t.name for t in tools],
                "authentication": "enabled" if MCP_BASE_URL == "http://127.0.0.1:8000" else "custom",
                "mcp_endpoints": {
                    "token_url": MCP_TOKEN_URL,
                    "server_url": MCP_SERVER_URL
                }
            }
    except Exception as e:
        return {
            "status": "degraded",
            "mcp_server": "disconnected",
            "error": str(e),
            "authentication": "failed"
        }
 
@app.get("/test-mcp")
async def test_mcp_endpoint():
    """Test endpoint that replicates your test script functionality."""
    try:
        print("🧪 Testing MCP connection like test script...")
       
        # Replicate your test script exactly
        token = await fetch_mcp_token()
        if not token:
            return {"status": "failed", "error": "Could not obtain token"}
       
        transport = StreamableHttpTransport(
            url=MCP_SERVER_URL,
            headers={"Authorization": f"Bearer {token}"}
        )
       
        async with Client(transport) as c:
            result = await c.call_tool("ping")
            return {
                "status": "success",
                "message": "MCP connection test successful",
                "server_response": result.data,
                "token_obtained": True,
                "endpoints": {
                    "token_url": MCP_TOKEN_URL,
                    "server_url": MCP_SERVER_URL
                }
            }
           
    except Exception as e:
        return {
            "status": "failed",
            "error": str(e),
            "token_obtained": token is not None if 'token' in locals() else False
        }
 
@app.get("/")
async def root():
    return {"status": "ok", "message": "AG-UI FastAPI server is running with MCP authentication"}
 
if __name__ == "__main__":
    print("🚀 FastAPI AG-UI server starting on http://127.0.0.1:8001")
    print("🔐 Azure authentication integration enabled")
    print(f"🔗 MCP Server: {MCP_BASE_URL}")
    print(f"🎫 Token URL: {MCP_TOKEN_URL}")
    print(f"📡 MCP URL: {MCP_SERVER_URL}")
    print("💡 TIP: Visit /test-mcp to test authentication like your test script")
    print("📡 Ready to receive requests...")
   
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,
        log_level="info",
        access_log=True,
    )