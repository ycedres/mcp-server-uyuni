import asyncio
import json
import os
import sys
import traceback
from typing import Any, Dict

from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import AsyncOpenAI

load_dotenv()

SERVER_COMMAND = "uv"
SERVER_ARGS = ["run", "--quiet", "mcp-server-uyuni"]

env = os.environ.copy()
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
env["PYTHONPATH"] = os.path.join(project_root, "src")

async def run_mcp_agent(prompt: str, context: Dict[str, Any]) -> Dict[str, Any]:
    config = context.get("config", {})
    
    model = config.get("model", "gemini-2.5-flash-lite")
    
    api_key = config.get("apiKey") or os.environ.get("GOOGLE_API_KEY")
    base_url = config.get("apiBaseUrl") or "https://generativelanguage.googleapis.com/v1beta/openai/"

    if not api_key:
        return {"error": "GOOGLE_API_KEY not found in config or environment"}

    client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    server_params = StdioServerParameters(
        command=SERVER_COMMAND,
        args=SERVER_ARGS,
        env=env
    )

    try:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                tools_result = await session.list_tools()
                openai_tools = []
                for tool in tools_result.tools:
                    openai_tools.append({
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": tool.inputSchema
                        }
                    })

                messages = [{"role": "user", "content": prompt}]
                
                max_turns = 10
                for _ in range(max_turns):
                    response = await client.chat.completions.create(
                        model=model,
                        messages=messages,
                        tools=openai_tools,
                    )
                    
                    message = response.choices[0].message
                    messages.append(message)

                    if not message.tool_calls:
                        return {"output": message.content}

                    for tool_call in message.tool_calls:
                        tool_name = tool_call.function.name
                        tool_args = json.loads(tool_call.function.arguments)
                        
                        result = await session.call_tool(tool_name, tool_args)
                        
                        content_text = ""
                        if result.content:
                            for content in result.content:
                                if content.type == "text":
                                    content_text += content.text
                                else:
                                    content_text += str(content)
                        
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": content_text
                        })
                
                return {"error": "Max turns reached"}
                
    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        return {"error": f"Exception during MCP execution: {str(e)}"}

def call_api(prompt, options, context):
    try:
        return asyncio.run(run_mcp_agent(prompt, context))
    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        return {"error": f"Unhandled exception: {str(e)}"}