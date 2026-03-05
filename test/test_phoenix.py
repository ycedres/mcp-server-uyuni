import pytest
import json
import os
import sys
import asyncio
import pandas as pd
from google import genai
from google.genai import types
from phoenix.evals import QAEvaluator, run_evals, LiteLLMModel
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

TEST_CASES_FILE = 'test_cases_sys.json'
TEST_CONFIG_FILE = 'test_config.json'
MCP_CONFIG_FILE = 'config.json'

def load_vars():
    config_path = os.path.join(os.path.dirname(__file__), TEST_CONFIG_FILE)
    placeholders = {}

    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config_data = json.load(f)
            for key, value in config_data.items():
                if isinstance(value, str):
                    placeholders[key] = value
            if "systems" in config_data:
                for sys_key, sys_values in config_data["systems"].items():
                    for attr_key, attr_value in sys_values.items():
                        placeholders[f"{sys_key}_{attr_key}"] = attr_value
            if "activation_keys" in config_data:
                for key_name, key_value in config_data["activation_keys"].items():
                    placeholders[f"key_{key_name}"] = key_value
    return placeholders

VARS = load_vars()

def sanitize_gemini_schema(schema):
    """
    Sanitizes a JSON schema dictionary to be compatible with Gemini's Schema protobuf.
    Removes unsupported fields like 'anyOf', 'default', 'title' and ensures only allowed fields remain.
    """
    if not isinstance(schema, dict):
        return schema
    
    ALLOWED_FIELDS = {'type', 'format', 'description', 'nullable', 'enum', 'properties', 'required', 'items'}
    new_schema = {}

    if "anyOf" in schema:
        options = schema["anyOf"]
        selected = options[0]
        for opt in options:
            if isinstance(opt, dict) and opt.get("type") == "string":
                selected = opt
                break
        new_schema.update(sanitize_gemini_schema(selected))
    
    for k, v in schema.items():
        if k in ALLOWED_FIELDS:
            if k == "properties":
                new_schema[k] = {pk: sanitize_gemini_schema(pv) for pk, pv in v.items()}
            elif k == "items":
                new_schema[k] = sanitize_gemini_schema(v)
            else:
                new_schema[k] = v
                
    return new_schema

async def run_mcp_agent(prompt: str, model: str = None) -> str:
    if not model:
        model = os.environ.get("AGENT_MODEL", "gemini-1.5-flash-latest")
    server_params = StdioServerParameters(
        command="uv",
        args=["run", "mcp-server-uyuni"],
        env={**os.environ, "UYUNI_MCP_WRITE_TOOLS_ENABLED": "true"}
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            mcp_tools = await session.list_tools()

            function_declarations = []
            for tool in mcp_tools.tools:
                function_declarations.append(
                    types.FunctionDeclaration(
                        name=tool.name,
                        description=tool.description,
                        parameters=sanitize_gemini_schema(tool.inputSchema)
                    )
                )

            client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))

            chat = client.aio.chats.create(
                model=model,
                config=types.GenerateContentConfig(
                    tools=[types.Tool(function_declarations=function_declarations)]
                )
            )

            response = await chat.send_message(prompt)
            
            while response.candidates and response.candidates[0].content.parts:
                parts = response.candidates[0].content.parts
                fc_parts = [p for p in parts if p.function_call]
                if not fc_parts:
                    break
                
                async def call_and_prepare_response(fc_part):
                    fc = fc_part.function_call
                    result = await session.call_tool(fc.name, dict(fc.args))
                    tool_output = "\n".join([c.text for c in result.content if c.type == "text"])
                    return types.Part(
                        function_response=types.FunctionResponse(
                            name=fc.name,
                            response={"result": tool_output}
                        )
                    )

                tool_responses = await asyncio.gather(
                    *(call_and_prepare_response(fc_part) for fc_part in fc_parts)
                )
                
                response = await chat.send_message(tool_responses)

            return response.text

def query_mcp_server(prompt: str) -> str:
    return asyncio.run(run_mcp_agent(prompt))

def load_test_cases():
    json_path = os.path.join(os.path.dirname(__file__), TEST_CASES_FILE)
    if not os.path.exists(json_path):
        return []
    
    with open(json_path, 'r') as f:
        return json.load(f)

@pytest.mark.parametrize("test_case", load_test_cases())
def test_uyuni_mcp_phoenix(test_case):
    prompt_template = test_case.get("prompt")
    expected_template = test_case.get("expected_output")
    test_id = test_case.get("id", "unknown")

    if not prompt_template or not expected_template:
        pytest.skip(f"Skipping malformed test case: {test_id}")

    prompt = prompt_template.format(**VARS)
    expected_output = expected_template.format(**VARS)

    actual_output = query_mcp_server(prompt)

    judge_model = os.environ.get("JUDGE_MODEL", "gemini-1.5-flash-latest")
    if not judge_model.startswith("gemini/"):
        judge_model = f"gemini/{judge_model}"
    eval_model = LiteLLMModel(model=judge_model)
    qa_correctness_evaluator = QAEvaluator(model=eval_model)

    test_df = pd.DataFrame([{
        "input": prompt,
        "output": actual_output,
        "reference": expected_output,
    }])

    eval_results = run_evals(
        dataframe=test_df,
        evaluators=[qa_correctness_evaluator],
    )

    correctness_df = eval_results[0]
    result_row = correctness_df.iloc[0]
    label = str(result_row["label"])
    score = result_row.get("score", "N/A")
    explanation = result_row.get("explanation") or result_row.get("reasoning")

    if not explanation:
        explanation = f"No explanation provided. Score: {score}. Available columns: {list(correctness_df.columns)}"

    if label.lower() != "correct":
        error_message = (
            f"\n--- Phoenix Test Failed ---\n"
            f"Test Case ID: {test_id}\n"
            f"Prompt: {prompt}\n"
            f"Expected Output Hint: {expected_output}\n"
            f"----- ACTUAL OUTPUT -----\n{actual_output}\n"
            f"----- END ACTUAL OUTPUT -----\n"
            f"Reason: {explanation}"
        )
        pytest.fail(error_message)
