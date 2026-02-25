import json
import os

def generate_tests(context=None):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    test_cases_file = os.path.join(base_dir, "test_cases_sys.json")
    config_file = os.path.join(base_dir, "test_config.json")
    
    placeholders = {}
    if os.path.exists(config_file):
        with open(config_file, "r") as f:
            config_data = json.load(f)
            if "systems" in config_data:
                for sys_key, sys_values in config_data["systems"].items():
                    for attr_key, attr_value in sys_values.items():
                        placeholders[f"{sys_key}_{attr_key}"] = attr_value
            if "activation_keys" in config_data:
                for key_name, key_value in config_data["activation_keys"].items():
                    placeholders[f"key_{key_name}"] = key_value
    
    with open(test_cases_file, "r") as f:
        data = json.load(f)
        
    tests = []
    for item in data:
        prompt = item.get("prompt")
        expected_output = item.get("expected_output")

        if placeholders:
            if prompt and isinstance(prompt, str):
                try:
                    prompt = prompt.format(**placeholders)
                except (KeyError, ValueError):
                    pass
            if expected_output and isinstance(expected_output, str):
                try:
                    expected_output = expected_output.format(**placeholders)
                except (KeyError, ValueError):
                    pass

        assertion_type = item.get("assertion_type", "llm-rubric")

        assertion = {
            "type": assertion_type
        }

        if expected_output is not None:
            assertion["value"] = expected_output

        if "threshold" in item:
            assertion["threshold"] = item["threshold"]

        if "assertion_config" in item:
            assertion.update(item["assertion_config"])

        tests.append({
            "vars": {
                "prompt": prompt
            },
            "assert": [assertion]
        })
    return tests