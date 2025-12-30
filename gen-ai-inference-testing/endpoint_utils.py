
import time
import requests
import json
import re
import os
from pprint import pprint
import sys
from requests.exceptions import RequestException
from urllib.parse import urlparse
from importlib import import_module

def get_health_endpoint(endpoint_url: str) -> str:
    """
    Maps an inference endpoint URL to its corresponding health check endpoint.
    """
    # Remove trailing slash for consistent string matching
    endpoint_url = endpoint_url.rstrip('/')
    
    # 1. DJL Serving / SageMaker Style
    if "/invocations" in endpoint_url:
        return endpoint_url.replace("/invocations", "/ping")
    
    # 2. OpenAI / vLLM Style - handle all /v1 routes
    if "/v1/" in endpoint_url:
        # Extract everything up to and including /v1
        # This handles /v1/chat/completions, /v1/completions, /v1/embeddings, etc.
        v1_index = endpoint_url.find("/v1/")
        base_url = endpoint_url[:v1_index]
        # /health is standard for vLLM liveness
        # Optional: Use /v1/models if you want to ensure the model is actually loaded
        return base_url + "/health"
    
    # 3. Triton Inference Server (KServe v2)
    if "/v2/models/" in endpoint_url:
        # We extract the base (scheme + netloc) to get the global server health
        parsed = urlparse(endpoint_url)
        return f"{parsed.scheme}://{parsed.netloc}/v2/health/ready"

    # Default fallback: common convention is /health
    # We split by the last slash to replace the action (e.g., /generate) with /health
    if "/" in endpoint_url:
        return endpoint_url.rsplit('/', 1)[0] + "/health"
    
    return endpoint_url + "/health"

def wait_for_inference_ready(endpoint_url, timeout_seconds=1800, interval=5):
    """
    Polls the health endpoint until the server is ready or timeout is reached.
    """
    health_url = get_health_endpoint(endpoint_url)
    start_time = time.time()
    
    print(f"Checking health at: {health_url}")

    while time.time() - start_time < timeout_seconds:
        try:
            # 1. Added a small request timeout (2s) so the script doesn't hang
            response = requests.get(health_url, timeout=2)
            
            # 2. Check for 200 OK
            if response.status_code == 200:
                print(f"✅ Inference server is up and ready! (Took {int(time.time() - start_time)}s)")
                return True
            
            print(f"⏳ Server returned {response.status_code}, still initializing...")
            
        except RequestException:
            # 3. Specific exception handling (ConnectionError, Timeout, etc.)
            print("📡 Waiting for network connection...")

        time.sleep(interval)

    print("❌ Timeout: Inference server failed to start in time.")
    return False

def get_prompt_generator(config: dict):
    prompt_module_dir = config['module_dir']
    sys.path.append(prompt_module_dir)
    
    prompt_module_name = config['module_name']
    prompt_module=import_module(prompt_module_name)
    
    prompt_generator_name = config['prompt_generator']
    prompt_generator_class = getattr(prompt_module, prompt_generator_name)

    return prompt_generator_class()()
      
def fill_template(template: dict, template_keys:list, inputs:list) -> dict:
        
    assert len(template_keys) <= len(inputs), f"template_keys: {template_keys}, prompts: {inputs}"
    for i, template_key in enumerate(template_keys):
        _template = template
        keys = template_key.split(".")
        for key in keys[:-1]:
            m = re.match(r'\[(\d+)\]', key)
            if m:
                key = int(m.group(1))
            _template = _template[key]

        _template[keys[-1]] = inputs[i]
    
    return template

def inference_request(config: dict):
    prompt_generator = get_prompt_generator(config=config)
    inputs = next(prompt_generator)
    inputs = [inputs] if isinstance(inputs, str) else inputs

    template = config['template']
    assert template is not None

    template_keys = config['template_keys']
    assert template_keys is not None
    
    if "model" in template_keys:
        inputs.insert(0, os.environ['MODEL_ID'])
    data = fill_template(template=template, template_keys=template_keys, inputs=inputs)

    body = json.dumps(data).encode("utf-8")
    pprint(body)
    headers = {"Content-Type":  "application/json"}
    response = requests.post(config['endpoint_url'], data=body, headers=headers)
    return response