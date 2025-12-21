"""
Custom LiteLLM handler for the endpoint_user.py style API
This adapts the original Locust-based endpoint client to work with LiteLLM proxy
"""
import os
import json
import re
import sys
from importlib import import_module
import httpx
import litellm
from typing import Iterator
from litellm import CustomLLM


class CustomEndpointLLM(CustomLLM):
    """
    Custom LLM handler that mimics the endpoint_user.py behavior
    but integrates with LiteLLM's interface
    """
    
    def __init__(self):
        super().__init__()
        self.content_type = os.getenv("CONTENT_TYPE", "application/json")
        
    def _fill_template(self, template: dict, template_keys:list, inputs:list) -> dict:
        
        assert len(template_keys) == len(inputs), f"template_keys: {template_keys}, prompts: {inputs}"
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
    
    def _messages_to_prompt(self, messages: list) -> str:
        """Convert OpenAI-style messages to a single prompt string"""
        if not messages:
            return ""
        
        # Simple conversion - concatenate all message contents
        prompt_parts = []
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role and content:
                prompt_parts.append(f"{role}: {content}")
            elif content:
                prompt_parts.append(content)
        
        return "\n".join(prompt_parts)
    
    def _prepare_request_data(self, messages: list, model: str, **kwargs) -> dict:
        """
        Prepare the request data by filling the template
        
        Args:
            messages: OpenAI-style messages
            model: Model name
            **kwargs: Additional parameters (temperature, max_tokens, etc.)
        """
        # Get template configuration from environment
        template_str = os.getenv('TEMPLATE', '{}')
        template = json.loads(template_str)
        
        template_keys_str = os.getenv('TEMPLATE_KEYS', '[]')
        template_keys = json.loads(template_keys_str)
        
        if not template or not template_keys:
            raise ValueError(
                "TEMPLATE and TEMPLATE_KEYS environment variables must be set. "
                "Example: TEMPLATE='{\"prompt\":\"\",\"max_tokens\":100}' "
                "TEMPLATE_KEYS='[\"prompt\"]'"
            )
        
        # Prepare inputs based on template keys
        inputs = []
        
        # Convert messages to prompt
        prompt = self._messages_to_prompt(messages)
        inputs.append(prompt)
        
        # Add model if it's in the template keys
        if "model" in template_keys:
            inputs.insert(0, os.getenv("MODEL", model))
        
        # Fill the template with inputs
        data = self._fill_template(
            template=template,
            template_keys=template_keys,
            inputs=inputs
        )
        
        # Merge any additional kwargs that might be in the template
        for key, value in kwargs.items():
            if key in data:
                data[key] = value
        
        return data
    
    async def acompletion(
        self,
        model: str,
        messages: list,
        api_base: str,
        custom_prompt_dict: dict = {},
        **kwargs
    ) -> litellm.ModelResponse:
        """
        Async completion method required by LiteLLM
        
        Args:
            model: Model identifier
            messages: List of message dicts with 'role' and 'content'
            api_base: Base URL for the API endpoint
            custom_prompt_dict: Custom prompt configurations
            **kwargs: Additional parameters (temperature, max_tokens, etc.)
        """
        # Prepare the request payload
        data = self._prepare_request_data(messages, model, **kwargs)
        
        # Make the HTTP request
        headers = {
            "Content-Type": self.content_type
        }
        
        # Add any custom headers from environment
        custom_headers = os.getenv("CUSTOM_HEADERS", "{}")
        if custom_headers:
            try:
                headers.update(json.loads(custom_headers))
            except json.JSONDecodeError:
                pass
        
        print(f"api_base: {api_base}, data: {data}, headers: {headers}")
        async with httpx.AsyncClient(timeout=kwargs.get("timeout", 600)) as client:
            response = await client.post(
                api_base,
                json=data,
                headers=headers
            )
            response.raise_for_status()
            
            response_json = response.json()
            
            # Extract the generated text from the response
            # This depends on your API's response format
            # Common patterns:
            text = self._extract_text_from_response(response_json)
            
            # Create LiteLLM-compatible response
            model_response = litellm.ModelResponse()
            model_response.choices = [
                litellm.Choices(
                    finish_reason="stop",
                    index=0,
                    message=litellm.Message(
                        content=text,
                        role="assistant"
                    )
                )
            ]
            model_response.model = model
            model_response.usage = litellm.Usage(
                prompt_tokens=0,  # Set if available in response
                completion_tokens=0,  # Set if available in response
                total_tokens=0
            )
            
            return model_response
    
    
    def completion(
        self,
        model: str,
        messages: list,
        api_base: str,
        custom_prompt_dict: dict = {},
        **kwargs
    ) -> litellm.ModelResponse:
        """
        Async completion method required by LiteLLM
        
        Args:
            model: Model identifier
            messages: List of message dicts with 'role' and 'content'
            api_base: Base URL for the API endpoint
            custom_prompt_dict: Custom prompt configurations
            **kwargs: Additional parameters (temperature, max_tokens, etc.)
        """
        # Prepare the request payload
        data = self._prepare_request_data(messages, model, **kwargs)
        
        # Make the HTTP request
        headers = {
            "Content-Type": self.content_type
        }
        
        # Add any custom headers from environment
        custom_headers = os.getenv("CUSTOM_HEADERS", "{}")
        if custom_headers:
            try:
                headers.update(json.loads(custom_headers))
            except json.JSONDecodeError:
                pass
        
        print(f"api_base: {api_base}, data: {data}, headers: {headers}")
        with httpx.Client(timeout=kwargs.get("timeout", 600)) as client:
            response = client.post(
                api_base,
                json=data,
                headers=headers
            )
            response.raise_for_status()
            
            response_json = response.json()
            
            # Extract the generated text from the response
            # This depends on your API's response format
            # Common patterns:
            text = self._extract_text_from_response(response_json)
            
            # Create LiteLLM-compatible response
            model_response = litellm.ModelResponse()
            model_response.choices = [
                litellm.Choices(
                    finish_reason="stop",
                    index=0,
                    message=litellm.Message(
                        content=text,
                        role="assistant"
                    )
                )
            ]
            model_response.model = model
            model_response.usage = litellm.Usage(
                prompt_tokens=0,  # Set if available in response
                completion_tokens=0,  # Set if available in response
                total_tokens=0
            )
            
            return model_response
        
    def streaming(
        self,
        model: str,
        messages: list,
        api_base: str,
        custom_prompt_dict: dict,
        **kwargs
    ) -> Iterator[litellm.GenericStreamingChunk]:
        """
        Synchronous streaming method
        """
        # Prepare the request payload
        data = self._prepare_request_data(messages, model, **kwargs)
        data["stream"] = True  # Enable streaming in the request
        
        # Get headers
        # Make the HTTP request
        headers = {
            "Content-Type": self.content_type
        }
        custom_headers = os.getenv("CUSTOM_HEADERS", "{}")
        if custom_headers:
            try:
                headers.update(json.loads(custom_headers))
            except json.JSONDecodeError:
                pass
        
        print(f"api_base: {api_base}, data: {data}, headers: {headers}")
        
        # Make the streaming HTTP request
        with httpx.Client(timeout=kwargs.get("timeout", 600)) as http_client:
            with http_client.stream(
                "POST",
                api_base,
                json=data,
                headers=headers
            ) as response:
                response.raise_for_status()
                
                # Process the streaming response
                for line in response.iter_lines():
                    if not line or line.strip() == "":
                        continue
                    
                    # Handle SSE format (Server-Sent Events)
                    if line.startswith("data: "):
                        line = line[6:]  # Remove "data: " prefix
                    
                    # Skip done signal
                    if line.strip() == "[DONE]":
                        break
                    
                    try:
                        chunk_data = json.loads(line)
                        
                        # Extract text from the chunk
                        text = self._extract_text_from_response(chunk_data)
                        
                        # Create streaming chunk
                        streaming_chunk = litellm.GenericStreamingChunk(
                            text=text,
                            is_finished=chunk_data.get("finish_reason") is not None,
                            finish_reason=chunk_data.get("finish_reason", "stop"),
                            usage=None,
                            index=0,
                            tool_use=None,
                        )
                        
                        yield streaming_chunk
                        
                    except json.JSONDecodeError:
                        # Skip malformed JSON lines
                        continue
    
    async def astreaming(
        self,
        model: str,
        messages: list,
        api_base: str,
        custom_prompt_dict: dict,
        **kwargs
    ):  # Changed: removed the Iterator return type
        """
        Asynchronous streaming method
        """
        # Prepare the request payload
        data = self._prepare_request_data(messages, model, **kwargs)
        data["stream"] = True  # Enable streaming in the request
        
        # Get headers
        headers = {
            "Content-Type": self.content_type
        }
        custom_headers = os.getenv("CUSTOM_HEADERS", "{}")
        if custom_headers:
            try:
                headers.update(json.loads(custom_headers))
            except json.JSONDecodeError:
                pass
        
        print(f"api_base: {api_base}, data: {data}, headers: {headers}")
        
        # Make the streaming HTTP request
        async with httpx.AsyncClient(timeout=kwargs.get("timeout", 600)) as http_client:
            async with http_client.stream(
                "POST",
                api_base,
                json=data,
                headers=headers
            ) as response:
                response.raise_for_status()
                
                # Process the streaming response
                async for line in response.aiter_lines():  # Changed: use aiter_lines() instead of iter_lines()
                    if not line or line.strip() == "":
                        continue
                    
                    # Handle SSE format (Server-Sent Events)
                    if line.startswith("data: "):
                        line = line[6:]  # Remove "data: " prefix
                    
                    # Skip done signal
                    if line.strip() == "[DONE]":
                        break
                    
                    try:
                        chunk_data = json.loads(line)
                        
                        # Extract text from the chunk
                        text = self._extract_text_from_response(chunk_data)
                        
                        # Create streaming chunk
                        streaming_chunk = litellm.GenericStreamingChunk(
                            text=text,
                            is_finished=chunk_data.get("finish_reason") is not None,
                            finish_reason=chunk_data.get("finish_reason", "stop"),
                            usage=None,
                            index=0,
                            tool_use=None,
                        )
                        
                        yield streaming_chunk
                        
                    except json.JSONDecodeError:
                        # Skip malformed JSON lines
                        continue

    def _extract_text_from_response(self, response_json: dict) -> str:
        """
        Extract generated text from the API response
        Configure via RESPONSE_TEXT_PATH environment variable
        
        Default paths to try:
        - output
        - text
        - generated_text
        - choices[0].text
        - choices[0].message.content
        """
        # Try custom path first
        response_path = os.getenv("RESPONSE_TEXT_PATH", "")
        if response_path:
            keys = response_path.split(".")
            result = response_json
            for key in keys:
                m = re.match(r'\[(\d+)\]', key)
                if m:
                    key = int(m.group(1))
                result = result[key]
            return str(result)
        
        # Try common response patterns
        common_paths = [
            "output",
            "text",
            "generated_text",
            "completion",
            "response",
        ]
        
        for path in common_paths:
            if path in response_json:
                return str(response_json[path])
        
        # Try nested paths
        if "choices" in response_json and len(response_json["choices"]) > 0:
            choice = response_json["choices"][0]
            if "text" in choice:
                return str(choice["text"])
            if "message" in choice and "content" in choice["message"]:
                return str(choice["message"]["content"])
        
        # If nothing works, return the whole response as a string
        return json.dumps(response_json)


# Create an instance of the custom LLM
custom_endpoint_llm = CustomEndpointLLM()