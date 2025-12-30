import copy
from importlib import import_module
import re
import time
import os
import json
import sys

from locust.contrib.fasthttp import FastHttpUser
from locust import task, events
import requests


class EndpointClient:
    def __init__(self, url):
        
        self.url = url
        self.content_type =os.getenv("CONTENT_TYPE", "application/json")
        self.__init_prompt_generator()
    
    def __init_prompt_generator(self):
        prompt_module_dir = os.getenv("PROMPT_MODULE_DIR", "")
        sys.path.append(prompt_module_dir)
        
        prompt_module_name = os.getenv("PROMPT_MODULE_NAME", None)
        prompt_module=import_module(prompt_module_name)
        
        prompt_generator_name = os.getenv('PROMPT_GENERATOR_NAME', None)
        prompt_generator_class = getattr(prompt_module, prompt_generator_name)
        self.text_input_generator = prompt_generator_class()()

        self.kwargs = json.loads(os.getenv('PROMPT_KWARGS', "{}"))

    def _fill_template(self, template: dict, template_keys:list, inputs:list) -> dict:
        
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

    def __inference_request(self, request_meta:dict):
        inputs = next(self.text_input_generator)
        template = os.getenv('TEMPLATE', "{}")
        template = json.loads(template)
        assert template

        template_keys = os.getenv('TEMPLATE_KEYS', "[]")
        template_keys = json.loads(template_keys)
        assert template_keys

        if "model" in template_keys:
            inputs.insert(0, os.getenv("MODEL", ""))
        data = self._fill_template(template=template, template_keys=template_keys, inputs=inputs)

        body = json.dumps(data).encode("utf-8")
        headers = {"Content-Type":  self.content_type}
        response = requests.post(self.url, data=body, headers=headers)
        request_meta['response'] = { "status_code": response.status_code }
        request_meta['response_length'] = len(response.json())

    def send(self):

        request_meta = {
            "request_type": "Post",
            "name": "Local",
            "start_time": time.time(),
            "response_length": 0,
            "response": None,
            "context": {},
            "exception": None,
        }
        start_perf_counter = time.perf_counter()

        try: 
            self.__inference_request(request_meta)
        except StopIteration as se:
            self.__init_prompt_generator()
            request_meta["exception"] = se
        except Exception as e:
            request_meta["exception"] = e

        request_meta["response_time"] = (
            time.perf_counter() - start_perf_counter
        ) * 1000

        events.request.fire(**request_meta)


class EndpointUser(FastHttpUser):
    abstract = True

    def __init__(self, env):
        super().__init__(env)
        self.client = EndpointClient(self.host)


class LocalEndpointUser(EndpointUser):
    @task
    def send_request(self):
        self.client.send()