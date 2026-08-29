import os
import json
from dotenv import load_dotenv
from openai import OpenAI
from tools import function_register
from tools.registerData import registerData
from tools.dataLoad import dataLoad, load_MDD_edf, load_Sleep_edf
from tools.baseInfo import baseInfo, get_age_factor
from prompt import getSystemPrompt
from utils.parseCalling import extract_tool_calls, has_config_parameter
from utils.messageMerge import messageMerge
import yaml
import time

class EEGAgent:
    def __init__(self, config_path: str, file_name: str):
        """初始化对象状态。"""
        load_dotenv()
        api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("api_key")
        base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        model = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
        if not api_key:
            raise ValueError(
                "Missing DeepSeek API key. Set DEEPSEEK_API_KEY in .env."
            )

        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        with open(os.path.join(self.config['report_template_path'], 'report_template.yaml'), 'r') as file:
            report_template = yaml.safe_load(file)
        self.file_path = os.path.join(self.config['dataPath'], file_name)

        # 加载数据。
        # 配置区。
        data = dataLoad(self.file_path, self.config)
        # 配置区。
        registerData(data)

        # 保留的开发备注。
        self.info = baseInfo(self.file_path)
        self.tool_schemas = function_register.export_tool_schemas()

        # 保留的开发备注。
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        self.model = model
        self.rag_retriever = None

        # 保留的开发备注。
        prior_knowlwdge = self.config['prior knowledge'].copy()
        if 'age' in self.info.keys():
            age_factor = get_age_factor(self.info['age'], self.config['prior knowledge']['Age factor'])
            prior_knowlwdge['Age factor'] = age_factor
        self.system_prompt = getSystemPrompt(self.info, self.tool_schemas, prior_knowlwdge, report_template)
        self.messages = [{'role': 'system', 'content': self.system_prompt}]
    
    def prepare_user_message(self, user_query: str):
        """处理 prepare user message 相关逻辑。"""
        self.messages.append({'role': 'user', 'content': user_query})

    def prepare_rag_user_message(self, user_query: str):
        """Add retrieval evidence to only the current user turn."""
        from RAG.retriever import EEGRetriever, format_temporary_context
        from RAG.retrieval_policy import decide_retrieval

        previous_user_query = next(
            (
                str(message.get("content", ""))
                for message in reversed(self.messages)
                if message.get("role") == "user" and message.get("content")
            ),
            None,
        )
        decision = decide_retrieval(
            user_query,
            has_eeg_session=True,
            previous_user_query=previous_user_query,
        )

        results = []
        if decision.mode != "skip":
            if self.rag_retriever is None:
                self.rag_retriever = EEGRetriever()
            results = self.rag_retriever.retrieve(
                decision.retrieval_query,
                require_faiss_probe=decision.mode == "probe",
            )
        message = {
            "role": "user",
            "content": format_temporary_context(user_query, results),
        }
        self.messages.append(message)
        return message, results

    def call_model(self):
        """处理 call model 相关逻辑。"""
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            # 保留的开发备注。
            timeout=60
        )
        response = completion.choices[0].message.content
        self.messages.append({
            "role": "assistant",
            "content": response
        })
        return response

    def handle_tool_calls(self, response, on_tool_start=None, on_tool_end=None):
        """处理 handle tool calls 相关逻辑。"""
        calls = extract_tool_calls(response)
        if not calls:
            return False 

        function_return = []
        for call in calls:
            function_name = call['name'].strip()
            args = call['args'].copy()
            function = function_register.get_function(function_name)
            if not function:
                print(f"No function named {function_name}")
                continue

            # 配置区。
            if has_config_parameter(function) and 'config' not in args:
                args['config'] = self.config

            try:
                if on_tool_start:
                    on_tool_start(function_name)
                output = function(**args)
                new_call = call.copy()
                new_call['return'] = output
                function_return.append(new_call)
                if on_tool_end:
                    on_tool_end(function_name)
            except Exception as e:
                if on_tool_end:
                    on_tool_end(function_name)
                print(f"{function_name} execution unsuccessful: {e}")
                raise e
        messageMerge(function_return, self.messages)
        return True  

    def run(self, user_query):
        """执行当前任务流程。"""
        user_message, retrieval_results = self.prepare_rag_user_message(user_query)
        try:
            total_rounds = 0
            total_time = 0.0
            local_tool_time = 0.0

            start_total = time.time()
            while True:
                round_start = time.time()
                response = self.call_model()
                round_end = time.time()
                total_rounds += 1
                total_time += (round_end - round_start)

                tool_start = time.time()
                has_more_tools = self.handle_tool_calls(response)
                local_tool_time += (time.time() - tool_start)

                if not has_more_tools:
                    break

            return {
                "response": response,
                "rounds": total_rounds,
                "model_time": total_time,
                "local_tool_time": local_tool_time,
                "total_time": time.time() - start_total,
                "retrieved_sources": [item["source"] for item in retrieval_results],
            }
        finally:
            user_message["content"] = user_query

    def call_model_stream(self, on_delta=None):
        """处理 call model stream 相关逻辑。"""
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            timeout=60,
            stream=True,
        )
        response_parts = []
        for chunk in stream:
            if not chunk.choices:
                continue
            content = chunk.choices[0].delta.content
            if content:
                response_parts.append(content)
                if on_delta:
                    on_delta(content)

        response = "".join(response_parts)
        self.messages.append({"role": "assistant", "content": response})
        return response

    def run_stream(self, user_query, on_delta=None, on_tool_start=None,
                   on_tool_end=None, on_tool_call_detected=None):
        """以流式方式执行当前任务流程。"""
        user_message, retrieval_results = self.prepare_rag_user_message(user_query)
        try:
            total_rounds = 0
            total_time = 0.0
            local_tool_time = 0.0
            start_total = time.time()

            while True:
                round_start = time.time()
                response = self.call_model_stream(on_delta=on_delta)
                total_time += time.time() - round_start
                total_rounds += 1

                if extract_tool_calls(response) and on_tool_call_detected:
                    on_tool_call_detected()

                tool_start = time.time()
                has_more_tools = self.handle_tool_calls(
                    response,
                    on_tool_start=on_tool_start,
                    on_tool_end=on_tool_end,
                )
                local_tool_time += time.time() - tool_start
                if not has_more_tools:
                    break

            return {
                "response": response,
                "rounds": total_rounds,
                "model_time": total_time,
                "local_tool_time": local_tool_time,
                "total_time": time.time() - start_total,
                "retrieved_sources": [item["source"] for item in retrieval_results],
            }
        finally:
            user_message["content"] = user_query
        

if __name__ == "__main__":
    agent = EEGAgent(
        config_path="config/config.json",
        file_name="data/gped_049_a_6.edf",
    )
    user_question = "Can epileptic discharges be observed within the first minute? If so, where?"
    print("Human:", user_question)
    result = agent.run(user_question)
    print("Assistant:", result["response"])
    print("耗时:", round(result["total_time"], 2), "秒")
