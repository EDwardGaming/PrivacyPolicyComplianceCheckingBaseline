from Config import Config
import requests
import time
import numpy as np
import json
import re
from tqdm import tqdm

# ==================== Llama 批处理分类器 ====================
class LlamaBatchClassifier:
    """
    Llama API 分类器 - 支持批处理 (Batch Inference)
    使用 meta/meta-llama-3.1-405b-instruct 模型
    """
    
    def __init__(self):
        self.api_key = Config.LLM_API_KEY
        self.base_url = Config.LLM_BASE_URL
        self.model_id = Config.LLM_MODEL_ID
        self.batch_size = 1
        self.max_retries = Config.LLM_MAX_RETRIES
        self.timeout = Config.LLM_TIMEOUT
        self.delay = Config.LLM_DELAY
        
        if self.api_key == "your-api-key-here":
            raise ValueError("请在Config中配置LLM_API_KEY和LLM_BASE_URL")
            
        self.base_prompt = self._build_base_prompt_template()
    
    def _build_base_prompt_template(self):
        """构建基础System Prompt，包含标签定义"""
        label_explanations = "\n".join([f"- {k}: {v}" for k, v in Config.LABEL_NAMES.items()])
        
        prompt = f"""You are a professional GDPR privacy policy compliance analysis expert. You need to classify the given privacy policy sentence.
Classification Category Definitions (0-10):
{label_explanations}
Task Requirements:
I will provide a sentence.
You need to analyze the sentence and determine its corresponding category number (0-10).
The output format must be strictly a single integer.
Do not output any explanations, code block markers, or other text; only output the integer."""
        return prompt.strip()
    
    def _build_user_prompt(self, sentence):
        """构建单条User Prompt"""
        clean_sent = sentence.replace('\n', ' ').strip()
        return f"Sentence: {clean_sent}"

    def _call_llm_api(self, user_prompt, system_prompt=None):
        """调用OpenAI兼容接口"""
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        data = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": system_prompt if system_prompt else self.base_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.1,  # 低温以保证格式稳定
            "max_tokens": 99999,   # 单个数字
            "stream": False
        }
        
        for retry in range(self.max_retries):
            try:
                response = requests.post(url, headers=headers, json=data, timeout=self.timeout)
                
                if response.status_code == 200:
                    result = response.json()
                    if 'choices' in result and len(result['choices']) > 0:
                        choice = result['choices'][0]
                        if choice.get("finish_reason") == "MAX_TOKENS":
                            print("⚠️ Warning: LLM output truncated by MAX_TOKENS")
                        return result['choices'][0]['message']['content'].strip()
                else:
                    print(f"API Error {response.status_code}: {response.text}")
                    time.sleep(2)
                    
            except Exception as e:
                print(f"Request Exception: {e}")
                time.sleep(2)
                
        return None

    def _parse_response(self, response_text):
        """解析API返回的单个整数"""
        if not response_text:
            return 0
            
        try:
            # 清理可能的Markdown标记和空白
            clean_text = response_text.replace("```json", "").replace("```", "").strip()
            # 提取数字
            numbers = re.findall(r'\d+', clean_text)
            if numbers:
                val = int(numbers[0])
                return val if 0 <= val <= 10 else 0
        except Exception as e:
            print(f"Parse Error: {e}")
            
        return 0

    def predict(self, X_test, sample_size=None, delay=None):
        """执行批处理预测"""
        predictions = []
        
        # 1. 准备数据
        X_test_arr = np.asarray(X_test)
        total_original = len(X_test_arr)

        if sample_size is not None and total_original > sample_size:
            rng = np.random.RandomState(Config.RANDOM_STATE)
            indices = rng.choice(total_original, sample_size, replace=False)
            X_target = X_test_arr[indices]
            sampled_indices = indices
            print(f"已抽样: {sample_size} 条数据")
        else:
            X_target = X_test_arr
            sampled_indices = np.arange(total_original)
            
        total_target = len(X_target)
        effective_delay = delay if delay is not None else self.delay
        
        print(f"开始 {self.model_id} 逐条预测 (Total: {total_target})...")
        
        # 2. 逐条循环
        for i in tqdm(range(total_target), desc="Processing"):
            sentence = X_target[i]
            
            # 构建Prompt
            user_prompt = self._build_user_prompt(sentence)
            
            # 调用API
            response_text = self._call_llm_api(user_prompt,system_prompt=self.base_prompt)
            
            # 解析结果
            label = self._parse_response(response_text)
            predictions.append(label)
            
            if effective_delay > 0:
                time.sleep(effective_delay)
                
        # 3. 结果对齐
        full_predictions = np.zeros(total_original, dtype=int)
        for idx, pred in zip(sampled_indices, predictions):
            full_predictions[idx] = int(pred)
            
        print(f"预测完成。成功获取 {len(predictions)} 个标签。")
        return full_predictions, np.array(sampled_indices)