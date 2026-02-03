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
        self.batch_size = Config.LLM_API_BATCH_SIZE
        self.max_retries = Config.LLM_MAX_RETRIES
        self.timeout = Config.LLM_TIMEOUT
        self.delay = Config.LLM_DELAY
        
        if self.api_key == "your-api-key-here":
            raise ValueError("请在Config中配置LLM_API_KEY和LLM_BASE_URL")
            
        self.base_prompt = self._build_base_prompt_template()
    
    def _build_base_prompt_template(self):
        """构建基础System Prompt，包含标签定义"""
        label_explanations = "\n".join([f"- {k}: {v}" for k, v in Config.LABEL_NAMES.items()])
        
        prompt = f"""You are a professional GDPR privacy policy compliance analysis expert. You need to perform batch classification on the given multiple privacy policy sentences.
Classification Category Definitions (0-10):
{label_explanations}
Task Requirements:
I will provide a set of numbered sentences.
You need to analyze each sentence and determine its corresponding category number (0-10).
The output format must be strictly a JSON integer list, and the order of labels in the list must correspond one-to-one with the order of the input sentences.
Do not output any explanations, code block markers (such as ```json), or other text; only output the list.

Example Input:
We collect your email.
We retain data for 3 years.
Example Output:
[1, 2]"""
        return prompt.strip()
    
    def _build_batch_user_prompt(self, sentences):
        """将一批句子构建为User Prompt"""
        content = "请对以下句子进行分类，返回对应的JSON标签列表：\n\n"
        for i, sent in enumerate(sentences):
            # 去除句子中的换行符，防止破坏格式
            clean_sent = sent.replace('\n', ' ').strip()
            content += f"{i+1}. {clean_sent}\n"
        return content

    def _call_llm_api(self, user_prompt):
        """调用OpenAI兼容接口"""
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        data = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": self.base_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.1,  # 低温以保证格式稳定
            "max_tokens": 10000,   # 足够容纳20个数字的JSON列表
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

    def _parse_batch_response(self, response_text, expected_count):
        """解析API返回的JSON列表"""
        if not response_text:
            return [0] * expected_count
            
        try:
            # 1. 尝试直接解析JSON
            # 清理可能的Markdown标记
            clean_text = response_text.replace("```json", "").replace("```", "").strip()
            labels = json.loads(clean_text)

            # 验证是否为列表且长度匹配
            if isinstance(labels, list):
                # 处理长度不一致的情况（截断或补0）
                if len(labels) > expected_count:
                    labels = labels[:expected_count]
                elif len(labels) < expected_count:
                    print(f"⚠️ Warning: Received {labels} labels, expected {expected_count}. Padding with 0s.")
                    labels.extend([0] * (expected_count - len(labels)))
                
                # 确保都是整数且在范围内
                valid_labels = []
                for l in labels:
                    try:
                        val = int(l)
                        valid_labels.append(val if 0 <= val <= 10 else 0)
                    except:
                        valid_labels.append(0)
                return valid_labels
                
        except json.JSONDecodeError:
            # 2. 如果JSON解析失败，使用正则提取所有数字
            numbers = re.findall(r'\d+', response_text)
            labels = [int(n) for n in numbers]
            
            # 过滤掉不合理的数字（比如可能是编号），这一步比较冒险，建议依赖JSON
            # 这里简单处理：取前 expected_count 个有效范围内的数字
            valid_labels = [x for x in labels if 0 <= x <= 10]
            
            if len(valid_labels) >= expected_count:
                return valid_labels[:expected_count]
            else:
                return valid_labels + [0] * (expected_count - len(valid_labels))
                
        except Exception as e:
            print(f"Parse Error: {e}")
            
        return [0] * expected_count

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
        
        print(f"开始 {self.model_id} 批处理预测 (Batch Size: {self.batch_size})...")
        
        # 2. 批次循环
        num_batches = (total_target + self.batch_size - 1) // self.batch_size
        
        for i in tqdm(range(num_batches), desc="Batch Processing"):
            start_idx = i * self.batch_size
            end_idx = min((i + 1) * self.batch_size, total_target)
            
            batch_sentences = X_target[start_idx:end_idx]
            current_batch_size = len(batch_sentences)
            
            # 构建Prompt
            user_prompt = self._build_batch_user_prompt(batch_sentences)
            
            # 调用API
            response_text = self._call_llm_api(user_prompt)
            
            # 解析结果
            batch_labels = self._parse_batch_response(response_text, current_batch_size)
            predictions.extend(batch_labels)
            
            if effective_delay > 0:
                time.sleep(effective_delay)
                
        # 3. 结果对齐
        full_predictions = np.zeros(total_original, dtype=int)
        for idx, pred in zip(sampled_indices, predictions):
            full_predictions[idx] = int(pred)
            
        print(f"预测完成。成功获取 {len(predictions)} 个标签。")
        return full_predictions, np.array(sampled_indices)