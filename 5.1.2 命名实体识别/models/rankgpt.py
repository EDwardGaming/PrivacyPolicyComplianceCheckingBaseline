import requests
import time
import json
from tqdm import tqdm
from .config import gptConfig

class GPTNER:
    """
    使用 GPT 模型进行命名实体识别
    """

    def __init__(self):
        self.api_key = gptConfig.LLM_API_KEY
        self.base_url = gptConfig.LLM_BASE_URL
        self.model_id = gptConfig.LLM_MODEL_ID
        self.max_retries = gptConfig.LLM_MAX_RETRIES
        self.timeout = gptConfig.LLM_TIMEOUT

        if self.api_key == "your-api-key-here":
            raise ValueError("请在Config中配置LLM_API_KEY和LLM_BASE_URL")

        self.base_prompt = self._build_base_prompt_template()

    def _build_base_prompt_template(self):
        prompt = f"""You are an expert in Named Entity Recognition (NER) for legal texts, specifically privacy policies.
        Task: You will be given only 1 sentences. 
        For this sentence:
            - Assign a tag to EACH token. The number of tags must exactly match the number of tokens.
            - Allowed entity types are: `data`, `handler`, `purpose`, `condition`, `collect`, `share`, `subjects`.
            - Use the BIOES tagging scheme:
                - tag list: ['O', 'B-data', 'I-data', 'E-data', 'B-handler', 'E-handler', 'B-purpose', 'I-purpose', 'E-purpose', 'B-subjects', 'I-subjects', 'B-share', 'E-share', 'B-condition', 'I-condition', 'E-condition', 'B-collect', 'E-collect', 'E-subjects', 'I-handler', 'I-share', 'I-collect', 'S-collect', 'S-handler', 'S-subjects', 'S-purpose', 'S-data', 'S-condition']
            - Your output must be a valid python list[str], where each inner list contains the tags for one sentence.

        Rules:
            - ONLY output the list. Do not include any other text, explanations, or code blocks like ```json.
            - The number of tags in each list must equal the number of tokens in the corresponding input sentence.
            - Do not skip or merge any tokens. Every input token, including punctuation and spaces, needs a corresponding tag.
            - Do NOT repeat the input sentence.

        Example:
        - Input: 当你使用微光APP时
        - Output: ['B-condition', 'I-condition', 'I-condition', 'I-condition', 'I-condition', 'I-condition', 'I-condition', 'I-condition', 'I-condition', 'E-condition']

        You need to answer the following input tokens:
        """
        return prompt.strip()


    def _call_gpt_api(self, sentences):
        """调用OpenAI兼容接口"""
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        user_prompt = self._build_user_prompt(sentences)
        data = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": self.base_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0,
            "max_tokens": 50000,
            "stream": False
        }

        for retry in range(self.max_retries):
            try:
                response = requests.post(url, headers=headers, json=data, timeout=self.timeout)

                if response.status_code == 200:
                    result = response.json()
                    if 'choices' in result and len(result['choices']) > 0:
                        return result['choices'][0]['message']['content'].strip()
                else:
                    print(f"API Error {response.status_code}: {response.text}")
                    time.sleep(2)

            except Exception as e:
                print(f"Request Exception: {e}")
                time.sleep(2)

        return None

    def _build_user_prompt(self, batch_tokens):
        """
        batch_tokens: List[List[str]]
        """
        content = ""
        for i, tokens in enumerate(batch_tokens):
            token_str = ''.join(tokens)
            content += f"{token_str}"
        return content

    def predict(self, tokens, start_index):
        """
        tokens: List[str]
        start_index: int, current batch start index for debugging
        return: List[str]
        """
        # 包装成列表以适配 _call_gpt_api
        batch_tokens = [tokens]
        response_text = self._call_gpt_api(batch_tokens)

        if not response_text:
            print("⚠️ API 返回为空")
            print(f"样本 {start_index} 执行失败将使用全O返回")
            return ["O"] * len(tokens)

        # Clean up common LLM output formatting
        response_text = response_text.replace("```json", "").replace("```", "").strip()

        try:
            result_tags = eval(response_text)
             
            # 长度校验
            if len(result_tags) != len(tokens):
                print(f"样本 {start_index} 执行不正常: 长度不一致")
                print(f"输入长度: {len(tokens)}, 预测长度: {len(result_tags)}")
                if len(result_tags) > len(tokens):
                    print("⚠️ 预测标签过长，将进行截断。")
                    result_tags = result_tags[:len(tokens)]
                else:
                    print("⚠️ 预测标签过短，将使用'O'标签补全。")
                    result_tags = result_tags + ["O"] * (len(tokens) - len(result_tags))
                    print(result_tags)
            else:
                print(f"样本 {start_index} 执行正常")
                
            return result_tags

        except Exception as e:
            # 尝试修复
            print(f"样本 {start_index} 执行失败尝试修复ing")
            repaired_text = response_text
            
            if repaired_text.startswith("[") and not repaired_text.endswith("]"):
                repaired_text += "]"
            
            try:
                pred = eval(repaired_text)
                print(f"✅修复成功。")
                return pred
            except:
                print(f"修复失败将使用全O返回")
                return ["O"] * len(tokens)