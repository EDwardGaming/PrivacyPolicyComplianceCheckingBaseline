import json

# 定义你的JSON字符串（注意Python中多行字符串用三引号）
json_str = '''{
  "choices": [
    {
      "finish_reason": "stop",
      "index": 0,
      "logprobs": null,
      "message": {
        "content": "Hello! How can I help you today?",
        "role": "assistant"
      }
    }
  ],
  "created": 1742631811,
  "id": "0217426318107460cfa43dc3f3683b1de1c09624ff49085a456ac",
  "model": "doubao-1-5-pro-32k-250115",
  "service_tier": "default",
  "object": "chat.completion",
  "usage": {
    "completion_tokens": 9,
    "prompt_tokens": 19,
    "total_tokens": 28,
    "prompt_tokens_details": {
      "cached_tokens": 0
    },
    "completion_tokens_details": {
      "reasoning_tokens": 0
    }
  }
}'''

try:
    # 1. 解析JSON字符串为Python字典（核心步骤）
    data = json.loads(json_str)
    
    # 2. 提取关键信息（按层级索引）
    # 提取助手的回复内容（最核心）
    assistant_content = data["choices"][0]["message"]["content"]
    print(f"助手回复内容：{assistant_content}")
    
    # 提取回复的角色
    assistant_role = data["choices"][0]["message"]["role"]
    print(f"回复角色：{assistant_role}")
    
    # 提取模型名称
    model_name = data["model"]
    print(f"使用的模型：{model_name}")
    
    # 提取Token使用量
    total_tokens = data["usage"]["total_tokens"]
    prompt_tokens = data["usage"]["prompt_tokens"]
    completion_tokens = data["usage"]["completion_tokens"]
    print(f"总Token数：{total_tokens}（提示词：{prompt_tokens}，回复：{completion_tokens}）")
    
    # 提取创建时间（Unix时间戳，可转换为普通时间）
    import datetime
    create_time = datetime.datetime.fromtimestamp(data["created"])
    print(f"回复创建时间：{create_time}")

except json.JSONDecodeError as e:
    print(f"JSON格式错误：{e}")
except KeyError as e:
    print(f"键不存在：{e}，请检查JSON结构是否正确")