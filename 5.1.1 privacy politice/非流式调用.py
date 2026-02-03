import requests
import os
import json

def call_ark_chat_stream():
    url = "https://ark.cn-beijing.volces.com/api/v3/chat/completions"
    ark_api_key = os.getenv("ARK_API_KEY")
    if not ark_api_key:
        raise ValueError("未找到ARK_API_KEY环境变量，请先设置")

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {ark_api_key}"
    }

    # 核心区别：添加 "stream": true 参数
    data = {
        "model": "doubao-1-5-pro-32k-250115",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"}
        ],
        "stream": True  # 开启流式返回
    }

    try:
        # 发送流式请求：stream=True 保持连接不关闭
        response = requests.post(
            url=url,
            headers=headers,
            json=data,
            stream=True,  # 关键参数：开启流式响应
            timeout=60    # 流式调用超时时间可以设长一点
        )
        response.raise_for_status()

        print("流式回复实时输出：", end="", flush=True)
        full_content = ""  # 用于拼接完整回复

        # 逐行解析流式响应
        for line in response.iter_lines():
            if line:
                # 去掉前缀 "data: "，解析JSON
                line_data = line.decode("utf-8").lstrip("data: ")
                # 跳过空数据或结束标记
                if line_data in ["", "[DONE]"]:
                    continue
                # 解析单条片段数据
                chunk = json.loads(line_data)
                # 提取当前片段的回复内容
                delta = chunk["choices"][0]["delta"]
                if "content" in delta:
                    content_chunk = delta["content"]
                    full_content += content_chunk
                    # 实时打印（flush=True 强制刷新输出，避免缓存）
                    print(content_chunk, end="", flush=True)

        print(f"\n\n完整回复内容：{full_content}")
        return full_content

    except Exception as e:
        print(f"\n调用出错：{e}")

if __name__ == "__main__":
    call_ark_chat_stream()