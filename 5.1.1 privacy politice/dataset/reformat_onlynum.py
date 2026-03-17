import json
import os
import re

def extract_label_from_response(response_str):
    """
    从长文本中提取最后的数字标签。
    例如从 "... Data Processing Purposes (DPP)->3" 中提取 "3"
    """
    if not isinstance(response_str, str):
        return str(response_str)
        
    # 优先精确匹配 '->' 后面的数字，因为你的格式是 "... ->3"
    match = re.search(r'->\s*(\d+)', response_str)
    if match:
        return match.group(1)
        
    # 如果部分数据格式稍有不同没有 '->'，退而求其次寻找文本中的最后一个连续数字
    numbers = re.findall(r'\d+', response_str)
    if numbers:
        return numbers[-1]
        
    return response_str

def process_dataset(input_file, output_file):
    """
    逐行读取 jsonl，替换 response 后写出到新文件
    """
    if not os.path.exists(input_file):
        print(f"⚠️ 文件未找到: {input_file}")
        return
        
    processed_count = 0
    with open(input_file, 'r', encoding='utf-8') as fin, \
         open(output_file, 'w', encoding='utf-8') as fout:
        
        for line in fin:
            line = line.strip()
            if not line:
                continue
                
            try:
                data = json.loads(line)
                
                # 兼容字段名首字母大写或小写的情况
                if "Response" in data:
                    data["Response"] = extract_label_from_response(data["Response"])
                elif "response" in data:
                    data["response"] = extract_label_from_response(data["response"])
                    
                # 写回到新文件，保证中文不被 Unicode 转义
                fout.write(json.dumps(data, ensure_ascii=False) + '\n')
                processed_count += 1
                
            except json.JSONDecodeError:
                print(f"JSON解析错误，跳过该行: {line[:50]}...")
                
    print(f"✅ 成功处理 {input_file} -> {output_file}，共清洗了 {processed_count} 条数据。")

if __name__ == "__main__":
    # 定义需要处理的文件列表
    # 请确保这两个文件与此 python 脚本放在同一目录下，或补全绝对路径
    files_to_process = [
        "dataset(RAG).jsonl",
        "dataset(RAG_ablation).jsonl"
    ]
    
    for filename in files_to_process:
        # 为了安全起见，将结果保存到新文件
        output_filename = filename.replace(".jsonl", "_cleaned.jsonl")
        
        print(f"正在处理 {filename} ...")
        process_dataset(filename, output_filename)
