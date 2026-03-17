import json
import os
import re
from collections import Counter

# ==================== 配置 ====================
DATA_DIR = "./dataset"

# 需清除 0 标签并统计的数据集
TRAIN_VAL_FILES = [
    "ablation_train_dedup.jsonl",
    "ablation_val.jsonl",
    "rag_train_dedup.jsonl",
    "rag_val.jsonl",
    "ablation_test_llm.jsonl",
    "rag_test_llm.jsonl"
]

# 仅需执行文本替换和 Response 模板扩展，不清除 0 的数据集
TEST_FILES = [
    "ablation_test.jsonl",
    "rag_test.jsonl"
]

# Response 对应的扩展推理模板 (注意：模板陈述部分避免出现除标签外的其他数字，以防干扰正则提取)
RESPONSE_TEMPLATES = {
    "1": "Based on the provided context, the target sentence details the collection of personal information. Therefore, the most appropriate legal category is: Collect Personal Information (CPI)->1",
    "2": "Based on the provided context, the target sentence details precisely how long the collected data is stored on the servers before deletion. Therefore, the most appropriate legal category is: Data Retention Period (DRP)->2",
    "3": "Based on the provided context, the target sentence describes the specific purposes for which the collected personal data is processed. Therefore, the most appropriate legal category is: Data Processing Purposes (DPP)->3",
    "4": "Based on the provided context, the target sentence provides the contact details of the data controller or privacy officer. Therefore, the most appropriate legal category is: Contact Details (CD)->4",
    "5": "Based on the provided context, the target sentence explains the user's right to access their personal data. Therefore, the most appropriate legal category is: Right to Access (RA)->5",
    "6": "Based on the provided context, the target sentence mentions the user's right to rectify inaccurate data or request erasure of their personal data. Therefore, the most appropriate legal category is: Right to Rectify or Erase (RRE)->6",
    "7": "Based on the provided context, the target sentence details the user's right to request the restriction of processing of their personal data. Therefore, the most appropriate legal category is: Right to Restrict of Processing (RRP)->7",
    "8": "Based on the provided context, the target sentence outlines the user's right to object to the processing of their personal data. Therefore, the most appropriate legal category is: Right to Object to Processing (ROP)->8",
    "9": "Based on the provided context, the target sentence specifies the user's right to receive their personal data in a structured, commonly used and machine-readable format. Therefore, the most appropriate legal category is: Right to Data Portability (RDP)->9",
    "10": "Based on the provided context, the target sentence informs the user of their right to lodge a complaint with a supervisory authority. Therefore, the most appropriate legal category is: Right to Lodge a Complaint (RLC)->10"
}

def process_string(text):
    """处理文本字段，替换类别范围并移除 0 类别"""
    if not isinstance(text, str):
        return text
    # 替换 (0-10) 为 (1-10)
    text = text.replace("(0-10)", "(1-10)")
    text = text.replace("0-10", "1-10")
    # 从 Instruction 文本中移除 Other 类别的说明行
    text = text.replace("- 0: Other\n", "")
    return text

def process_dict_fields(d):
    """递归处理字典中的所有字符串字段"""
    for k, v in d.items():
        if isinstance(v, str):
            d[k] = process_string(v)
        elif isinstance(v, dict):
            process_dict_fields(v)
        elif isinstance(v, list):
            d[k] = [process_string(i) if isinstance(i, str) else i for i in v]
    return d

def process_datasets():
    all_files = TRAIN_VAL_FILES + TEST_FILES
    
    for filename in all_files:
        filepath = os.path.join(DATA_DIR, filename)
        if not os.path.exists(filepath):
            print(f"警告: 找不到文件 {filepath}")
            continue
            
        print(f"\n正在处理文件: {filepath} ...")
        is_train_val = filename in TRAIN_VAL_FILES
        
        lines_to_keep = []
        label_counter = Counter()
        removed_count = 0
        
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                    
                # 解析原始 Response 数字标签
                original_response = str(item.get("Response", "")).strip()
                match = re.search(r'\d+', original_response)
                label_key = match.group() if match else original_response
                
                # 要求 1: 直接物理删除训练集和验证集中 Response 为 "0" 的整行数据
                if is_train_val and label_key == "0":
                    removed_count += 1
                    continue  # 直接跳过，不保存该行，实现整行数据的物理删除
                
                # 要求 2: 统计剩余类别的数量
                if is_train_val:
                    label_counter[label_key] += 1
                    
                # 要求 3: 扫描所有字段修改 (0-10) -> (1-10) (包括测试集)
                item = process_dict_fields(item)
                
                # 要求 4: 扩展 Response 为带有推理逻辑的文本 (仅限训练/验证集)
                # 注意：切勿修改测试集的 Response，否则测试脚本中的 int(item["Response"]) 会抛出错误
                if is_train_val and label_key in RESPONSE_TEMPLATES:
                    item["Response"] = RESPONSE_TEMPLATES[label_key]
                    
                lines_to_keep.append(item)
        
        # 将处理后的内容覆写回文件
        with open(filepath, 'w', encoding='utf-8') as f:
            for item in lines_to_keep:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        # 输出打印报告
        print(f"[{filename}] 处理完成！")
        if is_train_val:
            print(f" => 已清除 O 类别 (0) 数据: {removed_count} 条")
            print(f" => 剩余各类别数据量统计:")
            # 按标签数字大小排序打印
            for label in sorted(label_counter.keys(), key=lambda x: int(x) if x.isdigit() else 99):
                print(f"    标签 {label}: {label_counter[label]} 条")
        else:
            print(f" => 测试集处理完成，O 类别 (0) 已保留，模板已扩展。")

if __name__ == "__main__":
    process_datasets()