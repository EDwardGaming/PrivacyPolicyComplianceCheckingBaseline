import os
import torch
import pandas as pd
import numpy as np
import json
import re
from sklearn.metrics import precision_recall_fscore_support
from datasets import load_dataset
from unsloth import FastLanguageModel
from tqdm import tqdm

# ==================== 配置参数 ====================
class Config:
    # 路径配置
    # 注意：请确保已挂载 Google Drive 并且数据集已上传到相应目录
    DATA_PATH = "./dataset/data.tsv"
    RAG_TEST_PATH = "./dataset/rag_test.jsonl"

    MODEL_PATH = "./llama_finetune_gdpr"
    
    # 模型参数
    MAX_SEQ_LENGTH = 2048
    DTYPE = None # 自动检测
    LOAD_IN_4BIT = True
    
    # 标签定义
    LABEL_NAMES = {
        0: "Other",
        1: "Collect Personal Information (CPI)",
        2: "Data Retention Period (DRP)",
        3: "Data Processing Purposes (DPP)",
        4: "Contact Details (CD)",
        5: "Right to Access (RA)",
        6: "Right to Rectify or Erase (RRE)",
        7: "Right to Restrict of Processing (RRP)",
        8: "Right to Object to Processing (ROP)",
        9: "Right to Data Portability (RDP)",
        10: "Right to Lodge a Complaint (RLC)"
    }
    
    # 合规性规则
    COMPLIANCE_RULES = {
        "Rule 1": (1, 2),   # CPI -> DRP
        "Rule 2": (1, 3),   # CPI -> DPP
        "Rule 3": (1, 4),   # CPI -> CD
        "Rule 4": (1, 5),   # CPI -> RA
        "Rule 5": (1, 6),   # CPI -> RRE
        "Rule 6": (1, 7),   # CPI -> RRP
        "Rule 7": (1, 8),   # CPI -> ROP
        "Rule 8": (1, 9),   # CPI -> RDP
        "Rule 9": (1, 10),  # CPI -> RLC
    }

# ==================== 评估函数 ====================
def evaluate_classification(y_true, y_pred):
    """评估分类性能 (Precision, Recall, F1)"""
    print(f"\n{'='*60}")
    print(f"分类性能评估 (Classification Evaluation)")
    print(f"{'='*60}")
    
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(11)), average=None, zero_division=0
    )
    
    results = []
    for i in range(11):
        results.append({
            'Label': Config.LABEL_NAMES[i],
            'Precision': f"{precision[i]:.4f}",
            'Recall': f"{recall[i]:.4f}",
            'F1-Score': f"{f1[i]:.4f}",
            'Support': support[i]
        })
    
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    
    non_other_mask = np.arange(11) != 0
    avg_precision = precision[non_other_mask].mean()
    avg_recall = recall[non_other_mask].mean()
    avg_f1 = f1[non_other_mask].mean()
    
    print(f"\n{'='*60}")
    print(f"10个GDPR标签的平均指标 (Average Metrics for 10 GDPR Tags):")
    print(f"Precision: {avg_precision:.4f}")
    print(f"Recall:    {avg_recall:.4f}")
    print(f"F1-Score:  {avg_f1:.4f}")
    print(f"{'='*60}\n")

def check_compliance_violations(label_set):
    """检查单个文档的标签集合是否违反规则"""
    violations = []
    if 1 in label_set:
        for rule_name, (antecedent, consequent) in Config.COMPLIANCE_RULES.items():
            if consequent not in label_set:
                violations.append(rule_name)
    return violations

def evaluate_compliance(y_true, y_pred, filenames):
    """执行合规性检测评估 (基于文档聚合)"""
    print(f"\n{'='*60}")
    print("合规性检测评估 (Compliance Detection Evaluation)")
    print(f"{'='*60}")
    
    file_data = {}
    for i, filename in enumerate(filenames):
        if i >= len(y_pred): break
        if filename not in file_data:
            file_data[filename] = {'true': [], 'pred': []}
        file_data[filename]['true'].append(y_true[i])
        file_data[filename]['pred'].append(y_pred[i])
    
    print(f"评估文档总数: {len(file_data)}")
    
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    total_true_violations = 0
    
    for fname, data in file_data.items():
        true_set = set(data['true'])
        pred_set = set(data['pred'])
        
        true_violations = set(check_compliance_violations(true_set))
        pred_violations = set(check_compliance_violations(pred_set))
        
        total_true_violations += len(true_violations)
        
        tp = len(true_violations & pred_violations)
        fp = len(pred_violations - true_violations)
        fn = len(true_violations - pred_violations)
        
        true_positives += tp
        false_positives += fp
        false_negatives += fn
        
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"真实违规总数: {total_true_violations}")
    print(f"合规性 Precision: {precision:.4f}")
    print(f"合规性 Recall:    {recall:.4f}")
    print(f"合规性 F1-Score:  {f1:.4f}")
    print(f"{'='*60}\n")

# ==================== 主程序 ====================
def main():
    # 设置工作目录
    print(f"Current working directory: {os.getcwd()}")

    # 1. 加载微调后的模型
    print(f">>> Loading Fine-tuned Llama 3.1 8B model from {Config.MODEL_PATH}...")
    
    try:
        # Unsloth 会自动处理 adapter 的加载
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = Config.MODEL_PATH,
            max_seq_length = Config.MAX_SEQ_LENGTH,
            dtype = Config.DTYPE,
            load_in_4bit = Config.LOAD_IN_4BIT,
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please ensure Google Drive is mounted and the path is correct.")
        return

    # 启用原生推理加速
    FastLanguageModel.for_inference(model)
    
    # 2. 准备 Prompt 模板
    alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

    EOS_TOKEN = tokenizer.eos_token

    # 3. 加载测试数据
    print(f">>> Loading test dataset from {Config.RAG_TEST_PATH}...")
    if not os.path.exists(Config.RAG_TEST_PATH):
        print(f"Error: Test file {Config.RAG_TEST_PATH} not found.")
        print("Please ensure the dataset is uploaded to the correct location.")
        return
        
    test_dataset = load_dataset("json", data_files={"test": Config.RAG_TEST_PATH}, split="test")
    
    predictions = []
    true_labels = []
    filenames = []
    
    print(">>> Starting inference on test set...")
    
    # 4. 逐条推理
    for item in tqdm(test_dataset, desc="Inference"):
        instruction = item["Instruction"]
        input_text = item["Input"]
        filename = item.get("filename", "unknown")
        true_label = int(item["Response"])
        
        # 构建 Prompt (Response 部分留空)
        prompt = alpaca_prompt.format(instruction, input_text, "")
        
        inputs = tokenizer([prompt], return_tensors = "pt").to("cuda")
        
        # 生成
        outputs = model.generate(**inputs, max_new_tokens = 10, use_cache = True)
        
        # 解码
        decoded = tokenizer.batch_decode(outputs)[0]
        
        # 提取 Response 部分
        response_part = decoded.split("### Response:\n")[-1].replace(EOS_TOKEN, "").strip()
        
        # 提取数字
        numbers = re.findall(r'\d+', response_part)
        if numbers:
            pred = int(numbers[0])
            # 确保在 0-10 范围内
            pred = max(0, min(10, pred))
        else:
            print(f"\n[Warning] 解析失败: 模型未回答数字。回答内容: '{response_part}'")
            pred = 0 # 解析失败默认为 0 (Other)
            
        predictions.append(pred)
        true_labels.append(true_label)
        filenames.append(filename)

    # 5. 执行评估
    y_true = np.array(true_labels)
    y_pred = np.array(predictions)
    
    # 5.1 分类指标评估
    evaluate_classification(y_true, y_pred)
    
    # 5.2 合规性检测评估
    try:
        evaluate_compliance(y_true, y_pred, filenames)
    except Exception as e:
        print(f"合规性评估失败: {e}")

if __name__ == "__main__":
    main()
