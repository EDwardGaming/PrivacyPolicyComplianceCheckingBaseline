import os
import torch
import pandas as pd
import numpy as np
import json
import re
from sklearn.metrics import precision_recall_fscore_support
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from unsloth import FastLanguageModel

# ==================== 配置参数 ====================
class Config:
    # 路径配置
    RAG_TEST_PATH = "./dataset/rag_test.jsonl"
    LLAMA_MODEL_PATH = "/content/drive/MyDrive/llama_finetune_gdpr"
    DEBERTA_MODEL_PATH = "./deberta_model"
    
    # 模型参数
    MAX_SEQ_LENGTH = 2048
    DTYPE = None
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
    
    # 合规性规则 (CPI 触发其他义务)
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
    print(f"\n{'='*60}\n分类性能评估 (Classification Evaluation)\n{'='*60}")
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(11)), average=None, zero_division=0)
    results = []
    for i in range(11):
        results.append({
            'Label': Config.LABEL_NAMES[i], 'Precision': f"{precision[i]:.4f}",
            'Recall': f"{recall[i]:.4f}", 'F1-Score': f"{f1[i]:.4f}", 'Support': support[i]
        })
    print(pd.DataFrame(results).to_string(index=False))
    
    non_other_mask = np.arange(11) != 0
    print(f"\n{'='*60}\n10个GDPR标签的平均指标 (Average Metrics for 10 GDPR Tags):")
    print(f"Precision: {precision[non_other_mask].mean():.4f}")
    print(f"Recall:    {recall[non_other_mask].mean():.4f}")
    print(f"F1-Score:  {f1[non_other_mask].mean():.4f}\n{'='*60}\n")

def evaluate_compliance(y_true, y_pred, filenames):
    print(f"\n{'='*60}\n合规性检测评估 (Compliance Detection Evaluation)\n{'='*60}")
    file_data = {}
    for i, filename in enumerate(filenames):
        if i >= len(y_pred): break
        if filename not in file_data: file_data[filename] = {'true': [], 'pred': []}
        file_data[filename]['true'].append(y_true[i])
        file_data[filename]['pred'].append(y_pred[i])
    
    print(f"评估文档总数: {len(file_data)}")
    tp, fp, fn, total_true = 0, 0, 0, 0
    
    def check_violations(label_set):
        return [rule for rule, (_, cons) in Config.COMPLIANCE_RULES.items() if cons not in label_set] if 1 in label_set else []
        
    for fname, data in file_data.items():
        true_v = set(check_violations(set(data['true'])))
        pred_v = set(check_violations(set(data['pred'])))
        total_true += len(true_v)
        tp += len(true_v & pred_v)
        fp += len(pred_v - true_v)
        fn += len(true_v - pred_v)
        
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    
    print(f"真实违规总数: {total_true}\n合规性 Precision: {prec:.4f}")
    print(f"合规性 Recall:    {rec:.4f}\n合规性 F1-Score:  {f1_score:.4f}\n{'='*60}\n")

# ==================== 模型推理类 ====================

class DeBERTaFilter:
    def __init__(self, model_path=Config.DEBERTA_MODEL_PATH):
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        
        threshold_path = os.path.join(model_path, "optimal_threshold.txt")
        if os.path.exists(threshold_path):
            with open(threshold_path, "r") as f:
                self.optimal_threshold = float(f.read().strip())
        else:
            print("Warning: optimal_threshold.txt not found, using default 0.5")
            self.optimal_threshold = 0.5
    
    def predict(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=1).cpu().numpy()[0]
        
        return 1 if probabilities[1] >= self.optimal_threshold else 0

class LlamaClassifier:
    def __init__(self, model_path=Config.LLAMA_MODEL_PATH):
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=Config.MAX_SEQ_LENGTH,
            dtype=Config.DTYPE,
            load_in_4bit=Config.LOAD_IN_4BIT,
        )
        FastLanguageModel.for_inference(self.model)
        self.eos_token = self.tokenizer.eos_token
        self.alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{}\n\n### Input:\n{}\n\n### Response:\n{}"""
    
    def predict(self, instruction, input_text):
        prompt = self.alpaca_prompt.format(instruction, input_text, "")
        inputs = self.tokenizer([prompt], return_tensors="pt").to("cuda")
        
        outputs = self.model.generate(**inputs, max_new_tokens=10, use_cache=True)
        decoded = self.tokenizer.batch_decode(outputs)[0]
        
        response_part = decoded.split("### Response:\n")[-1].replace(self.eos_token, "").strip()
        
        numbers = re.findall(r'\d+', response_part)
        if numbers:
            pred = int(numbers[0])
            return max(0, min(10, pred))
        else:
            print(f"Warning: No valid number found in Llama response. Response was: '{response_part}' Returning 0 by default.")
            return 0

def main():
    print(f"Working directory set to: {os.getcwd()}")
    
    print(">>> Stage 1: Loading DeBERTa Filter...")
    deberta_filter = DeBERTaFilter()
    
    print(f">>> Stage 2: Loading Fine-tuned Llama Classifier from {Config.LLAMA_MODEL_PATH}...")
    try:
        llama_classifier = LlamaClassifier()
    except Exception as e:
        print(f"Error loading Llama model: {e}\nEnsure model path in Config is correct.")
        return
    
    print(f">>> Loading test dataset from {Config.RAG_TEST_PATH}...")
    test_dataset = load_dataset("json", data_files={"test": Config.RAG_TEST_PATH}, split="test")
    
    predictions, true_labels, filenames = [], [], []
    
    print(">>> Starting Two-Stage Inference on test set...")
    for item in tqdm(test_dataset, desc="Inference"):
        instruction = item["Instruction"]
        input_text = item["Input"]
        filename = item.get("filename", "unknown")
        true_label = int(item["Response"])
        
        # ================= 两阶段推理 =================
        # 第一阶段: 使用 DeBERTa 判断是否为 Other (0)
        stage1_pred = deberta_filter.predict(input_text)
        if stage1_pred == 0:
            pred = 0
        else:
            # 第二阶段: 是潜在隐私政策内容时，送入 Llama 判断类别 (1-10)
            pred = llama_classifier.predict(instruction, input_text)
            
        predictions.append(pred)
        true_labels.append(true_label)
        filenames.append(filename)
        
    y_true = np.array(true_labels)
    y_pred = np.array(predictions)
    
    # 6. 执行评估 (分类性能 & 合规检测)
    evaluate_classification(y_true, y_pred)
    try:
        evaluate_compliance(y_true, y_pred, filenames)
    except Exception as e:
        print(f"合规性评估失败: {e}")

if __name__ == "__main__":
    main()