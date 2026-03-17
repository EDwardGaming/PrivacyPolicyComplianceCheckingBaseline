import os
import pandas as pd
import numpy as np
import re
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, classification_report
from datasets import load_dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from tqdm import tqdm

# ==================== 配置参数 ====================
class Config:
    # 路径配置
    DATA_PATH = "./dataset/data.tsv"
    ALL_DATA_PATH = "./dataset/dataset(RAG-LLM).jsonl"
    TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_DIR = f"./llama_finetune_gdpr_{TIMESTAMP}"
    
    # 数据划分参数 (需与生成数据集时保持一致)
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    
    # 模型参数
    MAX_SEQ_LENGTH = 2048 # 根据您的数据长度估算，2048 足够
    DTYPE = None # 自动检测 (Float16 或 Bfloat16)
    LOAD_IN_4BIT = True # 4bit 量化以节省显存
    
    # 标签定义
    LABEL_NAMES = {
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
    """评估分类性能 (Precision, Recall, F1)"""
    print(f"\n{'='*60}")
    print(f"分类性能评估 (Classification Evaluation)")
    print(f"{'='*60}")
    
    labels_to_eval = list(range(1, 11))
    target_names = [Config.LABEL_NAMES[i] for i in labels_to_eval]
    # 额外输出包含每个单独标签性能指标的详细分类报告
    print("\n详细分类报告 (Detailed Classification Report):")
    print(classification_report(y_true, y_pred, target_names=target_names, labels=labels_to_eval, zero_division=0))
    print(f"\n{'-'*60}")
    
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels_to_eval, average=None, zero_division=0
    )
    
    results = []
    for i, label_id in enumerate(labels_to_eval):
        results.append({
            'Label': Config.LABEL_NAMES[label_id],
            'Precision': f"{precision[i]:.4f}",
            'Recall': f"{recall[i]:.4f}",
            'F1-Score': f"{f1[i]:.4f}",
            'Support': support[i]
        })
    
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    
    avg_precision = precision.mean()
    avg_recall = recall.mean()
    avg_f1 = f1.mean()
    
    print(f"\n{'='*60}")
    print(f"10个GDPR标签的平均指标 (Average Metrics for 10 GDPR Tags):")
    print(f"Precision: {avg_precision:.4f}")
    print(f"Recall:    {avg_recall:.4f}")
    print(f"F1-Score:  {avg_f1:.4f}")
    print(f"{'='*60}\n")

def check_compliance_violations(label_set):
    """检查单个文档的标签集合是否违反规则"""
    violations = []
    if 1 in label_set: # 如果包含 CPI (收集个人信息)
        for rule_name, (antecedent, consequent) in Config.COMPLIANCE_RULES.items():
            if consequent not in label_set:
                violations.append(rule_name)
    return violations

def evaluate_compliance(y_true, y_pred, filenames):
    """执行合规性检测评估 (基于文档聚合)"""
    print(f"\n{'='*60}")
    print("合规性检测评估 (Compliance Detection Evaluation)")
    print(f"{'='*60}")
    
    # 按文件名聚合标签
    file_data = {}
    for i, filename in enumerate(filenames):
        if i >= len(y_pred): break
        if filename not in file_data:
            file_data[filename] = {'true': [], 'pred': []}
        file_data[filename]['true'].append(y_true[i])
        file_data[filename]['pred'].append(y_pred[i])
    
    print(f"评估文档总数: {len(file_data)}")
    
    true_positives = 0  # 正确检测到的违规
    false_positives = 0 # 误报
    false_negatives = 0 # 漏检
    total_true_violations = 0
    
    for fname, data in file_data.items():
        true_set = set(data['true'])
        pred_set = set(data['pred'])
        
        true_violations = set(check_compliance_violations(true_set))
        pred_violations = set(check_compliance_violations(pred_set))
        
        total_true_violations += len(true_violations)
        
        # 计算 TP, FP, FN
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
    print(f"Working directory set to: {os.getcwd()}")

    # 1. 加载模型
    print(">>> Loading Llama 3.1 8B model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "unsloth/Llama-3.1-8B",
        max_seq_length = Config.MAX_SEQ_LENGTH,
        dtype = Config.DTYPE,
        load_in_4bit = Config.LOAD_IN_4BIT,
    )

    # 2. 添加 LoRA 适配器
    print(">>> Adding LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r = 16,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 16,
        lora_dropout = 0,
        bias = "none",
        use_gradient_checkpointing = "unsloth",
        random_state = 3407,
        use_rslora = False,
        loftq_config = None,
    )

    # 3. 准备数据
    alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

    EOS_TOKEN = tokenizer.eos_token
    
    def formatting_prompts_func(examples):
        instructions = examples["Instruction"]
        inputs       = examples["Input"]
        outputs      = examples["Response"]
        texts = []
        for instruction, input, output in zip(instructions, inputs, outputs):
            text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
            texts.append(text)
        return { "text" : texts, }

    print(">>> Loading and splitting datasets...")
    # 加载本地统一 JSONL 文件
    full_dataset = load_dataset("json", data_files=Config.ALL_DATA_PATH, split="train")
    
    # 划分数据集 8:1:1
    train_testval = full_dataset.train_test_split(test_size=0.2, seed=Config.RANDOM_STATE)
    test_val = train_testval["test"].train_test_split(test_size=0.5, seed=Config.RANDOM_STATE)
    
    train_dataset = train_testval["train"].map(formatting_prompts_func, batched=True)
    val_dataset = test_val["train"].map(formatting_prompts_func, batched=True)
    test_dataset = test_val["test"]  # 测试集保持原始列以供后续推理提取 Instruction/Input 等

    # 4. 开始微调
    print(">>> Starting training...")
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = train_dataset,
        eval_dataset = val_dataset,
        dataset_text_field = "text",
        max_seq_length = Config.MAX_SEQ_LENGTH,
        packing = False,
        args = SFTConfig(
            per_device_train_batch_size = 2,
            gradient_accumulation_steps = 4,
            warmup_steps = 5,
            num_train_epochs = 1,
            learning_rate = 2e-4,
            logging_steps = 10,
            optim = "adamw_8bit",
            weight_decay = 0.001,
            lr_scheduler_type = "linear",
            seed = 3407,
            output_dir = Config.OUTPUT_DIR,
            report_to = "none",
            eval_strategy = "steps",
            eval_steps = 100,
        ),
    )
    
    trainer_stats = trainer.train()
    print(">>> Training complete.")
    # 保存最终微调出的 LoRA 权重和分词器
    print(f">>> Saving model to {Config.OUTPUT_DIR}...")
    model.save_pretrained(Config.OUTPUT_DIR)
    tokenizer.save_pretrained(Config.OUTPUT_DIR)
    

    # 5. 在测试集上进行推理
    print(">>> Starting inference on test set...")
    FastLanguageModel.for_inference(model) # 启用原生推理加速
    
    predictions = []
    true_labels = []
    filenames = []
    
    # 逐条推理
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
        # 假设输出格式为 "... ### Response:\n<number><eos>"
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

    # 6. 执行评估
    y_true = np.array(true_labels)
    y_pred = np.array(predictions)
    
    # 6.1 分类指标评估
    evaluate_classification(y_true, y_pred)
    
    '''
    # 6.2 合规性检测评估
    try:
        evaluate_compliance(y_true, y_pred, filenames)
            
    except Exception as e:
        print(f"合规性评估失败: {e}")
        print("请检查 dataset/data.tsv 是否存在以及划分逻辑是否一致。")
    '''
if __name__ == "__main__":
    main()