import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments, DataCollatorWithPadding
from datasets import load_dataset, Dataset
import numpy as np
from sklearn.metrics import f1_score, precision_recall_curve, average_precision_score
import pandas as pd
from sklearn.preprocessing import label_binarize

class CustomTrainer(Trainer):
    def __init__(self, class_weights=None, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        
        # 1. 弹出 labels，防止模型内部进行冗余的 Loss 计算
        # 必须创建一个新的字典，避免修改原始 inputs 导致报错
        clean_inputs = {k: v for k, v in inputs.items() if k != "labels"}
        outputs = model(**clean_inputs)

        logits = outputs.get("logits")
        
        # 2. 计算自定义 Loss
        if self.class_weights is not None and model.training:
            class_weights = self.class_weights.to(logits.device)
            loss = F.cross_entropy(logits, labels, weight=class_weights)
        else:
            loss = F.cross_entropy(logits, labels)
        
        # 3. 极其重要：将你自定义的 loss 覆盖回 outputs 中
        outputs.loss = loss
        return (loss, outputs) if return_outputs else loss

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    
    # 使用数值稳定的 Softmax 将 logits 转换为概率
    probabilities = torch.nn.functional.softmax(torch.tensor(logits), dim=-1).numpy()
    
    y_score = probabilities[:, 1]
    y_true = labels
    
    # 计算 PR 曲线和不同阈值下的 F1 分数
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    precision = precision[:-1]
    recall = recall[:-1]
    
    denominator = precision + recall
    f1_scores = np.divide(2 * precision * recall, denominator, out=np.zeros_like(denominator), where=denominator!=0)
    
    # 动态获取当前 Epoch 最佳阈值及其对应的 F1
    if len(f1_scores) > 0:
        optimal_idx = np.argmax(f1_scores)
        optimal_binary_f1 = f1_scores[optimal_idx]
        optimal_threshold = thresholds[optimal_idx]
    else:
        print("警告: 没有有效的 F1 分数，可能是由于所有预测概率相同导致的。默认使用 0.5 作为阈值。")
        optimal_binary_f1 = 0.0
        optimal_threshold = 0.5

    # 计算 Average Precision (即 PR-AUC)，更能平滑反映极度不平衡下的性能
    pr_auc = average_precision_score(y_true, y_score)
    
    # 基于动态最佳阈值计算 Macro F1
    opt_predictions = (y_score >= optimal_threshold).astype(int)
    macro_f1 = f1_score(y_true, opt_predictions, average='macro', zero_division=0)
    
    return {
        "binary_f1": optimal_binary_f1,
        "pr_auc": pr_auc,
        "macro_f1": macro_f1,
        "opt_thresh": optimal_threshold
    }

def find_optimal_threshold(trainer, val_dataset):
    print("开始预测验证集寻找最佳阈值...")
    predictions = trainer.predict(val_dataset)
    all_logits = predictions.predictions
    all_labels = predictions.label_ids
    
    # 计算每个类别的概率
    probabilities = torch.nn.functional.softmax(torch.tensor(all_logits), dim=1).numpy()
    
    # 对于二分类任务，我们关注类别1的概率
    y_score = probabilities[:, 1]
    y_true = (all_labels != 0).astype(int)  # 0类为Other，其他为正类
    
    # 计算PR曲线
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    
    # 移除最后一个没有对应阈值的点 (precision_recall_curve返回的precision和recall长度比thresholds多1)
    precision = precision[:-1]
    recall = recall[:-1]
    
    # 计算 F1 分数
    denominator = precision + recall
    f1 = np.divide(2 * precision * recall, denominator, out=np.zeros_like(denominator), where=denominator!=0)
    
    # 优化：直接选择最大 F1，移除 tolerance 逻辑
    optimal_idx = np.argmax(f1)
    optimal_threshold = thresholds[optimal_idx]
    
    print(f"最佳阈值: {optimal_threshold:.4f}")
    print(f"对应的最佳 F1 分数: {f1[optimal_idx]:.4f}")
    print(f"对应的精确率: {precision[optimal_idx]:.4f}")
    print(f"对应的召回率: {recall[optimal_idx]:.4f}")
    
    return optimal_threshold

def main():
    # 加载数据集
    df = pd.read_csv('./dataset/data.tsv', sep='\t')
    hf_dataset = Dataset.from_pandas(df)
    
    # 预处理数据集
    def preprocess_function(examples):
        return {
            "sentence": examples["sentence"],
            "label": 0 if examples["label"] == 0 else 1  # 二分类：0=Other, 1=Privacy
        }
    
    hf_dataset = hf_dataset.map(preprocess_function)
    
    # 划分训练集和验证集
    dataset = hf_dataset.train_test_split(test_size=0.2, seed=42)
    train_dataset = dataset["train"]
    val_dataset = dataset["test"]
    
    # 统计类别分布
    label_counts = np.bincount(train_dataset["label"])
    print(f"训练集类别分布: {label_counts}")
    
    # 使用标准的 Balanced 权重算法应对极端不平衡 (与 scikit-learn 逻辑一致)
    total_samples = len(train_dataset)
    balanced_weights = total_samples / (len(label_counts) * label_counts)
    # 对权重进行归一化，使得权重的平均值为 1，防止 Loss 数值尺度发生剧变
    balanced_weights = balanced_weights / np.sum(balanced_weights) * len(label_counts)
    class_weights = torch.tensor(balanced_weights, dtype=torch.float)
    print(f"类别权重 (Balanced): {class_weights}")
    
    # 加载模型和分词器
    model_name = "microsoft/deberta-v3-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    
    # 分词
    def tokenize_function(examples):
        # 移除静态 padding，采用动态长度对齐提升速度和内存利用率
        return tokenizer(examples["sentence"], truncation=True, max_length=512)
    
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True)
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # 设置训练参数
    training_args = TrainingArguments(
        output_dir="./deberta_output",
        learning_rate=1.5e-5, # 稍微降低一点学习率，DeBERTa-v3 对其非常敏感
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4, # 累积梯度，使得等效 Batch Size = 32，稳定极端分布下的梯度更新
        per_device_eval_batch_size=16, # 增加验证 batch size 加速评估
        num_train_epochs=5, # 减少 epoch 防止过拟合
        weight_decay=0.01,
        warmup_ratio=0.1, # 采用相对比例进行预热，防止陷入退化解
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="pr_auc",
        greater_is_better=True,
        logging_steps=50, # 增加日志输出频率，便于监控 Loss 是否健康下降
        logging_dir="./logs",
        max_grad_norm=1.0, # 加入梯度裁剪，防止焦点损失加大权重时引发梯度爆炸
        report_to="none",  # 禁用 wandb 等外部日志记录器
        fp16=False, # 确保禁用FP16以防止 DeBERTaV3 出现梯度溢出或NaN
    )
    
    # 创建训练器
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        class_weights=class_weights,
        data_collator=data_collator,
    )
    
    # 训练模型
    trainer.train()
    
    # 找到最佳阈值
    optimal_threshold = find_optimal_threshold(trainer, val_dataset)
    
    # 保存模型和阈值
    model.save_pretrained("./deberta_model")
    tokenizer.save_pretrained("./deberta_model")
    
    with open("./deberta_model/optimal_threshold.txt", "w") as f:
        f.write(str(optimal_threshold))
    
    print("模型训练完成并保存")

if __name__ == "__main__":
    main()