import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_recall_fscore_support
from transformers import BertTokenizer, BertModel
from torch.optim import AdamW
import warnings
import json

# 修改这里：导入新的分类器类
from kfc import LlamaBatchClassifier 
from Config import Config
warnings.filterwarnings('ignore')


def load_data(file_path):
    """加载TSV数据"""
    print(f"加载数据: {file_path}")
    df = pd.read_csv(file_path, sep='\t', encoding='utf-8')
    print(f"数据集大小: {len(df)}")
    print(f"\n标签分布:")
    label_counts = df['label'].value_counts().sort_index()
    for label, count in label_counts.items():
        print(f"  {label} ({Config.LABEL_NAMES[label]}): {count}")
    
    print(f"\n文件数量: {df['filename'].nunique()}")
    return df


def compute_class_weights(labels, num_classes=11):
    """
    改进的类别权重计算 - 使用balanced方式避免极端权重
    """
    label_counts = np.bincount(labels, minlength=num_classes)
    # 避免除零
    label_counts = np.maximum(label_counts, 1)
    
    # 使用balanced方式计算权重
    total = len(labels)
    weights = total / (num_classes * label_counts)
    
    # 限制权重范围，避免过大或过小
    weights = np.clip(weights, 0.5, 5.0)
    
    # 归一化
    weights = weights / weights.mean()
    
    return torch.FloatTensor(weights)

# ==================== SVM模型 ====================
class SVMClassifier:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 3))
        self.model = LinearSVC(random_state=Config.RANDOM_STATE, max_iter=5000)
    
    def fit(self, X_train, y_train):
        X_train_vec = self.vectorizer.fit_transform(X_train)
        self.model.fit(X_train_vec, y_train)
    
    def predict(self, X_test):
        X_test_vec = self.vectorizer.transform(X_test)
        return self.model.predict(X_test_vec)


# ==================== BiLSTM模型 ====================
class BiLSTMDataset(Dataset):
    def __init__(self, texts, labels, vocab=None, max_length=100):
        self.texts = texts
        self.labels = labels
        self.max_length = max_length
        
        if vocab is None:
            self.vocab = self.build_vocab(texts)
        else:
            self.vocab = vocab
    
    def build_vocab(self, texts):
        vocab = {'<PAD>': 0, '<UNK>': 1}
        for text in texts:
            for word in text.lower().split():
                if word not in vocab:
                    vocab[word] = len(vocab)
        return vocab
    
    def text_to_indices(self, text):
        words = text.lower().split()[:self.max_length]
        indices = [self.vocab.get(word, 1) for word in words]
        if len(indices) < self.max_length:
            indices += [0] * (self.max_length - len(indices))
        return indices
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        indices = self.text_to_indices(text)
        return torch.LongTensor(indices), torch.LongTensor([label])


class BiLSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes, num_layers=2, dropout=0.5):
        super(BiLSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, 
                           batch_first=True, bidirectional=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        pooled = torch.max(lstm_out, dim=1)[0]
        pooled = self.dropout(pooled)
        logits = self.fc(pooled)
        return logits


class BiLSTMClassifier:
    def __init__(self, class_weights=None):
        self.vocab = None
        self.model = None
        self.class_weights = class_weights
    
    def fit(self, X_train, y_train, X_val, y_val):
        train_dataset = BiLSTMDataset(X_train.tolist(), y_train)
        self.vocab = train_dataset.vocab
        val_dataset = BiLSTMDataset(X_val.tolist(), y_val, vocab=self.vocab)
        
        train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE)
        
        self.model = BiLSTMModel(
            vocab_size=len(self.vocab),
            embedding_dim=Config.EMBEDDING_DIM,
            hidden_dim=Config.HIDDEN_DIM,
            num_classes=11,
            num_layers=Config.LSTM_LAYERS,
            dropout=Config.DROPOUT
        ).to(Config.DEVICE)
        
        if self.class_weights is not None:
            criterion = nn.CrossEntropyLoss(weight=self.class_weights.to(Config.DEVICE))
        else:
            criterion = nn.CrossEntropyLoss()
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=Config.LR_BILSTM)
        
        best_val_loss = float('inf')
        patience = 3
        patience_counter = 0
        
        for epoch in range(Config.EPOCHS):
            self.model.train()
            train_loss = 0
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(Config.DEVICE)
                batch_y = batch_y.view(-1).to(Config.DEVICE)
                
                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item()
            
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.to(Config.DEVICE)
                    batch_y = batch_y.view(-1).to(Config.DEVICE)
                    outputs = self.model(batch_x)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            
            print(f"  Epoch {epoch+1}/{Config.EPOCHS} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                self.best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break
        
        self.model.load_state_dict(self.best_model_state)
    
    def predict(self, X_test):
        test_dataset = BiLSTMDataset(X_test.tolist(), np.zeros(len(X_test)), vocab=self.vocab)
        test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE)
        
        self.model.eval()
        predictions = []
        with torch.no_grad():
            for batch_x, _ in test_loader:
                batch_x = batch_x.to(Config.DEVICE)
                outputs = self.model(batch_x)
                preds = torch.argmax(outputs, dim=1)
                predictions.extend(preds.cpu().numpy())
        
        return np.array(predictions)


# ==================== BERT模型 ====================
class BERTDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'label': torch.LongTensor([label])
        }


class BERTClassifier(nn.Module):
    def __init__(self, num_classes=11):
        super(BERTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(Config.BERT_MODEL)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.fc(pooled_output)
        return logits


class BERTClassifierWrapper:
    def __init__(self, class_weights=None):
        self.tokenizer = BertTokenizer.from_pretrained(Config.BERT_MODEL)
        self.model = None
        self.class_weights = class_weights
    
    def fit(self, X_train, y_train, X_val, y_val):
        train_dataset = BERTDataset(X_train.tolist(), y_train, self.tokenizer, Config.MAX_LENGTH)
        val_dataset = BERTDataset(X_val.tolist(), y_val, self.tokenizer, Config.MAX_LENGTH)
        
        train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE)
        
        self.model = BERTClassifier(num_classes=11).to(Config.DEVICE)
        
        if self.class_weights is not None:
            criterion = nn.CrossEntropyLoss(weight=self.class_weights.to(Config.DEVICE))
        else:
            criterion = nn.CrossEntropyLoss()
        
        optimizer = AdamW(self.model.parameters(), lr=Config.LR_BERT)
        
        best_val_loss = float('inf')
        patience = 3
        patience_counter = 0
        
        for epoch in range(Config.EPOCHS):
            self.model.train()
            train_loss = 0
            for batch in train_loader:
                input_ids = batch['input_ids'].to(Config.DEVICE)
                attention_mask = batch['attention_mask'].to(Config.DEVICE)
                labels = batch['label'].view(-1).to(Config.DEVICE)
                
                optimizer.zero_grad()
                outputs = self.model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                
                # 检查loss是否有效
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"  Warning: Invalid loss detected at epoch {epoch+1}")
                    continue
                
                loss.backward()
                # 梯度裁剪防止梯度爆炸
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item()
            
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(Config.DEVICE)
                    attention_mask = batch['attention_mask'].to(Config.DEVICE)
                    labels = batch['label'].view(-1).to(Config.DEVICE)
                    outputs = self.model(input_ids, attention_mask)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            
            print(f"  Epoch {epoch+1}/{Config.EPOCHS} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                self.best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break
        
        self.model.load_state_dict(self.best_model_state)
    
    def predict(self, X_test):
        test_dataset = BERTDataset(X_test.tolist(), np.zeros(len(X_test)), 
                                  self.tokenizer, Config.MAX_LENGTH)
        test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE)
        
        self.model.eval()
        predictions = []
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(Config.DEVICE)
                attention_mask = batch['attention_mask'].to(Config.DEVICE)
                outputs = self.model(input_ids, attention_mask)
                preds = torch.argmax(outputs, dim=1)
                predictions.extend(preds.cpu().numpy())
        
        return np.array(predictions)

# ==================== 评估和合规性检测 ====================
def evaluate_model(y_true, y_pred, model_name):
    """评估模型性能"""
    print(f"\n{'='*60}")
    print(f"{model_name} 评估结果")
    print(f"{'='*60}")
    
    # 指定 labels 确保在只评估子集时仍返回所有 11 个类别的指标（0-10）
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(11)), average=None, zero_division=0
    )
    
    results = []
    for i in range(11):
        label_name = Config.LABEL_NAMES[i]
        results.append({
            'Label': label_name,
            'Precision': f"{precision[i]:.4f}",
            'Recall': f"{recall[i]:.4f}",
            'F1-Score': f"{f1[i]:.4f}",
            'Support': support[i]
        })
    
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    
    # 只计算GDPR相关类别(1-10)的平均指标
    non_other_mask = np.arange(11) != 0
    avg_precision = precision[non_other_mask].mean()
    avg_recall = recall[non_other_mask].mean()
    avg_f1 = f1[non_other_mask].mean()
    
    print(f"\n{'='*60}")
    print(f"10个GDPR标签的平均指标:")
    print(f"Precision: {avg_precision:.4f}")
    print(f"Recall: {avg_recall:.4f}")
    print(f"F1-Score: {avg_f1:.4f}")
    print(f"{'='*60}\n")
    
    return results_df, avg_precision, avg_recall, avg_f1


def check_compliance_violations(label_set):
    """
    检查给定标签集是否违反合规性规则
    返回违规的规则列表
    """
    violations = []
    
    # 如果包含CPI(标签1)，检查是否包含所有其他必需条款
    if 1 in label_set:
        for rule_name, (antecedent, consequent) in Config.COMPLIANCE_RULES.items():
            if consequent not in label_set:
                violations.append({
                    'rule': rule_name,
                    'missing_label': consequent,
                    'missing_name': Config.LABEL_NAMES[consequent]
                })
    
    return violations


def compliance_detection_evaluation(y_true, y_pred, filenames):
    """
    改进的合规性检测评估
    按filename分组，对每个文档的预测和真实标签应用合规规则，然后对比评估
    """
    print(f"\n{'='*60}")
    print("合规性检测评估")
    print(f"{'='*60}")
    
    # 按filename分组
    file_data = {}
    for i, filename in enumerate(filenames):
        if filename not in file_data:
            file_data[filename] = {
                'true_labels': [],
                'pred_labels': []
            }
        file_data[filename]['true_labels'].append(y_true[i])
        file_data[filename]['pred_labels'].append(y_pred[i])
    
    print(f"\n总共 {len(file_data)} 个隐私政策文档")
    
    # 对每个文档检测违规
    true_violations_count = 0
    pred_violations_count = 0
    true_positives = 0  # 正确检测到的违规
    false_positives = 0  # 误报的违规
    false_negatives = 0  # 漏检的违规
    
    violation_details = []
    
    for filename, data in file_data.items():
        true_label_set = set(data['true_labels'])
        pred_label_set = set(data['pred_labels'])
        
        # 检测真实标签的违规
        true_violations = check_compliance_violations(true_label_set)
        # 检测预测标签的违规
        pred_violations = check_compliance_violations(pred_label_set)
        
        true_violations_count += len(true_violations)
        pred_violations_count += len(pred_violations)
        
        # 将违规转换为集合便于比较
        true_violation_rules = {v['rule'] for v in true_violations}
        pred_violation_rules = {v['rule'] for v in pred_violations}
        
        # 计算TP, FP, FN
        tp = len(true_violation_rules & pred_violation_rules)
        fp = len(pred_violation_rules - true_violation_rules)
        fn = len(true_violation_rules - pred_violation_rules)
        
        true_positives += tp
        false_positives += fp
        false_negatives += fn
        
        # 记录详细信息（只记录前20个有违规的文档）
        if len(violation_details) < 20 and (true_violations or pred_violations):
            violation_details.append({
                'filename': filename,
                'true_violations': len(true_violations),
                'pred_violations': len(pred_violations),
                'correct': tp,
                'false_alarm': fp,
                'missed': fn
            })
    
    print(f"\n合规性统计:")
    print(f"  真实违规总数: {true_violations_count}")
    print(f"  预测违规总数: {pred_violations_count}")
    print(f"  正确检测 (TP): {true_positives}")
    print(f"  误报 (FP): {false_positives}")
    print(f"  漏检 (FN): {false_negatives}")
    
    # 计算合规性检测的Precision, Recall, F1
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\n合规性检测性能:")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    
    if violation_details:
        print(f"\n部分文档违规详情 (前20个):")
        details_df = pd.DataFrame(violation_details)
        print(details_df.to_string(index=False))
    
    print(f"\n{'='*60}\n")
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives
    }


# ==================== 主实验 ====================
def run_experiment():
    """运行完整实验 - 单轮训练版本"""
    print("="*60)
    print("GDPR Article 13 隐私政策合规性分析实验 (修复版 + 豆包API)")
    print("="*60)
    
    # 加载数据
    df = load_data(Config.DATA_PATH)
    X = df['sentence'].values
    y = df['label'].values
    filenames = df['filename'].values
    
    # 计算类别权重
    class_weights = compute_class_weights(y)
    print(f"\n类别权重:")
    for i, w in enumerate(class_weights):
        print(f"  {Config.LABEL_NAMES[i]}: {w:.4f}")
    
    # 划分数据集
    X_temp, X_test, y_temp, y_test, files_temp, files_test = train_test_split(
        X, y, filenames, test_size=Config.TEST_SIZE, random_state=Config.RANDOM_STATE, stratify=y
    )
    
    X_train, X_val, y_train, y_val, _, _ = train_test_split(
        X_temp, y_temp, files_temp, test_size=Config.VAL_SIZE/(1-Config.TEST_SIZE), 
        random_state=Config.RANDOM_STATE, stratify=y_temp
    )
    
    print(f"\n数据划分:")
    print(f"  训练集: {len(X_train)}")
    print(f"  验证集: {len(X_val)}")
    print(f"  测试集: {len(X_test)}")
    
    # 存储结果 - 同时保存分类性能和预测结果
    results = {}
    predictions = {}
    '''
    # ==================== SVM ====================
    print(f"\n{'='*60}")
    print("训练 SVM...")
    print(f"{'='*60}")
    svm = SVMClassifier()
    svm.fit(X_train, y_train)
    svm_pred = svm.predict(X_test)
    results['SVM'] = evaluate_model(y_test, svm_pred, "SVM")
    predictions['SVM'] = svm_pred
    
    # ==================== BiLSTM ====================
    print(f"\n{'='*60}")
    print("训练 BiLSTM...")
    print(f"{'='*60}")
    bilstm = BiLSTMClassifier(class_weights=None)
    bilstm.fit(X_train, y_train, X_val, y_val)
    bilstm_pred = bilstm.predict(X_test)
    results['BiLSTM'] = evaluate_model(y_test, bilstm_pred, "BiLSTM")
    predictions['BiLSTM'] = bilstm_pred
    
    # ==================== BiLSTM+LW ====================
    print(f"\n{'='*60}")
    print("训练 BiLSTM+LW (Loss Weighting)...")
    print(f"{'='*60}")
    bilstm_lw = BiLSTMClassifier(class_weights=class_weights)
    bilstm_lw.fit(X_train, y_train, X_val, y_val)
    bilstm_lw_pred = bilstm_lw.predict(X_test)
    results['BiLSTM+LW'] = evaluate_model(y_test, bilstm_lw_pred, "BiLSTM+LW")
    predictions['BiLSTM+LW'] = bilstm_lw_pred
    
    # ==================== BERT ====================
    print(f"\n{'='*60}")
    print("训练 BERT...")
    print(f"{'='*60}")
    bert = BERTClassifierWrapper(class_weights=None)
    bert.fit(X_train, y_train, X_val, y_val)
    bert_pred = bert.predict(X_test)
    results['BERT'] = evaluate_model(y_test, bert_pred, "BERT")
    predictions['BERT'] = bert_pred
    
    # ==================== BERT+LW ====================
    print(f"\n{'='*60}")
    print("训练 BERT+LW (Loss Weighting)...")
    print(f"{'='*60}")
    bert_lw = BERTClassifierWrapper(class_weights=class_weights)
    bert_lw.fit(X_train, y_train, X_val, y_val)
    bert_lw_pred = bert_lw.predict(X_test)
    results['BERT+LW'] = evaluate_model(y_test, bert_lw_pred, "BERT+LW")
    predictions['BERT+LW'] = bert_lw_pred
    '''
    # ==================== Llama 405B Batch Prediction ====================
    print(f"\n{'='*60}")
    print(f"使用 {Config.LLM_MODEL_ID} (Batch Size={Config.LLM_API_BATCH_SIZE}) 进行预测...")
    print(f"{'='*60}")
    
    if Config.LLM_API_KEY == "your-api-key-here":
        print("错误：请先在 Config.py 中配置 LLM API Key！")
    else:
        # 初始化分类器
        llama_classifier = LlamaBatchClassifier()
        
        # 执行批处理预测
        llama_full_pred, llama_sampled_idx = llama_classifier.predict(
            X_test, 
            sample_size=getattr(Config, 'TEST_SAMPLE_SIZE', None)
        )

        # 评估
        y_test_sample = y_test[llama_sampled_idx]
        llama_pred_sample = llama_full_pred[llama_sampled_idx]
        
        results[llama_classifier.model_id] = evaluate_model(y_test_sample, llama_pred_sample, f'{Config.LLM_MODEL_ID}')
        predictions[llama_classifier.model_id] = llama_full_pred
        sampled_indices_by_model = {f'{Config.LLM_API_KEY}': llama_sampled_idx}
    
    # ==================== 合规性检测评估 ====================
    print(f"\n{'='*60}")
    print("合规性检测评估")
    print(f"{'='*60}")
    
    compliance_results = {}
    for model_name, y_pred in predictions.items():
        if 'sampled_indices_by_model' in locals() and model_name in sampled_indices_by_model:
            idx = sampled_indices_by_model[model_name]
            y_true_eval = y_test[idx]
            y_pred_eval = y_pred[idx]
            files_eval = files_test[idx]
        else:
            y_true_eval = y_test
            y_pred_eval = y_pred
            files_eval = files_test

        compliance_results[model_name] = compliance_detection_evaluation(y_true_eval, y_pred_eval, files_eval)
    
    # 总结
    print(f"\n{'='*60}")
    print("实验总结")
    print(f"{'='*60}")
    print("\n分类性能 (10个GDPR标签平均):")
    print(f"{'Model':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-" * 51)
    for model_name, (_, avg_p, avg_r, avg_f1) in results.items():
        print(f"{model_name:<15} {avg_p:<12.4f} {avg_r:<12.4f} {avg_f1:<12.4f}")
    
    print(f"\n合规性检测性能 (所有模型):")
    print(f"{'Model':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-" * 51)
    for model_name, comp_result in compliance_results.items():
        print(f"{model_name:<15} {comp_result['precision']:<12.4f} {comp_result['recall']:<12.4f} {comp_result['f1']:<12.4f}")
    
    print(f"\n{'='*60}")
    print("实验完成！")
    print(f"{'='*60}\n")
    
    return results, compliance_results


if __name__ == "__main__":
    import os
    
    if not os.path.exists(Config.DATA_PATH):
        print(f"错误: 数据文件 {Config.DATA_PATH} 不存在！")
        print("请确保data.tsv文件在dataset目录下")
        print("如果没有数据，请先下载数据集")
    else:
        results, compliance_results = run_experiment()