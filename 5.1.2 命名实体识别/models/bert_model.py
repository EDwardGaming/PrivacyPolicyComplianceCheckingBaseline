import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from tqdm import tqdm
import numpy as np

from .bert_ner import BertNER
from .config import BERTConfig, TrainingConfig


class BertNERDataset(Dataset):
    """BERT NER数据集"""
    def __init__(self, word_lists, tag_lists, tokenizer, tag2id, max_len=128):
        self.word_lists = word_lists
        self.tag_lists = tag_lists
        self.tokenizer = tokenizer
        self.tag2id = tag2id
        self.max_len = max_len
        # 使用-100作为padding标签,这是PyTorch CrossEntropyLoss的默认ignore_index
        self.pad_token_label_id = -100

    def __len__(self):
        return len(self.word_lists)

    def __getitem__(self, idx):
        words = self.word_lists[idx]
        tags = self.tag_lists[idx]

        # 截断words和tags到合适长度(为[CLS]和[SEP]预留空间)
        max_seq_len = self.max_len - 2
        if len(words) > max_seq_len:
            words = words[:max_seq_len]
            tags = tags[:max_seq_len]

        # 将字符列表拼接成字符串
        text = ''.join(words)

        # 使用BERT tokenizer进行编码,不添加特殊token以便手动处理对齐
        encoding = self.tokenizer(
            text,
            padding=False,
            truncation=False,  # 我们已经手动截断了
            add_special_tokens=False,
            return_tensors=None
        )

        # 处理标签：为每个字符对应的token分配标签
        # 注意：使用tag2id['O']作为默认值,而不是0
        label_ids = [self.tag2id.get(tag, self.tag2id['O']) for tag in tags]

        # 确保encoding和label_ids长度一致
        # 由于中文字符,tokenizer应该产生相同数量的token
        token_len = len(encoding['input_ids'])
        if token_len > len(label_ids):
            # 如果token比标签多,截断token
            encoding['input_ids'] = encoding['input_ids'][:len(label_ids)]
        elif token_len < len(label_ids):
            # 如果token比标签少,截断标签
            label_ids = label_ids[:token_len]

        # 手动添加[CLS]和[SEP]
        input_ids = [self.tokenizer.cls_token_id] + encoding['input_ids'] + [self.tokenizer.sep_token_id]
        attention_mask = [1] * len(input_ids)
        token_type_ids = [0] * len(input_ids)

        # 为[CLS]和[SEP]添加特殊标签(使用pad_token_label_id忽略)
        label_ids = [self.pad_token_label_id] + label_ids + [self.pad_token_label_id]

        # 填充到max_len
        current_len = len(input_ids)
        padding_length = self.max_len - current_len

        if padding_length > 0:
            input_ids = input_ids + [self.tokenizer.pad_token_id] * padding_length
            attention_mask = attention_mask + [0] * padding_length
            token_type_ids = token_type_ids + [0] * padding_length
            label_ids = label_ids + [self.pad_token_label_id] * padding_length
        elif padding_length < 0:
            # 如果超长,需要截断(这不应该发生,但为了安全起见)
            input_ids = input_ids[:self.max_len]
            attention_mask = attention_mask[:self.max_len]
            token_type_ids = token_type_ids[:self.max_len]
            label_ids = label_ids[:self.max_len]

        # 确保所有列表长度都是max_len
        assert len(input_ids) == self.max_len
        assert len(attention_mask) == self.max_len
        assert len(token_type_ids) == self.max_len
        assert len(label_ids) == self.max_len

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'labels': torch.tensor(label_ids, dtype=torch.long)
        }


class BERT_Model:
    def __init__(self, vocab_size, out_size, bert_model_name='bert-base-chinese'):
        """
        vocab_size: 未使用，为了保持接口一致
        out_size: 标注种类数
        bert_model_name: BERT预训练模型名称
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 初始化tokenizer和模型
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        self.model = BertNER(
            bert_model_name=bert_model_name,
            out_size=out_size,
            dropout=BERTConfig.dropout
        ).to(self.device)

        # 训练配置
        self.epoches = TrainingConfig.bert_epoches
        self.print_step = TrainingConfig.print_step
        self.lr = TrainingConfig.bert_lr  # 使用BERT专用的小学习率
        self.batch_size = TrainingConfig.batch_size

    def train(self, word_lists, tag_lists, dev_word_lists, dev_tag_lists,
              word2id, tag2id):
        """训练BERT模型"""
        # 创建数据集
        train_dataset = BertNERDataset(
            word_lists, tag_lists, self.tokenizer, tag2id,
            max_len=BERTConfig.max_len
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )

        dev_dataset = BertNERDataset(
            dev_word_lists, dev_tag_lists, self.tokenizer, tag2id,
            max_len=BERTConfig.max_len
        )
        dev_loader = DataLoader(
            dev_dataset,
            batch_size=self.batch_size,
            shuffle=False
        )

        # 优化器
        optimizer = optim.AdamW(self.model.parameters(), lr=self.lr)

        # 损失函数,使用-100作为ignore_index(忽略[CLS], [SEP], [PAD]位置的标签)
        criterion = nn.CrossEntropyLoss(ignore_index=-100)

        # 训练
        for epoch in range(self.epoches):
            self.model.train()
            epoch_loss = 0

            for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.epoches}")):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                token_type_ids = batch['token_type_ids'].to(self.device)
                labels = batch['labels'].to(self.device)

                optimizer.zero_grad()

                logits = self.model(input_ids, attention_mask, token_type_ids)

                # 计算损失
                loss = criterion(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1)
                )

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            # 验证
            if (epoch + 1) % self.print_step == 0:
                dev_loss = self._evaluate(dev_loader, criterion)
                print(f"Epoch {epoch+1}: Train Loss = {epoch_loss/len(train_loader):.4f}, "
                      f"Dev Loss = {dev_loss:.4f}")

    def _evaluate(self, data_loader, criterion):
        """在验证集上评估"""
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                token_type_ids = batch['token_type_ids'].to(self.device)
                labels = batch['labels'].to(self.device)

                logits = self.model(input_ids, attention_mask, token_type_ids)
                loss = criterion(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1)
                )
                total_loss += loss.item()

        return total_loss / len(data_loader)

    def test(self, word_lists, tag_lists, word2id, tag2id):
        """测试BERT模型"""
        self.model.eval()

        # id到tag的映射
        id2tag = {v: k for k, v in tag2id.items()}

        pred_tag_lists = []

        # 逐个样本进行预测(避免使用BertNERDataset导致的标签混乱)
        with torch.no_grad():
            for words in word_lists:
                # 截断到合适长度
                max_seq_len = BERTConfig.max_len - 2
                original_len = len(words)
                if len(words) > max_seq_len:
                    words = words[:max_seq_len]

                # 拼接成文本
                text = ''.join(words)

                # 编码
                encoding = self.tokenizer(
                    text,
                    padding=False,
                    truncation=False,
                    add_special_tokens=False,
                    return_tensors=None
                )

                # 手动添加[CLS]和[SEP]
                input_ids = [self.tokenizer.cls_token_id] + encoding['input_ids'] + [self.tokenizer.sep_token_id]
                attention_mask = [1] * len(input_ids)
                token_type_ids = [0] * len(input_ids)

                # 填充到max_len
                current_len = len(input_ids)
                padding_length = BERTConfig.max_len - current_len

                if padding_length > 0:
                    input_ids = input_ids + [self.tokenizer.pad_token_id] * padding_length
                    attention_mask = attention_mask + [0] * padding_length
                    token_type_ids = token_type_ids + [0] * padding_length
                elif padding_length < 0:
                    input_ids = input_ids[:BERTConfig.max_len]
                    attention_mask = attention_mask[:BERTConfig.max_len]
                    token_type_ids = token_type_ids[:BERTConfig.max_len]

                # 转为tensor并添加batch维度
                input_ids = torch.tensor([input_ids], dtype=torch.long).to(self.device)
                attention_mask = torch.tensor([attention_mask], dtype=torch.long).to(self.device)
                token_type_ids = torch.tensor([token_type_ids], dtype=torch.long).to(self.device)

                # 预测
                tag_ids = self.model.test(input_ids, attention_mask, token_type_ids)
                tag_ids = tag_ids.cpu().numpy()[0]  # 取出batch中的第一个

                # 跳过[CLS],只取原始长度的标签
                pred_len = min(len(words), len(tag_ids) - 1)
                pred_tags = [id2tag.get(tid, 'O') for tid in tag_ids[1:1+pred_len]]

                # 如果原始长度超过了预测长度,用'O'填充
                if original_len > len(pred_tags):
                    pred_tags = pred_tags + ['O'] * (original_len - len(pred_tags))

                pred_tag_lists.append(pred_tags[:original_len])

        return pred_tag_lists, tag_lists
