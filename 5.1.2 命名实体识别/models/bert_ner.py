import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer


class BertNER(nn.Module):
    def __init__(self, bert_model_name, out_size, dropout=0.1):
        """初始化参数：
            bert_model_name: 预训练BERT模型名称
            out_size: 标注的种类
            dropout: dropout率
        """
        super(BertNER, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, out_size)

    def forward(self, input_ids, attention_mask, token_type_ids):
        """
        input_ids: [batch_size, seq_len]
        attention_mask: [batch_size, seq_len]
        token_type_ids: [batch_size, seq_len]
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        sequence_output = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)  # [batch_size, seq_len, out_size]
        return logits

    def test(self, input_ids, attention_mask, token_type_ids):
        """预测标签"""
        logits = self.forward(input_ids, attention_mask, token_type_ids)
        _, batch_tagids = torch.max(logits, dim=2)
        return batch_tagids
