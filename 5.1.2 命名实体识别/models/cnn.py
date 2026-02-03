import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, vocab_size, emb_size, hidden_size, out_size, kernel_size=3, num_filters=128):
        super(CNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        
        # CNN layer
        self.conv1d = nn.Conv1d(in_channels=emb_size, out_channels=num_filters, kernel_size=kernel_size, padding=kernel_size//2)
        self.relu = nn.ReLU()
        
        # The output from conv1d will have shape (batch_size, num_filters, seq_len)
        # We need to map this to (batch_size, seq_len, out_size)
        self.fc = nn.Linear(num_filters, out_size)

    def forward(self, sents_tensor, lengths=None):
        # sents_tensor: (batch_size, seq_len)
        embedded = self.embedding(sents_tensor)  # (batch_size, seq_len, emb_size)
        
        # Conv1d expects (batch_size, emb_size, seq_len)
        embedded = embedded.permute(0, 2, 1)
        
        conved = self.conv1d(embedded)  # (batch_size, num_filters, seq_len)
        conved = self.relu(conved)
        
        # Back to (batch_size, seq_len, num_filters)
        conved = conved.permute(0, 2, 1)
        
        output = self.fc(conved)  # (batch_size, seq_len, out_size)
        return output
