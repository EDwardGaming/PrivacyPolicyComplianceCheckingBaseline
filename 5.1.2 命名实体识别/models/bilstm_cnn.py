import torch
import torch.nn as nn

class BiLSTM_CNN(nn.Module):
    def __init__(self, vocab_size, emb_size, hidden_size, out_size, kernel_size=3, num_filters=128):
        super(BiLSTM_CNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        
        # CNN layer
        self.conv1d = nn.Conv1d(in_channels=emb_size, out_channels=num_filters, kernel_size=kernel_size, padding=kernel_size//2)
        self.relu = nn.ReLU()
        
        # BiLSTM output is (batch, seq_len, hidden_size * 2), but bilstm model from this project returns (batch, seq_len, out_size)
        # The output from conv1d will have shape (batch_size, num_filters, seq_len)
        # We will change the bilstm to not have the final fc layer
        
        self.bilstm_for_feature = nn.LSTM(emb_size, hidden_size, bidirectional=True, batch_first=True)

        # We need to map the concatenated features to (batch_size, seq_len, out_size)
        self.fc = nn.Linear(hidden_size * 2 + num_filters, out_size)

    def forward(self, sents_tensor, lengths):
        # sents_tensor: (batch_size, seq_len)
        embedded = self.embedding(sents_tensor)  # (batch_size, seq_len, emb_size)
        
        # BiLSTM part
        packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)
        lstm_out, _ = self.bilstm_for_feature(packed)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True) # (batch_size, seq_len, hidden_size*2)

        # CNN part
        # Conv1d expects (batch_size, emb_size, seq_len)
        cnn_input = embedded.permute(0, 2, 1)
        
        conved = self.conv1d(cnn_input)  # (batch_size, num_filters, seq_len)
        conved = self.relu(conved)
        
        # Back to (batch_size, seq_len, num_filters)
        conved = conved.permute(0, 2, 1)

        # Concatenate BiLSTM and CNN outputs
        combined_output = torch.cat((lstm_out, conved), dim=2)
        
        output = self.fc(combined_output)  # (batch_size, seq_len, out_size)
        return output
