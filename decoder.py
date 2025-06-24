import torch
import torch.nn as nn

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.3)

    def forward(self, features, captions):
        """
        features: [batch_size, embed_size]
        captions: [batch_size, seq_len]
        """
        embeddings = self.dropout(self.embed(captions))  # [batch_size, seq_len, embed_size]
        # Concatenate image features as the first input
        features = features.unsqueeze(1)  # [batch_size, 1, embed_size]
        inputs = torch.cat((features, embeddings[:, :-1, :]), dim=1)  # [batch_size, seq_len, embed_size]
        hiddens, _ = self.lstm(inputs)
        outputs = self.linear(hiddens)  # [batch_size, seq_len, vocab_size]
        return outputs
