import torch
import torch.nn as nn
import torch.nn.functional as F


class AzeNewsModel(nn.Module):
    def __init__(self, vocab_size, emb_size):
        super(AzeNewsModel, self).__init__()
        self.filter_sizes = [2, 3, 4]
        self.num_filters = 50
        self.num_classes = 5
        self.dropout_prob = 0.3
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=emb_size)
        self.convs = nn.ModuleList(
            [
                nn.Conv2d(1, self.num_filters, (f_size, emb_size))
                for f_size in self.filter_sizes
            ]
        )
        self.fc = nn.Linear(len(self.filter_sizes) * self.num_filters, self.num_classes)
        self.dropout = nn.Dropout(self.dropout_prob)

    def forward(self, input_ids):
        x = self.embedding(input_ids)  # [batch_size, seq_len, embedding_dim]
        x = x.unsqueeze(1)  # [batch_size, 1, seq_len, embedding_dim]
        x = [
            F.relu(conv(x)).squeeze(3) for conv in self.convs
        ]  # [(batch_size, num_filters, seq_len - filter_size + 1), ...]
        x = [
            F.max_pool1d(conv, conv.size(2)).squeeze(2) for conv in x
        ]  # [(batch_size, num_filters), ...]
        x = torch.cat(x, 1)  # [batch_size, len(filter_sizes) * num_filters]

        return self.fc(self.dropout(x))  # [batch_size, num_classes]
