import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import TensorDataset, DataLoader
from transformers import AutoModel

class BertRNNWithSpeakerAttention(nn.Module):
    def __init__(self, nlayer, nclass, dropout=0.5, nfinetune=0, emb_batch=0, attention_heads=4):
        super(BertRNNWithSpeakerAttention, self).__init__()

        self.bert = AutoModel.from_pretrained('roberta-base')
        nhid = self.bert.config.hidden_size

        for param in self.bert.parameters():
            param.requires_grad = False

        if nfinetune > 0:
            for param in self.bert.pooler.parameters():
                param.requires_grad = True
            for layer in self.bert.encoder.layer[-nfinetune:]:
                for param in layer.parameters():
                    param.requires_grad = True

        # Speaker attention
        # self.attention = nn.MultiheadAttention(nhid, num_heads=4, batch_first=True)
        self.attention = nn.MultiheadAttention(nhid, num_heads=attention_heads, batch_first=True)


        # RNN for sequence modeling
        self.encoder = nn.GRU(nhid, nhid // 2, num_layers=nlayer, dropout=dropout, bidirectional=True)
        self.fc = nn.Linear(nhid, nclass)

        self.dropout = nn.Dropout(p=dropout)
        self.nclass = nclass
        self.emb_batch = emb_batch

    def forward(self, input_ids, attention_mask, chunk_lens, speaker_ids, topic_labels=None):
        chunk_lens = chunk_lens.cpu()
        batch_size, chunk_size, seq_len = input_ids.shape

        # Flatten for BERT
        flat_input_ids = input_ids.reshape(-1, seq_len)
        flat_attention_mask = attention_mask.reshape(-1, seq_len)

        if self.training or self.emb_batch == 0:
            embeddings = self.bert(flat_input_ids, attention_mask=flat_attention_mask)[0][:, 0]
        else:
            embeddings_ = []
            loader = DataLoader(TensorDataset(flat_input_ids, flat_attention_mask), batch_size=self.emb_batch)
            for batch_ids, batch_mask in loader:
                emb = self.bert(batch_ids, attention_mask=batch_mask)[0][:, 0]
                embeddings_.append(emb)
            embeddings = torch.cat(embeddings_, dim=0)

        embeddings = embeddings.reshape(batch_size, chunk_size, -1)

        # Compute contextual speaker embeddings using attention
        contextual_emb = torch.zeros_like(embeddings)
        for b in range(batch_size):
            speaker_history = {}
            for t in range(chunk_size):
                spk = speaker_ids[b, t].item()
                current_emb = embeddings[b, t].unsqueeze(0).unsqueeze(0)  # [1,1,dim]

                # Get speaker history embeddings
                history = speaker_history.get(spk, [])
                if history:
                    history_embs = torch.stack(history).unsqueeze(0)  # [1, history_len, dim]

                    # Attention: query=current embedding, key/value=history embeddings
                    attn_output, _ = self.attention(current_emb, history_embs, history_embs)
                    contextual_emb[b, t] = attn_output.squeeze(0).squeeze(0)
                else:
                    contextual_emb[b, t] = torch.zeros_like(current_emb).squeeze(0).squeeze(0)

                # Update history
                speaker_history.setdefault(spk, []).append(embeddings[b, t])

        # Combine embeddings
        combined_embeddings = embeddings + contextual_emb

        # RNN
        combined_embeddings = combined_embeddings.permute(1, 0, 2)
        packed_emb = pack_padded_sequence(combined_embeddings, chunk_lens, enforce_sorted=False)
        self.encoder.flatten_parameters()
        outputs, _ = self.encoder(packed_emb)
        outputs, _ = pad_packed_sequence(outputs)

        # Padding if required
        if outputs.shape[0] < chunk_size:
            pad_size = chunk_size - outputs.shape[0]
            outputs = torch.cat([outputs, torch.zeros(pad_size, batch_size, outputs.shape[-1], device=outputs.device)], dim=0)

        outputs = self.dropout(outputs)
        outputs = self.fc(outputs)
        outputs = outputs.permute(1, 0, 2).reshape(-1, self.nclass)

        return outputs
