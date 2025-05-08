import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import TensorDataset, DataLoader
from torchcrf import CRF

class BCRF(nn.Module):
    def __init__(self, nlayer, nclass, dropout=0.5, nfinetune=0,
                 speaker_info='none', topic_info='none', emb_batch=0):
        super(BCRF, self).__init__()

        from transformers import AutoModel
        self.bert = AutoModel.from_pretrained('roberta-base')
        nhid = self.bert.config.hidden_size

        for param in self.bert.parameters():
            param.requires_grad = False

        n_layers = 12
        if nfinetune > 0:
            for param in self.bert.pooler.parameters():
                param.requires_grad = True
            for i in range(n_layers - 1, n_layers - 1 - nfinetune, -1):
                for param in self.bert.encoder.layer[i].parameters():
                    param.requires_grad = True

        self.encoder = nn.GRU(nhid, nhid // 2, num_layers=nlayer,
                              dropout=dropout if nlayer > 1 else 0.0,
                              bidirectional=True)

        self.fc = nn.Linear(nhid, nclass)
        self.crf = CRF(nclass, batch_first=True)

        self.speaker_emb = nn.Embedding(3, nhid)
        self.topic_emb = nn.Embedding(100, nhid)

        self.dropout = nn.Dropout(p=dropout)
        self.nclass = nclass
        self.speaker_info = speaker_info
        self.topic_info = topic_info
        self.emb_batch = emb_batch

    def forward(self, input_ids, attention_mask, chunk_lens,
                speaker_ids, topic_labels, labels=None):
        chunk_lens_cpu = chunk_lens.to('cpu')  # used for packing only
        batch_size, chunk_size, seq_len = input_ids.shape

        speaker_ids = speaker_ids.reshape(-1)
        topic_labels = topic_labels.reshape(-1)
        input_ids = input_ids.reshape(-1, seq_len)
        attention_mask = attention_mask.reshape(-1, seq_len)

        if self.training or self.emb_batch == 0:
            bert_output = self.bert(input_ids, attention_mask=attention_mask,
                                    output_hidden_states=True)[0][:, 0]
        else:
            embeddings_ = []
            dataset2 = TensorDataset(input_ids, attention_mask)
            loader = DataLoader(dataset2, batch_size=self.emb_batch)
            for _, batch in enumerate(loader):
                emb = self.bert(batch[0], attention_mask=batch[1],
                                output_hidden_states=True)[0][:, 0]
                embeddings_.append(emb)
            bert_output = torch.cat(embeddings_, dim=0)

        nhid = bert_output.shape[-1]

        if self.speaker_info == 'emb_cls':
            speaker_embeds = self.speaker_emb(speaker_ids)
            bert_output = bert_output + speaker_embeds

        bert_output = bert_output.reshape(batch_size, chunk_size, nhid)

        packed_emb = pack_padded_sequence(bert_output, chunk_lens_cpu,
                                          batch_first=True, enforce_sorted=False)
        packed_out, _ = self.encoder(packed_emb)
        outputs, _ = pad_packed_sequence(packed_out, batch_first=True)

        outputs = self.dropout(outputs)
        emissions = self.fc(outputs)  # (batch_size, seq_len, nclass)

        # FIX 1: Ensure mask is created on correct device
        mask = torch.arange(chunk_size, device=input_ids.device).expand(batch_size, chunk_size)
        mask = mask < chunk_lens.to(mask.device).unsqueeze(1)

        if labels is not None:
            # FIX 2: Ensure labels and mask are trimmed to match emissions
            labels = labels[:, :emissions.size(1)]
            mask = mask[:, :emissions.size(1)]
            loss = -self.crf(emissions, labels, mask=mask, reduction='mean')
            return loss
        else:
            mask = mask[:, :emissions.size(1)]
            predictions = self.crf.decode(emissions, mask=mask)
            return predictions
