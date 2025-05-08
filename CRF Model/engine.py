import torch
import torch.nn as nn
import os
import numpy as np
from datasets import data_loader
from models import BertRNN
from sklearn.metrics import accuracy_score
import copy


class Engine:
    def __init__(self, args):
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        os.makedirs('ckp', exist_ok=True)

        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        np.random.seed(args.seed)

        train_loader = data_loader(corpus=args.corpus,
                                   phase='train',
                                   batch_size=args.batch_size,
                                   chunk_size=args.chunk_size,
                                   shuffle=True) if args.mode != 'inference' else None
        val_loader = data_loader(corpus=args.corpus,
                                 phase='val',
                                 batch_size=args.batch_size_val,
                                 chunk_size=args.chunk_size) if args.mode != 'inference' else None
        test_loader = data_loader(corpus=args.corpus,
                                  phase='test',
                                  batch_size=args.batch_size_val,
                                  chunk_size=args.chunk_size)

        print('Done\n')

        if torch.cuda.device_count() > 0:
            print(f"Let's use {torch.cuda.device_count()} GPUs!")

        print('Initializing model....')
        model = BertRNN(nlayer=args.nlayer,
                        nclass=args.nclass,
                        dropout=args.dropout,
                        nfinetune=args.nfinetune,
                        speaker_info=args.speaker_info,
                        topic_info=args.topic_info,
                        emb_batch=args.emb_batch,
                        )

        model = nn.DataParallel(model)
        model.to(device)
        params = model.parameters()

        #from transformers import AdamW
        from torch.optim import AdamW

        optimizer = AdamW(params, lr=args.lr, weight_decay=5e-4)
        criterion = nn.CrossEntropyLoss(ignore_index=-1)

        self.device = device
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.args = args

    def train(self):
        best_epoch = 0
        best_epoch_acc = 0
        best_epoch_test_acc = 0
        best_acc = 0
        best_state_dict = copy.deepcopy(self.model.state_dict())
        for epoch in range(self.args.epochs):
            print(f"{'*' * 20}Epoch: {epoch + 1}{'*' * 20}")
            loss = self.train_epoch()
            acc = self.eval()
            test_acc = self.eval(False)
            if acc > best_epoch_acc:
                best_epoch = epoch
                best_epoch_acc = acc
                best_epoch_test_acc = test_acc
                best_state_dict = copy.deepcopy(self.model.state_dict())
            if test_acc > best_acc:
                best_acc = test_acc
            print(f'Epoch {epoch + 1}\tTrain Loss: {loss:.3f}\tVal Acc: {acc:.3f}\tTest Acc: {test_acc:.3f}\n'
                  f'Best Epoch: {best_epoch + 1}\tBest Epoch Val Acc: {best_epoch_acc:.3f}\t'
                  f'Best Epoch Test Acc: {best_epoch_test_acc:.3f}, Best Test Acc: {best_acc:.3f}\n')
            if epoch - best_epoch >= 10:
                break

        print('Saving the best checkpoint....')
        torch.save(best_state_dict, f"ckp/model_{self.args.corpus}.pt")
        self.model.load_state_dict(best_state_dict)
        acc = self.eval(False)
        print(f'Test Acc: {acc:.3f}')

    def train_epoch(self):
        self.model.train()
        epoch_loss = 0
        for i, batch in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            chunk_lens = batch['chunk_lens']
            speaker_ids = batch['speaker_ids'].to(self.device)
            topic_labels = batch['topic_labels'].to(self.device)

            # ✅ Call CRF-enabled model which returns loss
            loss = self.model(input_ids, attention_mask, chunk_lens, speaker_ids, topic_labels, labels)

            # ✅ CRF loss can be Tensor([val]) → convert to scalar
            loss = loss.sum() if loss.dim() > 0 else loss
            loss.backward()
            self.optimizer.step()

            if i % max(len(self.train_loader) // 20, 1) == 0 or i == len(self.train_loader) - 1:
                print(f'Batch: {i + 1}/{len(self.train_loader)}\tloss: {loss.item():.3f}')
            epoch_loss += loss.item()
        return epoch_loss / len(self.train_loader)


    def eval(self, val=True, inference=False):
        self.model.eval()
        y_pred = []
        y_true = []
        loader = self.val_loader if val else self.test_loader
        with torch.no_grad():
            for i, batch in enumerate(loader):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                chunk_lens = batch['chunk_lens']
                speaker_ids = batch['speaker_ids'].to(self.device)
                topic_labels = batch['topic_labels'].to(self.device)

                # ✅ Use module directly (CRF returns list[list[int]], not Tensor)
                predictions = self.model.module(input_ids, attention_mask, chunk_lens, speaker_ids, topic_labels)

                for pred_seq, true_seq, lens in zip(predictions, labels, chunk_lens):
                    y_pred.append(np.array(pred_seq[:lens]))
                    y_true.append(true_seq[:lens].cpu().numpy())

        y_pred = np.concatenate(y_pred)
        y_true = np.concatenate(y_true)

        mask = y_true != -1
        acc = accuracy_score(y_true[mask], y_pred[mask])

        if inference:
            import pickle
            pickle.dump(y_pred[mask].tolist(), open('preds_on_new.pkl', 'wb'))

        return acc


    def inference(self):
        ## using the trained model to inference on a new unseen dataset

        # load the saved checkpoint
        # change the model name to whatever the checkpoint is named
        self.model.load_state_dict(torch.load('ckp/model.pt'))

        # make predictions
        self.eval(val=False, inference=True)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus', type=str, default='swda', help='the dataset to use')
    parser.add_argument('--mode', type=str, choices=('train', 'inference'), default='train',
                        help='train the model or use the trained model to inference')
    parser.add_argument('--nclass', type=int, default=43, help='num of dialog act classes in the dataset')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size of training')
    parser.add_argument('--batch_size_val', type=int, default=32, help='batch size of evaluation')
    parser.add_argument('--emb_batch', type=int, default=0, help='batch size when embedding all the utterances')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--gpu', type=str, default='', help='GPUs to use')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--nlayer', type=int, default=1, help='num of layers of the GRU')
    parser.add_argument('--chunk_size', type=int, default=32,
                        help='chunk size used to slice the long conversations. 0 means not slicing the conversations')
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--speaker_info', type=str, choices=('none', 'emb_cls'), default='none',
                        help='how to use the speaker labels')
    parser.add_argument('--topic_info', type=str, choices=('none', 'emb_cls'), default='none',
                        help='how to use the topic labels')
    parser.add_argument('--nfinetune', type=int, default=2,
                        help='num of the BERT layers to finetune. 0 means finuetuning all layers')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    print(args)
    engine = Engine(args)
    if args.mode == 'train':
        engine.train()
    else:
        engine.inference()