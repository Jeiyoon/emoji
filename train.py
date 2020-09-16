"""
@author: k4ke
"""
import logging
import os, sys
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import setproctitle
setproctitle.setproctitle('[k4ke (debugger)] emoji_v0')
# setproctitle.setproctitle('[k4ke] emoji_v0')


import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import gluonnlp as nlp
import numpy as np
from tqdm import tqdm

from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model
from transformers import AdamW
from transformers.optimization import WarmupLinearSchedule

from sklearn.metrics import f1_score
from torch.utils.tensorboard import SummaryWriter
from utils import CheckpointManager, SummaryManager

# device = torch.device("cuda:0")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger = logging.getLogger(__name__)

class BERTClassifier(nn.Module):
    def __init__(self,
                 bert,
                 hidden_size = 768,
                 num_classes = 7,
                 dr_rate = None,
                 params = None):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate

        self.classifier = nn.Linear(hidden_size, num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)

    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)

        _, pooler = self.bert(input_ids=token_ids, token_type_ids=segment_ids.long(), attention_mask=attention_mask.float().to(token_ids.device))

        if self.dr_rate:
            out = self.dropout(pooler)
        else:
            out = pooler

        return self.classifier(out)

class BERTDataset(Dataset):
    def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer, max_len, pad, pair):

        transform = nlp.data.BERTSentenceTransform(bert_tokenizer, max_seq_length=max_len, pad=pad, pair=pair)
        # sent_idx: 0 (데이터)
        # label_idx : 1 (레이블)
        self.sentences = [transform([i[sent_idx]]) for i in dataset]
        self.labels = [np.int32(i[label_idx]) for i in dataset]

    def __getitem__(self, i):
        return (self.sentences[i] + (self.labels[i], ))

    def __len__(self):
        return (len(self.labels))

## Setting parameters
max_len = 64
batch_size = 64 # 64
warmup_ratio = 0.1
num_epochs = 999
max_grad_norm = 1
log_interval = 200
learning_rate =  5e-5

bertmodel, vocab = get_pytorch_kobert_model()

# gluonnlp.data.TSVDataset
# field_indices (list of int or None, default None)
# If set, for each sample, only fields with provided indices are selected as the output. Otherwise all fields are returned.

# num_discard_samples (int, default 0)
# Number of samples discarded at the head of the first file.
dataset = nlp.data.TSVDataset("sentiment2.tsv", field_indices=[1, 2], num_discard_samples = 1)

dataset_train = []
dataset_test = []

# trainset & testset
for j, d in enumerate(dataset):
    i = np.random.randint(2)
    if i == 0:
        dataset_train.append(d)
    elif i == 1 and len(dataset_test) <= 10000:
        dataset_test.append(d)
    else:
        dataset_train.append(d)

_count = [int(s[1]) for s in dataset_test[:]] # label
n_appear  = [_count.count(i) for i in range(7)] # total label


# dataset_train = nlp.data.TSVDataset("sentiment2.tsv", field_indices=[1, 2], num_discard_samples = 1)
# dataset_test = nlp.data.TSVDataset("sentiment2.tsv", field_indices=[1, 2], num_discard_samples = 1)

_count = [int(s[1]) for s in dataset_train[:]] # label
n_appear  = [_count.count(i) for i in range(7)] # total label

tokenizer = get_tokenizer()
tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower = False)

data_train = BERTDataset(dataset_train, 0, 1, tok, max_len, True, False)
data_test = BERTDataset(dataset_test, 0, 1, tok, max_len, True, False)

# num_workers: multi-process data loading
train_dataloader = torch.utils.data.DataLoader(data_train, batch_size = batch_size, num_workers = 5, shuffle = True)
test_dataloader = torch.utils.data.DataLoader(data_test, batch_size = batch_size, num_workers = 5, shuffle = True)

model = BERTClassifier(bertmodel, dr_rate = 0.5).to(device)

# Prepare optimizer and schedule (linear warmup and decay)
no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]

optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)

def weighted_corss_entropy_loss(n_appear: list):
    weights = [sum(n_appear) / n for n in n_appear]
    loss_fn = nn.CrossEntropyLoss(weight = torch.FloatTensor(weights).cuda())
    return loss_fn

loss_fn = weighted_corss_entropy_loss(n_appear)
t_total = len(train_dataloader) * num_epochs
warmup_step = int(t_total * warmup_ratio)

scheduler = WarmupLinearSchedule(optimizer, warmup_steps=warmup_step, t_total=t_total)

# torch.max(X, 1)

# tensor([[-1.2360, -0.2942, -0.1222,  0.8475],
#         [ 1.1949, -1.1127, -2.2379, -0.6702],
#         [ 1.5717, -0.9207,  0.1297, -1.8768],
#         [-0.6172,  1.0036, -0.6060, -0.2432]])
# >>> torch.max(a, 1)
# torch.return_types.max(values=tensor([0.8475, 1.1949, 1.5717, 1.0036]), indices=tensor([3, 0, 0, 1]))
def calc_accuracy(X, Y):
    max_vals, max_indices = torch.max(X, 1)
    train_acc = (max_indices == Y).sum().data.cpu().numpy()/max_indices.size()[0]
    return train_acc

def calc_F1(X, Y):
    """
    @author: k4ke
    """
    max_vals, max_indices = torch.max(X, 1)
    # train_f1 = f1_score(Y.cpu().numpy(), max_indices.cpu().numpy(), average = 'weighted')
    train_f1 = f1_score(Y.cpu().numpy(), max_indices.cpu().numpy(), average = 'micro')
    return train_f1

# save
# model_dir: Directory containing config.json of model
model_dir = "/home/k4ke/kobert/saves"
tb_writer = SummaryWriter('{}/runs'.format(model_dir))
checkpoint_manager = CheckpointManager(model_dir)
summary_manager = SummaryManager(model_dir)

# train
best_dev_f1 = -sys.maxsize # min

# early stop
# score = []

for e in range(num_epochs):
    train_acc = 0.0
    test_acc = 0.0
    train_f1 = 0.0
    test_f1 = 0.0
    _loss = 0.0

    model.train()

    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(tqdm(train_dataloader)):
        optimizer.zero_grad()
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)
        valid_length= valid_length
        label = label.long().to(device)
        out = model(token_ids, valid_length, segment_ids)
        loss = loss_fn(out, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        scheduler.step()  # Update learning rate schedule
        train_acc += calc_accuracy(out, label)
        train_f1 += calc_F1(out, label)

        if batch_id % log_interval == 0:
            print("epoch {} batch id {} loss {} train acc {} train F1 {}".format(e+1, batch_id+1, loss.data.cpu().numpy(), train_acc / (batch_id+1) , train_f1 / (batch_id+1)))

    print("epoch {} train acc {} train F1 {}".format(e+1, train_acc / (batch_id+1), train_f1 / (batch_id+1)))
    tr_summary = {'acc': train_acc / (batch_id + 1), 'f1': train_f1 / (batch_id+1)}

    model.eval()
    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(tqdm(test_dataloader)):
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)
        valid_length= valid_length
        label = label.long().to(device)
        out = model(token_ids, valid_length, segment_ids)
        test_acc += calc_accuracy(out, label)
        test_f1 += calc_F1(out, label)
    print("epoch {} test acc {} test_f1 {}".format(e+1, test_acc / (batch_id+1), test_f1 / (batch_id+1)))
    eval_summary = {'acc': test_acc / (batch_id+1), 'f1': test_f1 / (batch_id+1)}

    # save model
    output_dir = "/home/k4ke/kobert/saves/checkpoints/epoch-{}".format(e + 1)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print("model checkpoint: ", output_dir)

    state = {'global_step': e + 1,
             'model_state_dict': model.state_dict(),
             'opt_state_dict': optimizer.state_dict()}

    summary = {'train': tr_summary, 'eval': eval_summary}
    summary_manager.update(summary)
    print("summary: ", summary)
    summary_manager.save('summary.json')

    # save
    is_best = eval_summary['f1'] >= best_dev_f1

    if is_best:
        best_dev_f1 = eval_summary['f1']
        checkpoint_manager.save_checkpoint(state, 'best-epoch-{}-f1-{:.3f}.bin'.format(e + 1, best_dev_f1))
        print("model checkpoint has been saved: best-epoch-{}-f1-{:.3f}.bin".format(e + 1, best_dev_f1))

        ## print classification report and save confusion matrix
        # cr_save_path = model_dir / 'best-epoch-{}-f1-{:.3f}-cr.csv'.format(e + 1, best_dev_f1)
        # cm_save_path = model_dir / 'best-epoch-{}-f1-{:.3f}-cm.png'.format(e + 1, best_dev_f1)
    else:
        torch.save(state, os.path.join(output_dir, 'model-epoch-{}-f1-{:.3f}.bin'.format(e + 1, eval_summary["f1"])))
        print("model checkpoint has been saved: best-epoch-{}-f1-{:.3f}.bin".format(e + 1, eval_summary['f1']))

    # score.append(eval_summary['f1'])
    #
    # # early stop
    # if np.std(score[-min(5, len(score)):]) < 0.1:
    #     sys.exit()
    # else:
    #     print("not yet")
tb_writer.close()
print("done")
