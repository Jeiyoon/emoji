"""
@author: k4ke
"""
# import json
# import pickle
# import logging
import os, sys
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import setproctitle
setproctitle.setproctitle('[k4ke (inference)] emoji_v0')

import torch
from torch import nn
import torch.nn.functional as F
# import torch.optim as optim
from torch.utils.data import Dataset #, DataLoader
import gluonnlp as nlp
import numpy as np
from collections import defaultdict

from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model


from pathlib import Path
from utils import BERTClassifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    # _model_dir = "/home/k4ke/kobert/saves"
    # model_dir = Path(_model_dir)
    # model_config = Config(json_path = model_dir / 'config.json')

    # Vocab and Tokenizer
    tokenizer = get_tokenizer()
    bertmodel, vocab = get_pytorch_kobert_model()
    # token_to_idx = vocab.token_to_idx
    #
    # # vocab_size = len(token_to_idx)
    # print("len(toekn_to_idx): ", len(token_to_idx))
    #
    # with open(model_dir / "token2idx_vocab.json", 'w', encoding='utf-8') as f:
    #     json.dump(token_to_idx, f, ensure_ascii=False, indent=4)
    #
    # # save vocab & tokenizer
    # with open(model_dir / "vocab.pkl", 'wb') as f:
    #     pickle.dump(vocab, f)
    #
    # # load vocab & tokenizer
    # with open(model_dir / "vocab.pkl", 'rb') as f:
    #     vocab = pickle.load(f)

    # tokenizer = Tokenizer(vocab=vocab, split_fn=ptr_tokenizer, pad_fn=keras_pad_fn, maxlen=64)
    tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)

    model = BERTClassifier(bertmodel)

    # load model
    model_dict = model.state_dict()
    # checkpoint = torch.load("./experiments/base_model_with_crf_val/best-epoch-12-step-1000-acc-0.960.bin", map_location=torch.device('cpu'))
    # checkpoint = torch.load("/home/k4ke/kobert/saves/best-epoch-5-f1-0.916.bin", map_location = torch.device('cpu'))
    checkpoint = torch.load("/home/k4ke/kobert/best-epoch-54-f1-0.728.bin", map_location=torch.device('cpu'))

    convert_keys = {}
    for k, v in checkpoint['model_state_dict'].items():
        new_key_name = k.replace("module.", '')
        if new_key_name not in model_dict:
            print("{} is not int model_dict".format(new_key_name))
            continue
        convert_keys[new_key_name] = v

    model.load_state_dict(convert_keys)
    # model.load_state_dict(checkpoint)

    model.eval()
    model.to(device)

    emo_dict = {}
    emo_dict[0] = "중립"  # 48631
    emo_dict[1] = "혐오"  # 5650
    emo_dict[2] = "공포"  # 5566
    emo_dict[3] = "분노"  # 9299
    emo_dict[4] = "놀람"  # 10766
    emo_dict[5] = "슬픔"  # 7241
    emo_dict[6] = "기쁨"  # 7068

    while(True):
        _sentence = input("input: ")
        transform = nlp.data.BERTSentenceTransform(tok, max_seq_length=64, pad=True, pair=False)
        # self.sentences = [transform([i[sent_idx]]) for i in dataset]
        sentence = [transform([_sentence])]
        dataloader = torch.utils.data.DataLoader(sentence, batch_size=1)
        _token_ids = dataloader._index_sampler.sampler.data_source
        # print(_token_ids)
        # print(_token_ids[0])
        # print(_token_ids[0][0])
        _t = torch.from_numpy(_token_ids[0][0])
        _t = _t.tolist()
        token_ids = torch.tensor(_t, dtype=torch.long).unsqueeze(0).cuda()
        val_len = torch.tensor([len(token_ids[0])], dtype=torch.long).cuda()
        # val_len = torch.tensor([len(token_ids)], dtype=torch.long).cuda()

        _s = torch.from_numpy(_token_ids[0][1])
        _s = _s.tolist()
        segment_ids = torch.tensor(_s, dtype=torch.long).unsqueeze(0).cuda()
        # segment_ids = torch.from_numpy(_token_ids[0][1]).unsqueeze(0)
        # segment_ids = torch.zeros()
        # print(len(token_ids)) # 1

        out = model(token_ids, val_len, segment_ids)
        out_idx = np.argmax(out.cpu().detach().numpy())
        softmax = nn.Softmax(dim=1)
        score = softmax(out).cpu().detach().numpy()

        print("out: ", out)
        print(out_idx, emo_dict[out_idx])
        print("score: ", score)

if __name__ == "__main__":
    main()
    print("done")