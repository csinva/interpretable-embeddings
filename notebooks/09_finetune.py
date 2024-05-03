from qa_embedder import MutiTaskClassifier
import qa_embedder
import os
import pandas as pd
from os.path import join
import numpy as np
import joblib
from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW, AutoModel
from torch.utils.data import Dataset
from collections import defaultdict
import sys



# from peft import LoraConfig, LoraModel, get_peft_model, TaskType

path_to_file = os.path.dirname(os.path.abspath(__file__))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DataFrameDataset(Dataset):
    def __init__(self, df, tokenizer):
        self.df = df
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = self.df.index[idx]
        inputs = self.tokenizer(text, padding="max_length",
                                truncation=True, max_length=512, return_tensors="pt")
        labels_onehot = torch.tensor(np.eye(2)[self.df.iloc[idx].values])
        return inputs.to(device), labels_onehot.to(device)


def eval_acc(model, loader, gt_labels):
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in tqdm(loader):
            inputs, labels = batch
            outputs = model(
                input_ids=inputs['input_ids'].squeeze(),
                attention_mask=inputs['attention_mask'].squeeze(),
            )
            predictions.append(outputs.argmax(dim=-1).cpu().numpy())

    predictions = np.vstack(predictions)
    return np.mean(predictions == gt_labels)


def get_dfs(qa_questions_version='v3_boostexamples', train_frac=0.8):
    # set up data
    vals = np.load(
        join(path_to_file, f'../data/{qa_questions_version}_answers_numpy.npz'))['arr_0']
    meta = joblib.load(
        join(path_to_file, f'../data/{qa_questions_version}_metadata.pkl'))
    df = pd.DataFrame(vals.astype(int),
                      columns=meta['columns'], index=meta['index'])

    vals_test = np.load(
        join(path_to_file, f'../data/{qa_questions_version}_answers_test_numpy.npz'))['arr_0']
    meta_test = joblib.load(
        join(path_to_file, f'../data/{qa_questions_version}_test_metadata.pkl'))
    test_df = pd.DataFrame(vals_test.astype(int),
                           columns=meta_test['columns'], index=meta_test['index'])

    # df = df.iloc[:100]
    df = df
    idx_split = int(train_frac * len(df))
    train_df = df.iloc[:idx_split]
    tune_df = df.iloc[idx_split:]
    return train_df, tune_df, test_df


if __name__ == '__main__':
    train_df, tune_df, test_df = get_dfs()
    # checkpoint = 'bert-base-uncased'
    # checkpoint = 'roberta-base'
    checkpoint = 'roberta-large'
    # checkpoint = 'distilbert-base-uncased'
    # batch_size = 64 # 1 gpu
    save_dir = join(path_to_file, f'../qa_results/finetune')
    # batch_size = 256  # 4 gpus bert-base-uncased
    # batch_size = 256  # 4 gpus roberta-base
    batch_size = 80  # 4 gpus roberta-large
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint, return_token_type_ids=False)  # , device_map='auto')

    train_dset = DataFrameDataset(train_df, tokenizer)
    tune_dset = DataFrameDataset(tune_df, tokenizer)
    test_dset = DataFrameDataset(test_df, tokenizer)
    print('baseline tune acc', 1 - tune_df.values.mean())
    print('baseline test acc', 1 - test_df.values.mean())

    # seed everything
    torch.manual_seed(0)
    np.random.seed(0)

    train_loader = DataLoader(train_dset, batch_size=batch_size, shuffle=True)
    tune_loader = DataLoader(tune_dset, batch_size=int(
        batch_size * 1.3), shuffle=False)
    test_loader = DataLoader(test_dset, batch_size=int(
        batch_size * 1.3), shuffle=False)

    # training loop
    model = MutiTaskClassifier(
        checkpoint, num_binary_outputs=train_df.shape[1],
    )
    # model = model.to('cuda')
    model = torch.nn.DataParallel(model).to(device)
    torch.cuda.empty_cache()
    optimizer = AdamW(model.parameters(), lr=5e-5)

    # Training loop
    acc_best = 0
    accs = defaultdict(list)
    model.train()
    for epoch in range(200):
        for batch in tqdm(train_loader):
            inputs, labels = batch
            optimizer.zero_grad()
            outputs = model(
                input_ids=inputs['input_ids'].squeeze(),
                attention_mask=inputs['attention_mask'].squeeze(),
            )
            loss = sum(nn.BCEWithLogitsLoss()(output, label)
                       for output, label in zip(outputs, labels))
            loss.backward()
            optimizer.step()

        # Evaluation
        acc_tune = eval_acc(model, tune_loader, tune_df.values)
        print(f"Tune Accuracy: {acc_tune:.3f}")
        acc_test = eval_acc(model, test_loader, test_df.values)
        print(f"Test Accuracy: {acc_test:.3f}")
        accs['acc_tune'].append(acc_tune)
        accs['acc_test'].append(acc_test)
        if acc_tune > acc_best:
            acc_best = acc_tune
            os.makedirs(save_dir, exist_ok=True)
            torch.save(model.module.state_dict(),
                       join(save_dir, f'{checkpoint}.pt'), _use_new_zipfile_serialization=False)
            pd.DataFrame(accs).to_csv(join(save_dir, f'{checkpoint}.csv'))
