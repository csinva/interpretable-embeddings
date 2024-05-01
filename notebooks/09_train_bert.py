from collections import defaultdict
from torch.utils.data import Dataset
from transformers import AdamW, AutoTokenizer, AutoModel
from torch.utils.data import DataLoader
from torch import nn
import torch
from tqdm import tqdm
import joblib
import numpy as np
from os.path import join
import pandas as pd
import os
import sys
path_to_file = os.path.dirname(os.path.abspath(__file__))
sys.path.append('..')
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


class MutiTaskClassifier(nn.Module):
    def __init__(self, checkpoint, num_binary_outputs):
        super().__init__()
        self.model = AutoModel.from_pretrained(
            checkpoint, return_dict=True, output_hidden_states=True,
        )
        self.classifiers = nn.ModuleList(
            [nn.Linear(self.model.config.hidden_size, 2)
             for _ in range(num_binary_outputs)]
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids,
                             attention_mask=attention_mask)
        logits = torch.stack([classifier(outputs.pooler_output)
                              for classifier in self.classifiers])

        # swap axes to have batch first
        logits = logits.permute(1, 0, 2)
        return logits


if __name__ == '__main__':
    # set up data
    qa_questions_version = 'v3_boostexamples'
    vals = np.load(
        join(path_to_file, f'../data/{qa_questions_version}_answers_numpy.npz'))['arr_0']
    meta = joblib.load(
        join(path_to_file, f'../data/{qa_questions_version}_metadata.pkl'))
    df = pd.DataFrame(vals.astype(int),
                      columns=meta['columns'], index=meta['index'])
    save_dir = join(path_to_file, f'../qa_results/finetune')
    # df = df.iloc[:5000]
    df = df
    train_frac = 0.8
    idx_split = int(train_frac * len(df))
    train_data = df.iloc[:idx_split]
    val_data = df.iloc[idx_split:]

    # checkpoint = 'bert-base-uncased'
    checkpoint = 'roberta-base'
    # batch_size = 64 # 1 gpu
    batch_size = 256  # 4 gpus
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint, return_token_type_ids=False)  # , device_map='auto')

    train_dset = DataFrameDataset(train_data, tokenizer)
    val_dset = DataFrameDataset(val_data, tokenizer)
    print('baseline acc', val_data.values.mean())

    train_loader = DataLoader(train_dset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dset, batch_size=batch_size, shuffle=False)

    # training loop
    model = MutiTaskClassifier(
        checkpoint, num_binary_outputs=df.shape[1],
    )
    # model = model.to('cuda')
    model = torch.nn.DataParallel(model).to(device)
    torch.cuda.empty_cache()
    optimizer = AdamW(model.parameters(), lr=5e-5)

    # Training loop
    acc_best = 0
    accs = defaultdict(list)
    model.train()
    for epoch in range(3):
        for batch in tqdm(train_loader):
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)
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
        model.eval()
        predictions = []
        with torch.no_grad():
            for batch in tqdm(val_loader):
                inputs, labels = batch
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(
                    input_ids=inputs['input_ids'].squeeze(),
                    attention_mask=inputs['attention_mask'].squeeze(),
                )
                predictions.append(outputs.argmax(dim=-1).cpu().numpy())

        groundtruth_labels = val_data.values
        predictions = np.vstack(predictions)
        acc = np.mean(predictions == groundtruth_labels)
        print(f"Validation Accuracy: {acc:.3f}")

        accs['acc_val'].append(acc)
        if acc > acc_best:
            acc_best = acc
            os.makedirs(save_dir, exist_ok=True)
            torch.save(model.state_dict(), join(save_dir, 'model.pt'))
            pd.DataFrame(accs).to_csv(join(save_dir, 'accs.csv'))
