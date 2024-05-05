MODELS_RENAME = {
    'bert-base-uncased': 'BERT (Finetuned)',
    'bert-10__ndel=4fmri': 'BERT+fMRI (Finetuned)',
}


def feature_space_rename(x):
    FEATURE_SPACE_RENAME = {
        'bert-10': 'BERT',
        'eng1000': 'Eng1000',
        'finetune_roberta-base-10': 'QA-Emb (distill, probabilistic)',
        'finetune_roberta-base_binary-10': 'QA-Emb (distill, binary)',
    }
    if x in FEATURE_SPACE_RENAME:
        return FEATURE_SPACE_RENAME[x]
    x = x.replace('-10', '')
    x = x.replace('llama2-70B', 'LLaMA-2 (70B)')
    x = x.replace('llama2-7B', 'LLaMA-2 (7B)')
    x = x.replace('llama3-8B', 'LLaMA-3 (8B)')
    x = x.replace('mist-7B', 'Mistral (7B)')
    x = x.replace('ensemble1', 'Ensemble')
    if '_lay' in x:
        x = x.replace('_lay', ' (lay ') + ')'
        x = x.replace('(lay 6)', '(lay 06)')
    return x


def version_rename(x):
    if x == 'v1':
        return 'Prompts 1-3 (376 questions)'
    elif x == 'v2':
        return 'Prompts 1-5 (518 questions)'
    elif x == 'v3_boostexamples':
        return 'Prompts 1-6 (674 questions)'
    else:
        return x


DSETS_RENAME = {
    'tweet_eval': 'Tweet Eval',
    'sst2': 'SST2',
    'rotten_tomatoes': 'Rotten tomatoes',
    'moral_stories': 'Moral stories',
}


def dset_rename(x):
    if x in DSETS_RENAME:
        return DSETS_RENAME[x]
    else:
        x = x.replace('probing-', '')
        x = x.replace('_', ' ')
        return x.capitalize()
