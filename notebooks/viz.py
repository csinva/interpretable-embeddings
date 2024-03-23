MODELS_RENAME = {
    'bert-base-uncased': 'BERT (Finetuned)',
    'bert-10__ndel=4fmri': 'BERT+fMRI (Finetuned)',
}

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