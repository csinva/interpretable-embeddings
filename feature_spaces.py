import os
import sys
import datasets
import numpy as np
import json
from os.path import join, dirname
from functools import partial
from ridge_utils.DataSequence import DataSequence
from typing import Dict, List
from tqdm import tqdm
from ridge_utils.interpdata import lanczosinterp2D
from ridge_utils.SemanticModel import SemanticModel
from transformers.pipelines.pt_utils import KeyDataset
from ridge_utils.dsutils import apply_model_to_words, make_word_ds, make_phoneme_ds
from ridge_utils.stimulus_utils import load_textgrids, load_simulated_trfiles
from transformers import pipeline
import logging


# join(dirname(dirname(os.path.abspath(__file__))))
repo_dir = '/home/chansingh/mntv1/deep-fMRI'
nlp_utils_dir = '/home/chansingh/nlp_utils'
em_data_dir = join(repo_dir, 'em_data')
data_dir = join(repo_dir, 'data')
results_dir = join(repo_dir, 'results')


def get_story_wordseqs(stories) -> Dict[str, DataSequence]:
    grids = load_textgrids(stories, data_dir)
    with open(join(data_dir, "ds003020/derivative/respdict.json"), "r") as f:
        respdict = json.load(f)
    trfiles = load_simulated_trfiles(respdict)
    wordseqs = make_word_ds(grids, trfiles)
    return wordseqs


def get_story_phonseqs(stories):
    grids = load_textgrids(stories, data_dir)
    with open(join(data_dir, "ds003020/derivative/respdict.json"), "r") as f:
        respdict = json.load(f)
    trfiles = load_simulated_trfiles(respdict)
    wordseqs = make_phoneme_ds(grids, trfiles)
    return wordseqs


def downsample_word_vectors(stories, word_vectors, wordseqs):
    """Get Lanczos downsampled word_vectors for specified stories.

    Args:
            stories: List of stories to obtain vectors for.
            word_vectors: Dictionary of {story: <float32>[num_story_words, vector_size]}

    Returns:
            Dictionary of {story: downsampled vectors}
    """
    downsampled_semanticseqs = dict()
    for story in stories:
        downsampled_semanticseqs[story] = lanczosinterp2D(
            word_vectors[story],
            oldtime=wordseqs[story].data_times,  # timing of the old data
            newtime=wordseqs[story].tr_times,  # timing of the new data
            window=3
        )
    return downsampled_semanticseqs


def ph_to_articulate(ds, ph_2_art):
    """ Following make_phoneme_ds converts the phoneme DataSequence object to an 
    articulate Datasequence for each grid.
    """
    articulate_ds = []
    for ph in ds:
        try:
            articulate_ds.append(ph_2_art[ph])
        except:
            articulate_ds.append([""])
    return articulate_ds


articulates = [
    "bilabial", "postalveolar", "alveolar", "dental", "labiodental",
    "velar", "glottal", "palatal", "plosive", "affricative", "fricative",
    "nasal", "lateral", "approximant", "voiced", "unvoiced", "low", "mid",
    "high", "front", "central", "back"
]


def histogram_articulates(ds, data, articulateset=articulates):
    """Histograms the articulates in the DataSequence [ds]."""
    final_data = []
    for art in ds:
        final_data.append(np.isin(articulateset, art))
    final_data = np.array(final_data)
    return (final_data, data.split_inds, data.data_times, data.tr_times)


def get_articulation_vectors(allstories):
    """Get downsampled articulation vectors for specified stories.
    Args:
            allstories: List of stories to obtain vectors for.
    Returns:
            Dictionary of {story: downsampled vectors}
    """
    with open(join(em_data_dir, "articulationdict.json"), "r") as f:
        artdict = json.load(f)
    # (phonemes, phoneme_times, tr_times)
    phonseqs = get_story_phonseqs(allstories)
    downsampled_arthistseqs = {}
    for story in allstories:
        olddata = np.array(
            [ph.upper().strip("0123456789") for ph in phonseqs[story].data])
        ph_2_art = ph_to_articulate(olddata, artdict)
        arthistseq = histogram_articulates(ph_2_art, phonseqs[story])
        downsampled_arthistseqs[story] = lanczosinterp2D(
            arthistseq[0], arthistseq[2], arthistseq[3])
    return downsampled_arthistseqs


def get_phonemerate_vectors(allstories):
    """Get downsampled phonemerate vectors for specified stories.
    Args:
            allstories: List of stories to obtain vectors for.
    Returns:
            Dictionary of {story: downsampled vectors}
    """
    with open(join(em_data_dir, "articulationdict.json"), "r") as f:
        artdict = json.load(f)
    # (phonemes, phoneme_times, tr_times)
    phonseqs = get_story_phonseqs(allstories)
    downsampled_arthistseqs = {}
    for story in allstories:
        olddata = np.array(
            [ph.upper().strip("0123456789") for ph in phonseqs[story].data])
        ph_2_art = ph_to_articulate(olddata, artdict)
        arthistseq = histogram_articulates(ph_2_art, phonseqs[story])
        nphonemes = arthistseq[0].shape[0]
        phonemerate = np.ones([nphonemes, 1])
        downsampled_arthistseqs[story] = lanczosinterp2D(
            phonemerate, arthistseq[2], arthistseq[3])
    return downsampled_arthistseqs


def get_wordrate_vectors(allstories):
    """Get wordrate vectors for specified stories.

    Args:
            allstories: List of stories to obtain vectors for.

    Returns:
            Dictionary of {story: downsampled vectors}
    """
    eng1000 = SemanticModel.load(join(em_data_dir, "english1000sm.hf5"))
    wordseqs = get_story_wordseqs(allstories)
    vectors = {}
    for story in allstories:
        nwords = len(wordseqs[story].data)
        vectors[story] = np.ones([nwords, 1])
    return downsample_word_vectors(allstories, vectors, wordseqs)


def get_eng1000_vectors(allstories):
    """Get Eng1000 vectors (985-d) for specified stories.

    Args:
            allstories: List of stories to obtain vectors for.

    Returns:
            Dictionary of {story: downsampled vectors}
    """
    eng1000 = SemanticModel.load(join(em_data_dir, "english1000sm.hf5"))
    wordseqs = get_story_wordseqs(allstories)
    vectors = {}
    for story in allstories:
        sm = apply_model_to_words(wordseqs[story], eng1000, 985)
        vectors[story] = sm.data
    return downsample_word_vectors(allstories, vectors, wordseqs)


def get_glove_vectors(allstories):
    """Get glove vectors (300-d) for specified stories.

    Args:
            allstories: List of stories to obtain vectors for.

    Returns:
            Dictionary of {story: downsampled vectors}
    """
    glove = SemanticModel.load_np(join(nlp_utils_dir, 'glove'))
    wordseqs = get_story_wordseqs(allstories)
    vectors = {}
    for story in allstories:
        sm = apply_model_to_words(wordseqs[story], glove, 300)
        vectors[story] = sm.data
    return downsample_word_vectors(allstories, vectors, wordseqs)


def get_ngrams_list_from_words_list(words_list: List[str], ngram_size: int = 5) -> List[str]:
    """Concatenate running list of words into grams with spaces in between
    """
    ngrams_list = []
    for i in range(len(words_list)):
        l = max(0, i - ngram_size)
        ngram = ' '.join(words_list[l: i + 1])
        ngrams_list.append(ngram)
    return ngrams_list


def get_embs_list_from_text_list(text_list: List[str], embedding_function) -> List[np.ndarray]:
    """

    Params
    ------
    embedding_function
        ngram -> fixed size vector

    Returns
    -------
    embs: np.ndarray (len(text_list), embedding_size)
    """

    # ngrams_list = ngrams_list[:100]
    text = datasets.Dataset.from_dict({'text': text_list})

    # get embeddings
    def get_emb(x):
        return {'emb': embedding_function(x['text'])}
    embs_list = text.map(get_emb)['emb']  # embedding_function(text)


    # This allows for parallelization when passing batch_size, but sometimes throws "Killed" error
    """
    embs_list = []
    for out in tqdm(embedding_function(KeyDataset(text, "text")),
                    total=len(text)):  # , truncation="only_first"):
        embs_list.append(out)
    """
    return embs_list


def convert_embs_list_to_np_arr(embs_list: List[np.ndarray], avg_over_seq_len=True) -> np.ndarray:
    """Convert to np array by averaging over len
    """
    # Embeddings are already the same size
    if not avg_over_seq_len:
        embs = np.array(embs_list).squeeze() #.mean(axis=1)

    # Can't just convert this since seq lens vary
    # Need to avg over seq_len dim
    else:
        logging.info('\tPostprocessing embs...')
        embs = np.zeros((len(embs_list), embs_list[0].shape[0]))
        num_ngrams = len(embs_list)
        dim_size = len(embs_list[0][0][0])
        embs = np.zeros((num_ngrams, dim_size))
        for i in tqdm(range(num_ngrams)):
            # embs_list is (batch_size, 1, (seq_len + 2), 768) -- BERT adds initial / final tokens
            embs[i] = np.mean(embs_list[i], axis=1)  # avg over seq_len dim
    return embs


def get_llm_vectors(allstories, model='bert-base-uncased', ngram_size=5):
    """Get llm embedding vectors
    """
    if model.startswith('gpt3'):
        import openai
        from tenacity import retry, wait_random_exponential, stop_after_attempt

        @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(3))
        def get_embedding(text: str, engine="text-similarity-davinci-001") -> List[float]:
            text = text.replace("\n", " ")  # replace newlines
            if len(text) == 0:
                text = '  '
            emb = openai.Embedding.create(input=[text], engine=engine)[
                "data"][0]["embedding"]
            return np.array(emb)
        avg_over_seq_len = False
    else:
        get_embedding = pipeline("feature-extraction", model=model, device=0)
        avg_over_seq_len = True
    wordseqs = get_story_wordseqs(allstories)
    vectors = {}

    print(f'extracting {model} embs...')
    for story in tqdm(allstories):
        ds = wordseqs[story]
        ngrams_list = get_ngrams_list_from_words_list(
            ds.data, ngram_size=ngram_size)
        embs = get_embs_list_from_text_list(
            ngrams_list, embedding_function=get_embedding)
        embs = convert_embs_list_to_np_arr(
            embs, avg_over_seq_len=avg_over_seq_len)
        vectors[story] = DataSequence(
            embs, ds.split_inds, ds.data_times, ds.tr_times).data
    return downsample_word_vectors(allstories, vectors, wordseqs)


############################################
########## Feature Space Creation ##########
############################################
_FEATURE_VECTOR_FUNCTIONS = {
    "articulation": get_articulation_vectors,
    "phonemerate": get_phonemerate_vectors,
    "wordrate": get_wordrate_vectors,
    "eng1000": get_eng1000_vectors,
    'glove': get_glove_vectors,
}
_FEATURE_CHECKPOINTS = {
    'bert': 'bert-base-uncased',
    'roberta': 'roberta-large',
    'bert-sst2': 'textattack/bert-base-uncased-SST-2',
    'gpt3': 'gpt3',
}
BASE_KEYS = list(_FEATURE_CHECKPOINTS.keys())
for ngram_size in [3, 5, 10, 20]:
    for k in BASE_KEYS:
        _FEATURE_VECTOR_FUNCTIONS[f'{k}-{ngram_size}'] = partial(get_llm_vectors,
                                                                 ngram_size=ngram_size, model=_FEATURE_CHECKPOINTS[k])
        _FEATURE_CHECKPOINTS[f'{k}-{ngram_size}'] = _FEATURE_CHECKPOINTS[k]


def get_feature_space(feature, *args):
    return _FEATURE_VECTOR_FUNCTIONS[feature](*args)


if __name__ == '__main__':
    # feats = get_feature_space('bert-5', ['sloth'])
    print('configs', _FEATURE_VECTOR_FUNCTIONS.keys())
