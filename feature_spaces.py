from copy import deepcopy
import os
from dict_hash import sha256
import datasets
import joblib
import numpy as np
import json
from os.path import join, dirname
from functools import partial
import qa_questions
from ridge_utils.data_sequence import DataSequence
from typing import Dict, List
from tqdm import tqdm
from ridge_utils.interp_data import lanczosinterp2D
from ridge_utils.semantic_model import SemanticModel
from transformers.pipelines.pt_utils import KeyDataset
from ridge_utils.utils_ds import apply_model_to_words, make_word_ds, make_phoneme_ds
from ridge_utils.utils_stim import load_textgrids, load_simulated_trfiles
from transformers import pipeline
import logging
from qa_embedder import QuestionEmbedder
from config import repo_dir, nlp_utils_dir, em_data_dir, data_dir, results_dir, cache_embs_dir


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


def get_wordrate_vectors(allstories, **kwargs):
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


def get_eng1000_vectors(allstories, **kwargs):
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


def get_glove_vectors(allstories, **kwargs):
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


def get_embs_from_text_list(text_list: List[str], embedding_function) -> List[np.ndarray]:
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

    # Convert to np array by averaging over len
    # Embeddings are already the same size
    # Can't just convert this since seq lens vary
    # Need to avg over seq_len dim
    logging.info('\tPostprocessing embs...')
    embs = np.zeros((len(embs_list), len(embs_list[0])))
    num_ngrams = len(embs_list)
    dim_size = len(embs_list[0][0][0])
    embs = np.zeros((num_ngrams, dim_size))
    for i in tqdm(range(num_ngrams)):
        # embs_list is (batch_size, 1, (seq_len + 2), 768) -- BERT adds initial / final tokens
        embs[i] = np.mean(embs_list[i], axis=1)  # avg over seq_len dim
    return embs


def get_ngrams_list_from_words_list(words_list: List[str], ngram_size: int = 5) -> List[str]:
    """Concatenate running list of words into grams with spaces in between
    """
    ngrams_list = []
    for i in range(len(words_list)):
        l = max(0, i - ngram_size)
        ngram = ' '.join(words_list[l: i + 1])
        ngrams_list.append(ngram.strip())
    return ngrams_list


def _get_ngrams_list_from_chunks(chunks, num_trs=2):
    ngrams_list = []
    for i in range(len(chunks)):
        # print(chunks[i - num_trs:i])
        # sum(chunks[i - num_trs:i], [])
        chunk_block = chunks[i - num_trs:i]
        if len(chunk_block) == 0:
            ngrams_list.append('')
        else:
            chunk_block = np.concatenate(chunk_block)
            ngrams_list.append(' '.join(chunk_block))
    return ngrams_list


def get_llm_vectors(
        allstories,
        checkpoint='bert-base-uncased',
        num_ngrams_context=10,
        num_trs_context=None,
        qa_embedding_model='mistralai/Mistral-7B-v0.1',
        qa_questions_version='v1'
) -> Dict[str, np.ndarray]:
    """Get llm embedding vectors
    """

    def _get_embedding_model(checkpoint, qa_questions_version, qa_embedding_model):
        print('loading embedding_model...')
        if 'qa_embedder' in checkpoint:
            questions = qa_questions.get_questions(
                version=qa_questions_version)
            return QuestionEmbedder(
                checkpoint=qa_embedding_model, questions=questions)
        if not 'qa_embedder' in checkpoint:
            return pipeline("feature-extraction", model=checkpoint, device=0)

    # This loop function works at the level of individual words, embeds the ngram leading up to each word, and then interpolates them.
    # Alternatively, we could have used wordseqs[story].chunks() to combine each TR.
    print(f'getting wordseqs..')
    wordseqs = get_story_wordseqs(allstories)
    vectors = {}
    os.makedirs(cache_embs_dir, exist_ok=True)
    embedding_model = None  # only initialize if needed
    print(f'extracting {checkpoint} embs...')
    for story_num, story in enumerate(allstories):
        args_cache = {'story': story, 'model': checkpoint, 'ngram_size': num_ngrams_context,
                      'qa_embedding_model': qa_embedding_model, 'qa_questions_version': qa_questions_version}
        if num_trs_context is not None:
            args_cache['num_trs_context'] = num_trs_context
            args_cache['ngram_size'] = None
        cache_hash = sha256(args_cache)
        cache_file = join(
            cache_embs_dir, f'{cache_hash}.jl')
        if os.path.exists(cache_file):
            print(f'Loading cached {story_num}/{len(allstories)}: {story}')
            vectors[story] = joblib.load(cache_file)
        else:
            if embedding_model is None:
                embedding_model = _get_embedding_model(
                    checkpoint, qa_questions_version, qa_embedding_model)

            ds = wordseqs[story]

            if num_trs_context is not None:
                # replace each TR with text from the current TR and the TRs immediately before it
                ngrams_list = _get_ngrams_list_from_chunks(
                    ds.chunks(), num_trs=num_trs_context)
            else:
                # replace each word with an ngram leading up to that word
                ngrams_list = get_ngrams_list_from_words_list(
                    ds.data, ngram_size=num_ngrams_context)

            # embed the ngrams
            if 'qa_embedder' in checkpoint:
                print(f'Extracting {story_num}/{len(allstories)}: {story}')
                embs = embedding_model(ngrams_list, verbose=False)
            else:
                embs = get_embs_from_text_list(
                    ngrams_list, embedding_function=embedding_model)

            if num_trs_context is None:
                embs = DataSequence(
                    embs, ds.split_inds, ds.data_times, ds.tr_times).data

            vectors[story] = deepcopy(embs)
            joblib.dump(embs, cache_file)

    if num_trs_context is not None:
        return vectors
    else:
        return downsample_word_vectors(
            allstories, vectors, wordseqs)


############################################
########## Feature Space Creation ##########
############################################
_FEATURE_VECTOR_FUNCTIONS = {
    "wordrate": get_wordrate_vectors,
    "eng1000": get_eng1000_vectors,
    'glove': get_glove_vectors,
}
_FEATURE_CHECKPOINTS = {
    'bert': 'bert-base-uncased',
    'distil-bert': 'distilbert-base-uncased',
    'roberta': 'roberta-large',
    'bert-sst2': 'textattack/bert-base-uncased-SST-2',
    'qa_embedder': 'qa_embedder',
}
BASE_KEYS = list(_FEATURE_CHECKPOINTS.keys())
for context_length in [2, 3, 4, 5, 10, 20]:
    for k in BASE_KEYS:
        # context length by ngrams
        _FEATURE_VECTOR_FUNCTIONS[f'{k}-{context_length}'] = partial(
            get_llm_vectors,
            num_ngrams_context=context_length,
            checkpoint=_FEATURE_CHECKPOINTS[k])
        _FEATURE_CHECKPOINTS[f'{k}-{context_length}'] = _FEATURE_CHECKPOINTS.get(
            k, k)

        # context length by TRs
        _FEATURE_VECTOR_FUNCTIONS[f'{k}-tr{context_length}'] = partial(
            get_llm_vectors,
            num_trs_context=context_length,
            checkpoint=_FEATURE_CHECKPOINTS[k])
        _FEATURE_CHECKPOINTS[f'{k}-tr{context_length}'] = _FEATURE_CHECKPOINTS.get(
            k, k)


def get_features(feature, **kwargs):
    return _FEATURE_VECTOR_FUNCTIONS[feature](**kwargs)


if __name__ == '__main__':
    # feats = get_feature_space('bert-5', ['sloth'])
    print('configs', _FEATURE_VECTOR_FUNCTIONS.keys())
