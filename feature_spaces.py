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
from ridge_utils.interp_data import lanczosinterp2D, expinterp2D, kernel_density_interp2D, nearest_neighbor_interp2D
from ridge_utils.semantic_model import SemanticModel
from transformers.pipelines.pt_utils import KeyDataset
from ridge_utils.utils_ds import apply_model_to_words, make_word_ds, make_phoneme_ds
from ridge_utils.utils_stim import load_textgrids, load_simulated_trfiles
from transformers import pipeline
import logging
import imodelsx.llm
from qa_embedder import QuestionEmbedder, FinetunedQAEmbedder
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


def downsample_word_vectors(stories, word_vectors, wordseqs, strategy='lanczos'):
    """Get Lanczos downsampled word_vectors for specified stories.

    Args:
            stories: List of stories to obtain vectors for.
            word_vectors: Dictionary of {story: <float32>[num_story_words, vector_size]}

    Returns:
            Dictionary of {story: downsampled vectors}
    """
    downsampled_semanticseqs = dict()
    for story in stories:
        if strategy == 'lanczos':
            downsampled_semanticseqs[story] = lanczosinterp2D(
                word_vectors[story],
                oldtime=wordseqs[story].data_times,  # timing of the old data
                newtime=wordseqs[story].tr_times,  # timing of the new data
                window=3
            )
        elif strategy == 'exp':
            downsampled_semanticseqs[story] = expinterp2D(
                word_vectors[story],
                oldtime=wordseqs[story].data_times,  # timing of the old data
                newtime=wordseqs[story].tr_times,  # timing of the new data
                theta=1
            )
            # downsampled_semanticseqs[story] = kernel_density_interp2D(
            #     word_vectors[story],
            #     oldtime=wordseqs[story].data_times,  # timing of the old data
            #     newtime=wordseqs[story].tr_times,  # timing of the new data
            #     bandwidth=0.1
            # )
            # downsampled_semanticseqs[story] = nearest_neighbor_interp2D(
            #     word_vectors[story],
            #     oldtime=wordseqs[story].data_times,  # timing of the old data
            #     newtime=wordseqs[story].tr_times,  # timing of the new data
            # )
        else:
            raise ValueError(f"Strategy {strategy} not recognized")
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


def get_wordrate_vectors(allstories, downsample=True, **kwargs):
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
    if downsample:
        return downsample_word_vectors(allstories, vectors, wordseqs)
    else:
        return allstories, vectors, wordseqs


def get_eng1000_vectors(allstories, downsample='lanczos', **kwargs):
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
    if downsample:
        return downsample_word_vectors(allstories, vectors, wordseqs, strategy=downsample)
    else:
        return allstories, vectors, wordseqs


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


def get_ngrams_list_main(ds, num_trs_context, num_secs_context_per_word, num_ngrams_context):
    def _get_ngrams_list_from_words_list(words_list: List[str], ngram_size: int = 5) -> List[str]:
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

    def _get_ngrams_list_from_words_list_and_times(words_list: List[str], times_list: np.ndarray[float], sec_offset: float = 4) -> List[str]:
        words_arr = np.array(words_list)
        ngrams_list = []
        for i in range(len(times_list)):
            t = times_list[i]
            t_off = t - sec_offset
            idxs = np.where(np.logical_and(
                times_list >= t_off, times_list <= t))[0]
            ngrams_list.append(' '.join(words_arr[idxs]))
        return ngrams_list

    # get ngrams_list
    if num_trs_context is not None:
        # replace each TR with text from the current TR and the TRs immediately before it
        ngrams_list = _get_ngrams_list_from_chunks(
            ds.chunks(), num_trs=num_trs_context)
        assert len(ngrams_list) == len(ds.chunks())
    elif num_secs_context_per_word is not None:
        # replace each word with the ngrams in a time window leading up to that word
        ngrams_list = _get_ngrams_list_from_words_list_and_times(
            ds.data, ds.data_times, sec_offset=num_secs_context_per_word)
        assert len(ngrams_list) == len(ds.data)
    else:
        # replace each word with an ngram leading up to that word
        ngrams_list = _get_ngrams_list_from_words_list(
            ds.data, ngram_size=num_ngrams_context)
        assert len(ngrams_list) == len(ds.data)
    return ngrams_list


def get_llm_vectors(
        allstories,
        checkpoint='bert-base-uncased',
        num_ngrams_context=10,
        num_trs_context=None,
        num_secs_context_per_word=None,
        layer_idx=None,
        qa_embedding_model='mistralai/Mistral-7B-v0.1',
        qa_questions_version='v1',
        downsample='lanczos',
        use_cache=True,
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
        elif checkpoint.startswith('finetune_'):
            return FinetunedQAEmbedder(
                checkpoint.replace('finetune_', ''), qa_questions_version=qa_questions_version)
        if not 'qa_embedder' in checkpoint:
            if 'bert' in checkpoint.lower():
                return pipeline("feature-extraction", model=checkpoint, device=0)
            elif layer_idx is not None:
                return imodelsx.llm.LLMEmbs(checkpoint=checkpoint)

    assert not (
        num_trs_context and num_secs_context_per_word), 'num_trs_context and num_secs_context_per_word are mutually exclusive'
    logging.info(f'getting wordseqs..')
    wordseqs = get_story_wordseqs(allstories)
    vectors = {}
    ngrams_list_dict = {}
    embedding_model = None  # only initialize if needed
    if 'qa_embedder' in checkpoint:
        logging.info(
            f'extracting {checkpoint} {qa_questions_version} {qa_embedding_model} embs...')
    else:
        logging.info(f'extracting {checkpoint} {qa_questions_version} embs...')

    for story_num, story in enumerate(allstories):
        args_cache = {'story': story, 'model': checkpoint, 'ngram_size': num_ngrams_context,
                      'qa_embedding_model': qa_embedding_model, 'qa_questions_version': qa_questions_version,
                      'num_trs_context': num_trs_context, 'num_secs_context_per_word': num_secs_context_per_word}
        if layer_idx is not None:
            args_cache['layer_idx'] = layer_idx
        cache_hash = sha256(args_cache)
        cache_file = join(
            cache_embs_dir, qa_questions_version, checkpoint.replace('/', '_'), f'{cache_hash}.jl')
        loaded_from_cache = False
        if os.path.exists(cache_file) and use_cache:
            logging.info(
                f'Loading cached {story_num}/{len(allstories)}: {story}')
            try:
                vectors[story] = joblib.load(cache_file)
                loaded_from_cache = True
                if not downsample:
                    ngrams_list_dict[story] = get_ngrams_list_main(
                        wordseqs[story], num_trs_context, num_secs_context_per_word, num_ngrams_context)
                # print('Loaded', story, 'vectors', vectors[story].shape,
                #   'unique', np.unique(vectors[story], return_counts=True))
            except:
                print('Error loading', cache_file)

        if not loaded_from_cache:
            ngrams_list = get_ngrams_list_main(
                wordseqs[story], num_trs_context, num_secs_context_per_word, num_ngrams_context)

            # embed the ngrams
            if embedding_model is None:
                embedding_model = _get_embedding_model(
                    checkpoint, qa_questions_version, qa_embedding_model)
            if 'qa_embedder' in checkpoint:
                print(f'Extracting {story_num}/{len(allstories)}: {story}')
                embs = embedding_model(ngrams_list, verbose=False)
            elif checkpoint.startswith('finetune_'):
                embs = embedding_model.get_embs_from_text_list(ngrams_list)
                # embs = embs.argmax(axis=-1) # get yes/no binarized`
                embs = embs[:, :, 1]  # get logit for yes
            elif 'bert' in checkpoint:
                embs = get_embs_from_text_list(
                    ngrams_list, embedding_function=embedding_model)
            # elif 'finetune' in checkpoint:

            elif layer_idx is not None:
                embs = embedding_model(
                    ngrams_list, layer_idx=layer_idx, batch_size=8)
            else:
                raise ValueError(checkpoint)

            # if num_trs_context is None:
                # embs = DataSequence(
                # embs, ds.split_inds, ds.data_times, ds.tr_times).data
            vectors[story] = deepcopy(embs)
            if not downsample:
                ngrams_list_dict[story] = deepcopy(ngrams_list)
            # print(story, 'vectors', vectors[story].shape,
            #   'unique', np.unique(vectors[story], return_counts=True))
            os.makedirs(dirname(cache_file), exist_ok=True)
            joblib.dump(embs, cache_file)

    if num_trs_context is not None:
        return vectors
    elif not downsample:
        return allstories, vectors, wordseqs, ngrams_list_dict
    else:
        return downsample_word_vectors(
            allstories, vectors, wordseqs, strategy=downsample)


############################################
########## Feature Space Creation ##########
############################################
_FEATURE_VECTOR_FUNCTIONS = {
    "wordrate": get_wordrate_vectors,
    "eng1000": get_eng1000_vectors,
    'glove': get_glove_vectors,
}
_FEATURE_CHECKPOINTS = {
    'qa_embedder': 'qa_embedder',


    # embedding models (used when not qa_embedder)
    'bert': 'bert-base-uncased',
    'distil-bert': 'distilbert-base-uncased',
    'roberta': 'roberta-large',
    'bert-sst2': 'textattack/bert-base-uncased-SST-2',
    'llama2-7B': 'meta-llama/Llama-2-7b-hf',
    'llama2-13B': 'meta-llama/Llama-2-13b-hf',
    'llama2-70B': 'meta-llama/Llama-2-70b-hf',
    'llama3-8B': 'meta-llama/Meta-Llama-3-8B',
    'finetune_roberta-base': 'finetune_roberta-base',
}
BASE_KEYS = list(_FEATURE_CHECKPOINTS.keys())
for context_length in [2, 3, 4, 5, 10, 20, 25, 50, 75]:
    for k in BASE_KEYS:
        # context length by ngrams
        _FEATURE_VECTOR_FUNCTIONS[f'{k}-{context_length}'] = partial(
            get_llm_vectors,
            num_ngrams_context=context_length,
            checkpoint=_FEATURE_CHECKPOINTS[k])
        _FEATURE_CHECKPOINTS[f'{k}-{context_length}'] = _FEATURE_CHECKPOINTS.get(
            k, k)

        # llama-2: 7B has 32 layers, 13B has 40 layers, best model is likely between 20%-50% of layers
        # llama-3: 8B has 32 layers
        for layer_idx in [0, 6, 12, 18, 24, 30, 36, 48, 60]:

            # pass with layer
            _FEATURE_VECTOR_FUNCTIONS[f'{k}_lay{layer_idx}-{context_length}'] = partial(
                get_llm_vectors,
                num_ngrams_context=context_length,
                checkpoint=_FEATURE_CHECKPOINTS[k],
                layer_idx=layer_idx,
            )
            _FEATURE_CHECKPOINTS[f'{k}_lay{layer_idx}-{context_length}'] = _FEATURE_CHECKPOINTS.get(
                k)

        # context length by TRs
        _FEATURE_VECTOR_FUNCTIONS[f'{k}-tr{context_length}'] = partial(
            get_llm_vectors,
            num_trs_context=context_length,
            checkpoint=_FEATURE_CHECKPOINTS[k])
        _FEATURE_CHECKPOINTS[f'{k}-tr{context_length}'] = _FEATURE_CHECKPOINTS.get(
            k, k)

        # context length by seconds
        _FEATURE_VECTOR_FUNCTIONS[f'{k}-sec{context_length}'] = partial(
            get_llm_vectors,
            num_secs_context_per_word=context_length,
            checkpoint=_FEATURE_CHECKPOINTS[k])
        _FEATURE_CHECKPOINTS[f'{k}-sec{context_length}'] = _FEATURE_CHECKPOINTS.get(
            k, k)


def get_features(feature, **kwargs):
    return _FEATURE_VECTOR_FUNCTIONS[feature](**kwargs)


if __name__ == '__main__':
    # feats = get_feature_space('bert-5', ['sloth'])
    print('configs', _FEATURE_VECTOR_FUNCTIONS.keys())
