import numpy as np
import logging
import sys
import joblib
import matplotlib.pyplot as plt
import torch
from ridge_utils.DataSequence import DataSequence
from transformers import AutoTokenizer, AutoModelForCausalLM


### Warning, you are entering tokenization hell.

def compute_correct_tokens_opt(acc, acc_lookback, acc_offset, total_len):
    #print(acc)
    new_tokens = []
    new_tokens.append(2) # Special OPT start token
    acc_count_all = 0
    first_word = max(0,acc_offset-acc_lookback)
    last_word = min(acc_offset+1, total_len)
    acc_start = 0
    while acc_start != first_word + 1:
        if acc[acc_count_all] == 27:
            acc_start += 1
            acc_count_all += 1
        else:
            acc_count_all += 1
    
    acc2 = acc[acc_count_all:]
    acc_count8 = 0
    acc_count_all = 0
    while acc_count8 != (last_word - first_word):
        if acc2[acc_count_all] == 27:
            acc_count8 += 1
            acc_count_all += 1
        else:
            new_tokens.append(acc2[acc_count_all])
            acc_count_all += 1
    return new_tokens



def generate_efficient_feat_dicts_opt(wordseqs, tokenizer, lookback1, lookback2):
    text_dict = {}
    text_dict2 = {}
    text_dict3 = {}
    for story in wordseqs.keys():
        ds = wordseqs[story]
        newdata = []
        total_len = len(ds.data)
        acc = []
        acc8 = 0
        text = [" ".join(ds.data)]
        text_len = len(text[0])
        inputs = tokenizer(text, return_tensors="pt")
        tokens = np.array(inputs['input_ids'][0])
        assert (27 not in tokens)
        # Annotate word boundaries
        for ei,i in enumerate(tokens):
            # A lot of tokenization edge cases
            if (tokenizer.decode(torch.tensor([i]))[0] == ' ' and tokenizer.decode(torch.tensor([i])).strip() != '') or (tokenizer.decode(torch.tensor([i])) != '</s>' and ei == 1):
                acc.append(27)
                acc.append(i)
                acc8 += 1
            elif (ei==1860 and i == 2836) or (ei==349 and i == 1437) or (ei==365 and i == 1437) or (ei==1914 and i == 1437) or (ei==1305 and i == 1437) or (ei==300 and i==1437 and story=='beneaththemushroomcloud') or (ei==202 and i == 3432) or (ei==1316 and i==4514) or (ei==656 and i==2550) or (ei==1358 and i==6355) or (ei==2160 and i==8629) or (i==24929 and ei != 2):
                acc.append(27)
                acc.append(i)
                acc8 += 1
            else:
                acc.append(i)
        acc.append(27)
        #print(acc)
        lookback1 = 256
        lookback2 = 512
        acc_lookback = 0
        misc_offset = 0
        new_tokens = [2]
        #print(tokenizer.decode(new_tokens))
        for i, w in enumerate(ds.data):
            if w.strip() != '' and w != "'s":
                if acc_lookback < lookback1:
                    new_tokens = compute_correct_tokens_opt(acc, acc_lookback, i + misc_offset, total_len)
                    #print(tokenizer.decode(torch.tensor(new_tokens)))
                    text_dict[(story, i)] = new_tokens
                    text_dict2[(story, i)] = False
                    text_dict3[tuple(new_tokens)] = False
                elif lookback2 > acc_lookback and acc_lookback >= lookback1:
                    new_tokens = compute_correct_tokens_opt(acc, acc_lookback, i + misc_offset, total_len)
                    #print(tokenizer.decode(torch.tensor(new_tokens)))
                    text_dict[(story, i)] = new_tokens
                    text_dict2[(story, i)] = False
                    text_dict3[tuple(new_tokens)] = False
                elif acc_lookback == lookback2:
                    new_tokens = compute_correct_tokens_opt(acc, acc_lookback, i + misc_offset, total_len)
                    #print(tokenizer.decode(torch.tensor(new_tokens)))
                    acc_lookback = lookback1
                    text_dict[(story, i)] = new_tokens
                    text_dict2[(story, i)] = True
                    text_dict3[tuple(new_tokens)] = False
                else:
                    print("WARNING, LOOKBACK EDGE CASE 1", acc_lookback, "\n")
                    assert False
                    #print(max(0, i-acc_lookback), min(i+1, total_len))
                    #text = [" ".join(ds.data[max(0,i-acc_lookback):min(i+1,total_len)])][0]
                    #print(text)
                    text_dict[(story, i)] = new_tokens
                    text_dict2[(story, i)] = False
                    text_dict3[tuple(new_tokens)] = False
            else:
                #hidden_states = np.zeros((1024,))
                text_dict[(story, i)] = new_tokens
                text_dict2[(story, i)] = True
                text_dict3[tuple(new_tokens)] = False
                acc_lookback += 1
                misc_offset -= 1
                continue
            acc_lookback += 1
            if i == total_len - 1:
                text_dict2[(story, i)] = True
    return text_dict, text_dict2, text_dict3


def convert_to_feature_mats_opt(wordseqs, tokenizer, lookback1, lookback2, text_dict3):
    text_dict = {}
    text_dict2 = {}
    featureseqs = {}
    for story in wordseqs.keys():
        ds = wordseqs[story]
        newdata = []
        total_len = len(ds.data)
        acc = []
        acc8 = 0
        text = [" ".join(ds.data)]
        text_len = len(text[0])
        inputs = tokenizer(text, return_tensors="pt")
        tokens = np.array(inputs['input_ids'][0])
        assert (27 not in tokens)
        # Annotate word boundaries
        for ei,i in enumerate(tokens):
            # A lot of tokenization edge cases
            if (tokenizer.decode(torch.tensor([i]))[0] == ' ' and tokenizer.decode(torch.tensor([i])).strip() != '') or (tokenizer.decode(torch.tensor([i])) != '</s>' and ei == 1):
                acc.append(27)
                acc.append(i)
                acc8 += 1
            elif (ei==1860 and i == 2836) or (ei==349 and i == 1437) or (ei==365 and i == 1437) or (ei==1914 and i == 1437) or (ei==1305 and i == 1437) or (ei==300 and i==1437 and story=='beneaththemushroomcloud') or (ei==202 and i == 3432) or (ei==1316 and i==4514) or (ei==656 and i==2550) or (ei==1358 and i==6355) or (ei==2160 and i==8629) or (i==24929 and ei != 2):
                acc.append(27)
                acc.append(i)
                acc8 += 1
            else:
                acc.append(i)
        acc.append(27)
        lookback1 = 256
        lookback2 = 512
        acc_lookback = 0
        misc_offset = 0
        new_tokens = [2]
        for i, w in enumerate(ds.data):
            if w.strip() != '' and w != "'s":
                if acc_lookback < lookback1:
                    new_tokens = compute_correct_tokens_opt(acc, acc_lookback, i + misc_offset, total_len)
                    text_dict[(story, i)] = new_tokens
                    text_dict2[(story, i)] = False
                    newdata.append(text_dict3[tuple(new_tokens)])
                elif lookback2 > acc_lookback and acc_lookback >= lookback1:
                    new_tokens = compute_correct_tokens_opt(acc, acc_lookback, i + misc_offset, total_len)
                    text_dict[(story, i)] = new_tokens
                    text_dict2[(story, i)] = False
                    newdata.append(text_dict3[tuple(new_tokens)])
                elif acc_lookback == lookback2:
                    new_tokens = compute_correct_tokens_opt(acc, acc_lookback, i + misc_offset, total_len)
                    acc_lookback = lookback1
                    text_dict[(story, i)] = new_tokens
                    text_dict2[(story, i)] = True
                    newdata.append(text_dict3[tuple(new_tokens)])
                else:
                    print("WARNING, LOOKBACK EDGE CASE 1", acc_lookback, "\n")
                    assert False
                    text_dict[(story, i)] = new_tokens
                    text_dict2[(story, i)] = False
                    newdata.append(text_dict3[tuple(new_tokens)])
            else:
                text_dict[(story, i)] = new_tokens
                text_dict2[(story, i)] = True
                newdata.append(text_dict3[tuple(new_tokens)])
                acc_lookback += 1
                misc_offset -= 1
                continue
            acc_lookback += 1
            if i == total_len - 1:
                text_dict2[(story, i)] = True
        featureseqs[story] = DataSequence(np.array(newdata), ds.split_inds, ds.data_times, ds.tr_times)
    downsampled_featureseqs = {}
    for story in featureseqs:
        downsampled_featureseqs[story] = featureseqs[story].chunksums('lanczos', window=3)
    return downsampled_featureseqs


def compute_correct_tokens_llama(acc, acc_lookback, acc_offset, total_len):
    new_tokens = [1]
    acc_count_all = 0
    first_word = max(0,acc_offset-acc_lookback)
    last_word = min(acc_offset+1, total_len)
    acc_start = 0
    while acc_start != first_word + 1:
        if acc[acc_count_all] == 29947:
            acc_start += 1
            acc_count_all += 1
        else:
            acc_count_all += 1
    acc2 = acc[acc_count_all:]
    acc_count8 = 0
    acc_count_all = 0
    while acc_count8 != (last_word - first_word):
        if acc2[acc_count_all] == 29947:
            acc_count8 += 1
            acc_count_all += 1
        else:
            new_tokens.append(acc2[acc_count_all])
            acc_count_all += 1
    return new_tokens

def generate_efficient_feat_dicts_llama(wordseqs, tokenizer, lookback1, lookback2):
    text_dict = {}
    text_dict2 = {}
    text_dict3 = {}
    for es, story in enumerate(wordseqs.keys()):
        #print(story)
        ds = wordseqs[story]
        total_len = len(ds.data)
        text = [" ".join(ds.data)]
        inputs = tokenizer(text, return_tensors="pt")
        tokens = np.array(inputs['input_ids'][0])
        assert (29947 not in tokens) # Use a dummy token '8' for marking word cutoffs
        acc = [1] # Contexts should start with special START token
        acc8 = 0
        acc_words = 0
        for ei,i in enumerate(tokens):
            if tokenizer.convert_ids_to_tokens(torch.tensor([i]))[0][0] == '▁'  and (tokenizer.decode(torch.tensor([i])).strip() != ''):
                acc.append(29947)
                acc.append(i)
                acc8 += 1
            elif ei != (len(tokens) - 1):
                if (i == 29871) and (tokenizer.convert_ids_to_tokens(torch.tensor([tokens[ei+1]]))[0][0] != '▁'):
                    acc.append(29947)
                    acc.append(i)
                    acc8 += 1
                else:
                    acc.append(i)
            else:
                acc.append(i)
        decoded = tokenizer.decode(torch.tensor(acc))
        acc_words = 0
        for i in ds.data:
            if i.strip() != '':
                acc_words += 1
        #print(acc8, acc_words, story, es)
        assert acc8 == acc_words # Number of annotations should equal number of words
        acc.append(29947)
        acc_lookback = 0
        misc_offset = 0
        new_tokens = [1]
        for i, w in enumerate(ds.data):
            if w.strip() != '' and w != "'s":
                if acc_lookback < lookback1 or (lookback2 > acc_lookback and acc_lookback >= lookback1):
                    new_tokens = compute_correct_tokens_llama(acc, acc_lookback, i + misc_offset, total_len)
                    text_dict[(story, i)] = new_tokens
                    text_dict2[(story, i)] = False
                    text_dict3[tuple(new_tokens)] = False
                elif acc_lookback == lookback2:
                    new_tokens = compute_correct_tokens_llama(acc, acc_lookback, i + misc_offset, total_len)
                    acc_lookback = lookback1
                    text_dict[(story, i)] = new_tokens
                    text_dict2[(story, i)] = True
                    text_dict3[tuple(new_tokens)] = False
                else:
                    assert False
            else:
                text_dict[(story, i)] = new_tokens
                text_dict2[(story, i)] = True
                text_dict3[tuple(new_tokens)] = False
                acc_lookback += 1
                misc_offset -= 1
                continue
            acc_lookback += 1
            if i == total_len - 1:
                text_dict2[(story, i)] = True
    return text_dict, text_dict2, text_dict3

def convert_to_feature_mats_llama(wordseqs, tokenizer, lookback1, lookback2, text_dict3):
    text_dict = {}
    text_dict2 = {}
    featureseqs = {}
    for es, story in enumerate(wordseqs.keys()):
        #print(story)
        ds = wordseqs[story]
        newdata = []
        total_len = len(ds.data)
        text = [" ".join(ds.data)]
        inputs = tokenizer(text, return_tensors="pt")
        tokens = np.array(inputs['input_ids'][0])
        assert (29947 not in tokens) # Use a dummy token '8' for marking word cutoffs
        acc = [1] # Contexts should start with special START token
        acc8 = 0
        acc_words = 0
        for ei,i in enumerate(tokens):
            if tokenizer.convert_ids_to_tokens(torch.tensor([i]))[0][0] == '▁'  and (tokenizer.decode(torch.tensor([i])).strip() != ''):
                acc.append(29947)
                acc.append(i)
                acc8 += 1
            elif ei != (len(tokens) - 1):
                if (i == 29871) and (tokenizer.convert_ids_to_tokens(torch.tensor([tokens[ei+1]]))[0][0] != '▁'):
                    acc.append(29947)
                    acc.append(i)
                    acc8 += 1
                else:
                    acc.append(i)
            else:
                acc.append(i)
        decoded = tokenizer.decode(torch.tensor(acc))
        acc_words = 0
        for i in ds.data:
            if i.strip() != '':
                acc_words += 1
        #print(acc8, acc_words, story, es)
        assert acc8 == acc_words # Number of annotations should equal number of words
        acc.append(29947)
        acc_lookback = 0
        misc_offset = 0
        new_tokens = [1]
        for i, w in enumerate(ds.data):
            if w.strip() != '' and w != "'s":
                if acc_lookback < lookback1 or (lookback2 > acc_lookback and acc_lookback >= lookback1):
                    new_tokens = compute_correct_tokens_llama(acc, acc_lookback, i + misc_offset, total_len)
                    text_dict[(story, i)] = new_tokens
                    text_dict2[(story, i)] = False
                    newdata.append(text_dict3[tuple(new_tokens)])
                elif acc_lookback == lookback2:
                    new_tokens = compute_correct_tokens_llama(acc, acc_lookback, i + misc_offset, total_len)
                    acc_lookback = lookback1
                    text_dict[(story, i)] = new_tokens
                    text_dict2[(story, i)] = True
                    newdata.append(text_dict3[tuple(new_tokens)])
                else:
                    assert False
            else:
                text_dict[(story, i)] = new_tokens
                text_dict2[(story, i)] = True
                newdata.append(text_dict3[tuple(new_tokens)])
                acc_lookback += 1
                misc_offset -= 1
                continue
            acc_lookback += 1
            if i == total_len - 1:
                text_dict2[(story, i)] = True
        featureseqs[story] = DataSequence(np.array(newdata), ds.split_inds, ds.data_times, ds.tr_times)
        downsampled_featureseqs = {}
        for story in featureseqs:
            downsampled_featureseqs[story] = featureseqs[story].chunksums('lanczos', window=3)
        return downsampled_featureseqs
