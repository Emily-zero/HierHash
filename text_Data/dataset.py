"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
HDLTex: Hierarchical Deep Learning for Text Classification
load and tokenization module the input strings for deep learning model

* Copyright (C) 2018  Kamran Kowsari <kk7nc@virginia.edu>
* Last Update: Oct 26, 2018
* This file is part of  HDLTex project, University of Virginia.
* Free to use, change, share and distribute source code of RMDL
* Refrenced paper : HDLTex: Hierarchical Deep Learning for Text Classification
* Link: https://doi.org/10.1109/ICMLA.2017.0-134
* Comments and Error: email: kk7nc@virginia.edu
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
from __future__ import print_function

import re
# from HDLTex codes available

import os, sys, tarfile
import numpy as np
import torch

if sys.version_info >= (3, 0, 0):
    import urllib.request as urllib  # ugly but works
else:
    import urllib
from torch.utils import data

#print(sys.version_info)
from sklearn.feature_extraction.text import CountVectorizer
import text_Data.WOS as WOS
from sklearn.utils import shuffle
import numpy as np
import os
from torch.utils.data import Dataset,DataLoader,TensorDataset

from transformers import AutoTokenizer,AutoConfig,AutoModel
import pandas as pd

''' Location of the dataset'''
# path_WOS = WOS.download_and_extract()



'''
Base Dataset class for all the dataset used.

'''



def clean_str(string):
    """
    Tokenization/string cleaning for dataset
    Every dataset is lower cased except
    """
    string = re.sub(r"\\", "", string)
    string = re.sub(r"\'", "", string)
    string = re.sub(r"\"", "", string)
    return string.strip().lower()
def clean_tokenize_truncate(x, max_length):
    x = clean_string(x)
    x = x.split(" ")
    x = x[:max_length]
    return x


def clean_string(string):
    """
    Tokenization/string cleaning for yelp data set
    Based on https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'`\"]", " ", string)
    string = re.sub(r"\'s", " 's", string)
    string = re.sub(r"\'ve", " 've", string)
    string = re.sub(r"n\'t", " n't", string)
    string = re.sub(r"\'re", " 're", string)
    string = re.sub(r"\'d", " 'd", string)
    string = re.sub(r"\'ll", " 'll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\"\"", ' " ', string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.lower().strip()


def text_cleaner(text):
    """
    cleaning spaces, html tags, etc
    parameters: (string) text input to clean
    return: (string) clean_text
    Based on :
    """
    text = text.replace(".", "")
    text = text.replace("[", " ")
    text = text.replace(",", " ")
    text = text.replace("]", " ")
    text = text.replace("(", " ")
    text = text.replace(")", " ")
    text = text.replace("\"", "")
    text = text.replace("-", "")
    text = text.replace("=", "")
    rules = [
        {r'>\s+': u'>'},  # remove spaces after a tag opens or closes
        {r'\s+': u' '},  # replace consecutive spaces
        {r'\s*<br\s*/?>\s*': u'\n'},  # newline after a <br>
        {r'</(div)\s*>\s*': u'\n'},  # newline after </p> and </div> and <h1/>...
        {r'</(p|h\d)\s*>\s*': u'\n\n'},  # newline after </p> and </div> and <h1/>...
        {r'<head>.*<\s*(/head|body)[^>]*>': u''},  # remove <head> to </head>
        {r'<a\s+href="([^"]+)"[^>]*>.*</a>': r'\1'},  # show links instead of texts
        {r'[ \t]*<[^<]*?/?>': u''},  # remove remaining tags
        {r'^\s+': u''}  # remove spaces at the beginning
    ]
    for rule in rules:
        for (k, v) in rule.items():
            regex = re.compile(k)
            text = regex.sub(v, text)
        text = text.rstrip()
        text = text.strip()
    clean_text = text.lower()
    return clean_text

class dataset_NYT(Dataset):
    def __init__(self,DATA_DIR= '../baselines/data/nyt_c2f',max_length=50):
        self.max_length= max_length
        self.texts= []
        self.fine_labels= []
        self.coarse_labels = []
        self.DATA_DIR= DATA_DIR
        self.tokenizer = AutoTokenizer.from_pretrained("LM/bert-base-uncased")
        self.texts,self.coarse_labels,self.fine_labels = self.read_data(self.DATA_DIR,self.max_length)



    def __getitem__(self, index):
        # sentence = self.texts[index].lower()
        sentence = " ".join(self.texts[index]).lower()
        # print(index)
        # print(sentence)
        inputs = self.tokenizer(sentence, padding='max_length', truncation=True, max_length=self.max_length + 1,
                                return_tensors="pt")
        # print(inputs.get('input_ids').shape)
        # print(type(inputs))
        # print(inputs)
        # print(self.coarse_labels[index])
        return inputs, self.coarse_labels[index],self.fine_labels[index],index

    def __len__(self):
        return len(self.texts)

    def read_data(self, path, max_length):
        def label_fn(x):
            return x - 1

        rows = pd.read_csv(
            path,
            sep=",",
            error_bad_lines=False,
            header=None,
            skiprows=None,
            quoting=0,
            keep_default_na=False,
            encoding="utf-8",
        )
        label_fn = label_fn if label_fn is not None else (lambda x: x)
        label_coarse = rows[0].apply(lambda x: label_fn(x))
        label_fine = rows[1].apply(lambda x: label_fn(x))
        sentences = rows[2]

        sentences = sentences.apply(lambda x: clean_tokenize_truncate(x, max_length))
        #print(sentences[1])
        return sentences.tolist(), label_coarse.tolist(), label_fine.tolist()

class dataset_BERT(Dataset):
    def __init__(self,DATA_DIR= '../baselines/data/nyt_c2f',get_one_bert=True):
        self.get_one_bert = get_one_bert
        self.bert_0 = []
        self.bert_1 = []
        self.fine_labels= []
        self.coarse_labels = []
        self.DATA_DIR= DATA_DIR
        self.bert_dim = 768
        #self.bert_model.train()
        if get_one_bert:
            self.bert_0, _, self.coarse_labels, self.fine_labels = self.read_data(self.DATA_DIR)
        else:
            self.bert_0,self.bert_1,self.coarse_labels,self.fine_labels = self.read_data(self.DATA_DIR)



    def __getitem__(self, index):
        if self.get_one_bert:
            return self.bert_0[index], self.coarse_labels[index], self.fine_labels[index], index
        else:
            return self.bert_0[index],self.bert_1[index], self.coarse_labels[index],self.fine_labels[index],index

    def __len__(self):
        return len(self.fine_labels)


    def read_data(self, path):
        def label_fn(x):
            return int(x)
        def bert_float(x):
            return np.array(x)

        rows = pd.read_csv(
            path,
            sep=",",
            error_bad_lines=False,
            header=None,
            skiprows=None,
            quoting=0,
            keep_default_na=False
            # encoding="utf-8",
            # dtype='float'
        )
        label_fn = label_fn if label_fn is not None else (lambda x: x)
        label_coarse = rows[2*self.bert_dim].apply(lambda x: label_fn(x))
        label_fine = rows[2*self.bert_dim+1].apply(lambda x: label_fn(x))
        bert_0 = rows.iloc[:,0:self.bert_dim].values.astype(np.float32)#.apply(lambda x: bert_float(x))
        bert_1 = rows.iloc[:,self.bert_dim:2*self.bert_dim].values.astype(np.float32)#.apply(lambda x: bert_float(x))

        #print(sentences[1])
        return torch.tensor(bert_0), torch.tensor(bert_1),label_coarse.tolist(), label_fine.tolist()



