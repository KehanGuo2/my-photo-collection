import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from pytorch_transformers import *
import torch.utils.data as Data
import pickle

class Translator:
    """Backtranslation. Here to save time, we pre-processing and save all the translated data into pickle files.
    """

    def __init__(self, path, transform_type='BackTranslation'):
        # Pre-processed German data
        with open(path + 'de_1.pkl', 'rb') as f:
            self.de = pickle.load(f)
        # Pre-processed Russian data
        with open(path + 'ru_1.pkl', 'rb') as f:
            self.ru = pickle.load(f)

    def __call__(self, ori, idx):
        out1 = self.de[idx]
        out2 = self.ru[idx]
        return out1, out2, ori

# def get_data(n_labeled_per_class, max_seq_len=256, model='bert-base-uncased', train_aug=False):
#     """Read data, split the dataset, and build dataset for dataloaders.
#
#     Arguments:
#         data_path {str} -- Path to your dataset folder: contain a train.csv and test.csv
#         n_labeled_per_class {int} -- Number of labeled data per class
#
#     Keyword Arguments:
#         unlabeled_per_class {int} -- Number of unlabeled data per class (default: {5000})
#         max_seq_len {int} -- Maximum sequence length (default: {256})
#         model {str} -- Model name (default: {'bert-base-uncased'})
#         train_aug {bool} -- Whether performing augmentation on labeled training set (default: {False})
#
#     """
#     # Load the tokenizer for bert
#
#     data_path = './data/'
#     train_df = pd.read_csv(data_path+'train.csv', header = 0)
#     train_df.drop('Unnamed: 0', inplace=True, axis=1)
#
#     test_df = pd.read_csv(data_path+'test.csv', header = 0)
#     test_df.drop('Unnamed: 0', inplace=True, axis=1)
#
#     # val_df = pd.read_csv(data_path+'val.csv',header = 0)
#     # val_df.drop('Unnamed: 0', inplace=True, axis=1)
#
#
#     # Here we only use the bodies and removed titles to do the classifications
#     # train_labels = np.array([v-1 for v in train_df[0]])
#     train_labels_col = [col for col in train_df.columns if col not in ['Question_contend']]
#     train_labels_col = train_labels_col[1:]
#     train_text = np.array(train_df['Question_contend'])
#     train_labels = np.array(train_df[train_labels_col])
#
#     # val_labels_col =  [col for col in val_df.columns if col not in ['Question_contend']]
#     # val_labels_col = np.array(val_labels_col[1:])
#     # val_text = np.array(val_df['Question_contend'])
#     # val_labels = np.array(val_df[val_labels_col])
#
#     test_labels_col = [col for col in test_df.columns if col not in ['Question_contend']]
#     test_labels_col = np.array(test_labels_col[1:])
#     test_text = np.array(test_df['Question_contend'])
#     test_labels = np.array(test_df[test_labels_col])
#
#
#     n_labels = len(test_labels) + 1
#
#     # Build the dataset class for each set
#     train_labeled_dataset = loader_labeled(
#         train_text, train_labels, tokenizer, max_seq_len, train_aug)
#     # train_unlabeled_dataset = loader_unlabeled(
#     #     train_text[train_unlabeled_idxs], train_unlabeled_idxs, tokenizer, max_seq_len, Translator(data_path))
#     # val_dataset = loader_labeled(
#     #     train_text, train_labels, tokenizer, max_seq_len)
#     test_dataset = loader_labeled(
#         test_text, test_labels, tokenizer, max_seq_len)
#
#     print("#Labeled: {},  Test {}".format(len(
#         train_text),  len(test_labels)))
#
#     return train_df, test_df,n_labels

def get_data(train_df,val_df,max_seq_len=256,model='bert-base-uncased',train_aug = False,prefix):
    tokenizer = BertTokenizer.from_pretrained(model)
    prompt = ""

    train_labels_col = [col for col in train_df.columns if col not in ['Question_contend']]
    train_labels_col = train_labels_col[1:]

    task_prefix = np.array(train_labels_col)

    if prefix:
        for _ in task_prefix:
            prompt = prompt + " " + _

        train_text = [prompt + sentence for sentence in np.array(train_df['Question_contend'])]
    else:
        train_text = np.array(train_df['Question_contend'])

    train_labels = np.array(train_df[train_labels_col])

    val_labels_col =  [col for col in val_df.columns if col not in ['Question_contend']]
    val_labels_col = np.array(val_labels_col[1:])
    val_text = [prefix + sentence for sentence in np.array(val_df['Question_contend'])]
    val_labels = np.array(val_df[val_labels_col])

    # test_labels_col = [col for col in test_df.columns if col not in ['Question_contend']]
    # test_labels_col = np.array(test_labels_col[1:])
    # test_text = np.array(test_df['Question_contend'])
    # test_labels = np.array(test_df[test_labels_col])


    n_labels = len(train_labels) + 1

    # Build the dataset class for each set
    train_labeled_dataset = loader_labeled(
        train_text, train_labels, tokenizer, max_seq_len, train_aug)
    # train_unlabeled_dataset = loader_unlabeled(
    #     train_text[train_unlabeled_idxs], train_unlabeled_idxs, tokenizer, max_seq_len, Translator(data_path))
    val_dataset = loader_labeled(
        val_text, val_labels, tokenizer, max_seq_len)
    # test_dataset = loader_labeled(
    #     test_text, test_labels, tokenizer, max_seq_len)

    print("#Labeled: {},  val {}".format(len(
        train_text),  len(val_labels)))
    return train_labeled_dataset,val_dataset,n_labels

class loader_labeled(Dataset):
    # Data loader for labeled data
    def __init__(self, dataset_text, dataset_label, tokenizer, max_seq_len, aug=False):
        self.tokenizer = tokenizer
        self.text = dataset_text
        self.labels = dataset_label
        self.max_seq_len = max_seq_len

        self.aug = aug
        self.trans_dist = {}

        if aug:
            print('Aug train data by back translation of German')
            self.en2de = torch.hub.load(
                'pytorch/fairseq', 'transformer.wmt19.en-de.single_model', tokenizer='moses', bpe='fastbpe')
            self.de2en = torch.hub.load(
                'pytorch/fairseq', 'transformer.wmt19.de-en.single_model', tokenizer='moses', bpe='fastbpe')

    def __len__(self):
        return len(self.labels)

    def augment(self, text):
        if text not in self.trans_dist:
            self.trans_dist[text] = self.de2en.translate(self.en2de.translate(
                text,  sampling=True, temperature=0.9),  sampling=True, temperature=0.9)
        return self.trans_dist[text]
    # adding prefix to input sequence
    def get_tokenized(self, text):
        print(text)
        tokens = self.tokenizer.tokenize(text)
        if len(tokens) > self.max_seq_len:
            tokens = tokens[:self.max_seq_len]
        length = len(tokens)

        encode_result = self.tokenizer.convert_tokens_to_ids(tokens)
        padding = [0] * (self.max_seq_len - len(encode_result))
        encode_result += padding

        return encode_result, length

    def __getitem__(self, idx):
        if self.aug:
            text = self.text[idx]
            text_aug = self.augment(text)
            text_result, text_length = self.get_tokenized(text)
            text_result2, text_length2 = self.get_tokenized(text_aug)
            return ((torch.tensor(text_result), torch.tensor(text_result2)), (self.labels[idx], self.labels[idx]), (text_length, text_length2))
        else:
            text = self.text[idx]
            tokens = self.tokenizer.tokenize(text)
            if len(tokens) > self.max_seq_len:
                tokens = tokens[:self.max_seq_len]
            length = len(tokens)
            encode_result = self.tokenizer.convert_tokens_to_ids(tokens)
            padding = [0] * (self.max_seq_len - len(encode_result))
            encode_result += padding
            return (torch.tensor(encode_result), self.labels[idx], length)





