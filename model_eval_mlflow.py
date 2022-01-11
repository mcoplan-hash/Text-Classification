import re
from typing import Tuple, Dict
import mlflow
import numpy as np
import pandas as pd
import torch
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import TreebankWordTokenizer
from nltk.tokenize import word_tokenize as wt
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from tensorflow.keras.preprocessing.text import text_to_word_sequence
from torch import nn
from utils import load_embeddings, load_checkpoint, parse_opt

from datasets import get_clean_text, get_label_map
from utils import *

# data input shape
sentence_limit_per_doc = 25
word_limit_per_sentence = 50

data = []


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ML FLOW
mlflow_name = 'Pytorch'
mlflow.set_tracking_uri('http://10.200.6.63:5000/')  # @TODO pass host in as parameter
experiment_id = mlflow.get_experiment_by_name(name='Smoker_fact').experiment_id
mlflow.pytorch.autolog()

stemmer = PorterStemmer()
stopwords_list = set(stopwords.words('english'))
stopwords_list -= {'no', 'not', 'isn', 'haven', 'hasn', 'hadn', 'doesn', 'didn'}  # keep negation stopwords

# read CSV files of TEST data in
TEST = pd.read_csv('/mnt/data/mcoplan/Text-Classification/data/current_smoker_csv/test.csv', header=None)

TEST.columns = ['label', 'text']
TEST_label = TEST['label'].tolist()
TEST_text = TEST['text'].tolist()



def prepro_doc(
    document: str, word_map: Dict[str, int]
) -> Tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor]:
    """
    Preprocess a document into a hierarchial representation

    Parameters
    ----------
    document : str
        A document in text form

    word_map : Dict[str, int]
        Word2ix map

    Returns
    -------
    encoded_doc : torch.LongTensor
        Pre-processed tokenized document

    sentences_per_doc : torch.LongTensor
        Document lengths

    words_per_each_sentence : torch.LongTensor
        Sentence lengths
    """

    for paragraph in get_clean_text(document).splitlines():
        data = []
        for raw_word in text_to_word_sequence(paragraph):
            text = re.sub('[^A-Za-z]', ' ', raw_word)
            text = text.lower()
            tokenized_text = wt(text)
            text_processed = []
            for word in tokenized_text:
                if word not in stopwords_list:
                    text_processed.append((stemmer.stem(word)))
            text = " ".join(text_processed)
            if text != '':
                data.append(text)
    try:
        data = np.array(data)
    except:
        data = np.array(['NA'])
    zero = np.empty(sentence_limit_per_doc * word_limit_per_sentence, dtype='<U15')
    # print(data)
    # print(len(zero))
    # print(len(data))
    # print((len(zero) - len(data)))
    # print(80*'*')
    zero[(len(zero) - len(data)):len(zero)] = data
    doc = zero.reshape(sentence_limit_per_doc, word_limit_per_sentence)

    # number of sentences in the document
    sentences_per_doc = len(doc)
    sentences_per_doc = torch.LongTensor([sentences_per_doc]).to(device)  # (1)

    # number of words in each sentence
    words_per_each_sentence = list(map(lambda s: len(s), doc))
    words_per_each_sentence = torch.LongTensor(words_per_each_sentence).unsqueeze(0).to(device)  # (1, n_sentences)

    # encode document with indices from the word map
    encoded_doc = list(
        map(lambda s: list(
            map(lambda w: word_map.get(w, word_map['<unk>']), s)
        ) + [0] * (word_limit_per_sentence - len(s)), doc)
    ) + [[0] * word_limit_per_sentence] * (sentence_limit_per_doc - len(doc))
    encoded_doc = torch.LongTensor(encoded_doc).unsqueeze(0).to(device)

    return encoded_doc, sentences_per_doc, words_per_each_sentence


def prepro_sent(
    text: str, word_map: Dict[str, int]
) -> Tuple[torch.LongTensor, torch.LongTensor]:
    """
    Preprocess a sentence

    Parameters
    ----------
    text : str
        A sentence in text form

    word_map : Dict[str, int]
        Word2ix map

    Returns
    -------
    encoded_sent : torch.LongTensor
        Pre-processed tokenized sentence

    words_per_sentence : torch.LongTensor
        Sentence lengths
    """
    # tokenizers
    word_tokenizer = TreebankWordTokenizer()

    # tokenize sentences into words
    sentence = word_tokenizer.tokenize(text)[:word_limit]

    # number of words in sentence
    words_per_sentence = len(sentence)
    words_per_sentence = torch.LongTensor([words_per_sentence]).to(device)  # (1)

    # encode sentence with indices from the word map
    encoded_sent = list(
        map(lambda w: word_map.get(w, word_map['<unk>']), sentence)
    ) + [0] * (word_limit - len(sentence))
    encoded_sent = torch.LongTensor(encoded_sent).unsqueeze(0).to(device)

    return encoded_sent, words_per_sentence


def classify(
    text: str, model: nn.Module, model_name: str, dataset_name: str, word_map: Dict[str, int]
) -> str:
    """
    Classify a text using the given model.

    Parameters
    ----------
    text : str
        A document or sentence in text form

    model : nn.Module
        A loaded model

    model_name : str
        Name of the model

    dataset_name : str
        Name of the dataset

    word_map : Dict[str, int]
        Word2ix map

    Returns
    -------
    prediction : str
        The predicted category with its probability
    """
    _, rev_label_map = get_label_map(dataset_name)

    if model_name in ['han']:
        # preprocess document
        encoded_doc, sentences_per_doc, words_per_each_sentence = prepro_doc(text, word_map)
        # run through model
        scores, word_alphas, sentence_alphas = model(
            encoded_doc,
            sentences_per_doc,
            words_per_each_sentence
        )  # (1, n_classes), (1, n_sentences, max_sent_len_in_document), (1, n_sentences)
    else:
        # preprocess sentence
        encoded_sent, words_per_sentence = prepro_sent(text, word_map)
        # run through model
        scores = model(encoded_sent, words_per_sentence)

    scores = scores.squeeze(0)  # (n_classes)
    scores = nn.functional.softmax(scores, dim=0)  # (n_classes)

    # find best prediction and its probability
    score, prediction = scores.max(dim=0)

    prediction = 'Category: {category}, Probability: {score:.2f}%'.format(
        category=rev_label_map[prediction.item()],
        score=score.item() * 100
    )
    return prediction

for i in range(3,10):
    epoch = i
    print(epoch)
    with mlflow.start_run(experiment_id=experiment_id, run_name=mlflow_name) as run:
        # path to the checkpoint
        config = parse_opt()
        checkpoint_path = f'/mnt/share/sandbox/mcoplan/facts/current_smoker/pytorch_models/checkpoint_han_current_smoker_epoch_{epoch}.pth.tar'
        model, model_name, _, dataset_name, word_map, _ = load_checkpoint(checkpoint_path, device)
        model = model.to(device)
        model.eval()

        # prediction = classify(text, model, model_name, dataset_name, word_map)
        # visualize_attention(*classify(text, model, model_name, dataset_name, word_map))
        TEST_prediction = []
        for i in TEST_text:
            prediction = classify(i, model, model_name, dataset_name, word_map)
            x = prediction.split(", ")
            if x[0] == 'Category: Current smoker':
                TEST_prediction.append(float(x[1][-6:-1]))
            else:
                TEST_prediction.append(100 - float(x[1][-6:-1]))

        TEST_prediction_percent = [x / 100 for x in TEST_prediction]

        y_pred = []
        for i in TEST_prediction_percent:
            if i > 0.5:
                y_pred.append(1)
            else:
                y_pred.append(0)

        # Metrics
        f1_micro = f1_score(TEST_label, y_pred, average='micro')
        f1_macro = f1_score(TEST_label, y_pred, average='macro')
        precision_macro = precision_score(TEST_label, y_pred, average="macro")
        recall_macro = recall_score(TEST_label, y_pred, average="macro")
        roc_auc = roc_auc_score(TEST_label, y_pred)
        accuracy = accuracy_score(TEST_label, y_pred)

        mlflow_metrics = {'test_f1_micro_score': f1_micro
            , 'test_f1_macro_score': f1_macro
            , 'test_precision_macro_score': precision_macro
            , 'test_recall_macro_score': recall_macro
            , 'test_roc_auc_score': roc_auc
            , 'test_accuracy_score': accuracy}

        mlflow_params = {
            'num_epochs': config.num_epochs,
            'epoch': epoch,
            'model_name': config.model_name,
            'sentence_limit_per_doc': sentence_limit_per_doc,
            'word_limit_per_sentence': word_limit_per_sentence,
            'loss_function': 'CrossEntropyLoss',
            'optimizer': 'Adam',
            'lr_decay': config.lr_decay,
            'grad_clip': config.grad_clip,
            'word and sent NN': 'LSTM'

        }
        print(mlflow_metrics)
        print(mlflow_params)
        mlflow.log_metrics(mlflow_metrics)
        mlflow.log_params(mlflow_params)
        mlflow.end_run()
