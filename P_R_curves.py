from nltk.tokenize import TreebankWordTokenizer
from typing import Tuple, Dict
import torch
from torch import nn
import re
from datasets import get_clean_text, get_label_map
from utils import *
import numpy as np
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize as wt
from sklearn.metrics import confusion_matrix, classification_report
import re
from typing import Tuple, Dict
from sklearn.metrics import plot_precision_recall_curve
from sklearn.metrics import precision_recall_curve
from matplotlib import pyplot
from sklearn.metrics import auc
import numpy as np
import torch
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import TreebankWordTokenizer
from nltk.tokenize import word_tokenize as wt
from torch import nn
from tensorflow.keras.preprocessing.text import text_to_word_sequence

from datasets import get_clean_text, get_label_map
from utils import *
import pandas as pd

stemmer = PorterStemmer()
stopwords_list = set(stopwords.words('english'))
stopwords_list -= {'no', 'not', 'isn', 'haven', 'hasn', 'hadn', 'doesn', 'didn'}  # keep negation stopwords

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# path to the checkpoint
checkpoint_path = '/mnt/data/mcoplan/Text-Classification/checkpoints/checkpoint_han_current_smoker_epoch_7.pth.tar'


#read CSV files of TEST data in
TEST = pd.read_csv('/mnt/data/mcoplan/Text-Classification/data/current_smoker_csv/test.csv', header=None)

TEST.columns = ['label','text']
TEST_label = TEST['label'].tolist()
TEST_text = TEST['text'].tolist()

# pad limits
# only makes sense when model_name = 'han'
sentence_limit_per_doc = 25
word_limit_per_sentence = 50
# only makes sense when model_name != 'han'
word_limit = 200


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
    data = np.array(data)
    zero = np.empty(sentence_limit_per_doc * word_limit_per_sentence, dtype='<U15')
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

    # word_alphas = word_alphas.squeeze(0)  # (n_sentences, max_sent_len_in_document)
    # sentence_alphas = sentence_alphas.squeeze(0)  # (n_sentences)
    # words_per_each_sentence = words_per_each_sentence.squeeze(0)  # (n_sentences)

    # return doc, scores, word_alphas, sentence_alphas, words_per_each_sentence


if __name__ == '__main__':
    # text1 = 'the patient is a current smoker'
    # text2 = 'The patient is a non smoker.  The patient is a never smoker.'
    # text3 = 'the patient is not a smoker.'
    # text4 = 'The patient is a current everyday smoker. He smokes 1 pack per day. He smokers tobacco every day. Smoking Cessation recommended. He smokes cigarettes every day.'
    # text5 = '! Reviewed Social History i General IM and Zlka : lllicit drugs: No (Notes: pravious meth addiction- 15 months and 17 days ago) : Qeewpation: Unemployed : Edueation: 10 ; Marital status: Divarced | Sexual orientation: Heterosexual | Exercise level: Heavy Diet: Regular i General stress level: High Alcohol intake: None Caffeine intake: Heavy . Guns present in home: N ¢ Seat belts used routinely: Y | Sunscreen used routinely: N ; Smoke alarm in home: Y ¢ Advance directive: N ¢ Medical Power of Attorney: N i Performs monthly self-breast exam: N i Legally blind in ore or both eyes?: N : Hard of hearing or deaf in one or both ears?: N 15 the: patient ambulatory?: Yes: walks without restrictions i Tobacco Smoking Status: Current every day smoker | Smoker (1 1/2 PPD) | Mave you recently (within the Jast 12 weeks, or during a current pregnancy) fraveled to or lived in a Zika-affected area?: N Do you have symptoms associated with Zika virus (fever, rash, joint pain, or conjunctivitis)?: N ; Have you had sexual relations with anyone wha has been pasitivaly diagnosed with Zika virus within the last 6 manths?: N'
    # text6 = "8. Smoker Notes: Smoking cessation counselling given. The patient likes to eat apples"
    # text7 = 'the patient smokes 1 ppd'
    # text8 = 'the patient smoked but quit 2 years ago'
    # text9 = FN_list[0]
    #
    # text = [text1, text2, text3, text4, text5, text6, text7, text8, text9]

    # load model and word map
    model, model_name, _, dataset_name, word_map, _ = load_checkpoint(checkpoint_path, device)
    model = model.to(device)
    model.eval()

    #prediction = classify(text, model, model_name, dataset_name, word_map)
    # visualize_attention(*classify(text, model, model_name, dataset_name, word_map))
    TEST_prediction = []
    for i in TEST_text:
        prediction = classify(i, model, model_name, dataset_name, word_map)
        x = prediction.split(", ")
        if x[0] == 'Category: Current smoker':
            TEST_prediction.append(float(x[1][-6:-1]))
        else:
            TEST_prediction.append(100-float(x[1][-6:-1]))

    TEST_prediction_percent = [x / 100 for x in TEST_prediction]
    lr_precision, lr_recall, _ = precision_recall_curve(TEST_label, TEST_prediction_percent)
    pyplot.plot(lr_recall, lr_precision, marker='.', label='Smoker Fact Inference')
    # axis labels
    pyplot.xlabel('Recall')
    pyplot.ylabel('Precision')
    #y-axis
    pyplot.ylim(0,1.1)
    #add line for CADEN Recall
    pyplot.axvline(x=0.54,color='k', linestyle='--', label='CADEN Recall (0.54)')
    pyplot.axhline(y=0.64,color='b', linestyle='--', label='CADEN Precision (0.64)')
    # show the legend
    pyplot.legend()
    # show the plot
    pyplot.show()

    #confusion matrix
    y_pred = []
    for i in TEST_prediction_percent:
        if i > 0.5:
            y_pred.append(1)
        else:
            y_pred.append(0)
    cm = confusion_matrix(TEST_label, y_pred)
    cr = classification_report(TEST_label, y_pred)

    # df_FN = pd.DataFrame({'text': FN_list,'model_prediction': FN_prediction})
    # df_FN.to_csv('CADEN_FN_model_results.csv', index=False)
    #
    # df_FP = pd.DataFrame({'text': FP_list,'model_prediction': FP_prediction})
    # df_FP.to_csv('CADEN_FP_model_results.csv', index=False)

    print('Done')
