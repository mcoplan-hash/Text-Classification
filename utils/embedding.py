import os
from tqdm import tqdm
from typing import Dict, Tuple
import numpy as np
import torch
import fasttext
import json

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# NOTE: IF LOADING NEW TRAINING DATA:
# YOU MUST DELETE THE OLD .pth.tar FILE AND RE-RUN THIS SCRIPT
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

def init_embeddings(embeddings: torch.Tensor) -> None:
    """
    Fill embedding tensor with values from the uniform distribution.

    Parameters
    ----------
    embeddings : torch.Tensor
        Word embedding tensor
    """
    bias = np.sqrt(3.0 / embeddings.size(1))
    torch.nn.init.uniform_(embeddings, -bias, bias)

def load_embeddings(
    emb_file: str,
    word_map: Dict[str, int],
    output_folder: str
) -> Tuple[torch.Tensor, int]:
    """
    Create an embedding tensor for the specified word map, for loading into the model.

    Parameters
    ----------
    emb_file : str
        File containing embeddings (stored in GloVe format)

    word_map : Dict[str, int]
        Word2id map

    output_folder : str
        Path to the folder to store output files

    Returns
    -------
    embeddings : torch.Tensor
        Embeddings in the same order as the words in the word map

    embed_dim : int
        Dimension of the embeddings
    """
    emb_basename = os.path.basename(emb_file)
    cache_path = os.path.join(output_folder, emb_basename + '.pth.tar')
    embed_dim = 200

    # no cache, load embeddings from .txt file
    if not os.path.isfile(cache_path):
        # find embedding dimension
        model = fasttext.load_model(emb_file)
        vocab = set(word_map.keys())

        # create tensor to hold embeddings, initialize
        embeddings = torch.FloatTensor(len(vocab), embed_dim)
        init_embeddings(embeddings)

        # read embedding file
        for emb_word in model.words:
            # ignore word if not in train_vocab
            if emb_word not in vocab:
                continue
            embedding = model[emb_word]
            embeddings[word_map[emb_word]] = torch.FloatTensor(embedding)

        # create cache file so we can load it quicker the next time
        print('Saving vectors to {}'.format(cache_path))
        torch.save((embeddings, embed_dim), cache_path)

    # load embeddings from cache
    else:
        print('Loading embeddings from {}'.format(cache_path))
        embeddings, embed_dim = torch.load(cache_path)
        print(len(word_map))

    return embeddings, embed_dim

if __name__ == '__main__':
    #First delete old embedding: rm /mnt/share/sandbox/mcoplan/facts/current_smoker/pytorch_models/BioWordVec_PubMed_MIMICIII_d200.bin.pth.tar
    with open('/mnt/share/sandbox/mcoplan/facts/current_smoker/pytorch_models/word_map.json', 'r') as j:
        word_map = json.load(j)
    load_embeddings('/mnt/data/mcoplan/Text-Classification/data/glove/BioWordVec_PubMed_MIMICIII_d200.bin', word_map ,'/mnt/share/sandbox/mcoplan/facts/current_smoker/pytorch_models')
    print('done')
