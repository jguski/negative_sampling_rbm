import numpy as np
from RBM import RBM
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from pathlib import Path
import argparse

from pykeen.datasets import get_dataset
from pykeen.datasets.base import Dataset
from pykeen.typing import COLUMN_HEAD, COLUMN_TAIL, COLUMN_RELATION

parser = argparse.ArgumentParser(
    description='Ridle, learning a representation for entities using a target distributions over the usage of relations.',
)
parser.add_argument('--dataset', nargs='?', default='WN18', type=str)
parser = parser.parse_args()



for head_or_tail in [COLUMN_HEAD, COLUMN_TAIL]:
    if head_or_tail == COLUMN_HEAD:
        print('Learning Ridle Representations on {} (head)'.format(parser.dataset))
    else:
        print('Learning Ridle Representations on {} (tail)'.format(parser.dataset))
    # Loading Files
    #df = pd.read_pickle('./dataset/{}/dataset.pkl'.format(parser.dataset))[['S', 'P']].drop_duplicates()

    dataset_instance: Dataset = get_dataset(
            dataset=parser.dataset
        )

    mapped_triples = dataset_instance.training.mapped_triples.numpy()
    df = pd.DataFrame(data=mapped_triples[:,[head_or_tail,COLUMN_RELATION]],
        columns=['entity', 'relation'])

    # add a mock label that is removed later, so that each entity will get a row
    df_mock_label = pd.DataFrame(data={'entity': range(dataset_instance.num_entities), 'relation': -999})
    df = pd.concat([df,df_mock_label])
    df = df.sort_values(by=['entity', 'relation'])

    # Learning Representation
    mlb = MultiLabelBinarizer()
    mlb.fit([df['relation'].unique()])
    df_distr_s = df.groupby('entity')['relation'].apply(list).reset_index(name='Class')
    X = mlb.transform(df_distr_s['Class'])
    # remove mock label
    X = X [:,1:]
    rbm = RBM(n_hidden=50, n_iterations=100, batch_size=100, learning_rate=0.01)
    rbm.fit(X)

    ## Save Entity Representations
    r = 1*rbm.reconstruct_and_sample(X)
    Path('./RBM_embeddings/{}'.format(parser.dataset)).mkdir(parents=True, exist_ok=True)
    
    if head_or_tail == COLUMN_HEAD:
        np.savez('./RBM_embeddings/{}/reconstructed_head.npz'.format(parser.dataset), r)
    else:
        np.savez('./RBM_embeddings/{}/reconstructed_tail.npz'.format(parser.dataset), r)