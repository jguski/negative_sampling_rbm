""" A negative sampler based on the relaxed entity similarity-based sampler, 
that loads a precomputed matrix of Ridle samplings as a basis for similarity computation"""

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MultiLabelBinarizer
from pathlib import Path
import os

from pykeen.typing import COLUMN_HEAD, COLUMN_RELATION

from esns_relaxed import ESNSRelaxed
from RBM import RBM

class ESNSRidle(ESNSRelaxed):
    def __init__(
        self,
        embeddings_path: str = "RBM_embeddings",
        **kwargs,
    ) -> None:
        self.embeddings_path = embeddings_path
        super().__init__(**kwargs)
        
    
    def _create_relation_matrix(self, column: int) -> torch.LongTensor:
        # """
        # Provides relation matrix as a basis for the similarity measure. It is not computed here,
        # but loaded from existing npz files with Ridle embeddings.
        # """
        head_or_tail = ("_h" if column == COLUMN_HEAD else "_t")

        if os.path.exists(self.embeddings_path + '/{}/reconstructed{}.npz'.format(self.dataset, head_or_tail)):
            
            self.logger.info("Loading pre-computed Ridle representation for {} ({})".format(self.dataset, head_or_tail))
            relation_matrix = np.load(self.embeddings_path + "/" + self.dataset + "/reconstructed" + head_or_tail + ".npz")['arr_0']

        else:
            
            self.logger.info("Learning Ridle representation for {} ({})".format(self.dataset, head_or_tail))
            df = pd.DataFrame(data=self.mapped_triples.cpu().numpy()[:,[column,COLUMN_RELATION]],
                columns=['entity', 'relation'])

            # add a mock label that is removed later, so that each entity will get a row
            df_mock_label = pd.DataFrame(data={'entity': range(self.num_entities), 'relation': -999})
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
            relation_matrix = 1*rbm.reconstruct_and_sample(X)
            
            Path(self.embeddings_path + '/{}'.format(self.dataset)).mkdir(parents=True, exist_ok=True)
            np.savez(self.embeddings_path + '/{}/reconstructed{}.npz'.format(self.dataset, head_or_tail), relation_matrix)


        return(torch.Tensor(relation_matrix, device=self.model.device))