""" A relaxed version of the entity similarity-based sampler, 
counting number of shared relations instead of relation + target pairs for similarity measure"""

import torch
from scipy import sparse
from tqdm import tqdm
import numpy as np
from pathlib import Path

from pykeen.typing import COLUMN_TAIL

from .esns import ESNS
from ..similarity_metrics import BinarySimilarityFunction, RealSimilarityFunction


class ESNSRelaxed(ESNS):

    def _create_relation_matrix(self, column: int) -> torch.LongTensor:
        """
        Create a relation matrix as a basis for the similarity measure. This will typically
        be a |E|x|R| matrix with 1 if relation r_j is observed for entity e_i, and 0 otherwise.
        """
        # create a |E|x|R| 2D tensor that is 1 where there is a relation, and 0 where there is not
        # store all unique cominbations of (head/tail) entities with relations (to be used as indices for sparse tensor)
        relation_indices = self.mapped_triples.index_select(dim=1, index=torch.tensor([column,1], device=self.model.device)).unique(dim=0)
        # create the 2D tensor
        relation_matrix = torch.zeros(self.num_entities, self.num_relations, device=self.model.device)
        relation_matrix[relation_indices[:,0], relation_indices[:,1]] = 1

        return relation_matrix

    def _create_eii(self, column: int) -> None:
        """
        Create inverted indices (EII) containing a defined number of most similar entities e_j for each entity e_i.
        The logic of this method is based on matrix operations which speed up the computation of the EII as compared
        to the method from the original ESNS code.
        :param column: 
            Position of key entities (0 = head, 2 or -1 = tail). 
        """
       
        # initialize (3 x self.index_column_size x num_entities) EII tensor: Contains 3-rowed matrix for each entity, 
        # corresponding to index of the entity (row 0), indices (row 1) and similarity values (row 2) of top self.index_column_size entities
        #eii = torch.zeros(3, self.index_column_size, self.num_entities, device=self.model.device)
        eii = sparse.csr_matrix((self.num_entities, self.num_entities)).tolil()

        relation_matrix = self._create_relation_matrix(column)
        
        # fill the EII tensor for each e_i
        for i in tqdm(range(self.num_entities)):
            # compute similarity values of all e_j with e_i
            similarities = self.similarity_function.compute(relation_matrix=relation_matrix, mapped_triples=self.mapped_triples, head_or_tail=column, entity_id=i)
            # similarity of an entity to itself should be 0
            similarities[i] = 0
            topk_similarities = similarities.topk(k=self.max_index_column_size)
            eii[i, topk_similarities.indices.cpu()] = topk_similarities.values.cpu()

        return eii.tocsr()


    def quality_analysis(self, epoch, column=COLUMN_TAIL):

        if (self.ns_qual_analysis_every) != 0 and (epoch % self.ns_qual_analysis_every == 0):
            self.logger.info("Negative sampling quality analysis after epoch {}".format(epoch))

            if not any(self.random_triples_ids):
                # to ensure that always the same triples are considered
                np.random.seed(42)
                self.random_triples_ids = np.random.choice(self.mapped_triples.size()[0], len(self.random_triples_ids))

            relation_matrix = self._create_relation_matrix(column)
            for i in self.random_triples_ids:
                entity_to_replace = self.mapped_triples[i,column]
                # compute similarity values of all e_j with e_i
                similarities = self.similarity_function.compute(relation_matrix=relation_matrix, mapped_triples=self.mapped_triples, head_or_tail=column, entity_id=entity_to_replace)
                # similarity of an entity to itself should be -1 (so it won't be selected)
                similarities[entity_to_replace] = -1

                if issubclass(self.similarity_function, BinarySimilarityFunction):
                    # find entities with similarity values > 0 (similar neg. samples) or = 0 (non-similar neg. samples)
                    sns_entities = np.where(similarities.cpu() > 0)[0]
                    nns_entities = np.where(similarities.cpu()==0)[0]
                elif issubclass(self.similarity_function, RealSimilarityFunction):
                    sns_entities = np.where(similarities.cpu() > 0.5)[0]
                    nns_entities = np.where(abs(similarities.cpu())<=0.5)[0]
                
                # create tensors corresponding to sns and nns
                sns = self.mapped_triples[i].repeat(len(sns_entities),1)
                sns[:,column] = torch.Tensor(sns_entities)
                nns = self.mapped_triples[i].repeat(len(nns_entities),1)
                nns[:,column] = torch.Tensor(nns_entities)

                with torch.no_grad():
                    original_triple_score = self.model.score_hrt(self.mapped_triples[None,i]).detach()
                    minus_distances = {}
                    minus_distances["sns"] = [i[0] for i in (self.model.score_hrt(sns) - original_triple_score).cpu().detach().numpy()]
                    minus_distances["nns"] = [i[0] for i in (self.model.score_hrt(nns) - original_triple_score).cpu().detach().numpy()]

                save_path = self.ns_qual_analysis_path
                Path(save_path).mkdir(parents=True, exist_ok=True)
                np.savez(save_path + "/triple_{}_after_epoch_{}.npz".format(i, epoch), **minus_distances)




                

