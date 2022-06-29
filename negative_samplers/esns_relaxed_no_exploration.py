""" A relaxed version of the entity similarity-based sampler, 
counting number of shared relations instead of relation + target pairs for similarity measure"""

import torch

from pykeen.sampling import BernoulliNegativeSampler
from pykeen.typing import COLUMN_HEAD, COLUMN_TAIL, MappedTriples
from pykeen.models import Model
from negative_samplers.esns_relaxed import absolute_similarity, jaccard_similarity, similarity_dict

import os
import pickle
import logging
from tqdm import tqdm

class ESNSRelaxedNoExploration(BernoulliNegativeSampler):

    def __init__(
        self,
        *,
        mapped_triples: MappedTriples,
        index_path: str = "esns_indices",
        index_column_size: int,
        similarity_metric: str = "absolute",
        model: Model,
        **kwargs,
    ) -> None:
        super().__init__(mapped_triples=mapped_triples, **kwargs)

        self.index_path = index_path
        self.similarity_function=similarity_dict[similarity_metric]
        self.model = model
        self.mapped_triples = mapped_triples.to(self.model.device)
        self.index_column_size = index_column_size

        self._index_handling()
    
    def _index_handling(self) -> None:

        logger = logging.getLogger(self.__class__.__name__)
        logger.setLevel("INFO")

        # handling of pickle files with inverse indices
        if not os.path.exists(self.index_path):
            os.makedirs(self.index_path)

        filename_base = self.index_path + "/" + self.__class__.__name__ + "_" + self.similarity_function.__name__  + "_k" + str(self.index_column_size)
        if not (os.path.exists(filename_base + "_h.pkl") and os.path.exists(filename_base + "_t.pkl")):
            logger.info("Creating EII {}_h.pkl".format(filename_base))
            self.eii_h = self._create_eii(COLUMN_HEAD)#.view(-1, self.index_column_size*self.num_entities)
            pickle.dump(self.eii_h, open(filename_base + "_h.pkl", 'wb'))
            logger.info("Creating EII {}_t.pkl".format(filename_base))
            self.eii_t = self._create_eii(COLUMN_TAIL)#.view(-1, self.index_column_size*self.num_entities)
            pickle.dump(self.eii_t, open(filename_base + "_t.pkl", 'wb'))
        else: 
            logger.info("Loading EII {}_h.pkl".format(filename_base))
            self.eii_h = pickle.load(open(filename_base + "_h.pkl", 'rb'))
            logger.info("Loading EII {}_t.pkl".format(filename_base))
            self.eii_t = pickle.load(open(filename_base + "_t.pkl", 'rb'))

    def _create_eii(self, column: int) -> None:
        """
        Create inverted indices (EII) containing a defined number of most similar entities e_j for each entity e_i.
        :param column: 
            Position of key entities (0 = head, 2 or -1 = tail). 
        """
       
        # initialize (3 x self.index_column_size x num_entities) EII tensor: Contains 3-rowed matrix for each entity, 
        # corresponding to index of the entity (row 0), indices (row 1) and similarity values (row 2) of top self.index_column_size entities
        eii = dict.fromkeys(list(range(self.num_entities)))

        # create a |E|x|R| 2D tensor that is 1 where there is a relation, and 0 where there is not
        # store all unique cominbations of (head/tail) entities with relations (to be used as indices for sparse tensor)
        relation_indices = self.mapped_triples.index_select(dim=1, index=torch.tensor([column,1], device=self.model.device)).unique(dim=0)
        # create the 2D tensor
        relation_matrix = torch.zeros(self.num_entities, self.num_relations, device=self.model.device)
        relation_matrix[relation_indices[:,0], relation_indices[:,1]] = 1

        # fill the EII tensor for each e_i
        for i in tqdm(range(self.num_entities)):
            # compute similarity values of all e_j with e_i
            similarities = self.similarity_function(relation_matrix=relation_matrix, mapped_triples=self.mapped_triples, head_or_tail=column, entity_id=i)
            # set similarity of e_i with itself to 0
            similarities[i] = 0 

            # save indices of top self.index_column_size similar e_js
            topk = similarities.topk(min(self.num_entities, self.index_column_size))
            # save to i-th EII tensor slice: x-coordinate (i), y-coordinate (topk.indices) and values to be used to be able to create a sparse tensor from this EII
            eii[i] = topk.indices

        return eii

    def _esns_replacement(self, batch: torch.LongTensor, index: int, selection: slice, size: int, max_index: int) -> None:
        """
        Replace a column of a batch of indices by random indices.
        :param batch: shape: `(*batch_dims, d)`
            the batch of indices
        :param index:
            the index (of the last axis) which to replace
        :param selection:
            a selection of the batch, e.g., a slice or a mask
        :param size:
            the size of the selection
        :param max_index:
            the maximum index value at the chosen position
        """
        # return straight away if size of selection happens to be zero (function would crash otherwise)
        if size == 0:
            return

        # create quality sets (similar entities plus random subset of non-similar entities from uniform_sample_set)
        eii = (self.eii_h if index == COLUMN_HEAD else self.eii_t)

        # create tensor with one row per triple in selection, containing row from eii corresponding to head or tail value
        #similar_entities = index_select_sparse(eii, 0, batch[selection,index]).to_dense()
        entities_to_corrupt = batch[selection,index]
        replacement = torch.tensor([eii[i.item()][torch.randint(high=eii[i.item()].size()[0], size=(1,)).item()] for i in entities_to_corrupt], device=self.model.device)
       
        batch[selection, index] = replacement

    def corrupt_batch(self, positive_batch: torch.LongTensor) -> torch.LongTensor:
        # get number of positive triples in a batch
        batch_shape = positive_batch.shape[:-1]

        # Decide whether to corrupt head or tail
        head_corruption_probability = self.corrupt_head_probability[positive_batch[..., 1]].unsqueeze(dim=-1)
        head_mask = torch.rand(
            *batch_shape, self.num_negs_per_pos, device=self.model.device
        ) < head_corruption_probability.to(device=self.model.device)
        
        # clone positive batch for corruption (.repeat_interleave creates a copy)
        negative_batch = positive_batch.view(-1, 3).repeat_interleave(self.num_negs_per_pos, dim=0).to(self.model.device)
        # flatten mask
        head_mask = head_mask.view(-1)
        
        for index, mask in (
            (COLUMN_HEAD, head_mask),
            # Tails are corrupted if heads are not corrupted
            (COLUMN_TAIL, ~head_mask),
        ):  
            self._esns_replacement(
                batch=negative_batch,
                index=index,
                selection=mask,
                size=mask.sum(),
                max_index=self.num_entities,
            )

        return negative_batch.view(*batch_shape, self.num_negs_per_pos, 3).to(positive_batch.device)