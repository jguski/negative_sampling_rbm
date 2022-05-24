""" A relaxed version of the entity similarity-based sampler, 
counting number of shared relations instead of relation + target pairs for similarity measure"""

import torch

from pykeen.sampling import BernoulliNegativeSampler
from pykeen.typing import COLUMN_HEAD, COLUMN_TAIL, MappedTriples
from pykeen.models import Model

import os
import pickle
import logging
from tqdm import tqdm


def absolute_similarity(relation_matrix: torch.LongTensor, entity_id: int, head_or_tail: int, mapped_triples: torch.LongTensor):
    # store relations that are observed together with e_i
    relations = torch.unique(mapped_triples[(mapped_triples[:,head_or_tail]==entity_id), 1])
    # slice adjacency tensor, keeping only those columns corresponding to relations observed with e_i
    sliced_relation_matrix = relation_matrix.index_select(dim=1, index=relations)
    # sum over all rows (to get number of relations shared with e_i for each e_j)
    return sliced_relation_matrix.sum(dim=1)

def jaccard_similarity(relation_matrix: torch.LongTensor, entity_id: int, head_or_tail: int, mapped_triples: torch.LongTensor):
    vector_e_i = relation_matrix[entity_id,]
    # calculate intersection = adjacency matrix * vector_e_i
    intersection = torch.matmul(relation_matrix, vector_e_i.t())
    # calculate union = rowwise_sum(acjacency matrix + vector_e_i) - intersection
    union = torch.add(vector_e_i, relation_matrix).sum(dim=1).subtract(intersection)
    # now compute similarities (in terms of Jaccard distance): Intersection over Union
    return torch.div(intersection, union)

similarity_dict={"absolute": absolute_similarity, "jaccard": jaccard_similarity}


class ESNSRelaxed(BernoulliNegativeSampler):

    def __init__(
        self,
        *,
        mapped_triples: MappedTriples,
        index_path: str = "esns_indices",
        index_column_size: int,
        sampling_size: int,
        q_set_size: int,
        similarity_metric: str = "absolute",
        model: Model,
        **kwargs,
    ) -> None:
        super().__init__(mapped_triples=mapped_triples, **kwargs)

        self.index_path = index_path
        # if self.num_entities < index_column_size, only self.num_entities can be stored
        self.index_column_size = min(self.num_entities, index_column_size)
        self.sampling_size = sampling_size
        self.q_set_size = min(self.num_entities, q_set_size)
        self.similarity_function=similarity_dict[similarity_metric]
        self.model = model
        self.mapped_triples = mapped_triples.to(self.model.device)

        self._index_handling()
    
    def _index_handling(self) -> None:

        logger = logging.getLogger(self.__class__.__name__)
        logger.setLevel("INFO")

        # handling of pickle files with inverse indices
        if not os.path.exists(self.index_path):
            os.makedirs(self.index_path)

        filename_base = self.index_path + "/" + self.__class__.__name__ + "_" + self.similarity_function.__name__ + "_k" + str(self.index_column_size)
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
        eii = torch.zeros(3, self.index_column_size, self.num_entities, device=self.model.device)

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
            eii[0,...,i] = i
            eii[1,...,i] = topk.indices
            eii[2,...,i] = topk.values

        return eii

    def _get_best_by_score_fn(self, original_triple: torch.LongTensor, candidates: torch.LongTensor, column: int) -> int:
        corrupted_triples = original_triple.repeat(candidates.size()).view(-1, 3)
        corrupted_triples[...,column] = candidates
        index_of_best_candidate = self.model.score_hrt(corrupted_triples).argmax()

        return candidates[index_of_best_candidate]

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
        indices_similar_entities = eii.index_select(dim=2, index=batch[selection,index]).view(-1, self.index_column_size*size)
        similar_entities = torch.zeros(size, self.num_entities, device=self.model.device)
        similar_entities[torch.repeat_interleave(torch.tensor(list(range(size)), device=self.model.device), self.index_column_size), indices_similar_entities[1].long()] = indices_similar_entities[2] 
        # initialize uniform sample set with zeros
        uniform_sample_set = torch.zeros((size, self.num_entities), device=self.model.device)
        # determine the rows to be filled with ones in sample set (just self.sampling_size times the indices of the selection)
        rows_sample_set = torch.repeat_interleave(torch.tensor(list(range(size)), device=self.model.device), self.sampling_size)
        # determine the columns to be filled with ones in sample set (self.sampling_size random entity indices for each tuple in selection)
        columns_sample_set = torch.randint(high=max_index, size=(size*self.sampling_size,), device=self.model.device)
        # fill uniform_sample_set with ones at the preselected positions
        uniform_sample_set[rows_sample_set, columns_sample_set] = 1
        # compute matrix containing only similarity values at positions present in the random sample
        similar_entities_in_sample = similar_entities*uniform_sample_set
        # select non-similar entities in random samples: They have to be 0 in similar_entities and 1 in uniform_sample_set
        nonsimilar_entities_in_sample = torch.where(similar_entities==0, 1, 0)*uniform_sample_set
        # extract quality candiate sets now: top self.q_set_size similar entities. Nonsimilar entities are weighted with a small value so that 
        # they have a higher value than entities not in the sample, but a lower than similar entities in the sample
        # TODO: See if there is a way to shuffle order of the selected nonsimilar entities (atm, some are picked with a higher probability)
        quality_candidates = (similar_entities_in_sample + nonsimilar_entities_in_sample*(10**-10)).topk(k=self.q_set_size,dim=1,sorted=False)
        # select best candidate from quality candidates by the KGE model's score function
        triples=batch[selection]
        replacement = [self._get_best_by_score_fn(triples[i], quality_candidates.indices[i], index) for i in range(size)]

        batch[selection, index] = torch.stack(replacement)

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