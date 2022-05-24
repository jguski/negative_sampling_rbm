""" A relaxed version of the entity similarity-based sampler, 
counting number of shared relations instead of relation + target pairs for similarity measure"""

import torch

from pykeen.sampling import BernoulliNegativeSampler
from pykeen.typing import COLUMN_HEAD, COLUMN_TAIL, MappedTriples
from pykeen.models import Model

import os
import sys
import pickle
import logging
from tqdm import tqdm
from typing import Tuple, List

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


class ESNSRelaxedAlt(BernoulliNegativeSampler):

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

        self.mapped_triples = mapped_triples
        self.index_path = index_path
        # if self.num_entities < index_column_size, only self.num_entities can be stored
        self.index_column_size = min(self.num_entities, index_column_size)
        self.sampling_size = sampling_size
        self.q_set_size = min(self.num_entities, q_set_size)
        self.similarity_function=similarity_dict[similarity_metric]
        self.model = model

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
            self.eii_h = self._create_eii(COLUMN_HEAD)
            pickle.dump(self.eii_h, open(filename_base + "_h.pkl", 'wb'))
            logger.info("Creating EII {}_t.pkl".format(filename_base))
            self.eii_t = self._create_eii(COLUMN_TAIL)
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
        eii = dict.fromkeys(range(self.num_entities))

        # create a |E|x|R| 2D tensor that is 1 where there is a relation, and 0 where there is not
        # store all unique cominbations of (head/tail) entities with relations (to be used as indices for sparse tensor)
        relation_indices = self.mapped_triples.index_select(dim=1, index=torch.tensor([column,1], device=self.model.device)).unique(dim=0)
        # create the 2D tensor
        relation_matrix = torch.zeros(self.num_entities, self.num_relations, device=self.model.device)
        relation_matrix[relation_indices[:,0], relation_indices[:,1]] = 1

        # fill the EII dictionary for each e_i
        for i in tqdm(range(self.num_entities)):
            # compute similarity values of all e_j with e_i
            similarities = self.similarity_function(relation_matrix=relation_matrix, mapped_triples=self.mapped_triples, head_or_tail=column, entity_id=i)
            # set similarity of e_i with itself to 0
            similarities[i] = 0 
            # save indices of top self.index_column_size similar e_js
            topk = similarities.topk(min(self.num_entities, self.index_column_size))
            eii[i] = topk.indices[topk.values>0]

        return eii

    def _compute_intersection(self, tensor1: torch.LongTensor, tensor2: torch.LongTensor) -> Tuple[torch.LongTensor, torch.LongTensor]:
        # concatenate two tensors (.unique() so that duplicates in one of the tensors will not be misinterpreted as shared by both)
        combined = torch.cat((tensor1.unique(sorted=False), tensor2.unique(sorted=False)))
        uniques, counts = combined.unique(return_counts=True, sorted=False)
        # get values appearing in both tensors = intersection
        intersection = uniques[counts > 1]
        # get values appearing in only one of the tensors = difference
        difference = uniques[counts==1]

        return intersection, difference

    def _get_quality_candidates(self, from_index: torch.LongTensor, random_sample: torch.LongTensor) -> torch.LongTensor:
        similar_entities_in_random_sample, difference = self._compute_intersection(from_index, random_sample)
        nonsimilar_entities_in_random_sample, _ = self._compute_intersection(difference, random_sample)

        # select top self.q_set_size similar entities from the intersection, or all if fewer than self.q_set_size were sampled
        n_similar_entities = min(similar_entities_in_random_sample.size()[0], self.q_set_size)
        similar_entities = similar_entities_in_random_sample[:n_similar_entities]
        # fill up to self.q_set_size with random non-similar entities
        random_ids_nonsimilar_entities = torch.randperm(nonsimilar_entities_in_random_sample.size(0))[:(self.q_set_size-n_similar_entities)]
        nonsimilar_entities = nonsimilar_entities_in_random_sample[random_ids_nonsimilar_entities]

        return torch.cat((similar_entities, nonsimilar_entities))

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
        if size == 0:
            return

        # draw sets of entities uniformly at random
        uniform_sample_set = torch.randint(
            high=max_index - 1,
            size=(size,self.sampling_size),
            device=self.model.device,
        )

        # create quality sets (similar entities plus random subset of non-similar entities from uniform_sample_set)
        eii = (self.eii_h if index == COLUMN_HEAD else self.eii_t)
        quality_candidates = [self._get_quality_candidates(eii[x], uniform_sample_set[i]) 
                for i, x in enumerate(batch[selection,index].tolist())]

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
            *batch_shape, self.num_negs_per_pos, device=positive_batch.device
        ) < head_corruption_probability.to(device=positive_batch.device)
        
        # clone positive batch for corruption (.repeat_interleave creates a copy)
        negative_batch = positive_batch.view(-1, 3).repeat_interleave(self.num_negs_per_pos, dim=0)
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
        return negative_batch.view(*batch_shape, self.num_negs_per_pos, 3)