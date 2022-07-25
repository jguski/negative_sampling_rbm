""" A relaxed version of the entity similarity-based sampler, 
counting number of shared relations instead of relation + target pairs for similarity measure"""

import torch
import scipy
from scipy import sparse
import numpy as np
import os
import logging
from tqdm import tqdm
from pathlib import Path

from .similarity_metrics import similarity_factory

from pykeen.sampling import BernoulliNegativeSampler
from pykeen.typing import COLUMN_HEAD, COLUMN_TAIL, COLUMN_RELATION, MappedTriples
from pykeen.models import Model
from pykeen.datasets import Dataset

class ESNSRelaxed(BernoulliNegativeSampler):

    def __init__(
        self,
        *,
        mapped_triples: MappedTriples,
        index_path: str = "Output/EII",
        index_column_size: int,
        max_index_column_size: int = 1000,
        sampling_size: int,
        q_set_size: int,
        similarity_metric: str = "absolute",
        n_triples_for_ns_qual_analysis = 1,
        ns_qual_analysis_every = 100,
        ns_qual_analysis_path = "Output/NS_quality_analysis",
        model: Model,
        dataset: Dataset,
        logging_level: str = "INFO",
        **kwargs,
    ) -> None:
        super().__init__(mapped_triples=mapped_triples, **kwargs)

        self.index_path = index_path + "/" + dataset
        # if self.num_entities < index_column_size, only self.num_entities can be stored
        self.index_column_size = min(self.num_entities, index_column_size)
        self.max_index_column_size = min(self.num_entities, max_index_column_size)
        self.sampling_size = sampling_size
        self.q_set_size = min(self.num_entities, q_set_size)
        self.similarity_function=similarity_factory(similarity_metric)
        # for NS quality analysis: Init empty list to be filled with random triple ids later
        self.random_triples_ids = [None] * n_triples_for_ns_qual_analysis
        self.ns_qual_analysis_every = ns_qual_analysis_every
        self.ns_qual_analysis_path = ns_qual_analysis_path
        # some objects to be passed to the sampler
        self.model = model
        self.dataset = dataset
        self.mapped_triples = mapped_triples.to(self.model.device)

        self.logger = logging.getLogger(self.__class__.__name__ + "_" + str(id(self)))
        self.logger.setLevel(logging_level)
        
        self._index_handling()
    
    def _index_handling(self) -> None:

        # handling of npz files with inverse indices
        if not os.path.exists(self.index_path):
            os.makedirs(self.index_path)

        if hasattr(self, "rbm_layer"):
            filename_base = self.index_path + "/" + self.__class__.__name__ + "_" + self.similarity_function.__name__ + "_" + self.rbm_layer
        else:
            filename_base = self.index_path + "/" + self.__class__.__name__ + "_" + self.similarity_function.__name__
        if not (os.path.exists(filename_base + "_h.npz") and os.path.exists(filename_base + "_t.npz")):
            self.logger.info("Creating EII {}_h.npz".format(filename_base))
            self.eii_h = self._create_eii(COLUMN_HEAD)
            scipy.sparse.save_npz(filename_base + '_h.npz', self.eii_h)
            self.logger.info("Creating EII {}_t.npz".format(filename_base))
            self.eii_t = self._create_eii(COLUMN_TAIL)
            scipy.sparse.save_npz(filename_base + '_t.npz', self.eii_t)
            self.eii_h = self.eii_h.tolil()
            self.eii_t = self.eii_t.tolil()
        else: 
            self.logger.info("Loading EII {}_h.npz".format(filename_base))
            self.eii_h = scipy.sparse.load_npz(filename_base + '_h.npz').tolil()
            self.logger.info("Loading EII {}_t.npz".format(filename_base))
            self.eii_t = scipy.sparse.load_npz(filename_base + '_t.npz').tolil()

        # get top self.index_column_size similar entities
        if self.index_column_size != 0:
            ent = 0
            for data, row in zip(self.eii_h.data, self.eii_h.rows):
                if self.index_column_size < len(row):
                    d, r = zip(*sorted(zip(data, row), reverse=True)[:self.index_column_size])
                    self.eii_h.data[ent] = list(d)
                    self.eii_h.rows[ent] = list(r)
                ent +=1
            ent = 0
            for data, row in zip(self.eii_t.data, self.eii_t.rows):
                if self.index_column_size < len(row):
                    d, r = zip(*sorted(zip(data, row), reverse=True)[:self.index_column_size])
                    self.eii_t.data[ent] = list(d)
                    self.eii_t.rows[ent] = list(r)
                ent += 1

    def _create_relation_matrix(self, column: int) -> torch.LongTensor:
        """
        Create a relation matrix as a basis for the similarity measure. This will typically
        be a |E|x|R| matrix with 1 if relation r_j is observed for entity e_i, and 0 otherwise.
        """
        # create a |E|x|R| 2D tensor that is 1 where there is a relation, and 0 where there is not
        # store all unique cominbations of (head/tail) entities with relations (to be used as indices for sparse tensor)
        relation_indices = self.mapped_triples.index_select(dim=1, index=torch.tensor([column,1], device=self.model.device)).unique(dim=0)
        # create the 2D tensor
        relation_matrix = torch.zeros(self.num_entities, self.num_relations, device=self.model.device, dtype=torch.uint8)
        relation_matrix[relation_indices[:,0], relation_indices[:,1]] = 1

        return relation_matrix

    def _create_eii(self, column: int) -> None:
        """
        Create inverted indices (EII) containing a defined number of most similar entities e_j for each entity e_i.
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
            similarities = self.similarity_function(relation_matrix=relation_matrix, mapped_triples=self.mapped_triples, head_or_tail=column, entity_id=i)
            # similarity of an entity to itself should be 0
            similarities[i] = 0
            topk_similarities = similarities.topk(k=self.max_index_column_size)
            eii[i, topk_similarities.indices.cpu()] = topk_similarities.values.cpu()

        return eii.tocsr()


    def _similar_ent(self, original, candidate, index):
        shape = original.shape
        original = original.reshape(-1)
        candidate = candidate.reshape(-1)

        eii = (self.eii_h if index == COLUMN_HEAD else self.eii_t)
        
        return torch.from_numpy(eii[original, candidate].toarray()).reshape(shape)


    def _esns_replacement(self, batch, index, selection, size, max_index):
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

        head = batch[selection, COLUMN_HEAD]
        tail = batch[selection, COLUMN_TAIL]
        rela = batch[selection, COLUMN_RELATION]
        # uniform generate samples
        if index == COLUMN_HEAD:
            # draw self.sampling_size random samples for each triple in selection
            h_cand = np.random.choice(max_index, (head.shape[0], self.sampling_size))
            # move ids of original heads to numpy
            head_idx = head.cpu().numpy()
            # repeat head id sampling_size times
            head_i = np.tile(head_idx.reshape(-1, 1), self.sampling_size)
            probs = self._similar_ent(head_i, h_cand, index)
            _, h_new = torch.topk(probs, self.q_set_size, dim=-1)
            h_idx = torch.arange(0, head.shape[0]).type(torch.LongTensor).unsqueeze(1).expand(-1, self.q_set_size)
            # create array with q_set_size (columns) candidates per triple in selection (rows) and turn into a torch tensor
            h_rep = h_cand[h_idx, h_new]
            h_cand = torch.from_numpy(h_rep).type(torch.LongTensor).to(self.model.device)
            n = head.size(0)
            
            tail = tail.unsqueeze(1).expand_as(h_cand).flatten().to(self.model.device)
            rela = rela.unsqueeze(1).expand_as(h_cand).flatten().to(self.model.device)
            
            candidate_triples = torch.stack((h_cand.flatten(), rela, tail)).t()
            scores = self.model.score_hrt(candidate_triples)
            scores = scores.view(n, -1)

            probs = scores

            #probs = self.model.prob(h_cand, tail, rela)
            _, h_new = torch.topk(probs, 1, dim=-1)

            row_idx = torch.arange(0, n).type(torch.LongTensor)
            if row_idx.size() != h_new.size():
                row_idx = row_idx.unsqueeze(1).expand_as(h_new)
            h_idx = h_cand[row_idx, h_new].cpu().numpy()
            replacement = torch.LongTensor(h_idx).to(self.model.device)

        else:
            t_cand = np.random.choice(max_index, (head.shape[0], self.sampling_size))
            tail_idx = tail.cpu().numpy()
            tail_i = np.tile(tail_idx.reshape(-1, 1), self.sampling_size)
            probs = self._similar_ent(tail_i, t_cand, index)
            _, t_new = torch.topk(probs, self.q_set_size, dim=-1)
            t_idx = torch.arange(0, tail.shape[0]).type(torch.LongTensor).unsqueeze(1).expand(-1, self.q_set_size)
            t_rep = t_cand[t_idx, t_new]
            t_cand = torch.from_numpy(t_rep).type(torch.LongTensor).to(self.model.device)
            n = tail.size(0)
            head = head.unsqueeze(1).expand_as(t_cand).flatten()
            rela = rela.unsqueeze(1).expand_as(t_cand).flatten()

            candidate_triples = torch.stack((head, rela, t_cand.flatten())).t()

            scores = self.model.score_hrt(candidate_triples)
            scores = scores.view(n, -1)

            probs = scores

            _, t_new = torch.topk(probs, 1, dim=-1)
            row_idx = torch.arange(0, n).type(torch.LongTensor)
            if row_idx.size() != t_new.size():
                row_idx = row_idx.unsqueeze(1).expand_as(t_new)

            t_idx = t_cand[row_idx, t_new].cpu().numpy()
            replacement = torch.LongTensor(t_idx).to(self.model.device)
        
        batch[selection, index] = replacement.flatten()

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


    def quality_analysis(self, epoch, column=COLUMN_TAIL):

        if (self.ns_qual_analysis_every) != 0 and (epoch % self.ns_qual_analysis_every == 0):
            self.logger.info("Negative sampling quality analysis after epoch {}".format(epoch))

            if not any(self.random_triples_ids):
                self.random_triples_ids = np.random.choice(self.mapped_triples.size()[0], len(self.random_triples_ids))

            relation_matrix = self._create_relation_matrix(column)
            for i in self.random_triples_ids:
                entity_to_replace = self.mapped_triples[i,column]
                # compute similarity values of all e_j with e_i
                similarities = self.similarity_function(relation_matrix=relation_matrix, mapped_triples=self.mapped_triples, head_or_tail=column, entity_id=entity_to_replace)
                # similarity of an entity to itself should be -1 (so it won't be selected)
                similarities[entity_to_replace] = -1
                # find entities with similarity values > 0 (similar neg. samples) or = 0 (non-similar neg. samples)
                sns_entities = np.where(similarities.cpu() > 0)[0]
                nns_entities = np.where(similarities.cpu()==0)[0]
                # create tensors corresponding to sns and nns
                sns = self.mapped_triples[i].repeat(len(sns_entities),1)
                sns[:,-1] = torch.Tensor(sns_entities, device=self.model.device)
                nns = self.mapped_triples[i].repeat(len(nns_entities),1)
                nns[:,-1] = torch.Tensor(nns_entities, device=self.model.device)
                original_triple_score = self.model.score_hrt(self.mapped_triples[None,i]).detach()

                minus_distances = {}
                minus_distances["sns"] = [i[0] for i in (self.model.score_hrt(sns) - original_triple_score).cpu().detach().numpy()]
                
                minus_distances["nns"] = [i[0] for i in (self.model.score_hrt(nns) - original_triple_score).cpu().detach().numpy()]

                save_path = self.ns_qual_analysis_path + '/{}/{}'.format(self.dataset, self.__class__.__name__ + "_" + self.similarity_function.__name__)
                Path(save_path).mkdir(parents=True, exist_ok=True)
                np.savez(save_path + "/triple_{}_after_epoch_{}.npz".format(i, epoch), **minus_distances)

                

