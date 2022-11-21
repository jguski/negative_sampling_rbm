import torch
import numpy as np
import os
import logging

from ..similarity_metrics import SimilarityFactory
from pykeen.sampling import BernoulliNegativeSampler
from pykeen.typing import COLUMN_HEAD, COLUMN_TAIL, COLUMN_RELATION, MappedTriples
from pykeen.models import Model
from pykeen.datasets import Dataset

from ..similarity_metrics import SimilarityFactory


class ESNSNoExploration(BernoulliNegativeSampler):
    
    def __init__(
        self,
        *,
        mapped_triples: MappedTriples,
        index_path: str = "Output/EII",
        index_column_size: int,
        max_index_column_size: int = 100,
        similarity_metric: str = "absolute",
        n_triples_for_ns_qual_analysis = 0,
        ns_qual_analysis_every = 100,
        ns_qual_analysis_path = "Output/NS_quality_analysis",
        model: Model,
        dataset: Dataset,
        logging_level: str = "INFO",
        skip_create_eii: bool=False,
        **kwargs,
    ) -> None:
        super().__init__(mapped_triples=mapped_triples, **kwargs)
        self.index_path = index_path + "/" + dataset
        # if self.num_entities < index_column_size, only self.num_entities can be stored
        self.index_column_size = min(self.num_entities, index_column_size)
        self.max_index_column_size = min(self.num_entities, max_index_column_size)
        self.similarity_function=SimilarityFactory.get(similarity_metric)
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
        
        if not skip_create_eii:
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
            np.savez(filename_base + '_h.npz', self.eii_h)
            self.logger.info("Creating EII {}_t.npz".format(filename_base))
            self.eii_t = self._create_eii(COLUMN_TAIL)
            np.savez(filename_base + '_t.npz', self.eii_t)
        else: 
            self.logger.info("Loading EII {}_h.npz".format(filename_base))
            self.eii_h = np.load(filename_base + '_h.npz')['arr_0']
            self.logger.info("Loading EII {}_t.npz".format(filename_base))
            self.eii_t = np.load(filename_base + '_t.npz')['arr_0']

        # index_column_size must be > 0 
        assert self.index_column_size > 0

        # get top self.index_column_size similar entities
        self.eii_h = self.eii_h[:,:self.index_column_size]
        self.eii_t = self.eii_t[:,:self.index_column_size]


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

        # return straight away if size of selection happens to be zero (function would crash otherwise)
        if size == 0:
            return
        else:
            entities_to_replace = batch[selection, index].cpu().numpy()

            # randomly sample EII column indices
            samples_from_eii = np.random.choice(self.index_column_size, entities_to_replace.size)

            # read the entities from the EII according to the random samples
            eii = (self.eii_h if index == COLUMN_HEAD else self.eii_t)
            replacement = eii[entities_to_replace, samples_from_eii]

            # replace entities in batch with the sampled entities
            batch[selection, index] =  torch.LongTensor(replacement).to(self.model.device)


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