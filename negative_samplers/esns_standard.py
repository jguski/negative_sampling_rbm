""" The original version of the entity similarity-based sampler, 
counting number of relation + target pairs for similarity measure"""

import torch
from scipy import sparse
from collections import defaultdict
from tqdm import tqdm
import numpy as np
from pathlib import Path

from pykeen.typing import COLUMN_TAIL

from .esns import ESNS


class ESNSStandard(ESNS):

    def __init__(
        self,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        similarity_metric = kwargs.pop("similarity_metric")
        if similarity_metric != "absolute":
            self.logger.warning("'{}' provided as similarity metric, but only 'absolute' is currently supported. Switching to 'absolute' automatically.". format(similarity_metric))

    def _create_eii(self, column: int) -> torch.LongTensor:
        """
        Create inverted indices (EII) containing a defined number of most similar entities e_j for each entity e_i.
        The logic was adapted from the original ESNS code. It is slower than the ESNS relaxed code which is based
        on matrix operations, but every combination of E x R x E would overload the memory.
        :param column: 
            Position of key entities (0 = head, 2 or -1 = tail). 
        """

        eii = sparse.csr_matrix((self.num_entities, self.num_entities)).tolil()

        conf = defaultdict(set)
        for triple in self.mapped_triples:
            triple_list = triple.to("cpu").numpy().tolist()
            conf[triple[column].item()].add(tuple(triple_list[:column] + triple_list[column+1:]))

        for e1_idx in tqdm(range(self.num_entities)):
            for dist in range(self.num_entities - e1_idx - 1):
                score = conf[e1_idx].intersection(conf[e1_idx + dist + 1])
                if len(score) != 0 and not (len(score) == len(conf[e1_idx]) and len(score) == len(conf[e1_idx + dist + 1])):
                    eii[e1_idx, e1_idx + dist + 1] = len(score)
                    eii[e1_idx + dist + 1, e1_idx] = len(score)

        return eii.tocsr()


    def quality_analysis(self, epoch, column=COLUMN_TAIL):

        if (self.ns_qual_analysis_every) != 0 and (epoch % self.ns_qual_analysis_every == 0):
            self.logger.info("Negative sampling quality analysis after epoch {}".format(epoch))

            if not any(self.random_triples_ids):
                # to ensure that always the same triples are considered
                np.random.seed(42)
                self.random_triples_ids = np.random.choice(self.mapped_triples.size()[0], len(self.random_triples_ids))
                

            # create dictionary with similarities
            conf = defaultdict(set)
            for triple in self.mapped_triples:
                triple_list = triple.numpy().tolist()
                conf[triple[column].item()].add(tuple(triple_list[:column] + triple_list[column+1:]))


            for i in self.random_triples_ids:
                entity_to_replace = self.mapped_triples[i,column]
                # compute similarity values of all e_j with e_i
                similarities = [0]*self.num_entities
                for entity in range(self.num_entities):
                    if entity_to_replace != entity:
                        score = conf[entity_to_replace].intersection(conf[entity])
                        similarities[entity] = len(score)
                    else:
                        # similarity of an entity to itself should be -1 (so it won't be selected)
                        similarities[entity] = -1

                # find entities with similarity values > 0 (similar neg. samples) or = 0 (non-similar neg. samples)
                sns_entities = [n for n,s in enumerate(similarities) if s > 0]
                nns_entities = [n for n,s in enumerate(similarities) if s == 0]

                # create tensors corresponding to sns and nns
                sns = self.mapped_triples[i].repeat(len(sns_entities),1)
                sns[:,-1] = torch.Tensor(sns_entities)
                nns = self.mapped_triples[i].repeat(len(nns_entities),1)
                nns[:,-1] = torch.Tensor(nns_entities)
                original_triple_score = self.model.score_hrt(self.mapped_triples[None,i]).detach()

                minus_distances = {}
                minus_distances["sns"] = [i[0] for i in (self.model.score_hrt(sns) - original_triple_score).cpu().detach().numpy()]
                
                minus_distances["nns"] = [i[0] for i in (self.model.score_hrt(nns) - original_triple_score).cpu().detach().numpy()]

                save_path = self.ns_qual_analysis_path
                Path(save_path).mkdir(parents=True, exist_ok=True)
                np.savez(save_path + "/triple_{}_after_epoch_{}.npz".format(i, epoch), **minus_distances)
			



