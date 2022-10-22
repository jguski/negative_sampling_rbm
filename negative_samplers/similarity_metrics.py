from abc import abstractmethod
import torch



class SimilarityFunction():
    @staticmethod
    @abstractmethod
    def compute(relation_matrix: torch.Tensor, entity_id: int, head_or_tail: int, mapped_triples: torch.LongTensor):
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def typecheck(similarity_function, relation_matrix: torch.Tensor):
        raise NotImplementedError()

class BinarySimilarityFunction(SimilarityFunction):
    def typecheck(similarity_function, relation_matrix: torch.Tensor):
        # assert that the relation_matrix is binary
        unique_values = relation_matrix.unique()
        assert torch.equal(unique_values.to("cpu"), torch.Tensor([0,1])), "{} can only be computed for binary values, got matrix with values {} ...".format(similarity_function.__name__, unique_values.cpu().detach().numpy()[0:10])

class RealSimilarityFunction(SimilarityFunction):
    def typecheck(similarity_function, relation_matrix: torch.Tensor):
        # assert that the relation_matrix is of type torch.float (does not crash if it is binary)
        assert relation_matrix.dtype is torch.float, "{} can only be computed for torch.float, got {}.".format(similarity_function.__name__, relation_matrix.dtype)

class AbsoluteSimilarity(BinarySimilarityFunction):
    def compute(relation_matrix: torch.Tensor, entity_id: int, head_or_tail: int, mapped_triples: torch.LongTensor):
        super(AbsoluteSimilarity, AbsoluteSimilarity).typecheck(AbsoluteSimilarity, relation_matrix)

        # store relations that are observed together with e_i
        relations = torch.unique(mapped_triples[(mapped_triples[:,head_or_tail]==entity_id), 1])
        # slice adjacency tensor, keeping only those columns corresponding to relations observed with e_i
        sliced_relation_matrix = relation_matrix.index_select(dim=1, index=relations)
        # sum over all rows (to get number of relations shared with e_i for each e_j)
        return sliced_relation_matrix.sum(dim=1)

class JaccardSimilarity(BinarySimilarityFunction):
    def compute(relation_matrix: torch.Tensor, entity_id: int, head_or_tail: int, mapped_triples: torch.LongTensor):
        super(JaccardSimilarity, JaccardSimilarity).typecheck(JaccardSimilarity, relation_matrix)

        vector_e_i = relation_matrix[entity_id,]
        # calculate intersection = adjacency matrix * vector_e_i
        intersection = torch.matmul(relation_matrix, vector_e_i.t())
        # calculate union = rowwise_sum(acjacency matrix + vector_e_i) - intersection
        union = torch.add(vector_e_i, relation_matrix).sum(dim=1).subtract(intersection)
        # now compute similarities (in terms of Jaccard distance): Intersection over Union
        return torch.div(intersection, union)

class CosineSimilarity(RealSimilarityFunction):
    def compute(relation_matrix: torch.Tensor, entity_id: int, head_or_tail: int, mapped_triples:torch.LongTensor):
        super(CosineSimilarity, CosineSimilarity).typecheck(CosineSimilarity, relation_matrix)

        # https://en.wikipedia.org/wiki/Cosine_similarity
        vector_e_i = relation_matrix[entity_id,]
        # A*B
        nominator = torch.matmul(relation_matrix, vector_e_i)
        norms = torch.linalg.vector_norm(relation_matrix, dim=1)
        # ||A||* ||B||
        denominator = norms * norms[entity_id]
        return torch.div(nominator, denominator)

class SimilarityFactory():
    @staticmethod
    def get(similarity_metric) -> SimilarityFunction:   
        similarity_dict = {'absolute': AbsoluteSimilarity, 
            'jaccard': JaccardSimilarity, 
            'cosine': CosineSimilarity,
            '': None}
        try:
            return similarity_dict[similarity_metric]
        except KeyError:
            raise KeyError('The given similarity metric \"{}\" is currently not supported. Please use one of the following: {}.'
                .format(similarity_metric, ", ".join(list(similarity_dict.keys()))))