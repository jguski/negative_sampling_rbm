import torch

def similarity_factory(similarity_metric):
    similarity_dict = {'absolute': absolute_similarity, 
        'jaccard': jaccard_similarity, 
        'cosine': cosine_similarity}
    try:
        return similarity_dict[similarity_metric]
    except KeyError:
        raise KeyError('The given similarity metric \"{}\" is currently not supported. Please use one of the following: {}.'
            .format(similarity_metric, ", ".join(list(similarity_dict.keys()))))


def absolute_similarity(relation_matrix: torch.ByteTensor, entity_id: int, head_or_tail: int, mapped_triples: torch.LongTensor):
    #assert relation_matrix.dtype is torch.uint8, "Cosine similarity can only be computed for torch.uint8, got {}.".format(relation_matrix.dtype)
    # store relations that are observed together with e_i
    relations = torch.unique(mapped_triples[(mapped_triples[:,head_or_tail]==entity_id), 1])
    # slice adjacency tensor, keeping only those columns corresponding to relations observed with e_i
    sliced_relation_matrix = relation_matrix.index_select(dim=1, index=relations)
    # sum over all rows (to get number of relations shared with e_i for each e_j)
    return sliced_relation_matrix.sum(dim=1)


def jaccard_similarity(relation_matrix: torch.ByteTensor, entity_id: int, head_or_tail: int, mapped_triples: torch.LongTensor):
    # matmul seeems to only work with float
    #assert relation_matrix.dtype is torch.uint8, "Cosine similarity can only be computed for torch.uint8, got {}.".format(relation_matrix.dtype)
    vector_e_i = relation_matrix[entity_id,]
    # calculate intersection = adjacency matrix * vector_e_i
    intersection = torch.matmul(relation_matrix, vector_e_i.t())
    # calculate union = rowwise_sum(acjacency matrix + vector_e_i) - intersection
    union = torch.add(vector_e_i, relation_matrix).sum(dim=1).subtract(intersection)
    # now compute similarities (in terms of Jaccard distance): Intersection over Union
    return torch.div(intersection, union)


def cosine_similarity(relation_matrix: torch.FloatTensor, entity_id: int, head_or_tail: int, mapped_triples:torch.LongTensor):
    assert relation_matrix.dtype is relation_matrix.dtype is torch.double, "Cosine similarity can only be computed for torch.double, got {}.".format(relation_matrix.dtype)
    # https://en.wikipedia.org/wiki/Cosine_similarity
    vector_e_i = relation_matrix[entity_id,]
    # A*B
    nominator = torch.matmul(relation_matrix, vector_e_i)
    norms = torch.linalg.vector_norm(relation_matrix, dim=1)
    # ||A||* ||B||
    denominator = norms * norms[entity_id]

    return torch.div(nominator, denominator)