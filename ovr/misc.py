
import numpy as np
import torch

def dot_similarity(visual_emb: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
    """
    Calculate dot similarity.

    Args:
        visual_emb: Visual embedding with shape (num_images, num_boxes, dim_embedding)
        text_emb: Text embedding with shape (num_images, num_tokens, dim_embedding)

    Returns:
        Similaries with shape (num_images, num_boxes, num_tokens)
    """
    assert (visual_emb.shape[-1] == text_emb.shape[-1]), "Embedding sizes of images in text and visual do not match"

    feat_shape = visual_emb.shape[-1]
    visual_shape = visual_emb.shape[:-1]
    text_shape = text_emb.shape[:-1]

    visual_emb = visual_emb.view(-1, feat_shape)
    text_emb = text_emb.view(-1, feat_shape)
    similarity = visual_emb.mm(text_emb.t())

    similarity = similarity.view(visual_shape + text_shape)

    return similarity

def dot_similarity_np(visual_emb, text_emb):
    """
    Calculate dot similarity.

    Args:
        visual_emb: Visual embedding with shape (num_images, num_boxes, dim_embedding)
        text_emb: Text embedding with shape (num_images, num_tokens, dim_embedding)

    Returns:
        Similaries with shape (num_images, num_boxes, num_tokens)
    """
    assert (visual_emb.shape[-1] == text_emb.shape[-1]), "Embedding sizes of images in text and visual do not match"
    similarity = np.dot(visual_emb, text_emb.swapaxes(-2,-1))

    return similarity

def l2_normalize(vector: torch.Tensor, dimension: int = -1):
    # vector_norm = F.normalize(vector, p=2, dim=dimension)
    assert not torch.isnan(vector).any()
    v_norm = (vector ** 2).sum(dim=dimension, keepdim=True).sqrt().detach()
    v_norm = torch.where(
        v_norm == 0,
        torch.ones_like(v_norm),
        v_norm)
    vector_norm = vector / v_norm
    vector_norm = torch.where(
        torch.isnan(vector_norm),
        torch.zeros_like(vector_norm),
        vector_norm)
    return vector_norm
    
def l2_normalize_np(vector, dimension = -1):
    norm = np.linalg.norm(vector, ord=2, axis=dimension, keepdims=True)
    norm[norm==0] = 1.0
    vector = vector/norm
    return vector