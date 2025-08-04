import torch
import torch.nn as nn
import torch.nn.functional as F

class AFAE(nn.Module):
    def forward(self, audio, top_k_similarity):
        top_k_similarity_modified = top_k_similarity.clone()
        for b in range(top_k_similarity.size(0)):
            length = top_k_similarity[b, :, 0].size(0)
            k = min(16, length)
            topk_values, topk_indices = torch.topk(top_k_similarity[b, :, 0], k)

            for idx in topk_indices:
                center_value = top_k_similarity[b, idx, 0]
                for i in range(1, 4):
                    if idx - i >= 0:
                        top_k_similarity_modified[b, idx - i, 0] = center_value
                for i in range(1, 4):
                    if idx + i < length:
                        top_k_similarity_modified[b, idx + i, 0] = center_value

        are_equal = torch.equal(top_k_similarity, top_k_similarity_modified)
        audio = audio * top_k_similarity_modified

        return audio


