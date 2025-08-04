import torch
import torch.nn as nn
import torch.nn.functional as F


class CVKA(nn.Module):
    def forward(self, visual_only, text_features_all):
        # (64, 256, 7)
        similarity_matrix = torch.matmul(visual_only, text_features_all.transpose(0, 1))

        # (64, 7)
        similarity_sums = similarity_matrix.sum(dim=1)

        #(64, 2)
        top_two_similarity_indices = torch.topk(similarity_sums, 2, dim=1).indices

        # (64, 2, 512)
        max_similarity_text_features_1 = text_features_all[top_two_similarity_indices[:, 0]]
        max_similarity_text_features_2 = text_features_all[top_two_similarity_indices[:, 1]]


        # (64, 1, 256)
        similarity_matrix_1 = torch.bmm(
            max_similarity_text_features_1.unsqueeze(1),
            visual_only.transpose(1, 2)
        )

        # (softmax) (64, 1, 256)
        similarity_matrix_1_norm = F.softmax(similarity_matrix_1, dim=2)

        # (64, 256, 512)
        weighted_text_feature_1 = torch.bmm(
            similarity_matrix_1_norm.transpose(1, 2),
            max_similarity_text_features_1.unsqueeze(1)
        )

        # (64, 1, 256)
        similarity_matrix_2 = torch.bmm(
            max_similarity_text_features_2.unsqueeze(1),
            visual_only.transpose(1, 2)
        )

        # (64, 1, 256)
        similarity_matrix_2_norm = F.softmax(similarity_matrix_2, dim=2)

        # (64, 256, 512)
        weighted_text_feature_2 = torch.bmm(
            similarity_matrix_2_norm.transpose(1, 2),
            max_similarity_text_features_2.unsqueeze(1)
        )

        # (64, 256, 512)
        text_begin = weighted_text_feature_1 + weighted_text_feature_2


        top_k_similarity =(similarity_matrix_1_norm + similarity_matrix_2_norm).transpose(1,2)
        return text_begin, top_k_similarity
