import math
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class AttentionBlock(nn.Module):
    def __init__(self, dims, k_size, v_size, seq_len=None):
        super(AttentionBlock, self).__init__()
        self.key_layer = nn.Linear(dims, k_size)
        self.query_layer = nn.Linear(dims, k_size)
        self.value_layer = nn.Linear(dims, v_size)
        self.sqrt_k = math.sqrt(k_size)

    def forward(self, minibatch):
        keys = self.key_layer(minibatch)
        queries = self.query_layer(minibatch)
        values = self.value_layer(minibatch)
        logits = torch.bmm(queries, keys.transpose(2, 1))
        mask = np.triu(np.ones(logits.size()), k=1).astype('bool')
        mask = torch.from_numpy(mask).cuda()
        logits.data.masked_fill_(mask, float('-inf'))
        probs = F.softmax(logits, dim=1) / self.sqrt_k
        read = torch.bmm(probs, values)
        return minibatch + read


class MKAF(nn.Module):
    def __init__(self):
        super(MKAF, self).__init__()

        self.encoder1 = nn.Linear(512, 64)
        self.encoder2 = nn.Linear(512, 128)
        self.encoder3 = nn.Linear(512, 64)

        self.affine_a = nn.Linear(256, 256, bias=False)
        self.affine_v = nn.Linear(256, 256, bias=False)
        self.affine_t = nn.Linear(256, 256, bias=False)

        self.W_a = nn.Linear(256, 32, bias=False)
        self.W_v = nn.Linear(256, 32, bias=False)
        self.W_t = nn.Linear(256, 32, bias=False)
        self.W_ca = nn.Linear(256, 32, bias=False)
        self.W_cv = nn.Linear(256, 32, bias=False)
        self.W_ct = nn.Linear(256, 32, bias=False)

        self.W_ha = nn.Linear(32, 256, bias=False)
        self.W_hv = nn.Linear(32, 256, bias=False)
        self.W_ht = nn.Linear(32, 256, bias=False)

        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.vregressor = nn.Sequential(nn.Linear(256, 128),
                                        nn.Dropout(0.6),
                                        nn.Linear(128, 128))

        self.aregressor = nn.Sequential(nn.Linear(256, 128),
                                        nn.Dropout(0.6),
                                        nn.Linear(128, 128))

    # def first_init(self):
    #    nn.init.xavier_normal_(self.corr_weights)

    def forward(self, f1_norm, f2_norm, f3_norm):
        # f1 = f1.squeeze(1)
        # f2 = f2.squeeze(1)

        # f1_norm = F.normalize(f1_norm, p=2, dim=2, eps=1e-12)
        # f2_norm = F.normalize(f2_norm, p=2, dim=2, eps=1e-12)

        fin_audio_features = []
        fin_visual_features = []
        fin_text_features = []

        vsequence_outs = []
        asequence_outs = []
        tsequence_outs = []

        for i in range(f1_norm.shape[0]):
            audfts = f1_norm[i, :, :]  # .transpose(0,1)
            visfts = f2_norm[i, :, :]  # .transpose(0,1)
            texfts = f3_norm[i, :, :]  # .transpose(0,1)

            aud_fts = self.encoder1(audfts)
            vis_fts = self.encoder2(visfts)
            tex_fts = self.encoder3(texfts)

            aud_vis_fts = torch.cat((aud_fts, vis_fts, tex_fts), 1)  # i*256*256
            a_t = self.affine_a(aud_vis_fts.transpose(0, 1))
            att_aud = torch.mm(aud_fts.transpose(0, 1), a_t.transpose(0, 1))
            audio_att = self.tanh(torch.div(att_aud, math.sqrt(aud_vis_fts.shape[1])))

            aud_vis_fts = torch.cat((aud_fts, vis_fts, tex_fts), 1)
            v_t = self.affine_v(aud_vis_fts.transpose(0, 1))
            att_vis = torch.mm(vis_fts.transpose(0, 1), v_t.transpose(0, 1))
            vis_att = self.tanh(torch.div(att_vis, math.sqrt(aud_vis_fts.shape[1])))

            aud_vis_fts = torch.cat((aud_fts, vis_fts, tex_fts), 1)
            t_t = self.affine_t(aud_vis_fts.transpose(0, 1))
            att_tex = torch.mm(tex_fts.transpose(0, 1), t_t.transpose(0, 1))
            tex_att = self.tanh(torch.div(att_tex, math.sqrt(aud_vis_fts.shape[1])))

            H_a = self.relu(self.W_ca(audio_att) + self.W_a(aud_fts.transpose(0, 1)))
            H_v = self.relu(self.W_cv(vis_att) + self.W_v(vis_fts.transpose(0, 1)))
            H_t = self.relu(self.W_ct(tex_att) + self.W_t(tex_fts.transpose(0, 1)))

            att_audio_features = self.W_ha(H_a).transpose(0, 1) + aud_fts   # i*256*128
            att_visual_features = self.W_hv(H_v).transpose(0, 1) + vis_fts
            att_text_features = self.W_ht(H_t).transpose(0, 1) + tex_fts

            # a1 = torch.matmul(aud_fts.transpose(0,1), self.corr_weights)
            # cc_mat = torch.matmul(a1, vis_fts)

            # audio_att = F.softmax(cc_mat, dim=1)
            # visual_att = F.softmax(cc_mat.transpose(0,1), dim=1)

            # atten_audiofeatures = torch.matmul(aud_fts, audio_att)
            # atten_visualfeatures = torch.matmul(vis_fts, visual_att)

            # added_atten_audiofeatures = aud_fts.add(atten_audiofeatures)
            # added_atten_visualfeatures = vis_fts.add(atten_visualfeatures)

            #### apply tanh on features

            ### apply same dimensions
            # att_audio_features = self.tanh(atten_audiofeatures)
            # att_visual_features = self.tanh(atten_visualfeatures)

            # audiovisualfeatures = torch.cat((visfts, att_visual_features), 1)
            audiovisualfeatures = torch.cat((att_audio_features, att_visual_features, att_text_features), 1)   #i*256*256
            vouts = self.vregressor(audiovisualfeatures)  # .transpose(0,1))    i*256*128
            aouts = self.aregressor(audiovisualfeatures)  # .transpose(0,1))
            touts = self.aregressor(audiovisualfeatures)  # .transpose(0,1))
            # seq_outs, _ = torch.max(outs,0)
            # print(seq_outs)
            vsequence_outs.append(vouts)
            asequence_outs.append(aouts)
            tsequence_outs.append(touts)

            fin_audio_features.append(att_audio_features)
            fin_visual_features.append(att_visual_features)
            fin_text_features.append(att_text_features)

        final_aud_feat = torch.stack(fin_audio_features)
        final_vis_feat = torch.stack(fin_visual_features)
        final_text_feat = torch.stack(fin_text_features)

        # vfinal_outs = torch.stack(vsequence_outs)
        # afinal_outs = torch.stack(asequence_outs)
        # tfinal_outs = torch.stack(tsequence_outs)
        return final_vis_feat, final_aud_feat, final_text_feat  # final_aud_feat.transpose(1,2), final_vis_feat.transpose(1,2)

