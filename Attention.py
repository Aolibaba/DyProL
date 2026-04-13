import torch
import torch.nn as nn
import torch.nn.functional as F


class Geo_attention(nn.Module):
    """
    Geo-attention with learnable distance kernel, relative rotation encoding, proper multi-head shaping,
    and post-residual LayerNorm.
    """
    def __init__(self, Ni, Nd, Nh, *, act=nn.ReLU):
        super().__init__()

        assert Nd % Nh == 0, "Nd must be divisible by Nh for multi-head attention."
        self.Ni = Ni
        self.Nd = Nd
        self.Nh = Nh
        self.Hd = Nd // Nh

        self.feat_proj = nn.Linear(Ni, Nd)

        self.query_f = nn.Sequential(
            nn.Linear(Nd, Nd),
            act(),
            nn.Linear(Nd, Nd),
        )

        self.geo_encoder = nn.Sequential(
            nn.Linear(12, Nd),
            act(),
            nn.Linear(Nd, Nd),
        )

        self.dist_kernel = nn.Sequential(
            nn.Linear(1, Nd),
            act(),
            nn.Linear(Nd, Nd)
        )

        self.neigh_feat_proj = nn.Linear(Ni, Nd)

        self.key_f = nn.Sequential(
            nn.Linear(Nd, Nd),
            act(),
            nn.Linear(Nd, Nd),
        )
        self.value_f = nn.Sequential(
            nn.Linear(Nd, Nd),
            act(),
            nn.Linear(Nd, Nd),
        )

        self.decode_f = nn.Sequential(
            nn.Linear(Nd, Nd),
            act(),
            nn.Linear(Nd, Nd),
        )

        self.layer_norm = nn.LayerNorm(Nd)


    def forward(self, features, x, nuv, topk):
        """
        Args:
            features: [N, Ni]
            x:        [N, 3]
            nuv:      [N, 3, 3]  (local frames per residue)
            topk:     [N, nn] (neighbor indices, 0 denotes padding)
        Returns:
            [N, Nd]
        """
        N = features.size(0)
        nnbr = topk.size(1)

        # ---- 1) Project center & neighbor features to Nd ----
        feat_nd = self.feat_proj(features)                # [N, Nd]
        feat_nbr = features[topk]                         # [N, nn, Ni]
        feat_nbr_nd = self.neigh_feat_proj(feat_nbr)      # [N, nn, Nd]

        # ---- 2) Multi-head Query ----
        Q = self.query_f(feat_nd)                         # [N, Nd] = [N, Nh*Hd]
        Q = Q.view(N, self.Nh, self.Hd)                   # [N, Nh, Hd]

        # ---- 3) Geometric relative features ----
        x_nbr = x[topk]                                   # [N, nn, 3]
        delta_x = x_nbr - x.unsqueeze(1)                  # [N, nn, 3]
        # Relative position in center local frame: RL_x = nuv * Δx
        # nuv: [N, 3, 3], delta_x^T: [N, 3, nn] -> matmul -> [N, 3, nn] -> transpose -> [N, nn, 3]
        RL_x = torch.matmul(nuv, delta_x.transpose(1, 2)).transpose(1, 2)  # [N, nn, 3]

        # Relative rotation: R_rel = nuv^T @ nuv_nbr
        nuv_nbr = nuv[topk]                               # [N, nn, 3, 3]
        center_rot_T = nuv.transpose(1, 2).unsqueeze(1)   # [N, 1, 3, 3]
        R_rel = torch.matmul(center_rot_T, nuv_nbr)       # [N, nn, 3, 3]
        R_rel_flat = R_rel.reshape(N, nnbr, 9)            # [N, nn, 9]

        # Pack 12-dim geometric descriptor per neighbor
        geom12 = torch.cat([RL_x, R_rel_flat], dim=-1)    # [N, nn, 12]
        # Encode geometry to Nd
        geo_fea = self.geo_encoder(geom12)                # [N, nn, Nd]

        # ---- 4) Learnable distance kernel gate ----
        dist = torch.norm(delta_x, dim=-1, keepdim=True)  # [N, nn, 1]
        #sigma = 5.0
        #gaussian_dist = torch.exp(-dist ** 2 / (2 * sigma ** 2))
        dist_gate = torch.sigmoid(self.dist_kernel(dist)) # [N, nn, Nd] in (0,1)

        # ---- 5) Fuse: geometry ⊙ distance_gate ⊙ neighbor_feature ----
        fused = geo_fea * dist_gate * feat_nbr_nd         # [N, nn, Nd]

        # ---- 6) Keys & Values (per-head reshape) ----
        # Produce Nd then reshape -> [N, nn, Nh, Hd] -> [N, Nh, nn, Hd]
        K = self.key_f(fused).view(N, nnbr, self.Nh, self.Hd).transpose(1, 2)   # [N, Nh, nn, Hd]
        V = self.value_f(fused).view(N, nnbr, self.Nh, self.Hd).transpose(1, 2) # [N, Nh, nn, Hd]

        # ---- 7) Attention scores ----
        # scores[b,h,nbr] = <Q[b,h,:], K[b,h,nbr,:]> / sqrt(Hd)
        scores = (Q.unsqueeze(2) * K).sum(dim=-1) / (self.Hd ** 0.5)            # [N, Nh, nn]
        #scores = torch.matmul(Q.unsqueeze(2), K.transpose(-1, -2)).squeeze(2) / (self.Hd ** 0.5)
        mask = (topk == 0).unsqueeze(1)                                         # [N, 1, nn]
        scores = scores.masked_fill(mask, float('-inf'))

        # Softmax over neighbors
        attn = F.softmax(scores, dim=2)                                          # [N, Nh, nn]

        # ---- 8) Weighted sum of V ----
        context = (attn.unsqueeze(-1) * V).sum(dim=2)                            # [N, Nh, Hd]
        context = context.reshape(N, self.Nd)                                    # [N, Nd]

        # ---- 9) Output MLP + Residual + LayerNorm (post-norm) ----
        out = self.decode_f(context)                                             # [N, Nd]
        out = self.layer_norm(out + feat_nd)                                     # [N, Nd]
        return out


