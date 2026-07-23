import os

import torch
import torch_scatter
from torch import nn, optim
from torch.nn.functional import pad

from Attention import Geo_attention


def printf(args, *content):
    message = " ".join(str(item) for item in content)
    with open(os.path.join(args.checkpoints_dir, "log.txt"), "a+") as f_handler:
        print(message, file=f_handler)
    print(message)


class EarlyStopping:
    def __init__(self, opt, patience_stop=10, patience_lr=5, delta=0.0001, path="check1.pth"):
        self.opt = opt
        self.stop_patience = patience_stop
        self.lr_patience = patience_lr
        self.counter = 0
        self.best_fitness = None
        self.early_stop = False
        self.delta = delta
        self.path = path

    def __call__(self, val_fitness, model):
        if self.best_fitness is None:
            self.best_fitness = val_fitness
            self.save_checkpoint(model)
            printf(self.opt, "saving best model...")
            return True
        if val_fitness <= self.best_fitness + self.delta:
            self.counter += 1
            if self.counter == self.lr_patience:
                self.adjust_lr(model)
            if self.counter >= self.stop_patience:
                self.early_stop = True
            return False

        self.best_fitness = val_fitness
        self.save_checkpoint(model)
        self.counter = 0
        printf(self.opt, "saving best model...")
        return False

    def adjust_lr(self, model):
        lr = model.optimizer.param_groups[0]["lr"] / 10
        for param_group in model.optimizer.param_groups:
            param_group["lr"] = lr
        model.load_state_dict(torch.load(self.path, map_location=model.device))
        printf(self.opt, "loading best model, changing learning rate to %.7f" % lr)

    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.path)


class Main_model(nn.Module):
    def __init__(self, opt):
        super().__init__()

        self.opt = opt
        self.device = torch.device(opt.device) if opt.device else torch.device("cpu")

        token_dim = 21
        hidden_dim = opt.emb_dims
        fusion_in = token_dim + opt.llm_dim + opt.msa_dim + opt.pssm_dim

        self.embed_tokens = nn.Embedding(token_dim, token_dim).to(self.device)
        self.embed = nn.Linear(fusion_in, hidden_dim, bias=True).to(self.device)

        self.stru_net = nn.ModuleList(
            [Geo_attention(hidden_dim, hidden_dim, 2) for _ in range(opt.n_layers_structure)]
        ).to(self.device)
        self.cross_net = nn.ModuleList(
            [Geo_attention(hidden_dim, hidden_dim, 2) for _ in range(opt.n_layers_structure)]
        ).to(self.device)

        self.net_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2, bias=True),
        ).to(self.device)

        self.optimizer = optim.Adam(self.parameters(), lr=opt.lr, weight_decay=1e-16)
        self.criterion = nn.CrossEntropyLoss()

    def set_input(self, data):
        self.P = {}

        self.P["token"] = pad(data["token"], (1, 0), value=0).to(self.device)
        self.P["llm"] = pad(data["llm"].to(self.device), (0, 0, 1, 0), value=0)
        self.P["msa"] = pad(data["msa"].to(self.device), (0, 0, 1, 0), value=0)
        self.P["pssm"] = pad(data["pssm"].to(self.device), (0, 0, 1, 0), value=0)

        self.P["xyz"] = pad(data["xyz"].to(self.device), (0, 0, 1, 0), value=0)
        self.P["nuv"] = torch.nn.functional.pad(data["nuv"].to(self.device), (0, 0, 0, 0, 1, 0), value=0)
        self.P["y"] = data["y"].to(self.device)

        self.P["amino_acid_batch"] = data["amino_acid_batch"].to(self.device)
        self.P["topk"] = data["adjusted_topk"].to(self.device)
        self.P["cross_topk"] = torch.nn.functional.pad(
            data["cross_topk"].to(self.device), (0, 0, 1, 0), value=1
        )

    def fuse_features(self):
        token_features = self.embed_tokens(self.P["token"])
        fusion_input = torch.cat([token_features, self.P["llm"], self.P["msa"], self.P["pssm"]], dim=1)
        self.P["features"] = self.embed(fusion_input)

    def embed_stru(self):
        features = self.P["features"]
        for index, _ in enumerate(self.cross_net):
            features = self.stru_net[index](features, self.P["xyz"], self.P["nuv"], self.P["topk"])
            features = self.cross_net[index](features, self.P["xyz"], self.P["nuv"], self.P["cross_topk"])
        self.P["features"] = features

        features_no_pad = self.P["features"][1:]
        amino_acid_batch = self.P["amino_acid_batch"]

        query = torch_scatter.scatter_mean(features_no_pad, amino_acid_batch, dim=0)
        query_expanded = query[amino_acid_batch]
        similarities = (features_no_pad * query_expanded).sum(dim=1)
        similarities = similarities / (features_no_pad.shape[1] ** 0.5)
        alpha = torch_scatter.composite.scatter_softmax(similarities, amino_acid_batch)

        weighted_features = features_no_pad * alpha.unsqueeze(1)
        pooled_features = torch_scatter.scatter_sum(weighted_features, amino_acid_batch, dim=0)

        self.P["features"] = torch.cat([
            torch.zeros(1, pooled_features.shape[1], device=pooled_features.device),
            pooled_features,
        ], dim=0)

    def forward(self):
        self.fuse_features()
        self.embed_stru()
        self.P["pre_logits"] = self.net_out(self.P["features"][1:])
        return self.P["pre_logits"]

    def optimize_parameters(self):
        self.train()
        self.optimizer.zero_grad()
        self.forward()
        self.loss = self.compute_loss()
        self.loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1)
        self.optimizer.step()
        return self.loss

    def test(self):
        self.eval()
        with torch.no_grad():
            self.forward()
            self.loss = self.compute_loss()
        return self.loss, self.P["y"], self.P["pre_logits"]

    def compute_loss(self):
        return self.criterion(self.P["pre_logits"], self.P["y"])
