import torch
import os, sys
from torch import nn, optim
from torch.nn.functional import pad
import torch_scatter

from Attention import Geo_attention

def printf(args, *content):
    file = sys.stdout
    f_handler = open(os.path.join(args.checkpoints_dir, 'log.txt'), 'a+')
    sys.stdout = f_handler
    print(' '.join(content))
    f_handler.close()
    sys.stdout = file
    print(' '.join(content))

class EarlyStopping:
    def __init__(self, opt, patience_stop=10, patience_lr=5, verbose=False, delta=0.0001, path='check1.pth'):
        self.opt = opt
        self.stop_patience = patience_stop
        self.lr_patience = patience_lr
        self.verbose = verbose
        self.counter = 0
        self.best_fitness = None
        self.early_stop = False
        self.delta = delta
        self.path = path

    def __call__(self, val_fitness, model):
        if self.best_fitness is None:
            self.best_fitness = val_fitness
            self.save_checkpoint(model)
            printf(self.opt, 'saving best model...')
            return True
        elif val_fitness <= self.best_fitness + self.delta:
            self.counter += 1
            if self.counter == self.lr_patience:
                self.adjust_lr(model)
            if self.counter >= self.stop_patience:
                self.early_stop = True
            return False
        else:
            self.best_fitness = val_fitness
            self.save_checkpoint(model)
            self.counter = 0
            printf(self.opt, 'saving best model...')
            return False

    def adjust_lr(self, model):
        lr = model.optimizer.param_groups[0]['lr']
        lr = lr/10
        for param_group in model.optimizer.param_groups:
            param_group['lr'] = lr
        model.load_state_dict(torch.load(self.path))
        printf(self.opt, 'loading best model, changing learning rate to %.7f' % lr)

    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.path)


class Main_model(nn.Module):

    def __init__(self, opt):
        super(Main_model, self).__init__()

        self.opt = opt
        self.gpu_ids = opt.device
        self.device = torch.device('{}'.format(self.gpu_ids)) if self.gpu_ids else torch.device('cpu')

        in_channels = 21

        E = opt.emb_dims

        self.lr = opt.lr
        fusion_in = 0
        self.embed_tokens = nn.Embedding(in_channels, in_channels).to(self.device)
        if opt.use_token:
            fusion_in += in_channels
        if opt.use_llm:
            fusion_in += opt.llm_dim
        if opt.use_msa:
            fusion_in += opt.msa_dim
        if opt.use_pssm:
            fusion_in += opt.pssm_dim
        assert fusion_in > 0, "At least one type of feature needs to be used"

        self.embed = nn.Linear(fusion_in, E, bias=True).to(self.device)

        self.stru_net = nn.ModuleList([Geo_attention(E, E, 2) for i in range(opt.n_layers_structure)]).to(self.device)

        self.cross_net = nn.ModuleList([Geo_attention(E, E, 2) for i in range(opt.n_layers_structure)]).to(self.device)

        self.net_out = nn.Sequential(
            nn.Linear(E, E, bias=True),
            nn.ReLU(),
            nn.Linear(E, E, bias=True),
            nn.ReLU(),
            nn.Linear(E, 2, bias=True),
        ).to(self.device)

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr,  weight_decay=1e-16)
#        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, 20) # #of epochs 40
        self.criterion = torch.nn.CrossEntropyLoss()


    def set_input(self, data):
        self.P = {}
        self.P['token'] = data.get('token')
        self.P['token'] = pad(self.P['token'], (1, 0), value=0).to(self.device)
        if self.opt.use_llm:
            self.P['llm'] = data.get('llm').to(self.device)
            self.P['llm'] = pad(self.P['llm'], (0, 0, 1, 0), value=0).to(self.device)
        if self.opt.use_msa:
            self.P['msa'] = data.get('msa').to(self.device)
            self.P['msa'] = pad(self.P['msa'], (0, 0, 1, 0), value=0).to(self.device)
        if self.opt.use_pssm:
            self.P['pssm'] = data.get('pssm').to(self.device)
            self.P['pssm'] = pad(self.P['pssm'], (0, 0, 1, 0), value=0).to(self.device)        

        self.P['xyz'] = data.get('xyz').to(self.device)
        self.P['xyz'] = pad(self.P['xyz'], (0, 0, 1, 0), value=0).to(self.device)

        self.P['nuv'] = data.get('nuv').to(self.device)
        self.P['nuv'] = torch.nn.functional.pad(self.P['nuv'], (0, 0, 0, 0, 1, 0), value=0).to(self.device)

        self.P['y'] = data.get('y').to(self.device)
        self.P['batch'] = data.get('xyz_batch').to(self.device)

        if self.opt.dynamic_mode == 'dynamic':
            self.P['amino_acid_batch'] = data.get('amino_acid_batch').to(self.device)
            self.P['topk'] = data.get('adjusted_topk').to(self.device)

            cross_topk = data.get('cross_topk').to(self.device)
            self.P['cross_topk'] = torch.nn.functional.pad(cross_topk, (0, 0, 1, 0), value=1).to(self.device)

        elif self.opt.dynamic_mode == 'static':
            topk = data.get('topk').to(self.device)
            self.P['topk'] = self.collect_topk(topk, self.P['batch'])

    def collect_topk(self, topk, batch1, batch2=None):
        if torch.sum(batch1==1) == 0:
            return torch.nn.functional.pad(topk, (0,0,1,0), value=1)
        if batch2 is None:
            batch2 = batch1
        mask = topk==0
        indices1 = torch.nonzero(torch.eq(batch1[1:] - batch1[:-1], 1)).squeeze() + 1
        indices2 = torch.nonzero(torch.eq(batch2[1:] - batch2[:-1], 1)).squeeze() + 1

        if indices1.numel() == 0:
            topk = torch.nn.functional.pad(topk, (0, 0, 1, 0), value=1)
            return topk
        if len(indices1.shape) == 0:
            indices1 = indices1.unsqueeze(0)
            indices2 = indices2.unsqueeze(0)

        if indices1.shape[0] > 1:
            for item in range(indices1.shape[0]-1):
                topk[indices1[item]:indices1[item+1]] += indices2[item]

        if indices1.shape[0] > 0:
            topk[indices1[-1]:] += indices2[-1]
        topk[mask] = 0
        topk = torch.nn.functional.pad(topk, (0,0,1,0), value=1)
        return topk

    def fuse_features(self):
        feats = []
        if self.opt.use_token:
            self.P['f_token'] = self.embed_tokens(self.P['token'])
            feats.append(self.P['f_token'])
        if self.opt.use_llm:
            feats.append(self.P['llm'])
        if self.opt.use_msa:
            feats.append(self.P['msa'])
        if self.opt.use_pssm:
            feats.append(self.P['pssm'])

        fusion_input = torch.cat(feats, dim=1)
        if self.opt.fusion_type == 'gate':
            gate_logits = self.gate_net(fusion_input)
            gate = torch.softmax(gate_logits, dim=1)

            g_token = gate[:, 0:1]
            g_llm = gate[:, 1:2]
            g_msa = gate[:, 2:3]
            g_res = gate[:, 3:4]

            self.P['features'] = (
                    g_token * self.P['f_token'] +
                    g_llm * self.P['f_llm'] +
                    g_msa * self.P['f_msa'] +
                    g_res * self.P['f_resfeat']
            )
        else:
            self.P['features'] = self.embed(fusion_input)

    def embed_stru(self):
        if self.opt.dynamic_mode == 'dynamic':
            features = self.P['features']

            for index, _ in enumerate(self.cross_net):
                features = self.stru_net[index](features, self.P['xyz'], self.P['nuv'], self.P['topk'])
                features = self.cross_net[index](features, self.P['xyz'], self.P['nuv'], self.P['cross_topk'])
            self.P['features'] = features

            features_no_pad = self.P['features'][1:]
            amino_acid_batch = self.P['amino_acid_batch']

            Q = torch_scatter.scatter_mean(features_no_pad, amino_acid_batch, dim=0)
            Q_expanded = Q[amino_acid_batch]
            similarities = (features_no_pad * Q_expanded).sum(dim=1)
            similarities = similarities / (features_no_pad.shape[1] ** 0.5)
            alpha = torch_scatter.composite.scatter_softmax(similarities, amino_acid_batch)

            weighted_features = features_no_pad * alpha.unsqueeze(1)
            pooled_features = torch_scatter.scatter_sum(weighted_features, amino_acid_batch, dim=0)

            self.P['features'] = torch.cat([
                torch.zeros(1, pooled_features.shape[1], device=pooled_features.device),
                pooled_features
            ], dim=0)

        elif self.opt.dynamic_mode == 'static':
            for encoder in self.stru_net:
                self.P['features'] = encoder(self.P['features'], self.P['xyz'], self.P['nuv'], self.P['topk'])

    def forward(self):
        self.fuse_features()
        self.embed_stru()
        self.P["pre_logits"] = self.net_out(self.P['features'][1:]).squeeze()
        pass

    def optimize_parameters(self):
        self.train()
#        self.scheduler.step()
        self.forward()
        self.loss = self.compute_loss()
        self.loss.backward()

        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1)

        self.optimizer.step()
        self.optimizer.zero_grad()
        return self.loss

    def test(self):
        self.eval()
        with torch.no_grad():
            self.forward()
            self.loss = self.compute_loss()
        return self.loss, self.P['y'], self.P["pre_logits"]

    def compute_loss(self):
        loss = self.criterion(self.P["pre_logits"], self.P['y'])
        return loss
