import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import pytorch_lightning as pl
from argparse import Namespace
from utils import get_linear_schedule_with_warmup, aa_idx
from torch_geometric.utils import to_undirected
from utils import edge_index


class Linear3l_Classifier(nn.Module):
    def __init__(self, in_channels, out_channels1, out_channels2):
        super().__init__()
        self.lin1 = nn.Linear(in_channels, out_channels1)
        self.lin2 = nn.Linear(out_channels1, out_channels2)
        self.lin3 = nn.Linear(out_channels2, 1)

    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = self.lin3(x)
        return torch.sigmoid(x)
    

class GCN_net(nn.Module):
    def __init__(self, in_channels, conv1_channels, conv2_channels, use_edge_weight=False):
        super().__init__()
        self.conv1 = GCNConv(in_channels, conv1_channels)
        self.conv2 = GCNConv(conv1_channels, conv2_channels)
        self.use_edge_weight = use_edge_weight

    def forward(self, x, edge_index, edge_weight=None):
        if not self.use_edge_weight:
            edge_weight = None
        x = self.conv1(x, edge_index=edge_index, edge_weight=edge_weight)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index=edge_index, edge_weight=edge_weight)
        return x


class GNN_model(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        self.device = device
        self.conv1_channels = config.get('conv1_dim')
        self.conv2_channels = config.get('conv2_dim')
        self.layer1_size = config.get('layer1_size')
        self.layer2_size = self.layer1_size // 2
        self.bsz = config['batch_size']
        self.embedding = nn.Embedding(21, 10)
        self.conv = self._build_model()
        self.conv2linear = nn.Linear(45*self.conv2_channels, self.layer1_size)
        self.classifier = self._build_classifier()

    def forward(self, batch):
        conv_out = self.get_conv_embedding(batch)
        conv_out = conv_out.flatten(1, -1)
        out = F.leaky_relu(self.conv2linear(conv_out))
        out = self.classifier(out)
        return out

    def get_conv_embedding(self, batch):
        x_s = batch.x_s
        x_s = x_s.reshape((-1, 11))
        x_t = batch.x_t
        x_t = x_t.reshape((-1, 34))
        x = torch.cat((x_s, x_t), dim=1)
        ed_idx = to_undirected(edge_index).to(device=self.device)
        x = self.embedding(x)
        out = self.conv(x, ed_idx)
        return out

    def _build_model(self):
        return GCN_net(10, self.conv1_channels, self.conv2_channels, use_edge_weight=True)


    def _build_classifier(self):
        return Linear3l_Classifier(self.layer1_size, self.layer1_size, self.layer2_size)
        

class Attention(nn.Module):
    def __init__(self, atten_size, return_attention=False):
        super().__init__()
        self.w = nn.Parameter(torch.rand(atten_size, 1))
        self.b = nn.Parameter(torch.zeros(atten_size))
        self.return_attention = return_attention

    def forward(self, x: torch.Tensor):
        # [seq_len, batch, num_directions * hidden_size]
        assert len(x.size()) == 3
        assert self.w.size(0) == x.size(-1)
        elements = torch.matmul(x, self.w) + self.b
        elements = F.tanh(elements)  # [seq_len, batch, 1]
        alpha = F.softmax(elements, dim=0)  # [seq_len, batch, 1]
        out = x * alpha  # [seq_len, batch, num_directions * hidden_size]
        out = out.sum(dim=0)  # [batch, num_directions * hidden_size]
        assert out.size(0) == x.size(1)
        if self.return_attention:
            return out, alpha
        return out
    

class GRU_model(nn.Module):
    def __init__(self, config, dropout, bidirectional, use_attention=True):
        super().__init__()
        self.use_attention = use_attention
        self.n_direction = 2 if bidirectional else 1
        self.embed_dim = config['embed_dim']
        self.rnn_dim = config['rnn_dim']
        self.embedding = nn.Embedding(len(aa_idx), self.embed_dim)
        self.gru = nn.GRU(self.embedding.embedding_dim, self.rnn_dim, num_layers=3,
                          dropout=dropout, bidirectional=bidirectional)
        self.rnn_outsize = self.gru.hidden_size * self.n_direction
        self.linear = nn.Linear(self.rnn_outsize, 1)
        self.attention = Attention(atten_size=self.rnn_outsize)

    def forward(self, x):
        x = self.embedding(x)
        x = x.transpose(1, 0)
        output, hidden = self.gru(x)
        # output [seq_len, batch, num_directions * hidden_size]
        # hidden [num_layers * num_directions, batch, hidden_size]
        if self.n_direction == 2:
            # [batch, hidden_size * num_directions]
            hidden = torch.cat([hidden[-1], hidden[-2]], dim=1)
        else:
            hidden = hidden[-1]
        assert hidden.size(1) == self.n_direction*self.gru.hidden_size
        if self.use_attention:
            out = self.attention(output)
        else:
            out = hidden
        out = self.linear(out)
        out = torch.sigmoid(out)
        return out

    def get_rnn_feature(self, x, return_seq=False):
        x = self.embedding(x)
        x = x.transpose(1, 0)
        output, hidden = self.gru(x)
        # output [seq_len, batch, num_directions * hidden_size]
        # hidden [num_layers * num_directions, batch, hidden_size]
        if return_seq:
            return output.transpose(1, 0)
        if self.n_direction == 2:
            # [batch, hidden_size * num_directions]
            hidden = torch.cat([hidden[-1], hidden[-2]], dim=1)
        else:
            hidden = hidden[-1]
        assert hidden.size(1) == self.n_direction*self.gru.hidden_size
        if self.use_attention:
            out = self.attention(output)
        else:
            out = hidden
        return out


class Fusion_model(nn.Module):
    def __init__(self, config, device=None):
        super().__init__()
        self.device = device
        self.config = config
        self.dropout = config['dropout']
        self.gnn_weight = nn.Parameter(torch.tensor(
            config['gnn_weight']), requires_grad=True)
        self.rnn_weight = nn.Parameter(torch.tensor(
            config['rnn_weight']), requires_grad=True)

        self.gnn = GNN_model(config, device=self.device)
        self.rnn = GRU_model(
            config=config, dropout=self.dropout, bidirectional=True, use_attention=True)
        self.activation = nn.ReLU()
        self.indiv_fc_size = config['final_fc']
        self.gnn_fc = nn.Sequential(
            nn.Flatten(1, -1),
            nn.Linear(45*self.gnn.conv2_channels,
                      self.indiv_fc_size), self.activation
        )
        self.rnn_fc = nn.Sequential(
            nn.Linear(self.rnn.rnn_outsize, self.indiv_fc_size),
            self.activation
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.indiv_fc_size, 512),
            self.activation,
            nn.Linear(512, 512),
            self.activation,
            nn.Linear(512, 256),
            self.activation,
            nn.Linear(256, 1)
        )

    def forward(self, gnn_input, rnn_input):
        gnn_features = self.gnn.get_conv_embedding(gnn_input)
        rnn_features = self.rnn.get_rnn_feature(rnn_input)
        gnn_features = self.gnn_weight * gnn_features
        rnn_features = self.rnn_weight * rnn_features
        gnn_fc_out = F.dropout(self.gnn_fc(gnn_features), self.dropout, training=self.training)
        rnn_fc_out = F.dropout(self.rnn_fc(rnn_features), self.dropout, training=self.training)
        out = gnn_fc_out + rnn_fc_out
        out = self.classifier(out)
        out = torch.sigmoid(out)
        return out


class Fusion_PL(pl.LightningModule):
    def __init__(self, config, total_steps=None, warmup_ratio=0.1, device=None):
        super().__init__()
        self.total_steps = total_steps
        self.warmup_ratio = warmup_ratio
        self.warmup_steps = math.ceil(total_steps * warmup_ratio)
        self.model = Fusion_model(config, device)
        self.config = config
        self.save_hyperparameters(Namespace(**self.config))

    def forward(self, batch):
        gnn_input = batch['gnn_input']
        rnn_input = batch['rnn_input']
        out = self.model(gnn_input, rnn_input)
        return out

    def get_loss(self, pred, label):
        loss = nn.BCELoss()
        return loss(pred, label.float())

    def training_step(self, batch, batch_idx):
        label = batch['label']
        label = label.unsqueeze(dim=-1)
        pred = self.forward(batch)
        loss = self.get_loss(pred, label)
        sch = self.lr_schedulers()
        sch.step()
        opt = self.optimizers()
        self.log('lr', opt.state_dict()['param_groups'][0]['lr'])
        self.log(f'loss/train_loss', loss)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        label = batch['label']
        label = label.unsqueeze(dim=-1)
        pred = self.forward(batch)
        loss = self.get_loss(pred, label)
        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        self.log(f'loss/val_loss', avg_loss)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=1e-3, weight_decay=1e-2)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.warmup_steps, num_training_steps=self.total_steps)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def predict_step(self, batch, batch_idx):
        pred = self.forward(batch)
        return pred.detach().cpu().view(-1)