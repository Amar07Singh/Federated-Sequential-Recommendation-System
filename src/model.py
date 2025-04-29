import numpy as np
import torch
import json
import argparse


class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):

        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2)
        outputs += inputs
        return outputs


# parser = argparse.ArgumentParser()
# parser.add_argument('--dataset', default='Movies_and_TV')
# parser.add_argument('--batch_size', default=128, type=int)
# parser.add_argument('--lr', default=0.001, type=float)
# parser.add_argument('--maxlen', default=50, type=int)
# parser.add_argument('--hidden_units', default=50, type=int)
# parser.add_argument('--num_blocks', default=2, type=int)
# parser.add_argument('--num_epochs', default=200, type=int)
# parser.add_argument('--num_heads', default=1, type=int)
# parser.add_argument('--dropout_rate', default=0.5, type=float)
# parser.add_argument('--l2_emb', default=0.0, type=float)
# parser.add_argument('--device', default='cpu', type=str)
# parser.add_argument('--inference_only', default=False, action='store_true')
# parser.add_argument('--state_dict_path', default=None, type=str)

# arg = parser.parse_args()
    
class FARF(torch.nn.Module):
    def __init__(self, user_num, item_num, arguer):

        super(FARF, self).__init__()


        self.kwargs = {'user_num': user_num, 'item_num': item_num, 'args': arguer}
        self.user_num = user_num
        self.item_num = item_num
        self.arguer = arguer

        args_better = json.loads('{'+arguer.replace('&', '"')+'}')

        self.dev = args_better['device']

        self.item_emb = torch.nn.Embedding(self.item_num+1, args_better['hidden_units'], padding_idx=0)
        self.pos_emb = torch.nn.Embedding(args_better['maxlen'], args_better['hidden_units'])
        self.emb_dropout = torch.nn.Dropout(p=args_better['dropout_rate'])

        self.attention_layernorms = torch.nn.ModuleList()
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        self.last_layernorm = torch.nn.LayerNorm(args_better['hidden_units'], eps=1e-8)

        
        for _ in range(args_better['num_blocks']):
            new_attn_layernorm = torch.nn.LayerNorm(args_better['hidden_units'], eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer =  torch.nn.MultiheadAttention(args_better['hidden_units'],
                                                            args_better['num_heads'],
                                                            args_better['dropout_rate'])
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(args_better['hidden_units'], eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(args_better['hidden_units'], args_better['dropout_rate'])
            self.forward_layers.append(new_fwd_layer)

    def log2feats(self, log_seqs):
        seqs = self.item_emb(torch.LongTensor(log_seqs).to(self.dev))
        seqs *= self.item_emb.embedding_dim ** 0.5
        positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])
        seqs += self.pos_emb(torch.LongTensor(positions).to(self.dev))
        seqs = self.emb_dropout(seqs)

        timeline_mask = torch.BoolTensor(log_seqs == 0).to(self.dev)
        seqs *= ~timeline_mask.unsqueeze(-1)

        tl = seqs.shape[1]
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))

        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](Q, seqs, seqs, 
                                            attn_mask=attention_mask)

            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs *=  ~timeline_mask.unsqueeze(-1)

        log_feats = self.last_layernorm(seqs)
        return log_feats

    def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs, mode='default'):
        log_feats = self.log2feats(log_seqs)
        if mode == 'log_only':
            log_feats = log_feats[:, -1, :]
            return log_feats
            
        pos_embs = self.item_emb(torch.LongTensor(pos_seqs).to(self.dev))
        neg_embs = self.item_emb(torch.LongTensor(neg_seqs).to(self.dev))

        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)

        # pos_pred = self.pos_sigmoid(pos_logits)
        # neg_pred = self.neg_sigmoid(neg_logits)
        if mode == 'item':
            return log_feats.reshape(-1, log_feats.shape[2]), pos_embs.reshape(-1, log_feats.shape[2]), neg_embs.reshape(-1, log_feats.shape[2])
        else:
            return pos_logits, neg_logits

    def predict(self, user_ids, log_seqs, item_indices):
        log_feats = self.log2feats(log_seqs)

        final_feat = log_feats[:, -1, :]

        item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.dev))

        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

        return logits
