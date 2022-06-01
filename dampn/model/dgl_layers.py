"""Pytorch modules written in the context of dgl layers."""
from typing import Tuple

import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
from dgl.nn.pytorch import edge_softmax

# typing
from typing import Union

import logging
logger = logging.getLogger(__name__)

class DAMPLayer(nn.Module):
    """Distance Attentive Message passer.

    Update node state by messages from combined neighbor edges and node states.
    A Gated Recurrent Unit is applied to an attention mechanism to determine
    message weights. Edge states are assumed to embed distance. Optionally
    also update edge states. This is a bidirectional model, where each edge has
    two directed states.

    This layer is inspired by the first convolutional layer of the AttentiveFP
    model:

    <Xiong, Z., Wang, D., Liu, X., Zhong, F., Wan, X., Li, X., … Zheng, M.
        (2020). Pushing the Boundaries of Molecular Representation for Drug
        Discovery with the Graph Attention Mechanism. J. Med. Chem, 63, 21.
        https://doi.org/10.1021/acs.jmedchem.9b00959>

    In that model, only the first convolutional layer uses edge states to
    determine the messages, and convolves only over node states forever after.
    Here, edge features are given to the next layer, and optionally convolved
    themselves. This is mean for fully connected graphs of distances, allowing
    the distance to be attended to.

    This layer opperates as such for node v:

    (1) Embed node states:
        h^v = U(f^v); U:= activated layer shape (node feat, node hidden)
    (2) Embed neighbor states from neighbor node w and edge states to determine
    edge hidden states:
        h^{w \rightarrow v} = M(f^{w \rightarrow v}, f^w);
        M:= activated layer shape (edge feat + node feat, edge hidden)
    (3) Compute attention logits:
        l_c^{w \rightarrow v} = A(h^{(w \rightarrow v)'}_{(t)}, h^{v})
        A:= activated neuron shape (edge hidden + node hidden, 1)
    (4) Compute each edges attention using softmax:
        \alpha^{v} = softmax_{w\\in E(v)}(l_c^{i \rightarrow v})
    (5) Compute a context vector from the messages:
        C^{v} = elu(\\Sigma_{w \\in E(v)} \alpha^v_w * T(h^{w \rightarrow v}))
        T:= linear layer shape (edge feat, context hidden)
    (6) Update node states:
        h^{v'} = GRU(h_v, C^v)
        h^{v'} \rightarrow f^v
    (7 - optional) Update edge states:
        h^{w \rightarrow v} \rightarrow f^{w \rightarrow v}
    """

    def __init__(
        self,
        node_feature_size: int,
        edge_feature_size: int,
        node_hidden_size: int,
        edge_hidden_size: int,
        context_size: int,
        update_edges: bool = False,
        dropout: float = 0.0
    ):
        super().__init__()

        self.update_edges = update_edges
        # unit creation
        ########################################
        # produces embeded node state f^v -> h_v
        self.embed_node = nn.Sequential(
            nn.Linear(node_feature_size, node_hidden_size),
            nn.LeakyReLU()
        )

        # produces embded edge state
        # combines neighbor nodes and edges
        # f^w, f^{w->v} -> h^{w->v}
        self.embed_edge = nn.Sequential(
            nn.Linear(edge_feature_size + node_feature_size, edge_hidden_size),
            nn.LeakyReLU()
        )

        # produces logit for neighbor w as it relates to v
        # h^{w->v}, h^v -> l^{w->v}
        self.compute_logit = nn.Sequential(
            nn.Linear(edge_hidden_size + node_hidden_size, 1),
            nn.Dropout(dropout),
            nn.LeakyReLU()
        )

        # produces the raw message from neighbor w to node v
        # after this, we take the attention weighted sum
        # h^{w->v} -> m^{w->v}
        self.compute_raw_message = nn.Sequential(
            nn.Linear(edge_hidden_size, context_size),
            nn.Dropout(dropout)
        )

        # the gru updates the node state
        # maintains smooth gradient
        # h^v, C -> f^v
        self.gru = nn.GRUCell(context_size, node_hidden_size)
        return

    def reset_parameters(self):
        """Re-initialize all learnable parameters."""
        self.gru.reset_parameters()
        self.compute_raw_message[0].reset_parameters()
        self.compute_logit[0].reset_parameters()
        self.embed_edge[0].reset_parameters()
        self.embed_node[0].reset_parameters()
        return

    def _apply_embed_edge(self, edges):
        """Embed each edge.

        This is an edge-wise combination of features f_w->v and f_w, so it
        cannot simply be passed through as a single tensor.

        Expected to be called by DGLGraph batch apply_edges.

        Parameters
        ----------
        edges : EdgeBatch
            Container for a batch of edges.
        """
        out = {
            'h_w->v': torch.cat([edges.src['f_v'], edges.data['f_w->v']], dim=1)
        }
        return out

    def _apply_compute_logit(self, edges):
        """Compute logit for each edge

        This is an edge-wise combination of node and edge hidden states h_v, h_w->v
        cannot simply be passed through as a single tensor.

        Expected to be called by DGLGraph batch apply_edges.

        Parameters
        ----------
        edges : EdgeBatch
            Container for a batch of edges.
        """
        out = {
            'l_w->v': torch.cat([edges.src['h_v'], edges.data['h_w->v']], dim=1)
        }
        return out

    def forward(self, g, node_feats, edge_feats):
        """Forward pass through DAMPN layer.

        Parameters
        ----------
        g : DGLGraph
            Batch of graphs to pass through the network.
        node_feats : tensor of shape (N, node_feat_size)
            Node features for each node, where N is the number of nodes
            in this batch of graphs.
        edge_feats : tensor of shape (M, edge_feat_size)
            Edge features for each edge, where M is the number of edges
            in this batch of graphs.

        """

        # we wil be making temporary attribute updates to the graphs
        g = g.local_var()
        
        logging.debug(
            f"{type(self).__module__+'.'+type(self).__name__}:Features enter with shapes: node ({node_feats.shape}), edge ({edge_feats.shape})."
        )
        
        # assign input data
        g.ndata['f_v'] = node_feats
        g.edata['f_w->v'] = edge_feats

        # embed nodes and edges
        g.ndata["h_v"] = self.embed_node(node_feats)
        g.apply_edges(self._apply_embed_edge)
        g.edata["h_w->v"] = self.embed_edge(g.edata["h_w->v"])
        
        logging.debug(
            f"{type(self).__module__+'.'+type(self).__name__}:Features embedded shape: node ({g.ndata['h_v'].shape}), edge ({g.edata['h_w->v'].shape})."
        )
        
        # compute raw messages
        g.edata["m_w->v"] = self.compute_raw_message(g.edata["h_w->v"])
        
        logging.debug(
            f"{type(self).__module__+'.'+type(self).__name__}:Edge messages shape ({g.edata['m_w->v'].shape})."
        )
        
        # compute logits
        g.apply_edges(self._apply_compute_logit)
        g.edata['l_w->v'] = self.compute_logit(g.edata['l_w->v'])
        
        logging.debug(
            f"{type(self).__module__+'.'+type(self).__name__}:Edge logits ({g.edata['l_w->v'].shape})."
        )
        
        # compute attention
        g.edata['alpha'] = edge_softmax(g, g.edata['l_w->v'])
        
        logging.debug(
            f"{type(self).__module__+'.'+type(self).__name__}:Edge attention ({g.edata['alpha'].shape})."
        )
        
        # weigh the mesages
        g.edata["m_w->v"] = g.edata['alpha'] * g.edata["m_w->v"]
        
        logging.debug(
            f"{type(self).__module__+'.'+type(self).__name__}:Edge attended messages ({g.edata['m_w->v'].shape})."
        )
        
        # compute context vector
        g.update_all(
            fn.copy_edge(
                "m_w->v",
                "m_w->v_"),
            fn.sum(
                "m_w->v_",
                'C_v'))
        g.ndata['C_v'] = F.elu(g.ndata['C_v'])
        
        logging.debug(
            f"{type(self).__module__+'.'+type(self).__name__}:Node context size ({g.ndata['C_v'].shape})."
        )
        
        # compute updated node state
        g.ndata["h'_v"] = F.relu(self.gru(g.ndata['C_v'], g.ndata["h_v"]))
        
        _ = g.ndata["h'_v"]
        logging.debug(
            f"{type(self).__module__+'.'+type(self).__name__}:Updated node state size ({_.shape})."
        )
        
        if self.update_edges:
            return g.ndata["h'_v"], g.edata["h_w->v"]
        else:
            return g.ndata["h'_v"], g.edata["f_w->v"]

        
class SystemReadoutLayer(nn.Module):
    """System Level Readout layer

    Update system node state by messages from graph nodes states.
    A Gated Recurrent Unit is applied to an attention mechanism to determine
    message weights. System level features can be included as system node state

    This layer is inspired by the readout convolutional layer of the AttentiveFP
    model:

    <Xiong, Z., Wang, D., Liu, X., Zhong, F., Wan, X., Li, X., … Zheng, M.
        (2020). Pushing the Boundaries of Molecular Representation for Drug
        Discovery with the Graph Attention Mechanism. J. Med. Chem, 63, 21.
        https://doi.org/10.1021/acs.jmedchem.9b00959>

    Here, it is only minorly modified from the implimentation on dgl_life
    https://lifesci.dgl.ai/_modules/dgllife/model/readout/attentivefp_readout.html#AttentiveFPReadout
    
    Such that attention and projection layers can be MLP instead of linear layers.
    This is a necessary inclusion for system level features to be introduced. With only
    one neuron, system level features act as another bias term, so regardless of the node state
    the activation is only shifted, and the softmax produces near identical message weights
    regardless of system features. The nonlinearity of MLP allows for system level features
    (identical for all nodes ina. system) to attentuate each node's attention.
    

    This layer operates as such on the system state f_g and nodes v:

    (1) Compute node logits:
        l_c^{v} = A(f^{v}, f_g^{v})
        A:= activated MLP with input and output shape (graph feat + node feat, 1)
    (2) Compute attention:
        alpha^{v} = softmax_{v\\in V}(l_c^{v})
    (3) Compute messages:
        m^{v} = M(f^{v})
        A:= activated MLP with input and output shape (node feat, graph feat)
    (4) Compute context:
        C = elu(\\Sigma_{v \\in V} \alpha^{v} * m^{v})
    (5) Update graph state:
        h_g = GRU(C, f_g)
    """

    def __init__(
        self,
        node_feature_size: int,
        graph_feature_size: int,
        logit_mlp_sizes: Tuple[int] = (10,),
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.compute_messages = nn.Sequential(
            nn.Linear(node_feature_size, graph_feature_size),
            nn.Dropout(dropout)
        )
        
        self.compute_logits = nn.Sequential()
        size = node_feature_size + graph_feature_size
        for i, size_ in enumerate(logit_mlp_sizes):
            self.compute_logits.add_module(f"linear_{i}", nn.Linear(size, size_))
            self.compute_logits.add_module(f"dropout_{i}", nn.Dropout(dropout))
            self.compute_logits.add_module(f"leakyrelu_{i}", nn.LeakyReLU())
            size = size_
        self.compute_logits.add_module(f"linear_out", nn.Linear(size, 1))
        self.compute_logits.add_module(f"leakyrelu_out", nn.LeakyReLU())
        
        
        self.gru = nn.GRUCell(graph_feature_size, graph_feature_size)
        return
    
    def forward(self, g, node_feats, graph_feats):
        g = g.local_var()
        
        logging.debug(
            f"{type(self).__module__+'.'+type(self).__name__}:Features enter with shapes: node ({node_feats.shape}), graph ({graph_feats.shape})."
        )
        
        # logits
        g.ndata['l_v'] = self.compute_logits(
             torch.cat([dgl.broadcast_nodes(g, F.relu(graph_feats)), node_feats], dim=1)
        )
        
        logging.debug(
            f"{type(self).__module__+'.'+type(self).__name__}:Node logits size ({g.ndata['l_v'].shape})."
        )
        
        # attention
        g.ndata['alpha'] = dgl.softmax_nodes(g, 'l_v')
        
        logging.debug(
            f"{type(self).__module__+'.'+type(self).__name__}:Node attention size ({g.ndata['alpha'].shape})."
        )
        
        # compute and weigh messages
        g.ndata['m'] = self.compute_messages(node_feats)
        
        logging.debug(
            f"{type(self).__module__+'.'+type(self).__name__}:Node message size ({g.ndata['m'].shape})."
        )
        
        # get context
        context = dgl.sum_nodes(g, 'm', 'alpha')
        
        logging.debug(
            f"{type(self).__module__+'.'+type(self).__name__}:Context ({context.shape})."
        )
        
        # update graph states
        graph_feats = self.gru(context, graph_feats)
        
        logging.debug(
            f"{type(self).__module__+'.'+type(self).__name__}:Updated graph feats size ({graph_feats.shape})."
        )
        return graph_feats
    
    def reset_parameters(self):
        """Re-initialize all learnable parameters."""
        self.gru.reset_parameters()
        self.compute_messages[0].reset_parameters()
        for l in self.compute_logits[::3]:
            l.reset_parameters()
        return