from typing import Union, List, Tuple

import torch
from torch import nn

import dgl
import dampn.model.dgl_layers

import logging
logger = logging.getLogger(__name__)

class DAMPNModule(nn.Module):
    
    def __init__(
        self,
        node_feature_size: int,
        edge_feature_size: int,
        n_dampn_layers: int = 3,
        node_hidden_sizes: Union[int, List[int]] = 28,
        edge_hidden_sizes: Union[int, List[int]] = 28,
        context_sizes: Union[int, List[int]] = 28,
        classification: bool = False,
        readout_logit_mlp_sizes: Tuple[int] = (10,),
        n_readout_layers: int = 3,
        system_features_size: int = None,
        update_edges: bool = False,
        dropouts: Union[float, List[float]] = 0.0,
        readout_dropout: float = 0.0,
        **kwargs
    ):
        super().__init__()
        # handle hyps that could be multiples
        if hasattr(node_hidden_sizes, "__len__"):
            assert len(node_hidden_sizes) == n_dampn_layers, "Node hidden sizes specified does not match number of layers"
        else:
            node_hidden_sizes = [node_hidden_sizes]*n_dampn_layers
            
        if hasattr(edge_hidden_sizes, "__len__"):
            assert len(edge_hidden_sizes) == n_dampn_layers, "Edge hidden sizes specified does not match number of layers"
        else:
            edge_hidden_sizes = [edge_hidden_sizes]*n_dampn_layers
        
        if hasattr(context_sizes, "__len__"):
            assert len(context_sizes) == n_dampn_layers, "Context sizes specified does not match number of layers"
        else:
            context_sizes = [context_sizes]*n_dampn_layers
        
        if hasattr(dropouts, "__len__"):
            assert len(dropouts) == n_dampn_layers, "Dropout rates specified does not match number of layers"
        else:
            dropouts = [dropouts]*n_dampn_layers
            
        self.dampn_layers = nn.ModuleList()
        node_size_ = node_feature_size
        edge_size_ = edge_feature_size
        for i in range(n_dampn_layers):
            self.dampn_layers.append(
                dampn.model.dgl_layers.DAMPLayer(
                    node_feature_size=node_size_,
                    edge_feature_size=edge_size_,
                    node_hidden_size=node_hidden_sizes[i],
                    edge_hidden_size=edge_hidden_sizes[i],
                    context_size=context_sizes[i],
                    update_edges=update_edges,
                    dropout=dropouts[i]
                )
            )
            node_size_ = node_hidden_sizes[i]
            edge_size_ = edge_hidden_sizes[i]
        
        self.readout_layers = nn.ModuleList()
        graph_feature_size = 0 + node_size_
        if system_features_size is not None:
            graph_feature_size += system_features_size
        for i in range(n_readout_layers):
            self.readout_layers.append(
                dampn.model.dgl_layers.SystemReadoutLayer(
                    node_feature_size=node_size_,
                    graph_feature_size=graph_feature_size,
                    logit_mlp_sizes=readout_logit_mlp_sizes,
                    dropout=readout_dropout
                )
            )
        
        self.predictor = nn.Sequential(nn.Linear(graph_feature_size, 1))
        if classification:
            self.predictor.add_module('sigmoid', nn.Sigmoid())
        return
    
    def reset_parameters(self):
        for l in self.dampn_layers:
            l.reset_parameters()
        for l in self.readout_layers:
            l.reset_parameters()
        self.predictor[0].reset_parameters()
        return
    
    def forward(self, g, node_feats, edge_feats, system_feats=None):
        
        logging.debug(
            f"{type(self).__module__+'.'+type(self).__name__}:Starting features enter with shapes: node ({node_feats.shape}), edge ({edge_feats.shape})."
        )
        if system_feats is not None:
            logging.debug(
            f"{type(self).__module__+'.'+type(self).__name__}:Starting system features enter with shapes: ({system_feats.shape})."
        )
        
        # do distance attentive message passing
        for l in self.dampn_layers:
            node_feats, edge_feats = l(g, node_feats, edge_feats)
        
        logging.debug(
            f"{type(self).__module__+'.'+type(self).__name__}:States finish message passing with shapes: node ({node_feats.shape}), edge ({edge_feats.shape})."
        )
        
        # readout nodes
        # first create graph state
        g = g.local_var()
        g.ndata['h_v'] = node_feats
        graph_feats = dgl.sum_nodes(g, 'h_v')
        if system_feats is not None:
            graph_feats = torch.cat([graph_feats, system_feats], axis=1)
            
        logging.debug(
            f"{type(self).__module__+'.'+type(self).__name__}:Initial graph feats enter readout with shape: ({graph_feats.shape})."
        )
        
        for l in self.readout_layers:
            graph_feats = l(g, node_feats, graph_feats)
        
        logging.debug(
            f"{type(self).__module__+'.'+type(self).__name__}:Graph feats leave readout with shape: ({graph_feats.shape})."
        )
        
        out = self.predictor(graph_feats)
        
        return out
        