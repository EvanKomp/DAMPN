class DAMPNModule(nn.Module):
    
    def __init__(
        self,
        node_feature_size: int,
        edge_feature_size: int,
        output_size: int,
        n_dampn_layers: int = 3,
        node_hidden_sizes: Union[int, List[int]] = 28,
        edge_hidden_sizes: Union[int, List[int]] = 28,
        context_sizes: Union[int, List[int]] = 28,
        classification: bool = False,
        system_features_size: int = None,
        update_edges: bool = False,
        dropouts: Union[float, List[float]] = 0.0,
        **kwargs
    ):
        # handle hyps that could be multiples
        if hasattr(node_hidden_sizes, __len__):
            assert len(node_hidden_sizes) == n_dampn_layers, "Node hidden sizes specified does not match number of layers"
        else:
            node_hidden_sizes = [node_hidden_sizes]*n_dampn_layers
            
        if hasattr(edge_hidden_sizes, __len__):
            assert len(edge_hidden_sizes) == n_dampn_layers, "Edge hidden sizes specified does not match number of layers"
        else:
            edge_hidden_sizes = [edge_hidden_sizes]*n_dampn_layers
        
        if hasattr(context_sizes, __len__):
            assert len(context_sizes) == n_dampn_layers, "Context sizes specified does not match number of layers"
        else:
            context_sizes = [context_sizes]*n_dampn_layers
        
        if hasattr(dropouts, __len__):
            assert len(dropouts) == n_dampn_layers, "Dropout rates specified does not match number of layers"
        else:
            dropouts = [dropouts]*n_dampn_layers
            
        self.dampn_layers = []
        for i in range(n_dampn_layers):