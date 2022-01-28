import pytest
import dampn.layers.dgl_modules
import torch
import torch.nn as nn
import dgl
import copy


@pytest.fixture
def DAMPLayer():
    layer = dampn.layers.dgl_modules.DAMPLayer(
        node_feature_size=5,
        edge_feature_size=2,
        node_hidden_size=8,
        edge_hidden_size=3,
        context_size=5,
        system_features_size=None,
        update_edges=False,
        dropout=0.0
    )
    return layer


@pytest.fixture
def BATCH():
    graph1 = dgl.graph(([0, 1, 2, 1], [1, 2, 0, 0]), num_nodes=3)
    graph1.edata['e'] = torch.rand((4, 2))
    graph1.ndata['n'] = torch.rand((3, 5))
    graph2 = dgl.graph(([0], [1]), num_nodes=2)
    graph2.edata['e'] = torch.rand((1, 2))
    graph2.ndata['n'] = torch.rand((2, 5))
    batch = dgl.batch([graph1, graph2])
    return batch


class TestDAMPLayer:
    """Tests expected sizes, parameter updates of the layer."""

    def test_modules(self, DAMPLayer):
        """Test that sub modules are created with the correct sizes."""
        assert DAMPLayer.embed_node(
            torch.ones((1, 5))
        ).shape[1] == 8, "embed node incorrect shape"

        assert DAMPLayer.embed_edge(
            torch.ones((1, 5 + 2))
        ).shape[1] == 3, "embed edge incorrect shape"

        assert DAMPLayer.compute_logit(
            torch.ones((1, 8 + 3))
        ).shape[1] == 1, "compute logit incorrect shape"

        assert DAMPLayer.compute_raw_message(
            torch.ones((1, 3))
        ).shape[1] == 5, "compute message incorrect shape"

        assert DAMPLayer.gru(
            torch.ones((1, 5)), torch.ones((1, 8))
        ).shape[1] == 8, "gru incorrect shape"
        return

    def test_forward(self, DAMPLayer, BATCH):
        """A full forward pass produces the correct shape."""
        output = DAMPLayer(
            BATCH, BATCH.ndata['n'], BATCH.edata['e']
        )
        assert output[0].shape == (5, 8), "node hidden state incorrect shape"
        assert output[1].shape == (
            5, 2), "edge hidden state incorrect shape, should not have been updated"

        DAMPLayer.update_edges = True
        output = DAMPLayer(
            BATCH, BATCH.ndata['n'], BATCH.edata['e']
        )
        assert output[0].shape == (5, 8), "node hidden state incorrect shape"
        assert output[1].shape == (
            5, 3), "edge hidden state incorrect shape, should have been updated"
        return

    def test_backpropegate(self, DAMPLayer, BATCH):
        """Test that we have a full parameter update"""
        initial_state = copy.deepcopy(DAMPLayer.state_dict())
        print(initial_state)

        opt = torch.optim.SGD(DAMPLayer.parameters(), lr=.01)
        loss_fn = nn.MSELoss()

        h_n, h_e = DAMPLayer(BATCH, BATCH.ndata['n'], BATCH.edata['e'])
        out = torch.cat([torch.flatten(h_n), torch.flatten(h_e)])
        y = torch.rand(out.shape)

        loss = loss_fn(out, y)
        opt.zero_grad()
        loss.backward()
        opt.step()

        final_state = DAMPLayer.state_dict()
        for key, params in initial_state.items():
            assert not torch.equal(
                params, final_state[key]), f"{key} params not updated"
