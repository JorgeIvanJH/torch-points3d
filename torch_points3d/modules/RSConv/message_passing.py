import torch
from torch.nn import ReLU
from torch_geometric.nn import MessagePassing


from torch_points3d.core.base_conv.message_passing import *
from torch_points3d.core.spatial_ops import *


class Convolution(MessagePassing):
    r"""The Relation Shape Convolution layer from "Relation-Shape Convolutional Neural Network for Point Cloud Analysis"
    https://arxiv.org/pdf/1904.07601

    local_nn - an MLP which is applied to the relation vector h_ij between points i and j to determine
    the weights applied to each element of the feature for x_j

    global_nn - an optional MPL for channel-raising following the convolution

    """

    def __init__(self, local_nn, activation=ReLU(), global_nn=None, aggr="max", **kwargs):
        super(Convolution, self).__init__(aggr=aggr)
        self.local_nn = MLP(local_nn)
        self.activation = activation
        self.global_nn = MLP(global_nn) if global_nn is not None else None

    def forward(self, x, pos, edge_index):
        # print("[RSConv] pos[0].shape: ", pos[0].shape, "pos[1].shape: ", pos[1].shape)
        # print("[RSConv] x shape:", x.shape)           # Could be [N, C] or None
        # print("[RSConv] edge_index shape:", edge_index.shape)  # Expected: [2, E]
        # print("[RSConv] edge_index max:", edge_index.max().item())
        # print("[RSConv] edge_index min:", edge_index.min().item())

        pos_i, pos_j = pos  # Unpack the tuple
        # print("[RSConv] pos_i shape:", pos_i.shape)  # Expected: [N, 3]
        # print("[RSConv] pos_j shape:", pos_j.shape)  # Expected: [N, 3]
        return self.propagate(edge_index, x=x, pos=(pos_j, pos_i), size=(pos_j.size(0), pos_i.size(0)))


    def message(self, pos_i, pos_j, x_j):
        # print("[message] pos_i shape:", pos_i.shape)   # Should be [E, 3]
        # print("[message] pos_j shape:", pos_j.shape)   # Should be [E, 3]
        # print("[message] x_j shape:", x_j.shape)       # Should be [E, C]
        if x_j is None:
            x_j = pos_j

        vij = pos_i - pos_j
        # print("[message] vij shape:", vij.shape)       # Should be [E, 3]
        dij = torch.norm(vij, dim=1).unsqueeze(1)
        # print("[message] dij shape:", dij.shape)       # Should be [E, 1]

        hij = torch.cat(
            [
                dij,
                vij,
                pos_i,
                pos_j,
            ],
            dim=1,
        )
        # print("[message] hij shape:", hij.shape)       # Should be [E, 9] if pos_i and pos_j are [E, 3]

        M_hij = self.local_nn(hij)
        # print("[message] M_hij shape:", M_hij.shape)

        msg = M_hij * x_j
        # print("[message] msg shape:", msg.shape)       # Should be [E, C]

        return msg

    def update(self, aggr_out):
        x = self.activation(aggr_out)
        if self.global_nn is not None:
            x = self.global_nn(x)
        return x


class RSConvDown(BaseConvolutionDown):
    def __init__(self, ratio=None, radius=None, local_nn=None, down_conv_nn=None, *args, **kwargs):
        super(RSConvDown, self).__init__(FPSSampler(ratio), RadiusNeighbourFinder(radius), *args, **kwargs)

        self._conv = Convolution(local_nn=local_nn, global_nn=down_conv_nn)

    def conv(self, x, pos, edge_index, batch):
        # print("[RSConvDown] x:", x.shape)
        # print("[RSConvDown] pos[0].shape: ", pos[0].shape, "pos[1].shape: ", pos[1].shape)
        # print("[RSConvDown] edge_index:", edge_index.shape)
        return self._conv(x, pos, edge_index)
