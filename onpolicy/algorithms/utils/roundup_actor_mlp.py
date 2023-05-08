import torch.nn as nn
import torch
import numpy as np
from .util import init, get_clones

"""MLP modules.
   input: 6+2+5*(N-1) = 28  output: 1*32
"""

class A_MLPLayer(nn.Module):
    def __init__(self, input_dim, hidden_size, layer_N, use_orthogonal, use_ReLU):
        super(A_MLPLayer, self).__init__()
        self._layer_N = layer_N

        active_func = [nn.Tanh(), nn.ReLU()][use_ReLU]
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        gain = nn.init.calculate_gain(['tanh', 'relu'][use_ReLU])

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)
        
        self.fc1 = nn.Sequential(
            init_(nn.Linear(72, 64)), active_func, nn.LayerNorm(64))
        
        self.fc2 = nn.Sequential(
            init_(nn.Linear(64, 32)), active_func, nn.LayerNorm(32))

    def forward(self, x):        
        # print('actor x shape is:{} '.format(x.shape))
        x = self.fc1(x)
        x = self.fc2(x)
        
        return x


class A_MLPBase(nn.Module):
    def __init__(self, args, obs_dim, cat_self=True, attn_internal=False):
        super(A_MLPBase, self).__init__()

        self._use_feature_normalization = args.use_feature_normalization
        self._use_orthogonal = args.use_orthogonal
        self._use_ReLU = args.use_ReLU
        self._stacked_frames = args.stacked_frames
        self._layer_N = args.layer_N
        self.hidden_size = args.hidden_size


        if self._use_feature_normalization:
            self.feature_norm = nn.LayerNorm(obs_dim)

        self.mlp = A_MLPLayer(obs_dim, self.hidden_size,
                              self._layer_N, self._use_orthogonal, self._use_ReLU)

    def forward(self, x):
        if self._use_feature_normalization:
            x = self.feature_norm(x)

        x = self.mlp(x)

        return x