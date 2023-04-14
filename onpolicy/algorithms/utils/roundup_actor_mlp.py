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

        # input dim is 6+2+5*(N-1), N is num of agents
        self.phi = nn.Sequential(
            init_(nn.Linear(5, 64)), active_func, nn.LayerNorm(64))  # 输入5， 输出64
        
        self.fc1 = nn.Sequential(
            init_(nn.Linear(72, 64)), active_func, nn.LayerNorm(64))
        
        self.fc2 = nn.Sequential(
            init_(nn.Linear(64, 32)), active_func, nn.LayerNorm(32))

    def forward(self, x):
        # print(x.shape) #[320, 28]
        x0 = x[:,0:8]  # o_loc+o_ext
        x1 = x[:,8:]  # o_ij
        len_x1 = x1.size(1)
        # print(len_x1)  # 20
        assert np.mod(len_x1, 5) == 0, "wrong length in Actor Network"
        N = int(len_x1/5)
        phi_stacker = torch.zeros(64).cuda()  # 1*64, you cannot say torch.zeros(1, 64)
        # print(phi_stacker)
        for i in range(N):
            x_ = x1[:, 5*i: 5*i+5]
            # print(self.phi(x_).shape)
            phi_stacker = phi_stacker + self.phi(x_).cuda()
        phi_stacker = phi_stacker/N
        x = torch.cat((x0, phi_stacker), dim=1)  # 8+64=72 dim
        # print('actor x shape is:{} '.format(x.shape))
        x = self.fc1(x)
        x = self.fc2(x)
        
        return x


class A_MLPBase(nn.Module):
    def __init__(self, args, obs_shape, cat_self=True, attn_internal=False):
        super(A_MLPBase, self).__init__()

        self._use_feature_normalization = args.use_feature_normalization
        self._use_orthogonal = args.use_orthogonal
        self._use_ReLU = args.use_ReLU
        self._stacked_frames = args.stacked_frames
        self._layer_N = args.layer_N
        self.hidden_size = args.hidden_size

        obs_dim = obs_shape[0]

        if self._use_feature_normalization:
            self.feature_norm = nn.LayerNorm(obs_dim)

        self.mlp = A_MLPLayer(obs_dim, self.hidden_size,
                              self._layer_N, self._use_orthogonal, self._use_ReLU)

    def forward(self, x):
        if self._use_feature_normalization:
            x = self.feature_norm(x)

        x = self.mlp(x)

        return x