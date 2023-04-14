import torch.nn as nn
import torch
import numpy as np
from .util import init, get_clones

"""MLP modules.
   input: (6+2+5*(N-1)) * N   output: V(s)
"""

class C_MLPLayer(nn.Module):
    def __init__(self, input_dim, hidden_size, layer_N, use_orthogonal, use_ReLU):
        super(C_MLPLayer, self).__init__()
        self._layer_N = layer_N

        active_func = [nn.Tanh(), nn.ReLU()][use_ReLU]
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        gain = nn.init.calculate_gain(['tanh', 'relu'][use_ReLU])

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)

        self.phi_loc = nn.Sequential(
            init_(nn.Linear(6, 64)), active_func, nn.LayerNorm(64))  # 输入6， 输出64
        
        self.phi_oij = nn.Sequential(
            init_(nn.Linear(5, 64)), active_func, nn.LayerNorm(64))  # 输入5， 输出64
        
        self.fc1 = nn.Sequential(
            init_(nn.Linear(128, 64)), active_func, nn.LayerNorm(64))
        
        self.fc2 = nn.Sequential(
            init_(nn.Linear(64, 32)), active_func, nn.LayerNorm(32))

    def forward(self, x):
        len_x = x.size(1)  # col. the dim of x 
        N = (np.sqrt(9+20*len_x)-3)/10  # num of agents
        assert np.mod(N, 1.0) == 0.0, "wrong length in Critic Network"
        N = int(N)
        dim_obs = int(len_x/N)  # dim of x for each agent
        o_loc_stacker = torch.zeros(64).cuda()  # divide N
        # o_ext_stacker = torch.zeros(2).cuda()  # divide N
        o_ij_stacker = torch.zeros(64).cuda() # divide N*(N-1)
        for i in range(N):
            xi = x[:, dim_obs*i: dim_obs*i+dim_obs]  # obs x of agent i
            o_loc_stacker = o_loc_stacker + self.phi_loc(xi[:, 0:6]).cuda()  # o_loc
            # o_ext_stacker = o_ext_stacker + xi[:, 6:8]
            xi_oij = xi[:, 8:]
            for k in range(N-1):
                xi_oij_k = xi_oij[:, 5*k:5*k+5]
                o_ij_stacker = o_ij_stacker + self.phi_oij(xi_oij_k).cuda()
        o_loc_stacker = o_loc_stacker/N
        # o_ext_stacker = o_ext_stacker/N
        o_ij_stacker = o_ij_stacker/(N*(N-1))
        x = torch.cat((o_loc_stacker, o_ij_stacker), dim=1)  # 64+64 s
        # print('critic x shape is:{} '.format(x.shape))
        x = self.fc1(x)
        x = self.fc2(x)
        return x


class C_MLPBase(nn.Module):
    def __init__(self, args, obs_shape, cat_self=True, attn_internal=False):
        super(C_MLPBase, self).__init__()

        self._use_feature_normalization = args.use_feature_normalization
        self._use_orthogonal = args.use_orthogonal
        self._use_ReLU = args.use_ReLU
        self._stacked_frames = args.stacked_frames
        self._layer_N = args.layer_N
        self.hidden_size = args.hidden_size

        obs_dim = obs_shape[0]

        if self._use_feature_normalization:
            self.feature_norm = nn.LayerNorm(obs_dim)

        self.mlp = C_MLPLayer(obs_dim, self.hidden_size,
                              self._layer_N, self._use_orthogonal, self._use_ReLU)

    def forward(self, x):
        if self._use_feature_normalization:
            x = self.feature_norm(x)

        x = self.mlp(x)

        return x