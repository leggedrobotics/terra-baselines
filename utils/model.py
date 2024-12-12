import torch
import torch.nn as nn
import torch.nn.functional as F

def normalize(x, x_min, x_max):
    return 2.0 * (x - x_min) / (x_max - x_min) - 1.0


class MLP(nn.Module):
    """
    MLP without activation on the last layer. Supports optional layer normalization.
    """
    def __init__(self, hidden_dim_layers, use_layer_norm=False, last_layer_init_scaling=1.0):
        super().__init__()
        self.use_layer_norm = use_layer_norm

        layers = []
        # Construct linear layers
        for i in range(len(hidden_dim_layers)-1):
            in_dim = hidden_dim_layers[i]
            out_dim = hidden_dim_layers[i+1]
            linear = nn.Linear(in_dim, out_dim)
            nn.init.orthogonal_(linear.weight, gain=1.0)
            nn.init.constant_(linear.bias, 0.0)
            layers.append(linear)

            # Add layer norm if requested (for all but the last layer)
            if self.use_layer_norm and i < len(hidden_dim_layers)-2:
                layers.append(nn.LayerNorm(out_dim))
        self.layers = nn.ModuleList(layers)

        # Apply last_layer_init_scaling to the last layer's weights if needed
        if last_layer_init_scaling != 1.0:
            with torch.no_grad():
                self.layers[-1].weight.mul_(last_layer_init_scaling)
                self.layers[-1].bias.mul_(last_layer_init_scaling)

    def forward(self, x):
        # For all but last linear, apply relu
        for i, layer in enumerate(self.layers):
            # If this is a LayerNorm or Linear
            if isinstance(layer, nn.Linear):
                # If not last linear layer, apply relu after
                if i < len(self.layers)-1:
                    x = F.relu(layer(x))
                else:
                    # last layer, no activation
                    x = layer(x)
            else:
                # LayerNorm just apply it
                x = layer(x)
        return x


class AgentStateNet(nn.Module):
    def __init__(self, num_embeddings, loaded_max, mlp_use_layernorm=False,
                 num_embedding_features=8,
                 hidden_dim_layers_mlp_one_hot=(16,32),
                 hidden_dim_layers_mlp_continuous=(16,32)):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.loaded_max = loaded_max
        self.embedding = nn.Embedding(num_embeddings, num_embedding_features)

        # one-hot features = 5 categorical features embedded -> shape [B,5*num_embedding_features]
        self.mlp_one_hot = MLP([5*num_embedding_features] + list(hidden_dim_layers_mlp_one_hot),
                               use_layer_norm=mlp_use_layernorm)
        self.mlp_continuous = MLP([1] + list(hidden_dim_layers_mlp_continuous),
                                  use_layer_norm=mlp_use_layernorm)

    def forward(self, agent_state):
        # agent_state: [B,6], last is loaded, first 5 are categorical
        x_one_hot = agent_state[..., :-1].long()  # [B,5]
        x_loaded = agent_state[..., [-1]].float()  # [B,1]

        # Embed the 5 categorical features
        x_emb = self.embedding(x_one_hot) # [B,5,num_embedding_features]
        x_emb = x_emb.view(x_emb.size(0), -1) # [B,5*num_embedding_features]
        x_one_hot_features = self.mlp_one_hot(x_emb)

        # Normalize loaded
        x_loaded_norm = normalize(x_loaded, 0, self.loaded_max)
        x_continuous = self.mlp_continuous(x_loaded_norm)

        return torch.cat([x_one_hot_features, x_continuous], dim=-1)


class LocalMapNet(nn.Module):
    def __init__(self, map_min_max, mlp_use_layernorm=False, hidden_dim_layers_mlp=(256,32)):
        super().__init__()
        self.map_min_max = map_min_max
        self.mlp = MLP([6] + list(hidden_dim_layers_mlp),
                       use_layer_norm=mlp_use_layernorm)

    def forward(self, local_map_action_neg, local_map_action_pos,
                local_map_target_neg, local_map_target_pos,
                local_map_dumpability, local_map_obstacles):
        # Normalize inputs
        neg_an = normalize(local_map_action_neg, self.map_min_max[0], self.map_min_max[1])
        pos_an = normalize(local_map_action_pos, self.map_min_max[0], self.map_min_max[1])
        neg_tn = normalize(local_map_target_neg, self.map_min_max[0], self.map_min_max[1])
        pos_tn = normalize(local_map_target_pos, self.map_min_max[0], self.map_min_max[1])

        # Each input is [B, angles, layers], let's flatten them
        # Concatenate along last dimension first
        # shape after cat: [B, angles, layers*6] but we only have one layer typically
        # Actually we have 6 features each is [B, angles, layers], so cat along last dim = [B, angles, layers*6].
        # But original code flattens everything:
        x = torch.cat([neg_an.unsqueeze(-1),
                       pos_an.unsqueeze(-1),
                       neg_tn.unsqueeze(-1),
                       pos_tn.unsqueeze(-1),
                       local_map_dumpability.unsqueeze(-1),
                       local_map_obstacles.unsqueeze(-1)], dim=-1)
        # flatten [B, angles, layers*6] to [B, angles*layers*6]
        x = x.view(x.size(0), -1)
        x = self.mlp(x)
        return x


class AtariCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # In original code, input to CNN is [B,W,H,5], we will permute to [B,5,W,H].
        self.conv1 = nn.Conv2d(5, 8, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(8,16,kernel_size=4,stride=2)
        self.conv3 = nn.Conv2d(16,16,kernel_size=3,stride=1)
        # We'll determine the linear input size at runtime

        self.fc1 = None
        self.fc2 = None

    def _get_flattened_size(self, x):
        # Helper to infer fc size after conv
        with torch.no_grad():
            x = self.conv1(x)
            x = F.relu(x)
            x = self.conv2(x)
            x = F.relu(x)
            x = self.conv3(x)
            x = F.relu(x)
            return x.view(x.size(0), -1).size(1)

    def forward(self, x):
        # x: [B,W,H,5]
        x = x.permute(0,3,1,2)  # -> [B,5,W,H]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)

        # Lazy init fc layers
        if self.fc1 is None:
            in_feats = x.size(1)
            self.fc1 = nn.Linear(in_feats,64)
            self.fc2 = nn.Linear(64,32)
            nn.init.orthogonal_(self.fc1.weight, gain=1.0)
            nn.init.constant_(self.fc1.bias, 0.0)
            nn.init.orthogonal_(self.fc2.weight, gain=1.0)
            nn.init.constant_(self.fc2.bias, 0.0)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class MapsNet(nn.Module):
    def __init__(self, map_min_max):
        super().__init__()
        self.map_min_max = map_min_max
        self.cnn = AtariCNN()

    def forward(self, action_map, target_map, traversability_map, do_prediction, dig_map, dumpability_mask):
        # [B,W,H] each
        # Concatenate: traversability_map, target_map, do_prediction, dig_map, dumpability_mask
        # result: [B,W,H,5]
        x = torch.stack([traversability_map, target_map, do_prediction, dig_map, dumpability_mask], dim=-1)
        x = self.cnn(x)
        return x


class SimplifiedCoupledCategoricalNet(nn.Module):
    def __init__(self, mask_out_arm_extension, num_embeddings_agent, map_min_max,
                 local_map_min_max, loaded_max, action_type,
                 hidden_dim_pi=(128,32),
                 hidden_dim_v=(128,32,1),
                 mlp_use_layernorm=False):
        super().__init__()
        self.mask_out_arm_extension = mask_out_arm_extension
        self.num_embeddings_agent = num_embeddings_agent
        self.map_min_max = map_min_max
        self.local_map_min_max = local_map_min_max
        self.loaded_max = loaded_max
        self.action_type = action_type
        self.mlp_use_layernorm = mlp_use_layernorm

        num_actions = self.action_type.get_num_actions()

        self.mlp_v = MLP(hidden_dim_v, use_layer_norm=mlp_use_layernorm, last_layer_init_scaling=0.01)
        self.mlp_pi = MLP(list(hidden_dim_pi)+[num_actions], use_layer_norm=mlp_use_layernorm, last_layer_init_scaling=0.01)

        self.local_map_net = LocalMapNet(local_map_min_max, mlp_use_layernorm)
        self.agent_state_net = AgentStateNet(num_embeddings_agent, loaded_max, mlp_use_layernorm)
        self.maps_net = MapsNet(map_min_max)

    def forward(self, obs):
        # obs is a list of 13 elements as defined above
        # Extract them for clarity
        agent_state = obs[0]
        local_map_action_neg = obs[1]
        local_map_action_pos = obs[2]
        local_map_target_neg = obs[3]
        local_map_target_pos = obs[4]
        local_map_dumpability = obs[5]
        local_map_obstacles = obs[6]
        action_map = obs[7]
        target_map = obs[8]
        traversability_map = obs[9]
        do_prediction = obs[10]
        dig_map = obs[11]
        dumpability_mask = obs[12]

        x_agent_state = self.agent_state_net(agent_state)

        # maps_net input: (action_map, target_map, traversability_map, do_prediction, dig_map, dumpability_mask)
        x_maps = self.maps_net(action_map, target_map, traversability_map, do_prediction, dig_map, dumpability_mask)

        # local_map_net input:
        x_local_map = self.local_map_net(local_map_action_neg, local_map_action_pos,
                                         local_map_target_neg, local_map_target_pos,
                                         local_map_dumpability, local_map_obstacles)

        x = torch.cat([x_agent_state, x_maps, x_local_map], dim=-1)
        x = F.relu(x)

        v = self.mlp_v(x)  # [B,1]
        xpi = self.mlp_pi(x) # [B,num_actions]

        if self.mask_out_arm_extension:
            # mask last two arm extension actions
            xpi[..., -2] = -1e8
            xpi[..., -3] = -1e8

        return v, xpi
