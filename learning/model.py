"""Neural network architectures for DDPG agent.

Defines noisy linear layers and actor-critic networks with optional
distributional learning and exploration noise.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from learning.hp import INIT_SIGMA, FACTORIZED, LINEAR, N_ATOMS, DISTRIBUTIONAL


class NoisyLinear(nn.Module):
    """Noisy linear layer for parameter space exploration (Fortunato et al., 2017)."""
    def __init__(self, in_features, out_features, bias=True, initial_sigma=INIT_SIGMA, factorized=FACTORIZED):
        """Initialize noisy linear layer.
        
        Args:
            in_features: Input dimension
            out_features: Output dimension
            bias: Whether to include bias term
            initial_sigma: Initial noise parameter scale
            factorized: Use factorized Gaussian noise (more efficient)
        """
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.initial_sigma = initial_sigma
        self.factorized = factorized
        
        # Learnable parameters
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.noisy_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
            self.noisy_bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
            self.register_parameter('noisy_bias', None)
        self.reset_parameters()

        # Toggle for enabling/disabling noise
        self.noise = True  

    def reset_parameters(self):
        """Initialize weight parameters using appropriate schemes."""
        sqrt_input_size = math.sqrt(self.weight.size(1))
        bound = 1 / sqrt_input_size
        
        if self.factorized:
            # Factorized Gaussian initialization
            nn.init.uniform_(self.weight, -bound, bound)
            nn.init.constant_(self.noisy_weight, self.initial_sigma / sqrt_input_size)
            if self.bias is not None:
                nn.init.uniform_(self.bias, -bound, bound)
                nn.init.constant_(self.noisy_bias, self.initial_sigma / sqrt_input_size)
        else:
            # Independent Gaussian initialization
            nn.init.uniform_(self.weight, -bound, bound)
            nn.init.uniform_(self.noisy_weight, -bound, bound)
            if self.bias is not None:
                nn.init.uniform_(self.bias, -bound, bound)
                nn.init.uniform_(self.noisy_bias, -bound, bound)

    def forward(self, input):
        if self.noise:
            if self.factorized:
                input_noise  = torch.randn(1, self.noisy_weight.size(1), dtype=self.noisy_weight.dtype).to(self.noisy_weight.device)
                input_noise  = input_noise.sign().mul(input_noise.abs().sqrt())
                output_noise = torch.randn(self.noisy_weight.size(0), dtype=self.noisy_weight.dtype).to(self.noisy_weight.device)
                output_noise = output_noise.sign().mul(output_noise.abs().sqrt())
                weight_noise = input_noise.mul(output_noise.unsqueeze(1))
                bias_noise = output_noise
            else:
                weight_noise = torch.randn(self.noisy_weight.size(), dtype=self.noisy_weight.dtype).to(self.noisy_weight.device)
                bias_noise = None if self.bias is None else torch.randn(self.noisy_bias.size(), dtype=self.noisy_bias.dtype).to(self.noisy_bias.device)

            if self.bias is None:
                return F.linear(
                               input,
                               self.weight.add(self.noisy_weight.mul(weight_noise)),
                               None
                           )
            else:
                return F.linear(
                               input,
                               self.weight.add(self.noisy_weight.mul(weight_noise)),
                               self.bias.add(self.noisy_bias.mul(bias_noise))
                           )

        return F.linear(input, self.weight, self.bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}, initial_sigma={}, factorized={}'.format(
            self.in_features, self.out_features, self.bias is not None, self.initial_sigma, self.factorized
        )


class Actor(nn.Module):
    """Policy network that maps states to continuous actions.
    
    Uses batch normalization and optional noisy layers for exploration.
    """

    def __init__(self, state_size, action_size, linear_type=LINEAR, initial_sigma=INIT_SIGMA, factorized=FACTORIZED):
        """Initialize actor network.
        
        Args:
            state_size: Dimensionality of state observations
            action_size: Dimensionality of action space
            linear_type: Layer type - 'linear' or 'noisy'
            initial_sigma: Noise parameter initialization scale
            factorized: Whether to use factorized Gaussian noise
        """
        super(Actor, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.linear_type = linear_type.lower()
        self.initial_sigma = initial_sigma
        self.factorized = bool(factorized)

        # Layer factory for conditional noisy layer usage
        def noisy_layer(in_features, out_features):
            return NoisyLinear(in_features, out_features, True, initial_sigma, factorized)
        layer_constructor = {'linear': nn.Linear, 'noisy': noisy_layer}[self.linear_type]

        # Network architecture: state -> 256 -> 128 -> 64 -> action
        self.bn0 = nn.BatchNorm1d(state_size)
        self.fc0 = nn.Linear(state_size, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc1 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc2 = layer_constructor(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.fc3 = layer_constructor(64, action_size)

        self.hidden_activation = nn.ReLU()
        self.output_activation = nn.Tanh()

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize network weights using He initialization for hidden layers."""
        nn.init.kaiming_normal_(self.fc0.weight.data, a=0.01, mode='fan_in')
        nn.init.kaiming_normal_(self.fc1.weight.data, a=0.01, mode='fan_in')
        nn.init.kaiming_normal_(self.fc2.weight.data, a=0.01, mode='fan_in')
        # Small uniform initialization for output layer (common in policy networks)
        nn.init.uniform_(self.fc3.weight.data, -3e-3, 3e-3)

    def forward(self, state):
        x = state
        x = self.bn0(x)
        x = self.hidden_activation(self.bn1(self.fc0(x)))
        x = self.hidden_activation(self.bn2(self.fc1(x)))
        x = self.hidden_activation(self.bn3(self.fc2(x)))
        x = self.output_activation(self.fc3(x))

        return x

    def noise(self, enable):
        """Enable or disable parameter noise in noisy layers.
        
        Args:
            enable: Boolean flag to toggle noise
        """
        enable = bool(enable)
        for module in self.children():
            if isinstance(module, NoisyLinear):
                module.noise = enable


class Critic(nn.Module):
    """Value network that estimates Q-values for state-action pairs.
    
    Supports optional distributional learning (C51) for modeling value distributions.
    """

    def __init__(self, state_size, action_size, linear_type=LINEAR, initial_sigma=INIT_SIGMA, 
                 factorized=FACTORIZED, n_atoms=N_ATOMS, distributional=DISTRIBUTIONAL):
        """Initialize critic network.
        
        Args:
            state_size: Dimensionality of state observations
            action_size: Dimensionality of action space
            linear_type: Layer type - 'linear' or 'noisy'
            initial_sigma: Noise parameter initialization scale
            factorized: Whether to use factorized Gaussian noise
            n_atoms: Number of atoms for distributional learning
            distributional: Enable distributional value estimation
        """
        super(Critic, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.linear_type = linear_type.lower()
        self.initial_sigma = initial_sigma
        self.factorized = bool(factorized)
        self.n_atoms = n_atoms
        self.distributional = bool(distributional)

        # Layer factory for conditional noisy layer usage
        def noisy_layer(in_features, out_features):
            return NoisyLinear(in_features, out_features, True, initial_sigma, factorized)
        layer_constructor = {'linear': nn.Linear, 'noisy': noisy_layer}[self.linear_type]

        # Network architecture: [state -> 256] + action -> 128 -> 64 -> Q-value(s)
        self.bn  = nn.BatchNorm1d(state_size, momentum=0.02)
        self.fcs0 = nn.Linear(state_size, 256)
        self.fc1  = nn.Linear(256 + action_size, 128)  # Concatenate processed state with action
        self.fc2  = layer_constructor(128, 64)
        self.fc3  = layer_constructor(64, n_atoms if distributional else 1)

        self.hidden_activation = nn.ReLU()

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize network weights using He initialization for hidden layers."""
        nn.init.kaiming_normal_(self.fcs0.weight.data, a=0.01, mode='fan_in')
        nn.init.kaiming_normal_(self.fc1.weight.data, a=0.01, mode='fan_in')
        nn.init.kaiming_normal_(self.fc2.weight.data, a=0.01, mode='fan_in')
        # Small uniform initialization for output layer
        nn.init.uniform_(self.fc3.weight.data, -3e-3, 3e-3)

    def forward(self, state, action):
        xs = state
        xs = self.bn(xs)
        xs = self.hidden_activation(self.fcs0(xs))
        x  = torch.cat((xs, action), dim=-1)
        x  = self.hidden_activation(self.fc1(x))
        x  = self.hidden_activation(self.fc2(x))
        q  = self.fc3(x)

        return q