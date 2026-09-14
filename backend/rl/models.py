"""Actor-Critic PyTorch model with Action Masking support."""
from typing import Tuple, Optional, Union
import torch
import torch.nn as nn
from torch.distributions import Categorical

from ..harness.types import ACTION_SPACE_SIZE


class ActorCriticPolicy(nn.Module):
    """Actor-Critic policy network with action masking for discrete action spaces.
    
    Architecture:
      Shared Trunk: Linear(state_dim, hidden_dim) -> Tanh -> Linear(hidden_dim, hidden_dim) -> Tanh
      Actor Head:   Linear(hidden_dim, action_dim) -> masked logits -> Categorical
      Critic Head:  Linear(hidden_dim, 1) -> state value V(s)
    """

    def __init__(
        self,
        state_dim: int = 30,
        action_dim: int = ACTION_SPACE_SIZE,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Shared feature representation
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )

        # Actor head
        self.actor = nn.Linear(hidden_dim, action_dim)

        # Critic head
        self.critic = nn.Linear(hidden_dim, 1)

        # Initialize orthogonal weights (standard PPO practice)
        self._init_weights()

    def _init_weights(self):
        for m in self.trunk.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain("tanh"))
                nn.init.constant_(m.bias, 0.0)
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.constant_(self.actor.bias, 0.0)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.constant_(self.critic.bias, 0.0)

    def forward(
        self,
        state: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute raw logits and state value.
        
        Args:
            state: Float tensor of shape (batch, state_dim) or (state_dim,)
            mask:  Bool tensor of shape (batch, action_dim) or (action_dim,)
        Returns:
            masked_logits, value
        """
        features = self.trunk(state)
        logits = self.actor(features)
        value = self.critic(features).squeeze(-1)

        if mask is not None:
            # A terminal state's all-false mask must not become a uniform
            # distribution over forbidden actions after softmax.
            if not torch.all(mask.any(dim=-1)):
                raise ValueError('No legal actions: do not evaluate a policy after the episode ends.')
            # Set masked action logits to very large negative number
            masked_logits = torch.where(mask, logits, torch.full_like(logits, -1e9))
        else:
            masked_logits = logits

        return masked_logits, value

    def get_action_and_value(
        self,
        state: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        action: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample action (or evaluate given action), log_prob, entropy, and state value.
        
        Args:
            state: shape (batch, state_dim) or (state_dim,)
            mask: shape (batch, action_dim) or (action_dim,)
            action: optional int tensor of chosen actions
            deterministic: if True, pick argmax of legal logits
        Returns:
            action, log_prob, entropy, value
        """
        masked_logits, value = self.forward(state, mask=mask)
        dist = Categorical(logits=masked_logits)

        if action is None:
            if deterministic:
                # Argmax over legal logits
                action = torch.argmax(masked_logits, dim=-1)
            else:
                action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        return action, log_prob, entropy, value

    def get_value(self, state: torch.Tensor) -> torch.Tensor:
        """Evaluate state value V(s) only."""
        features = self.trunk(state)
        return self.critic(features).squeeze(-1)
