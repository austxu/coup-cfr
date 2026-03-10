import torch
import torch.nn as nn
import torch.nn.functional as F


class CoupLSTMPPOv2(nn.Module):
    """
    Gen 6 Actor-Critic Network for N-player Coup.
    
    Two-step action selection:
    - Action head: picks what to do (15 actions)
    - Target head: picks who to target (5 opponent slots)
    - Critic head: state value estimate
    """
    def __init__(self, input_dim=155, hidden_dim=512, num_actions=15, num_targets=5):
        super(CoupLSTMPPOv2, self).__init__()

        # Feature Extractor
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        # Memory
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)

        # Action Head (what to do)
        self.actor_action = nn.Linear(hidden_dim, num_actions)

        # Target Head (who to target)
        self.actor_target = nn.Linear(hidden_dim, num_targets)

        # Critic Head (state value)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, x, hidden_state=None):
        """
        x: (batch, seq_len, input_dim) or (batch, input_dim) or (input_dim,)
        Returns: action_logits, target_logits, value, hidden_state
        """
        is_unbatched = False
        if len(x.shape) == 1:
            is_unbatched = True
            x = x.unsqueeze(0).unsqueeze(0)
        elif len(x.shape) == 2:
            x = x.unsqueeze(1)

        batch_size, seq_len, _ = x.shape
        x = x.view(batch_size * seq_len, -1)

        feat = F.relu(self.fc1(x))
        feat = F.relu(self.fc2(feat))

        feat = feat.view(batch_size, seq_len, -1)
        lstm_out, hidden_state = self.lstm(feat, hidden_state)

        action_logits = self.actor_action(lstm_out)
        target_logits = self.actor_target(lstm_out)
        value = torch.tanh(self.critic(lstm_out))

        if is_unbatched:
            action_logits = action_logits.squeeze(0).squeeze(0)
            target_logits = target_logits.squeeze(0).squeeze(0)
            value = value.squeeze(0).squeeze(0)

        return action_logits, target_logits, value, hidden_state

    def reset_hidden(self, batch_size=1, device='cpu'):
        h0 = torch.zeros(1, batch_size, self.lstm.hidden_size).to(device)
        c0 = torch.zeros(1, batch_size, self.lstm.hidden_size).to(device)
        return (h0, c0)
