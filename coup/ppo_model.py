import torch
import torch.nn as nn
import torch.nn.functional as F

class CoupLSTMPPO(nn.Module):
    """
    Actor-Critic Network for Coup.
    Uses an LSTM to encode the sequence of game states and actions,
    allowing the agent to build a mental model of the opponent's
    playstyle (e.g., estimating their bluff_rate or challenge_rate).
    """
    def __init__(self, input_dim=35, hidden_dim=64, num_actions=15):
        super(CoupLSTMPPO, self).__init__()
        
        # 1. Feature Extractor
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        # 2. Memory / Recurrent Layer
        # batch_first=True means tensors are (batch, seq, feature)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        
        # 3. Actor Head (Policy Distribution)
        self.actor = nn.Linear(hidden_dim, num_actions)
        
        # 4. Critic Head (Value Estimate -1 to 1)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, x, hidden_state=None):
        """
        x: Input tensor of shape (batch, seq_len, input_dim)
        hidden_state: tuple of (h_n, c_n) for the LSTM
        """
        # Ensure we have a sequence dimension
        # If input is (batch, input_dim), unsqueeze to (batch, 1, input_dim)
        is_unbatched = False
        if len(x.shape) == 1:
            is_unbatched = True
            x = x.unsqueeze(0).unsqueeze(0)
        elif len(x.shape) == 2:
            x = x.unsqueeze(1)
            
        # Feature extraction (apply to all elements in sequence)
        batch_size, seq_len, _ = x.shape
        x = x.view(batch_size * seq_len, -1)
        
        feat = F.relu(self.fc1(x))
        feat = F.relu(self.fc2(feat))
        
        # Reshape for LSTM
        feat = feat.view(batch_size, seq_len, -1)
        
        # Pass through LSTM
        lstm_out, hidden_state = self.lstm(feat, hidden_state)
        
        # Apply heads to all steps in the sequence
        # Shape: (batch, seq_len, num_actions)
        action_logits = self.actor(lstm_out)
        value = torch.tanh(self.critic(lstm_out))  # Bound between -1 and 1
        
        if is_unbatched:
            action_logits = action_logits.squeeze(0).squeeze(0)
            value = value.squeeze(0).squeeze(0)
            
        return action_logits, value, hidden_state

    def reset_hidden(self, batch_size=1, device='cpu'):
        """Helper to create a fresh, zeroed-out hidden state."""
        h0 = torch.zeros(1, batch_size, self.lstm.hidden_size).to(device)
        c0 = torch.zeros(1, batch_size, self.lstm.hidden_size).to(device)
        return (h0, c0)
