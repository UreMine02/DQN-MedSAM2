# import random
# import pickle
# from collections import defaultdict

# class SimpleQAgent:
#     def __init__(self, max_memory, alpha=0.1, gamma=0.9, epsilon=0.2):
#         self.q_table = defaultdict(lambda: defaultdict(float))
#         self.alpha = alpha
#         self.gamma = gamma
#         self.epsilon = epsilon
#         self.max_memory = max_memory

#     def select_action(self, state, available_actions):
#         if random.random() < self.epsilon:
#             return random.choice(available_actions)
#         q_values = {a: self.q_table[state][a] for a in available_actions}
#         return max(q_values, key=q_values.get)

#     def update(self, state, action, reward, next_state, next_available):
#         max_next_q = max([self.q_table[next_state][a] for a in next_available], default=0.0)
#         current_q = self.q_table[state][action]
#         self.q_table[state][action] = current_q + self.alpha * (
#             reward + self.gamma * max_next_q - current_q
#         )

#     def save(self, path):
#         with open(path, 'wb') as f:
#             pickle.dump(dict(self.q_table), f)

#     def load(self, path):
#         with open(path, 'rb') as f:
#             data = pickle.load(f)
#             self.q_table = defaultdict(lambda: defaultdict(float), data)

# Deep Q-Learning Agent 
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque

# -------------------------
# Define Action Space
# -------------------------
ACTION_SPACE = {
    0: "skip",
    1: "add",
    2: "add_drop_oldest",
    3: "add_drop_lowest_iou",
    4: "add_drop_random",
}
ACTION_DIM = len(ACTION_SPACE)


# -------------------------
# Q-Network (MLP)
# -------------------------
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(QNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x):
        return self.net(x)


# -------------------------
# Deep Q-Learning Agent
# -------------------------
class DeepQAgent:
    def __init__(self, state_dim, ACTION_DIM, lr=1e-3, gamma=0.99, epsilon=0.1,
                 buffer_size=10000, batch_size=64):
        self.q_net = QNetwork(state_dim, ACTION_DIM).cuda()
        self.target_net = QNetwork(state_dim, ACTION_DIM).cuda()
        self.target_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.gamma = gamma
        self.epsilon = epsilon
        self.replay_buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        for name, param in self.q_net.named_parameters():
            print(name, param.requires_grad)


    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, ACTION_DIM - 1)
        state = torch.FloatTensor(state).unsqueeze(0).cuda()
        q_values = self.q_net(state)
        return q_values.argmax().item()

    def store(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states).cuda()
        actions = torch.LongTensor(actions).unsqueeze(1).cuda()
        rewards = torch.FloatTensor(rewards).unsqueeze(1).cuda()
        next_states = torch.FloatTensor(next_states).cuda()
        dones = torch.FloatTensor(dones).unsqueeze(1).cuda()

        with torch.enable_grad():
            
            q_values = self.q_net(states).gather(1, actions)
            with torch.no_grad():
                max_next_q = self.target_net(next_states).max(1, keepdim=True)[0]
                target_q = rewards + self.gamma * max_next_q * (1 - dones)

            q_values = q_values.squeeze(-1)   # [batch]
            target_q = target_q.squeeze(-1)   # [batch]

            loss_fn = nn.MSELoss()
            loss = loss_fn(q_values, target_q)

        self.optimizer.zero_grad()
        print("q_values.requires_grad:", q_values.requires_grad)
        print("target_q.requires_grad:", target_q.requires_grad)
        print("loss.requires_grad:", loss.requires_grad)
        loss.backward(retain_graph=True)   
        self.optimizer.step()

    def update_target(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

    def save(self, path):
        torch.save(self.q_net.state_dict(), path)

    def load(self, path):
        self.q_net.load_state_dict(torch.load(path))


def map_action(action, obj_output_dict, storage_key, current_out):
    """
    Map an integer action to a drop_key (frame index) for the given storage_key.
    Returns None if no drop should happen (or no candidate to drop).
    Safety: never index into empty lists and prefer per-entry iou if available.
    """
    # get current keys in memory (ordered)
    mem_dict = obj_output_dict.get(storage_key, {})
    sorted_keys = sorted(mem_dict.keys())
    if len(sorted_keys) == 0:
        return None

    # actions:
    # 0: skip
    # 1: add (if full, fallback to drop oldest)
    # 2: add_drop_oldest
    # 3: add_drop_lowest_iou
    # 4: add_drop_random

    # drop oldest (for action 1 or 2)
    if action in [1, 2]:
        return sorted_keys[0]

    # drop lowest IoU (action == 3)
    if action == 3:
        # try to collect IoU per stored entry (prefer entry['iou'] or entry['object_score_logits'])
        iou_list = []
        for k in sorted_keys:
            entry = mem_dict.get(k, {})
            i_val = None
            # 1) check explicit 'iou' in stored entry
            try:
                if isinstance(entry, dict) and "iou" in entry and entry["iou"] is not None:
                    v = entry["iou"]
                    if hasattr(v, "detach"):
                        i_val = float(v.detach().cpu().item())
                    else:
                        i_val = float(v)
                # 2) fallback to object_score_logits if present (may be tensor or scalar)
                elif isinstance(entry, dict) and "object_score_logits" in entry and entry["object_score_logits"] is not None:
                    v = entry["object_score_logits"]
                    if hasattr(v, "detach"):
                        # if vector, take mean
                        try:
                            vv = v.detach().cpu().numpy()
                            if np.ndim(vv) > 0:
                                i_val = float(np.mean(vv))
                            else:
                                i_val = float(vv)
                        except Exception:
                            try:
                                i_val = float(v.detach().cpu().item())
                            except Exception:
                                i_val = None
                    else:
                        try:
                            i_val = float(v)
                        except Exception:
                            i_val = None
            except Exception:
                i_val = None

            # 3) fallback to current_out['ious'] if shaped compatibly
            if i_val is None:
                try:
                    if "ious" in current_out and current_out["ious"] is not None:
                        ious = current_out["ious"]
                        # make numpy
                        if hasattr(ious, "detach"):
                            ious_np = ious.detach().cpu().numpy()
                        else:
                            ious_np = np.array(ious)
                        # if length matches sorted_keys, try map by index
                        if len(ious_np) == len(sorted_keys):
                            idx = sorted_keys.index(k)  # this is safe because same ordering assumption
                            i_val = float(ious_np[idx])
                except Exception:
                    i_val = None

            # final fallback: very large so it won't be selected as minimum
            if i_val is None:
                i_val = float("inf")
            iou_list.append(i_val)

        # if all inf (no usable iou), fallback to oldest
        if all(np.isinf(iou_list)):
            return sorted_keys[0]
        min_idx = int(np.argmin(np.array(iou_list)))
        return sorted_keys[min_idx]

    # drop random (action == 4)
    if action == 4:
        return random.choice(sorted_keys)

    # action 0 or unknown -> no drop
    return None

