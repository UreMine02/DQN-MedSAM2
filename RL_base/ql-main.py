"""
Q-learning memory selector for SAM2 + in-context learning (ICL)

This version is modified for *online memory management*:
- At each new frame, environment proposes a candidate memory (support + new frame features)
- Agent decides: ADD or SKIP.
- If ADD -> agent must also choose which existing memory (if bank is full) to DROP.
- Reward is computed as *relative loss*: compare SAM2 loss before and after memory update.

Notes:
- The reward_fn must compute relative loss given current memory bank and new memory candidate.
- Q-learning uses discrete states; consider feature engineering to represent memory bank state.
"""

import random
import pickle
from typing import List, Tuple, Callable, Dict, Any

import numpy as np

# -----------------------------
# ENVIRONMENT: Online Memory Management
# -----------------------------
class MemoryManagementEnv:
    """
    Online memory management environment for SAM2.

    - state: representation of memory bank + candidate memory features
    - action space: [0 = skip, 1..N = add & drop index (if full)]
    - reward: relative loss improvement after taking action
    """

    def __init__(self,
                 max_memory: int,
                 reward_fn: Callable[[List[int], Dict[str, Any]], float]):
        self.max_memory = max_memory
        self.reward_fn = reward_fn
        self.memory_bank = []  # list of memory indices or feature refs
        self.context = {}
        self.candidate = None

    def reset(self, initial_memory: List[Any] = None, context: Dict[str, Any] = None):
        self.memory_bank = [] if initial_memory is None else list(initial_memory)
        self.context = {} if context is None else context
        self.candidate = None
        return self._encode_state()

    def propose_candidate(self, candidate: Any):
        """Provide a new candidate memory item (features) from current frame."""
        self.candidate = candidate
        return self._encode_state()

    def available_actions(self) -> List[int]:
        """0 = skip, 1..len(memory_bank)=drop index (if memory full)."""
        if len(self.memory_bank) < self.max_memory:
            # can add without dropping
            return [0, 1]  # 0=skip, 1=add
        else:
            # must choose drop index
            return [0] + [i+1 for i in range(len(self.memory_bank))]

    def step_action(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        action=0: skip; action>0: add candidate, possibly drop memory[action-1] if full.
        Return next_state, reward, done=False, info.
        """
        pre_loss = self.context.get("loss_before", None)

        if action == 0:
            # skip adding
            reward = self.reward_fn(self.memory_bank, {**self.context, "action": "skip"})
        else:
            if len(self.memory_bank) >= self.max_memory:
                drop_idx = action - 1
                self.memory_bank.pop(drop_idx)
            self.memory_bank.append(self.candidate)
            reward = self.reward_fn(self.memory_bank, {**self.context, "action": "add"})

        self.candidate = None
        next_state = self._encode_state()
        return next_state, reward, False, {"memory_bank": self.memory_bank}

    def _encode_state(self) -> np.ndarray:
        """
        Return a state representation as np.array.
        Example: mean embedding of memory bank + candidate embedding (if exists).
        """
        # Simplest encoding: length + presence flag
        length = len(self.memory_bank)
        candidate_flag = 1.0 if self.candidate is not None else 0.0
        return np.array([length, candidate_flag], dtype=np.float32)


# -----------------------------
# Q-LEARNING AGENT (function approximation ready)
# -----------------------------
class QLearningAgent:
    def __init__(self,
                 state_dim: int,
                 n_actions: int,
                 alpha: float = 0.1,
                 gamma: float = 0.99,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.05,
                 epsilon_decay_steps: int = 1000):
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.total_steps = 0

        # simple table: discretize state via tuple(state) (for small state space)
        self.Q: Dict[Tuple, np.ndarray] = {}

    def _epsilon(self):
        t = min(self.total_steps, self.epsilon_decay_steps)
        return self.epsilon_end + (self.epsilon_start - self.epsilon_end) * (1 - t / self.epsilon_decay_steps)

    def select_action(self, state: np.ndarray, available_actions: List[int]) -> int:
        key = tuple(state)
        if key not in self.Q:
            self.Q[key] = np.zeros(self.n_actions, dtype=float)
        eps = self._epsilon()
        self.total_steps += 1
        if random.random() < eps:
            return random.choice(available_actions)
        qvals = self.Q[key][available_actions]
        return int(available_actions[np.argmax(qvals)])

    def update(self, state, action, reward, next_state, available_next_actions):
        key = tuple(state)
        next_key = tuple(next_state)
        if key not in self.Q:
            self.Q[key] = np.zeros(self.n_actions)
        if next_key not in self.Q:
            self.Q[next_key] = np.zeros(self.n_actions)

        q_next = 0 if len(available_next_actions) == 0 else np.max(self.Q[next_key][available_next_actions])
        td_target = reward + self.gamma * q_next
        td_error = td_target - self.Q[key][action]
        self.Q[key][action] += self.alpha * td_error

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self.Q, f)

    def load(self, path: str):
        with open(path, "rb") as f:
            self.Q = pickle.load(f)


# -----------------------------
# EXAMPLE RELATIVE LOSS REWARD
# -----------------------------

def relative_loss_reward(memory_bank: List[Any], context: Dict[str, Any]) -> float:
    """
    Compute relative loss improvement: (loss_before - loss_after) / loss_before.
    You must set context['loss_before'] and context['loss_after'] externally.
    """
    loss_before = context.get('loss_before', 1.0)
    loss_after = context.get('loss_after', loss_before)
    return (loss_before - loss_after) / max(1e-8, loss_before)


# -----------------------------
# TRAINING LOOP
# -----------------------------
def train(env: MemoryManagementEnv, agent: QLearningAgent, data_stream: List[Any], n_epochs=10):
    for epoch in range(n_epochs):
        state = env.reset()
        for frame in data_stream:
            # 1) propose candidate
            state = env.propose_candidate(frame)

            # 2) compute loss_before, update env.context externally
            env.context['loss_before'] = compute_loss(env.memory_bank, frame)
            available = env.available_actions()
            action = agent.select_action(state, available)

            # 3) simulate taking action and recompute loss_after
            if action != 0:
                temp_bank = env.memory_bank.copy()
                if len(temp_bank) >= env.max_memory:
                    temp_bank.pop(action-1)
                temp_bank.append(frame)
                env.context['loss_after'] = compute_loss(temp_bank, None)
            else:
                env.context['loss_after'] = env.context['loss_before']

            next_state, reward, _, info = env.step_action(action)
            available_next = env.available_actions()

            agent.update(state, action, reward, next_state, available_next)
            state = next_state

# placeholder: replace with SAM2 inference loss computation
def compute_loss(memory_bank, candidate):
    return random.random()
