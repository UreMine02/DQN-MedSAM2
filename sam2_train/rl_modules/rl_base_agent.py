from collections import deque

from sam2_train.rl_modules.rl_components import RLReplayInstance, RLStates

class BaseAgent:
    # Whether the caller must build the successor state when it closes a transition.
    # Off-policy agents push straight to the replay buffer and need it there and then;
    # BasePOAgent holds transitions in a Trajectory and fills next_state in by reference
    # from the following transition's state instead.
    materialize_next_state = True

    def __init__(
        self,
        num_maskmem, 
        lr=1e-4, 
        gamma=0.99, 
        beta=0.9995, 
        buffer_size=500, 
        batch_size=32, 
        device="cpu"
    ):
        self.num_maskmem = num_maskmem
        self.device = device
        self.lr = lr
        self.gamma = gamma
        self.buffer_size = buffer_size
        self.replay_buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.beta = beta
            
        self.await_replay_instance = None
        self.epoch = 0
    
    def init_new_replay_instance(self, **instance_info):
        self.await_replay_instance = RLReplayInstance(**instance_info)
        
    def update_await_replay_instance(self, loss_after, next_state):
        self.await_replay_instance.update(loss_after, next_state)
        self.replay_buffer.append(self.await_replay_instance.get())
        self.await_replay_instance = None
    
    def set_await_done(self, loss_after):
        """Terminate the pending transition at the end of a chunk."""
        if self.await_replay_instance is None:
            return
        self.await_replay_instance.set_done(loss_after)
        self.replay_buffer.append(self.await_replay_instance.get())
        self.await_replay_instance = None

    def final_trajectory(self):
        """Off-policy agents write straight to the buffer, so there is nothing to close
        out at the end of a volume."""
        pass

    def select_action(self, state: RLStates, valid_actions, bank_size, training=False):
        raise NotImplementedError
    
    def to(self, device):
        self.device = device

    def state_dict(self):
        raise NotImplementedError
    
    def load_state_dict(self, state_dict):
        raise NotImplementedError
        
    def set_epoch(self, epoch, distributed=False):
        self.epoch = epoch