from collections import deque

from sam2_train.rl_modules.rl_components import RLReplayInstance, RLStates

class BaseAgent:
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
    
    def clear_await(self, loss_after):
        self.await_replay_instance.set_done(loss_after)
        self.replay_buffer.append(self.await_replay_instance.get())
        self.await_replay_instance = None
        
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