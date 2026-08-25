import contextlib
from collections import deque

from torch.nn.parallel import DistributedDataParallel as DDP

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
        self.buffer_size = buffer_size
        self.replay_buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.beta = beta
            
        self.await_replay_instance = None
        self.epoch = 0
    
    def init_new_replay_instance(self, **instance_info):
        """Open a transition, closing the pending one first.

        The state we are opening with is, by construction, the successor of the
        transition that is still pending: it is built from the same frame, straight
        after that transition's action was applied to the memory bank. So opening is
        exactly the moment the previous transition's next_state becomes available, and
        no second prepare_rl_state is needed.
        """
        self.close_await_replay_instance(next_state=instance_info.get("state"))
        self.await_replay_instance = RLReplayInstance(**instance_info)

    def close_await_replay_instance(self, next_state=None, done=False):
        if self.await_replay_instance is None:
            return
        self.await_replay_instance.close(next_state=next_state, done=done)
        self.store_transition(self.await_replay_instance)
        self.await_replay_instance = None

    def store_transition(self, instance):
        """Off-policy agents write straight to the replay buffer."""
        self.replay_buffer.append(instance.get())

    def set_await_done(self):
        """Terminate the pending transition at the end of a volume.

        The tracking state now spans every chunk of a volume, so a chunk boundary is not
        a terminal -- the memory bank the agent leaves behind is the one the next chunk
        starts from, and that transition is closed by the next chunk's first decision.
        Only the last frame of the volume ends the episode.
        """
        self.close_await_replay_instance(done=True)

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

    @contextlib.contextmanager
    def unwrapped_modules(self):
        """Expose the raw sub-modules hiding behind any DDP wrappers, for the duration
        of the block.

        Validation runs on one rank only, and DDP's forward broadcasts buffers across
        every rank before it runs the wrapped module -- so a single rank calling a
        DDP-wrapped policy would block forever waiting on ranks that are sitting at a
        later barrier. The wrapped attributes are whatever `to_distributed` replaced
        (actor, or feat_summarizer/policy_net/value_net), found by type rather than by
        name so both agent hierarchies are covered; they are restored on the way out
        even if the body raises, since training after validation still needs them.
        """
        wrapped = {name: mod for name, mod in vars(self).items() if isinstance(mod, DDP)}
        for name, mod in wrapped.items():
            setattr(self, name, mod.module)
        try:
            yield self
        finally:
            for name, mod in wrapped.items():
                setattr(self, name, mod)