# GRPO (Group Relative Policy Optimization) agent over the SAM2 memory bank.
import numpy as np

import torch
import torch.distributed as dist
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel as DDP

from sam2_train.rl_modules.policy_optimization.base_po_agent import (
    BasePOAgent,
    RunningReturnScale,
    stack_state_feats,
)
from sam2_train.rl_modules.rl_components import RLReplayInstance


class GRPOReplayInstance(RLReplayInstance):
    def __init__(
        self,
        frame_idx=None,
        state=None,
        action=None,
        reward=None,
        log_probs=None,
        action_mask=None,
    ):
        super().__init__(frame_idx, state, action, None, None, None, reward)
        self.log_probs = log_probs
        # Which actions were legal at this decision. Once the global pool is on, the
        # legal set is a *subset* of the action space that changes frame to frame --
        # recall_every gates pool candidates off entirely, and entries already resident
        # in the bank are excluded -- so update() has to renormalize over exactly the
        # support the action was sampled from or the importance ratio compares two
        # different distributions. `done` is not stored: GRPO is a bandit over one
        # decision, with no bootstrapping for a terminal flag to mask.
        self.action_mask = action_mask

    def get(self):
        # Call tuple to create a copy
        return tuple((self.state, self.log_probs, self.action, self.reward, self.action_mask))


class GRPOGroup:
    """One decision's group of sampled actions, scored and normalized against each other.

    The group *is* the baseline: every member was rolled out from the same state, so
    subtracting the group mean removes the state's value without a critic. That is only
    an unbiased baseline if the members are i.i.d. draws from the policy, which is why
    select_action samples with replacement.
    """

    def __init__(self, range=0.05, advantage_norm="group_std"):
        self.range = range
        self.advantage_norm = advantage_norm
        self.group = []

    def add_instance(self, instance):
        self.group.append(instance)

    def finalize(self):
        # Raw reward statistics, in dice-loss units, captured before normalization. They
        # are the only readout of whether a decision carried real signal, so they are
        # recorded even for the groups dropped below; final_group folds them into the
        # agent's metrics.
        self.raw_spread = 0.0
        self.raw_std = 0.0

        if len(self.group) < 2:
            # A single-member group has no spread to normalize against: the unbiased std
            # is NaN, which would poison every minibatch that later draws this sample.
            # It also carries no signal -- r minus its own mean is exactly 0 -- so there
            # is nothing to keep.
            self.group = []
            return

        if len({ins.action for ins in self.group}) < 2:
            # Every member drew the same action, so each member's reward is its own mean
            # and every advantage is exactly 0. Such a group contributes no gradient, but
            # kept it would still occupy `group_size` minibatch slots and drag the mean
            # advantage of every real decision batched alongside it towards zero.
            # Sampling with replacement makes this common as soon as the policy sharpens,
            # which is why final_group counts the rate rather than dropping it silently.
            self.group = []
            return

        group_rewards = torch.tensor([float(ins.reward) for ins in self.group])
        self.raw_spread = (group_rewards.max() - group_rewards.min()).item()
        self.raw_std = group_rewards.std().item()

        # Centering is the part that has to happen here: the group mean is only knowable
        # inside the group, and it is what makes GRPO critic-free.
        group_rewards = group_rewards - group_rewards.mean(dim=0, keepdim=True)

        if self.advantage_norm == "group_std":
            # Textbook GRPO. Defensible for the bounded 0/1 verifier rewards it was
            # designed around; here the reward is an unbounded dice-loss delta that can
            # legitimately be zero, so this declares whatever spread the decision happened
            # to produce to be unit variance and nothing downstream can tell a 1e-2 group
            # from a 1e-4 one. Kept as the default so existing runs stay reproducible.
            group_std = group_rewards.std(dim=0, keepdim=True)
            group_rewards = self.range * group_rewards / (group_std + 1e-6)
        # Otherwise ("running_scale") the centered rewards stay in dice-loss units and
        # GRPOAgent.update divides them by a *global* running scale instead -- see
        # GRPOAgent._advantage_scale. Scaling cannot happen here: the scale is shared
        # across groups, so applying it at collection time would freeze early groups of a
        # volume into the buffer at a different scale from its late ones.

        for i, ins in enumerate(self.group):
            # Plain float, not a 0-dim tensor: these go into the replay buffer and are
            # re-batched with torch.as_tensor, which is far cheaper over floats.
            ins.reward = group_rewards[i].item()

    def get_instances(self):
        return [ins.get() for ins in self.group]


class GRPOActor(nn.Module):
    def __init__(self, feat_summarizer, policy_net):
        super().__init__()

        self.feat_summarizer = feat_summarizer
        self.policy_net = policy_net

    def forward(self, **state_feats):
        """`state_feats` is whatever stack_state_feats produced: the five bank/image
        tensors, plus the candidate/bank ages and the pool stack when the global pool is
        enabled. Taken as **kwargs so enabling the pool needs no change here."""
        curr_feats = self.feat_summarizer(**state_feats)
        return self.policy_net(**curr_feats)


class GRPOAgent(BasePOAgent):
    # Its select_action has its own group-sampling signature and it collects groups
    # through generate_rl_steps, not agent_update_first_stage's extra samples.
    supports_action_resampling = False

    def __init__(
        self,
        num_maskmem,
        policy_lr=0.0001,
        value_lr=0.001,
        gamma=0.99,
        beta=0.9995,
        tau=0.9,
        range=0.05,
        buffer_size=500,
        batch_size=64,
        device="cpu",
        entropy_weight=0.1,
        epsilon=0.2,
        lr_T_max=1000,
        min_lr=0.0,
        sam2_dim={},
        n_layers=2,
        target_kl=None,
        advantage_norm="group_std",
        adv_min_scale=1e-6,
    ):
        super().__init__(
            num_maskmem=num_maskmem,
            policy_lr=policy_lr,
            value_lr=value_lr,
            gamma=gamma,
            beta=beta,
            tau=tau,
            buffer_size=buffer_size,
            batch_size=batch_size,
            device=device,
            entropy_weight=entropy_weight,
            lr_T_max=lr_T_max,
            min_lr=min_lr,
            sam2_dim=sam2_dim,
            n_layers=n_layers,
            target_kl=target_kl,
        )
        self.epsilon = epsilon
        self.range = range

        if advantage_norm not in ("group_std", "running_scale"):
            raise ValueError(
                f"advantage_norm must be 'group_std' or 'running_scale', got "
                f"{advantage_norm!r}"
            )
        self.advantage_norm = advantage_norm
        self.adv_min_scale = float(adv_min_scale)
        # Std of the centered rewards over every group kept so far, used in place of each
        # group's own std under "running_scale". Deliberately the lifetime estimate rather
        # than an EMA: the whole point is to hold the units fixed while the per-decision
        # spread shrinks over training, and a window that tracked that shrinkage would
        # renormalize late noise straight back up to unit variance -- exactly the failure
        # being removed. Fed at collection time (final_group) so it has seen a full
        # buffer's worth before update() first divides by it; its var starts at 1.0 with a
        # count of ~0, so the first real batch swamps the prior and there is no cold start.
        self.adv_scale = RunningReturnScale()

        # Rehome the summarizer/policy pair super() built into a single actor module and
        # drop every attribute that belongs to the actor-critic path.
        #
        # This used to build a *second* pair and leave super()'s untouched, which cost
        # 4.8M dead parameters and -- worse -- silently broke the temporal prior:
        # get_network warm-starts `agent.feat_summarizer` from SAM2's maskmem_tpos_enc
        # and only falls back to `agent.actor.feat_summarizer` when the former is absent,
        # so the prior landed on the discarded module and the live policy kept a random
        # one. Deleting the attribute is what makes that fallback fire.
        self.actor = GRPOActor(self.feat_summarizer, self.policy_net)
        del self.feat_summarizer
        del self.policy_net
        # GRPO is critic-free: the group mean is the baseline. Dropping the optimizer too
        # is what actually frees the critic -- it holds the only other reference to it.
        del self.optimizer
        self.value_net = None

        self.policy_optimizer = optim.AdamW(self.actor.parameters(), lr=policy_lr, weight_decay=0.01)
        # The base scheduler tracks an optimizer this agent doesn't use: the actor has its
        # own optimizer, so it needs its own cosine schedule, stepped once per update().
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.policy_optimizer,
            T_max=lr_T_max,
            eta_min=min_lr,
        )

        self.await_group = None

        # Group diagnostics accumulated across decisions, drained by the next update().
        # See pop_group_stats.
        self._group_count = 0
        self._collapsed_count = 0
        self._raw_spread_sum = 0.0
        self._raw_std_sum = 0.0

        # update() calls that actually trained; seeds each one's minibatch shuffles.
        self._num_updates = 0

        # For distributed training
        self.rank = 0
        self.distributed = False

    def to(self, device, non_blocking=True):
        self.device = device
        self.actor.to(device=device, non_blocking=non_blocking)

    def to_dtype(self, dtype):
        """Record the compute dtype WITHOUT casting the parameters; see BasePOAgent."""
        self.dtype = dtype

    def freeze(self):
        for param in self.actor.parameters():
            param.requires_grad_(False)

    def init_new_group(self):
        self.await_group = GRPOGroup(range=self.range, advantage_norm=self.advantage_norm)

    def add_new_instance_to_group(self, **instance_info):
        self.await_group.add_instance(GRPOReplayInstance(**instance_info))

    def final_group(self):
        self.await_group.finalize()
        new_normalized_instances = self.await_group.get_instances()

        self._group_count += 1
        self._raw_spread_sum += self.await_group.raw_spread
        self._raw_std_sum += self.await_group.raw_std
        if not new_normalized_instances:
            self._collapsed_count += 1

        if new_normalized_instances and self.advantage_norm == "running_scale":
            # Only kept groups. A collapsed one is all-zero by construction, so feeding it
            # would drag the scale towards zero and inflate every later advantage --
            # a direct interaction with the drop in GRPOGroup.finalize.
            # index 3 is `reward`; see GRPOReplayInstance.get.
            self.adv_scale.update(
                torch.tensor([float(t[3]) for t in new_normalized_instances])
            )

        self.replay_buffer.extend(new_normalized_instances)
        self.await_group = None

    def _advantage_scale(self):
        """The divisor update() applies to the stored centered rewards.

        1.0 under "group_std", where finalize() already produced finished advantages.
        Under "running_scale" this is the global std, floored only to keep the division
        finite -- the floor is numerical hygiene, not the mechanism. What actually stops
        a noise-level decision from being trained on like a real one is that the scale is
        *shared*: a group whose spread is 25x smaller than the running average now yields
        advantages 25x smaller, instead of being rescaled to unit variance by its own std.

        Not synchronised across ranks. Each rank's estimate drifts by well under the noise
        of everything else feeding the gradient, and DDP averages the gradients anyway.
        """
        if self.advantage_norm != "running_scale":
            return 1.0
        return max(self.adv_scale.std, self.adv_min_scale)

    def pop_group_stats(self):
        """Raw, pre-normalization group statistics since the last call, then clear.

        `group_reward_spread_mean` is the number to watch. It is max-minus-min of the
        dice-loss deltas the sampled actions produced at one decision, in dice-loss
        units, and it has to be read against the precision of the measurement that
        produced it: the branch losses come out of a bf16 SAM2 forward, so a spread down
        near that resolution is quantization, not preference. finalize() divides each
        group by its own std, which rescales a spread of any size to unit-variance
        advantages -- so a noise-level spread is trained on exactly as hard as a real one
        and nothing downstream can tell them apart.

        `group_collapse_frac` is the other failure mode: it rises as the policy sharpens
        and every draw returns the same action. At 1.0 nothing reaches the buffer at all
        and update() silently stops running.
        """
        if self._group_count == 0:
            return {}
        stats = {
            "group_reward_spread_mean": self._raw_spread_sum / self._group_count,
            "group_reward_std_mean": self._raw_std_sum / self._group_count,
            "group_collapse_frac": self._collapsed_count / self._group_count,
            "n_groups": float(self._group_count),
        }
        self._group_count = 0
        self._collapsed_count = 0
        self._raw_spread_sum = 0.0
        self._raw_std_sum = 0.0
        return stats

    def _policy_module(self):
        """The bare actor behind any DDP wrapper.

        Rollout forwards must not go through DDP. Ranks make different numbers of
        decisions -- volumes differ in length and in object count -- so any collective
        DDP runs inside forward() desyncs them, and there is nothing to gain: the
        rollout is under no_grad and produces no gradient to reduce. update() keeps the
        wrapper on purpose, because every rank runs the same number of those.
        """
        return self.actor.module if isinstance(self.actor, DDP) else self.actor

    @staticmethod
    def _masked_readout(logits, mask, dim):
        """log-probs renormalized over exactly `mask`'s support, plus that support's -H.

        The same trick BasePOAgent.train_step uses, and it has to be applied identically
        at selection and at update time: with the pool on, the legal action set is a
        frame-dependent subset, so a full-support softmax would put mass on actions that
        were never on offer and the PPO ratio would compare two different distributions.
        nan_to_num handles the 0 * -inf that masked entries produce.
        """
        log_probs = torch.log_softmax(logits.masked_fill(~mask, float("-inf")), dim=dim)
        probs = log_probs.exp()  # exactly 0 on masked actions
        minus_entropy = (probs * torch.nan_to_num(log_probs, neginf=0.0)).sum(dim=dim)
        return log_probs, probs, minus_entropy

    @torch.no_grad()
    def select_action(self, state, valid_actions, num_samples=1, training=False):
        """Only called once the bank is full; the caller (generate_rl_steps) inserts the
        incoming frame directly without a decision while the bank is filling.

        `valid_actions` indexes the flat action space BasePolicyNetwork emits --
        0 = no-op, then 1 + c*M + j = admit candidate c into bank slot j, candidate 0
        being the incoming frame and 1.. the global pool entries. It is a strict subset
        once the pool is on, so everything below works over a mask rather than the full
        support.

        In training this returns a *group*: `num_samples` i.i.d. draws whose rewards
        GRPOGroup normalizes against each other, plus an independently drawn `main_action`
        that is the one actually applied to the live memory bank.
        """
        # Deliberately the unwrapped module, not self.actor; see _policy_module.
        actor = self._policy_module()
        actor.eval()

        device = next(actor.parameters()).device
        action_logits = actor(**stack_state_feats([state], device=device))
        action_logits = action_logits.squeeze(0).detach().float().cpu()

        valid_actions = torch.as_tensor(valid_actions, dtype=torch.int64)
        action_mask = torch.zeros_like(action_logits, dtype=torch.bool)
        action_mask[valid_actions] = True

        log_probs, probs, minus_entropy = self._masked_readout(action_logits, action_mask, dim=0)
        
        if not training:
            print("[GRPO Probs]", probs)

        if training:
            action_idx = torch.multinomial(probs, num_samples, replacement=True)
            return {
                "main_action": action_idx[0].item(),
                "action": action_idx.tolist(),
                "log_probs": log_probs[action_idx].tolist(),
                "action_mask": action_mask,
            }

        action = torch.argmax(probs).item()
        self.record_val_action(action, -minus_entropy.item())
        return {"main_action": action}

    # update() is called from inside train_sam, which runs under no_grad once SAM2 is
    # frozen (-stop_sam2_ep); the actor still needs a graph. Every other agent gets this
    # from a `with torch.enable_grad()` inside its train_step.
    @torch.enable_grad()
    def update(self, num_update):
        local_count = torch.tensor([len(self.replay_buffer)], dtype=torch.long, device=self.device)

        if self.distributed:
            dist.all_reduce(local_count, op=dist.ReduceOp.MIN)

        # On-policy, the same gate PPOAgent.update uses: train only once every rank's
        # buffer is full, then clear it. Collection spans as many volumes as it takes to
        # fill it, which is still on-policy -- the actor does not move between updates.
        #
        # The gate is also what keeps DDP in step. Every minibatch below runs collectives
        # (DDP's gradient all-reduce in backward, the target_kl all-reduce), so every rank
        # must run the same number of them. A full deque holds exactly `buffer_size` on
        # every rank; a partial one would not, since collapsed groups are dropped per rank.
        if local_count < self.buffer_size or num_update <= 0:
            return None

        self.actor.train()

        device = self.device
        buffer_size = len(self.replay_buffer)
        # `num_update` counts shuffled full passes over the buffer, not that many
        # independently drawn minibatches -- the same shape BasePOAgent.update uses. Every
        # sample in here cost a SAM2 forward to score, by far the most expensive thing in
        # this training loop, and the old random.sample form consumed roughly 40% of a
        # buffer's collection once before clear() threw the rest away unseen.
        #
        # Seeded per update, not per buffer size: the size is now always `buffer_size`,
        # so a size-based seed would replay the same shuffles on every update of an epoch.
        np.random.seed([self.rank, self.epoch, self._num_updates])
        self._num_updates += 1

        # Read once, before the loop: every minibatch of one update() has to be scaled
        # identically, or the ratio against old_log_probs means something different from
        # one pass to the next.
        adv_scale = self._advantage_scale()

        total_policy_loss, total_policy_gradnorm = 0, 0
        metric_sums, done_updates = {}, 0
        stopped_early, passes = False, 0
        for ep in range(num_update):
            passes = ep + 1
            shuffle_indice = np.random.permutation(buffer_size)

            for start in range(0, buffer_size, self.batch_size):
                batch_indice = shuffle_indice[start:start + self.batch_size]
                # A trailing minibatch of one has no advantage spread to report and would
                # take a full-weight optimizer step off a single sample.
                if len(batch_indice) < 2:
                    continue
                batch = [self.replay_buffer[idx] for idx in batch_indice]

                states, old_log_probs, actions, rewards, action_masks = zip(*batch)

                # Every member of a group shares one RLStates object, so a minibatch always
                # holds fewer distinct states than samples -- how many fewer depends on how
                # much of the buffer one volume filled (~25% duplicates on a full buffer,
                # far more early in a volume). The summarizer is by far the most expensive
                # part of the actor -- it cross-attends to a [1,256,64,64] image feature and
                # an 11-slot memory bank per state -- and every RL block here is built at
                # dropout 0, so one forward per unique state plus an index_select is exactly
                # equal to forwarding all of them, never more expensive, and correct for the
                # backward too (index_select accumulates gradient over duplicate rows).
                uniq_states, row_of, rows = [], {}, []
                for state in states:
                    key = id(state)
                    if key not in row_of:
                        row_of[key] = len(uniq_states)
                        uniq_states.append(state)
                    rows.append(row_of[key])
                rows = torch.as_tensor(rows, dtype=torch.int64, device=device)

                actions = torch.as_tensor(actions, dtype=torch.int64, device=device).unsqueeze(1)
                # GRPOGroup centred these against their own group, so they are advantages
                # rather than raw rewards. Under "group_std" it also scaled them and
                # adv_scale is 1.0; under "running_scale" they are still in dice-loss
                # units and this is where they get their (global) scale.
                advantages = torch.as_tensor(rewards, dtype=torch.float32, device=device).unsqueeze(1)
                if self.advantage_norm == "running_scale":
                    advantages = advantages * (self.range / adv_scale)
                old_log_probs_t = torch.as_tensor(
                    old_log_probs, dtype=torch.float32, device=device
                ).unsqueeze(1)
                action_masks_t = torch.stack(action_masks).to(device=device, non_blocking=True)

                feats = stack_state_feats(uniq_states, device=device)
                policy_logits = self.actor(**feats).index_select(0, rows)

                # Renormalized over the same support the action was sampled from -- see
                # _masked_readout. log_softmax, not log(softmax): the latter loses precision
                # exactly where the importance ratio is most sensitive.
                log_probs, _, minus_entropy = self._masked_readout(
                    policy_logits, action_masks_t, dim=1
                )
                log_action_probs = log_probs.gather(1, actions)

                policy_loss = self.compute_policy_loss(log_action_probs, advantages, old_log_probs_t)
                # Full-distribution -H = sum_a p_a log p_a over the legal support. The previous
                # version used only the taken action's p*log p, whose gradient (log p + 1) is
                # negative for p < 1/e -- over an action space this size that *sharpened* the
                # policy, the opposite of an entropy bonus.
                minus_entropy = minus_entropy.mean()
                policy_loss = policy_loss + minus_entropy * self.entropy_weight

                self.policy_optimizer.zero_grad()
                policy_loss.backward()
                gradnorm = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
                self.policy_optimizer.step()

                total_policy_loss += policy_loss.detach()
                total_policy_gradnorm += gradnorm
                done_updates += 1

                with torch.no_grad():
                    log_ratio = log_action_probs - old_log_probs_t
                    ratio = log_ratio.exp()
                    clipped = (ratio < 1.0 - self.epsilon) | (ratio > 1.0 + self.epsilon)
                    metrics = {
                        "policy_entropy": -minus_entropy.item(),
                        # Schulman's k3 estimator: unbiased and >=0 in expectation.
                        "approx_kl": ((ratio - 1) - log_ratio).mean().item(),
                        "clip_fraction": clipped.float().mean().item(),
                        "adv_std": advantages.std().item(),
                        "n_valid_actions": action_masks_t.sum(dim=1).float().mean().item(),
                        "unique_state_frac": len(uniq_states) / len(states),
                    }
                for k, v in metrics.items():
                    metric_sums[k] = metric_sums.get(k, 0.0) + v

                # Same early stop as the PPO path: every minibatch is drawn against the
                # old_log_probs recorded at collection time, so once the policy has moved this
                # far the remaining passes are optimizing a stale importance-sampling
                # estimate. None (the default) runs all num_update passes.
                if self.target_kl is not None:
                    stop = torch.tensor(
                        [float(metrics["approx_kl"] > self.target_kl)],
                        dtype=torch.float32, device=self.device,
                    )
                    if self.distributed:
                        # Every rank has to leave the loop on the same minibatch. Breaking
                        # independently leaves the ranks that kept going waiting forever on
                        # an all-reduce inside DDP's backward that no one else will join.
                        dist.all_reduce(stop, op=dist.ReduceOp.MAX)
                    if stop.item() > 0:
                        print(
                            f"Early stopping at pass {ep} after {done_updates} minibatch "
                            f"updates: approx_kl {metrics['approx_kl']:.4f} > "
                            f"target_kl {self.target_kl}"
                        )
                        stopped_early = True
                        break

            if stopped_early:
                break

        # Clear buffer after update for on-policy training
        self.replay_buffer.clear()

        # Logged before stepping so the value matches the LR the updates above ran at.
        current_lr = self.policy_optimizer.param_groups[0]["lr"]
        self.step_lr_scheduler()

        out = {
            "actor_loss": total_policy_loss / done_updates,
            "policy_gradnorm": total_policy_gradnorm / done_updates,
            "agent_lr": current_lr,
            "done_updates": done_updates,
            "buffer_passes": passes,
            # In dice-loss units under "running_scale", so it is directly comparable to
            # group_reward_spread_mean; a flat 1.0 under "group_std".
            "adv_scale": adv_scale,
        }
        for k, total in metric_sums.items():
            out[k] = total / done_updates
        # Raw, pre-normalization reward statistics for the groups collected since the last
        # update; see pop_group_stats for why they are the number to watch.
        out.update(self.pop_group_stats())
        return out

    def compute_policy_loss(self, log_prob, advantage, old_log_prob):
        advantage = advantage.detach()
        old_log_prob = old_log_prob.detach()

        ratio = (log_prob - old_log_prob).exp()
        surr_loss = ratio * advantage
        clipped_surr_loss = torch.clamp(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon) * advantage
        policy_loss = -torch.mean(torch.min(surr_loss, clipped_surr_loss))
        return policy_loss

    def state_dict(self):
        """Nested, so the advantage scale rides along with the weights.

        Under "running_scale" the policy is trained against advantages divided by
        `adv_scale`, so a resume that restarted it at 1.0 would silently rescale every
        advantage by ~1e3 -- the same trap BasePOAgent.state_dict calls out for its
        critic's return scale. The old flat layout is still readable; see
        load_state_dict.
        """
        return {
            "actor": self._policy_module().state_dict(),
            "adv_scale": self.adv_scale.state_dict(),
            "advantage_norm": self.advantage_norm,
        }

    def load_state_dict(self, state_dict):
        """Accepts all three layouts this agent has written.

        `actor`      -- current: nested, carries the advantage scale.
        `feat_summarizer` as a top-level key -- the split BasePOAgent layout.
        anything else -- a bare flat actor state dict, with the pre-rename `perceiver`
        keys mapped onto `qformer`. Note the middle test is an exact-key lookup, so a
        flat dict (whose keys are `feat_summarizer.<...>`) correctly falls through it.
        """
        actor = self._policy_module()

        if "actor" in state_dict:
            actor.load_state_dict(state_dict["actor"])
            if "adv_scale" in state_dict:
                self.adv_scale.load_state_dict(state_dict["adv_scale"])
            saved_norm = state_dict.get("advantage_norm")
            if saved_norm is not None and saved_norm != self.advantage_norm:
                print(
                    f"[agent] checkpoint was trained with advantage_norm={saved_norm!r} "
                    f"but this run uses {self.advantage_norm!r}; the policy is resuming "
                    f"into a differently scaled advantage space"
                )
        elif "feat_summarizer" in state_dict.keys():
            actor.feat_summarizer.load_state_dict(state_dict["feat_summarizer"])
            actor.policy_net.load_state_dict(state_dict["policy_net"])
        else:
            temp_state_dict = {}
            for k, v in state_dict.items():
                if 'perceiver' in k:
                    k = k.replace('perceiver', 'qformer')
                temp_state_dict[k] = v

            actor.load_state_dict(temp_state_dict)

        if "adv_scale" not in state_dict and self.advantage_norm == "running_scale":
            print(
                "[agent] checkpoint carries no advantage scale; it restarts from the "
                "prior and re-estimates over the first buffer"
            )

    def to_distributed(self, rank):
        self.distributed = True
        self.rank = rank
        # broadcast_buffers=False: the actor's only buffer is the summarizer's
        # non-persistent `slot_tpos_prior`, re-derived identically on every rank from the
        # same SAM2 checkpoint, so there is nothing to synchronise -- and DDP's buffer
        # broadcast is a collective inside forward(). _policy_module already keeps
        # rollouts off the wrapper; this is the belt to that's braces.
        self.actor = DDP(
            self.actor, device_ids=[rank], output_device=rank, broadcast_buffers=False,
        )

    def num_parameters(self):
        """This function expect modules didn't wrapped by DDP"""
        return sum(p.numel() for p in self.actor.parameters())
