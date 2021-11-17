"""
Code inspired by https://github.com/kchua/handful-of-trials/blob/master/dmbrl/controllers/MPC.py
                 https://github.com/quanvuong/handful-of-trials-pytorch/blob/master/MPC.py
                 https://github.com/kchua/handful-of-trials/blob/master/dmbrl/misc/optimizers/cem.py
"""

from .policy import Policy

import latentsafesets.utils.pytorch_utils as ptu
import latentsafesets.utils.spb_utils as spbu
from latentsafesets.utils.broil_utils import cvar_enumerate_pg, get_reward_distribution
from latentsafesets.utils.pointbot_reward_utils import PointBotReward
from latentsafesets.modules import VanillaVAE, PETSDynamics, ValueFunction, ConstraintEstimator, \
    GoalIndicator

import torch
import numpy as np
import gym

import logging

log = logging.getLogger('cem')


class CEMSafeSetPolicy(Policy):
    def __init__(self, env: gym.Env,
                 encoder: VanillaVAE,
                 safe_set,
                 value_function: ValueFunction,
                 dynamics_model: PETSDynamics,
                 constraint_function: ConstraintEstimator,
                 goal_indicator: GoalIndicator,
                 params):
        log.info("setting up safe set and dynamics model")

        self.env = env
        self.encoder = encoder
        self.safe_set = safe_set
        self.dynamics_model = dynamics_model
        self.value_function = value_function
        self.constraint_function = constraint_function
        self.goal_indicator = goal_indicator

        self.logdir = params['logdir']

        self.d_act = params['d_act']
        self.d_obs = params['d_obs']
        self.d_latent = params['d_latent']
        self.ac_ub, self.ac_lb = env.action_space.high, env.action_space.low
        self.plan_hor = params['plan_hor']
        self.random_percent = params['random_percent']
        self.popsize = params['num_candidates']
        self.num_elites = params['num_elites']
        self.max_iters = params['max_iters']
        self.safe_set_thresh = params['safe_set_thresh']
        self.safe_set_thresh_mult = params['safe_set_thresh_mult']
        self.safe_set_thresh_mult_iters = params['safe_set_thresh_mult_iters']
        self.constraint_thresh = params['constr_thresh']
        self.goal_thresh = params['gi_thresh']
        self.ignore_safe_set = params['safe_set_ignore']
        self.ignore_constraints = params['constr_ignore']

        self.mean = torch.zeros(self.d_act)
        self.std = torch.ones(self.d_act)
        self.ac_buf = np.array([]).reshape(0, self.d_act)
        self.prev_sol = np.tile((self.ac_lb + self.ac_ub) / 2, [self.plan_hor])
        self.init_var = np.tile(np.square(self.ac_ub - self.ac_lb) / 16, [self.plan_hor])

    @torch.no_grad()
    def act(self, obs):
        """
        Returns the action that this controller would take at time t given observation obs.

        Arguments:
            obs: The current observation. Cannot be a batch

        Returns: An action (and possibly the predicted cost)
        """

        # encode observation:
        obs = ptu.torchify(obs).reshape(1, *self.d_obs)
        emb = self.encoder.encode(obs)

        itr = 0
        reset_count = 0
        act_ss_thresh = self.safe_set_thresh
        while itr < self.max_iters:
            if itr == 0:
                # Action samples dim (num_candidates, planning_hor, d_act)
                if self.mean is None:
                    action_samples = self._sample_actions_random()
                else:
                    num_random = int(self.random_percent * self.popsize)
                    num_dist = self.popsize - num_random
                    action_samples_dist = self._sample_actions_normal(self.mean, self.std, n=num_dist)
                    action_samples_random = self._sample_actions_random(num_random)
                    action_samples = torch.cat((action_samples_dist, action_samples_random), dim=0)
            else:
                # Chop off the number of elites so we don't use constraint violating trajectories

                # num_constraint_satisfying = sum(values > -1e5)
                #iter_num_elites = min(num_constraint_satisfying, self.num_elites)
                iter_num_elites = self.num_elites
                # if num_constraint_satisfying == 0:
                #     reset_count += 1
                #     act_ss_thresh *= self.safe_set_thresh_mult
                #     if reset_count > self.safe_set_thresh_mult_iters:
                #         self.mean = None
                #         return self.env.action_space.sample()

                #     itr = 0
                #     self.mean, self.std = None, None
                #     continue

                # Sort
                sortid = values.argsort()
                actions_sorted = action_samples[sortid]
                elites = actions_sorted[-iter_num_elites:]

                # Refitting to Best Trajs
                self.mean, self.std = elites.mean(0), elites.std(0)

                action_samples = self._sample_actions_normal(self.mean, self.std)

            if itr < self.max_iters - 1:
                # dimension (num_models, num_candidates, planning_hor, d_latent)
                predictions = self.dynamics_model.predict(emb, action_samples, already_embedded=True)
                num_models, num_candidates, planning_hor, d_latent = predictions.shape

                last_states = predictions[:, :, -1, :].reshape(
                    (num_models * num_candidates, d_latent))
                #all_values = self.value_function.get_value(last_states, already_embedded=True) 
                all_values = self.compute_broil_weights(last_states) #use compute broil weights to compute values for cem candidates instead of value function
                nans = torch.isnan(all_values)
                all_values[nans] = -1e5
                values = torch.mean(all_values.reshape((num_models, num_candidates, 1)), dim=0)

                # Blow up cost for trajectories that are not constraint satisfying and/or don't end up
                #   in the safe set
                if not self.ignore_constraints:
                    constraints_all = torch.sigmoid(self.constraint_function(predictions, already_embedded=True))
                    constraint_viols = torch.sum(torch.max(constraints_all, dim=0)[0] > self.constraint_thresh, dim=1)
                else:
                    constraint_viols = torch.zeros((num_candidates, 1), device=ptu.TORCH_DEVICE)

                if not self.ignore_safe_set:
                    safe_set_all = self.safe_set.safe_set_probability(last_states, already_embedded=True)
                    safe_set_viols = torch.mean(safe_set_all
                                                .reshape((num_models, num_candidates, 1)),
                                                dim=0) < act_ss_thresh
                else:
                    safe_set_viols = torch.zeros((num_candidates, 1), device=ptu.TORCH_DEVICE)
                goal_preds = self.goal_indicator(predictions, already_embedded=True)
                goal_states = torch.sum(torch.mean(goal_preds, dim=0) > self.goal_thresh, dim=1)

                values = values + (constraint_viols + safe_set_viols) * -1e5 + goal_states
                values = values.squeeze()

            itr += 1

        # Return the best action
        action = actions_sorted[-1][0]
        return action.detach().cpu().numpy()

    def reset(self):
        # It's important to call this after each episode
        self.mean, self.std = None, None

    def _sample_actions_random(self, n=None):
        if n is None:
            n = self.popsize
        rand = torch.rand((n, self.plan_hor, self.d_act))
        scaled = rand * (self.ac_ub - self.ac_lb)
        action_samples = scaled + self.ac_lb
        return action_samples.to(ptu.TORCH_DEVICE)

    def _sample_actions_normal(self, mean, std, n=None):
        if n is None:
            n = self.popsize

        smp = torch.empty(n, self.plan_hor, self.d_act).normal_(
            mean=0, std=1).to(ptu.TORCH_DEVICE)
        mean = mean.unsqueeze(0).repeat(n, 1, 1).to(ptu.TORCH_DEVICE)
        std = std.unsqueeze(0).repeat(n, 1, 1).to(ptu.TORCH_DEVICE)

        # Sample new actions
        action_samples = smp * std + mean
        # TODO: Assuming action space is symmetric, true for maze and shelf for now
        action_samples = torch.clamp(
            action_samples,
            min=self.env.action_space.low[0],
            max=self.env.action_space.high[0])

        return action_samples

    def compute_broil_weights(self, states, weights=None):
        #for each candidate in states calculate the reward for that candidate with each reward hypothesis then calculate cvar for each candidate
        broil_lambda=0.5
        broil_alpha=0.95
        gamma=0.99
        broil_risk_metric = "cvar"
        expert_fcounts = None
        reward_distribution = PointBotReward()
        reward_dist = get_reward_distribution(reward_distribution) #hardcoded reward distribution
        '''batch_returns: list of numpy arrays of size num_rollouts x num_reward_fns
           weights: list of weights, e.g. advantages, rewards to go, etc by reward function over all rollouts,
            size is num_rollouts*ave_rollout_length x num_reward_fns
        '''
        #inputs are lists of numpy arrays
        #need to compute BROIL weights for policy gradient and convert to pytorch

        #first find the expected on-policy return for current policy under each reward function in the posterior
        # exp_batch_rets = np.mean(batch_rets.numpy(), axis=0)
        # posterior_reward_weights = reward_dist.posterior
        reward_hypotheses = [None] #either hardcoded or from demonstrations
        batch_rewards = np.zeros(list(states.shape).append(len(reward_hypotheses))) #(num_cand, horizon, ac_dim, num_reward_hypotheses)
        for cand in states:
            for i in reward_hypotheses:
                #TODO calculate reward for each cand under reward hypotheses i
                reward = None
                batch_rewards[:, :, :, i] = reward
        
        # if expert_fcounts is not None:
        #     reward_weights = reward_dist.get_posterior_weight_matrix()
        #     expert_returns = np.dot(reward_weights, expert_fcounts)
        #     exp_batch_rets -= expert_returns #baseline with what expert would have gotten under each reward function

        #TODO use cvar to calculate to compute a reward for each traj using the rewards from the difference hypotheses

        #calculate sigma and find either the conditional value at risk or entropic risk measure given the current policy
        if broil_risk_metric == "cvar":
            #Calculate policy gradient for conditional value at risk

            sigma, cvar = cvar_enumerate_pg(exp_batch_rets, posterior_reward_weights, broil_alpha) #TODO replace with rewards for each traj
           # print("sigma = {}, cvar = {}".format(sigma, cvar))

            #compute BROIL policy gradient weights
            total_rollout_steps = len(weights)
            broil_weights = np.zeros(total_rollout_steps, dtype=np.float64)
            for i, prob_r in enumerate(posterior_reward_weights):
                if sigma >= exp_batch_rets[i]:
                    w_r_i = broil_lambda + (1 - broil_lambda) / (1 - broil_alpha)
                else:
                    w_r_i = broil_lambda
                broil_weights += prob_r * w_r_i * np.array(weights)[:,i]

            return broil_values

        # elif broil_risk_metric == "erm":
        #     #calculate policy gradient for entropic risk measure
        #     erm = -1.0 / broil_alpha * np.log(np.dot(posterior_reward_weights, np.exp(-broil_alpha * exp_batch_rets)))

        #     #compute stable weighted soft-max
        #     exponents = -broil_alpha * exp_batch_rets
        #     z = exponents - max(exponents)
        #     numerators = np.exp(z)
        #     denominator = np.dot(numerators, posterior_reward_weights)
        #     softmax_probs = numerators / denominator

        #     #compute BROIL policy gradient weights for ERM
        #     total_rollout_steps = len(weights)
        #     erm_weights = broil_lambda * np.ones(len(posterior_reward_weights)) + (1-broil_lambda) * softmax_probs

        #     broil_weights = np.array(weights) * erm_weights
        #     broil_weights = np.dot(broil_weights, posterior_reward_weights)

        #     return broil_weights,erm

        else:
            print("Risk metric not implemented!")
            raise NotImplementedError

