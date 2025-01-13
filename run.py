import gymnasium as gym
from gymnasium import spaces
import numpy as np
import itertools
from scipy.stats import norm

import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
import numpy as np
import itertools
from scipy.stats import norm
from tqdm import tqdm

from mcts import TreeRootsWrapper

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--seed", default=0, type=int)

parser.add_argument("--height", default=20, type=int)
parser.add_argument("--ndim", default=4, type=int)

parser.add_argument("--mcts_rollouts", default=4, type=int)
parser.add_argument("--mcts_eps", default=0.01, type=float)

parser.add_argument("--algo", default='mcts_all', type=str) # no_mcts / mcts_all / mcts_train / mcts_inference



def Reward(state, side):
    ax = abs(state / (side - 1) - 0.5)
    return (ax > 0.25).prod(-1) * 0.5 + ((ax < 0.4) * (ax > 0.3)).prod(-1) * 2 + 1e-3

class HyperGridEnv(gym.Env):
    def __init__(self, dim=4, side=8, target_prob=Reward):
        self.dim = dim
        self.side = side
        self.target_prob = target_prob

        self.observation_space = spaces.MultiBinary(self.dim * self.side)
        self.action_space = spaces.Discrete(self.dim + 1,)
        
        self._state = np.array([0] * self.dim, dtype=np.int32)
        self._is_done = False
        
    def _get_info(self):
        return { "state": self._state }
    
    def _get_obs(self):
        state_ohe = np.zeros((self.dim * self.side), dtype=np.float32)
        state_ohe[np.arange(self.dim) * self.side + self._state] = 1.0
        return state_ohe
    
    def get_real_state(self):
        return self._state
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._is_done = False
        self._state = np.array([0] * self.dim, dtype=np.int32)

        observation = self._get_obs()
        info = self._get_info()

        return observation, info
    
    def step(self, action):
        if self._is_done:
            return self._get_obs(), 0.0, True, False, self._get_info()
        if action == self.dim or self._state[action] >= self.side - 1:
            self._is_done = True
            return self._get_obs(), np.log(self.target_prob(self._state, self.side)), True, False, self._get_info()
        else:
            self._state[action] += 1
            
            n_parents = sum([val != 0 for val in self._state])
            return self._get_obs(), np.log(1 / n_parents), False, False, self._get_info()
        
    def compute_true_distribution(self):
        probs = np.zeros((self.side,) * self.dim)
        
        for state in itertools.product(list(range(self.side)), repeat=self.dim):
            state = np.array(state)
            probs[tuple(state)] = self.target_prob(state, self.side)
            
        return probs / probs.sum()

def compute_empirical_distribution_error(true_dist, samples):
    empirical_dist = true_dist * 0.0
    for i in samples:
        empirical_dist[tuple(i)] += 1.0
    empirical_dist /= empirical_dist.sum()
    l1 = np.abs(empirical_dist - true_dist).mean().item()
    kl = (true_dist * np.log(true_dist / (empirical_dist + 1e-9))).sum().item()
    return l1, kl


class SimpleQLearningAgent():
    def __init__(self, env, hidden_size=256, use_q_target=True, update_target_every=100):
        self.env = env
        
        self.q = nn.Sequential(
            nn.Linear(env().dim * env().side, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, env().dim + 1)
        )
        
        self.use_q_target = use_q_target
        self.update_target_every = update_target_every
        if self.use_q_target:
            self.q_target = nn.Sequential(
                nn.Linear(env().dim * env().side, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, env().dim + 1)
            )
            self.q_target.load_state_dict(self.q.state_dict())
            
        
    def train(self, steps, true_dist, batch_size=16, lr=1e-3):
        envs = [self.env() for i in range(batch_size)]
        
        opt = torch.optim.Adam(self.q.parameters(), lr=lr)

        all_samples = []
        l1_history = []
        kl_history = []

        for it in tqdm(range(steps)):
            opt.zero_grad()

            states_ohe = torch.tensor([envs[i].reset()[0] for i in range(batch_size)], dtype=torch.float32)
            dones = [False] * batch_size
            loss = 0.0
            infos = [envs[i]._get_info() for i in range(batch_size)]
            total = 0

            while not all(dones):
                logits = self.q(states_ohe)
                with torch.no_grad():
                    mask = torch.tensor([list(v["state"]) + [0] for v in infos], dtype=torch.int32)
                    mask = (mask == envs[0].side - 1)
                    logits[mask] = -float("inf")
                    actions = Categorical(logits=logits).sample()#.squeeze()

                results = [envs[i].step(actions[i]) for i in range(batch_size)]
                next_states_ohe = torch.tensor([res[0] for res in results], dtype=torch.float32)
                rewards = torch.tensor([res[1] for res in results])
                dones_next = [res[2] for res in results]
                infos = [res[4] for res in results]

                with torch.no_grad():
                    model = self.q_target if self.use_q_target else self.q
                    next_logits = model(next_states_ohe)
                    mask = torch.tensor([list(v["state"]) + [0] for v in infos], dtype=torch.int32)
                    mask = (mask == envs[0].side - 1)
                    next_logits[mask] = -float("inf")
                    
                    next_mask = (1 - torch.tensor(dones_next, dtype=torch.float32))
                    y = rewards + next_mask * torch.logsumexp(next_logits, dim=-1)
                    
                mask = (1 - torch.tensor(dones, dtype=torch.float32))
                batch_losses = (logits[range(batch_size), actions] - y).pow(2) * mask
                #print(batch_losses.shape)
                loss += batch_losses.sum()
                total += mask.sum()

                states_ohe = next_states_ohe
                dones = dones_next

            # Adds information about the last state
            for info in infos:
                all_samples.append(info["state"])

            loss = loss / total
            loss.backward()
            opt.step()

            if (it + 1) % 100 == 0:
                l1, kl = compute_empirical_distribution_error(true_dist, all_samples[-200000:])
                print(f"states visited: {(it + 1) * batch_size}, empirical L1 distance: {l1}, KL: {kl}")
                
                l1_history.append(l1)
                kl_history.append(kl)

                np.save(f"new_results/{args.seed}_standard_{self.env().dim}_{self.env().side}_SDQN_uniform-pb_l1.npy", np.array(l1_history))
                
            if self.use_q_target and (it + 1) % self.update_target_every == 0:
                self.q_target.load_state_dict(self.q.state_dict())
                
        return l1_history, kl_history

class QLearningMCTSAgentCPP():
    def __init__(self, env, hidden_size=256, use_q_target=True, update_target_every=100, tree_max_size=4):
        self.env = env
        
        self.q = nn.Sequential(
            nn.Linear(env().dim * env().side, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, env().dim + 1)
        )
        
        self.use_q_target = use_q_target
        self.update_target_every = update_target_every
        if self.use_q_target:
            self.q_target = nn.Sequential(
                nn.Linear(env().dim * env().side, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, env().dim + 1)
            )
            self.q_target.load_state_dict(self.q.state_dict())
            
        self.tree_max_size = tree_max_size
            
        
    def train(self, steps, true_dist, batch_size=16, lr=1e-3, expl_eps=0.05, mcts_for_inf=True):
        envs = [self.env() for i in range(batch_size)]
        
        opt = torch.optim.Adam(self.q.parameters(), lr=lr)

        all_samples = []
        l1_history = []
        kl_history = []
        
        visited_terminals = 0

        for it in tqdm(range(steps)):
            opt.zero_grad()

            states_ohe = torch.tensor([envs[i].reset()[0] for i in range(batch_size)], dtype=torch.float32)
            dones = [False] * batch_size
            loss = 0.0
            infos = [envs[i]._get_info() for i in range(batch_size)]
            total = 0
            
            roots = TreeRootsWrapper(envs[0].dim, envs[0].side, expl_eps, batch_size, 
                                     self.tree_max_size, [info["state"] for info in infos])

            with torch.no_grad():
                preds = self.q_target(states_ohe)
                roots.set_root_q_values(preds.detach())

            while not all(dones):
                with torch.no_grad():
                    for _ in range(self.tree_max_size):
                        leaves, ohes = roots.expand(dones)
                        if leaves.size() == 0:
                            continue
                            
                        preds = self.q_target(torch.tensor(ohes, dtype=torch.float32))
                        
                        roots.backup(leaves, preds.detach())
                
                # Calculate loss using MCTS predicted q values
                preds = self.q(states_ohe)
                mcts_q_values = torch.tensor(roots.get_root_q_values(), dtype=torch.float32)
                
                mask_invalid = torch.tensor([list(v["state"]) + [0] for v in infos], dtype=torch.int32)
                mask_invalid = (mask_invalid == envs[0].side - 1)
                preds[mask_invalid] = 0
                mcts_q_values[mask_invalid] = 0
                
                mask_not_done = (1 - torch.tensor(dones, dtype=torch.float32))
                batch_losses = (preds - mcts_q_values).pow(2).mean(dim = 1) * mask_not_done
                loss += batch_losses.sum()
                total += mask_not_done.sum()
                
                # Update states
                if mcts_for_inf:
                    logits = mcts_q_values
                else:
                    logits = preds
                with torch.no_grad():
                    logits[mask_invalid] = -float("inf")
                    actions = Categorical(logits=logits).sample()#.squeeze()

                results = [envs[i].step(actions[i]) for i in range(batch_size)]
                next_states_ohe = torch.tensor([res[0] for res in results], dtype=torch.float32)
                rewards = torch.tensor([res[1] for res in results])
                dones_next = [res[2] for res in results]
                infos = [res[4] for res in results]
                
                new_dones = (np.array(dones_next, dtype=np.int32) - np.array(dones, dtype=np.int32)) == 1
                loss += ((preds[:, -1] - rewards).pow(2) * torch.tensor(new_dones, dtype=torch.float32)).sum()
                total += new_dones.sum()
                visited_terminals += new_dones.sum()

                states_ohe = next_states_ohe
                dones = dones_next
                
                
                # Update tree roots
                with torch.no_grad():
                    preds_next = self.q_target(states_ohe)
                    roots.move_to_children(actions, dones, [info["state"] for info in infos], preds_next.detach())

            # Adds information about the last state
            for info in infos:
                all_samples.append(info["state"])

            loss = loss / total
            loss.backward()
            opt.step()

            if (it + 1) % 100 == 0:
                l1, kl = compute_empirical_distribution_error(true_dist, all_samples[-200000:])
                print(f"terminal states visited: {visited_terminals}, empirical L1 distance: {l1}, KL: {kl}")
                
                l1_history.append(l1)
                kl_history.append(kl)

                algtype = 'MCTSALL' if mcts_for_inf else 'MCTSTRAIN'
                np.save(f"new_results/{args.seed}_standard_{self.env().dim}_{self.env().side}_{algtype}_SDQN_tree{self.tree_max_size}_eps{expl_eps}_uniform-pb_l1.npy", np.array(l1_history))
                
            if self.use_q_target and (it + 1) % self.update_target_every == 0:
                self.q_target.load_state_dict(self.q.state_dict())
                
        return l1_history, kl_history

class QLearningMCTSInfOnlyAgentCPP():
    def __init__(self, env, hidden_size=256, use_q_target=True, update_target_every=100, tree_max_size=4):
        self.env = env
        
        self.q = nn.Sequential(
            nn.Linear(env().dim * env().side, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, env().dim + 1)
        )
        
        self.use_q_target = use_q_target
        self.update_target_every = update_target_every
        if self.use_q_target:
            self.q_target = nn.Sequential(
                nn.Linear(env().dim * env().side, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, env().dim + 1)
            )
            self.q_target.load_state_dict(self.q.state_dict())
            
        self.tree_max_size = tree_max_size
            
        
    def train(self, steps, true_dist, batch_size=16, lr=1e-3, expl_eps=0.05):
        envs = [self.env() for i in range(batch_size)]
        
        opt = torch.optim.Adam(self.q.parameters(), lr=lr)

        all_samples = []
        l1_history = []
        kl_history = []
        
        visited_terminals = 0

        for it in tqdm(range(steps)):
            opt.zero_grad()

            states_ohe = torch.tensor([envs[i].reset()[0] for i in range(batch_size)], dtype=torch.float32)
            dones = [False] * batch_size
            loss = 0.0
            infos = [envs[i]._get_info() for i in range(batch_size)]
            total = 0
            
            roots = TreeRootsWrapper(envs[0].dim, envs[0].side, expl_eps, batch_size, 
                                     self.tree_max_size, [info["state"] for info in infos])

            with torch.no_grad():
                preds = self.q_target(states_ohe)
                roots.set_root_q_values(preds.detach())

            while not all(dones):
                with torch.no_grad():
                    for _ in range(self.tree_max_size):
                        leaves, ohes = roots.expand(dones)
                        if leaves.size() == 0:
                            continue
                            
                        preds = self.q_target(torch.tensor(ohes, dtype=torch.float32))
                        
                        roots.backup(leaves, preds.detach())
                
                # Calculate loss using MCTS predicted q values
                preds = self.q(states_ohe)
                mcts_q_values = torch.tensor(roots.get_root_q_values(), dtype=torch.float32)
                
                # Update states
                logits = mcts_q_values
                #logits = preds
                with torch.no_grad():
                    mask = torch.tensor([list(v["state"]) + [0] for v in infos], dtype=torch.int32)
                    mask = (mask == envs[0].side - 1)
                    logits[mask] = -float("inf")
                    actions = Categorical(logits=logits).sample()#.squeeze()

                results = [envs[i].step(actions[i]) for i in range(batch_size)]
                next_states_ohe = torch.tensor([res[0] for res in results], dtype=torch.float32)
                rewards = torch.tensor([res[1] for res in results])
                dones_next = [res[2] for res in results]
                infos = [res[4] for res in results]
                
                with torch.no_grad():
                    model = self.q_target if self.use_q_target else self.q
                    next_logits = model(next_states_ohe)
                    mask = torch.tensor([list(v["state"]) + [0] for v in infos], dtype=torch.int32)
                    mask = (mask == envs[0].side - 1)
                    next_logits[mask] = -float("inf")
                    
                    next_mask = (1 - torch.tensor(dones_next, dtype=torch.float32))
                    y = rewards + next_mask * torch.logsumexp(next_logits, dim=-1)
                    
                mask = (1 - torch.tensor(dones, dtype=torch.float32))
                batch_losses = (preds[range(batch_size), actions] - y).pow(2) * mask
                #print(batch_losses.shape)
                loss += batch_losses.sum()
                total += mask.sum()

                states_ohe = next_states_ohe
                dones = dones_next
                
                #new_dones = (np.array(dones_next, dtype=np.int32) - np.array(dones, dtype=np.int32)) == 1
                #loss += ((preds[:, -1] - rewards).pow(2) * torch.tensor(new_dones, dtype=torch.float32)).sum()
                #total += new_dones.sum()
                #visited_terminals += new_dones.sum()

                #states_ohe = next_states_ohe
                #dones = dones_next
                
                
                # Update tree roots
                with torch.no_grad():
                    preds_next = self.q_target(states_ohe)
                    roots.move_to_children(actions, dones, [info["state"] for info in infos], preds_next.detach())

            # Adds information about the last state
            for info in infos:
                all_samples.append(info["state"])

            loss = loss / total
            loss.backward()
            opt.step()

            if (it + 1) % 100 == 0:
                l1, kl = compute_empirical_distribution_error(true_dist, all_samples[-200000:])
                print(f"terminal states visited: {(it + 1) * batch_size}, empirical L1 distance: {l1}, KL: {kl}")
                
                l1_history.append(l1)
                kl_history.append(kl)

                np.save(f"new_results/{args.seed}_standard_{self.env().dim}_{self.env().side}_MCTSINF_SDQN_tree{self.tree_max_size}_eps{expl_eps}_uniform-pb_l1.npy", np.array(l1_history))
                
            if self.use_q_target and (it + 1) % self.update_target_every == 0:
                self.q_target.load_state_dict(self.q.state_dict())
                
        return l1_history, kl_history

def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    env = lambda : HyperGridEnv(dim=args.ndim, side=args.height)
    true_dist = env().compute_true_distribution()

    if args.algo == "no_mcts":
        trainer = SimpleQLearningAgent(env, hidden_size=256, use_q_target=True, update_target_every=3)
        l1, kl = trainer.train(steps=1000000//16, batch_size=16, lr=1e-3, true_dist=true_dist)

    elif args.algo == "mcts_all":
        trainer = QLearningMCTSAgentCPP(env, hidden_size=256, use_q_target=True, update_target_every=3, tree_max_size=args.mcts_rollouts)
        l1, kl = trainer.train(steps=1000000//16, batch_size=16, lr=1e-3, true_dist=true_dist, expl_eps=args.mcts_eps, mcts_for_inf=True)
        
    elif args.algo == "mcts_train":
        trainer = QLearningMCTSAgentCPP(env, hidden_size=256, use_q_target=True, update_target_every=3, tree_max_size=args.mcts_rollouts)
        l1, kl = trainer.train(steps=1000000//16, batch_size=16, lr=1e-3, true_dist=true_dist, expl_eps=args.mcts_eps, mcts_for_inf=False)
        
    elif args.algo == "mcts_inference":
        trainer = QLearningMCTSInfOnlyAgentCPP(env, hidden_size=256, use_q_target=True, update_target_every=3, tree_max_size=args.mcts_rollouts)
        l1, kl = trainer.train(steps=1000000//16, batch_size=16, lr=1e-3, true_dist=true_dist, expl_eps=args.mcts_eps)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)