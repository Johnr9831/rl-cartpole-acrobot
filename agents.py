"""Q-learning tabular y DQN; implementaciones propias para estudiar."""
import numpy as np
import torch
from torch import nn

class Tabular:
    def __init__(self, obs, actions, cfg):
        self.edges = [np.linspace(-r, r, cfg['bins']+1)[1:-1]
                      for r in [2.4, 3.0, 0.2095, 3.5]]
        self.q = np.zeros((cfg['bins'],)*4 + (actions,), dtype=np.float32)
        self.cfg = cfg
    def index(self, state):
        return tuple(np.digitize(x, edge) for x, edge in zip(state, self.edges))
    def act(self, state):
        return int(self.q[self.index(state)].argmax())
    def learn(self, s, a, r, nxt, terminal, step):
        idx = self.index(s)+(a,)
        target = r + self.cfg['gamma']*(1-terminal)*self.q[self.index(nxt)].max()
        self.q[idx] += self.cfg['alpha']*(target-self.q[idx])
    def save(self, path):
        np.save(path/'q.npy', self.q)
    def load(self, path):
        self.q = np.load(path/'q.npy')

class DQN:
    def __init__(self, obs, actions, cfg):
        self.cfg = cfg
        self.device = torch.device(cfg['device'])
        def network():
            return nn.Sequential(nn.Linear(obs,64),nn.ReLU(),nn.Linear(64,64),
                                 nn.ReLU(),nn.Linear(64,actions)).to(self.device)
        self.net, self.target = network(), network()
        self.target.load_state_dict(self.net.state_dict())
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=cfg['lr'])
        n = cfg['buffer']
        self.states = np.zeros((n,obs), np.float32)
        self.next_states = np.zeros_like(self.states)
        self.actions = np.zeros(n,np.int64)
        self.rewards = np.zeros(n,np.float32)
        self.terminals = np.zeros(n,np.float32)
        self.count = 0
    def act(self, state):
        with torch.no_grad():
            x = torch.as_tensor(state,dtype=torch.float32,device=self.device)
            return int(self.net(x).argmax().item())
    def learn(self, s, a, r, nxt, terminal, step):
        c = self.cfg
        i = self.count % c['buffer']
        self.states[i], self.next_states[i] = s, nxt
        self.actions[i], self.rewards[i], self.terminals[i] = a,r,terminal
        self.count += 1
        if step >= c['warmup'] and step % c['train_every'] == 0:
            ids = np.random.choice(min(self.count,c['buffer']),c['batch'],replace=False)
            def tensor(x):
                return torch.as_tensor(x[ids],device=self.device)
            q = self.net(tensor(self.states)).gather(1,tensor(self.actions)[:,None]).squeeze(1)
            with torch.no_grad():
                target_net = self.net if c['no_target'] else self.target
                y = tensor(self.rewards) + c['gamma']*(1-tensor(self.terminals))*target_net(tensor(self.next_states)).max(1).values
            loss = nn.functional.smooth_l1_loss(q,y)
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.net.parameters(),10)
            self.optimizer.step()
        if not c['no_target'] and step % c['target_every'] == 0:
            self.target.load_state_dict(self.net.state_dict())
    def save(self,path):
        torch.save(self.net.state_dict(),path/'model.pt')
    def load(self,path):
        self.net.load_state_dict(torch.load(path/'model.pt',map_location=self.device,weights_only=True))
