import argparse,json
from pathlib import Path
import gymnasium as gym
from PIL import Image
from agents import Tabular,DQN
p=argparse.ArgumentParser(); p.add_argument('run_folder'); a=p.parse_args(); path=Path(a.run_folder)
c=json.loads((path/'config.json').read_text()); c['device']='cpu'
env=gym.make(c['env'],render_mode='rgb_array')
agent=(Tabular if c['algorithm']=='tabular' else DQN)(env.observation_space.shape[0],env.action_space.n,c); agent.load(path)
s,_=env.reset(seed=777777); frames=[]; total=0
while True:
    frames.append(Image.fromarray(env.render()).resize((400,300)))
    s,r,term,trunc,_=env.step(agent.act(s)); total+=r
    if term or trunc: break
frames[0].save(path/'agent.gif',save_all=True,append_images=frames[1:],duration=round(1000/env.metadata['render_fps']),loop=0)
env.close(); print('Retorno del episodio ilustrativo:',total,'GIF:',path/'agent.gif')
