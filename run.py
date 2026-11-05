"""Ejecutar desde la carpeta del proyecto. Los resultados completos se omiten al reiniciar."""
import os, argparse, csv, json, random, time, platform, subprocess, sys
from pathlib import Path
import gymnasium as gym
import numpy as np
import torch
from agents import Tabular, DQN

def configs():
    base=dict(gamma=.99,alpha=.1,lr=.001,buffer=50000,batch=64,warmup=1000,
              train_every=4,target_every=1000,no_target=False,decay=.5,device='cpu',
              steps=200000,eval_every=10000,env='CartPole-v1',algorithm='dqn',bins=10)
    out={f'cart_bins{b}':dict(base,algorithm='tabular',bins=b) for b in [6,10,20]}
    out['cart_dqn']=dict(base)
    ac=dict(base,env='Acrobot-v1',steps=300000)
    out.update(ac_base=ac,ac_lr_low=dict(ac,lr=.0003),ac_lr_high=dict(ac,lr=.003),
               ac_decay_short=dict(ac,decay=.25),ac_decay_long=dict(ac,decay=.75),
               ac_no_target=dict(ac,no_target=True))
    return out

def evaluate(agent,env,seed,count):
    values=[]
    for ep in range(count):
        s,_=env.reset(seed=seed+ep); total=0
        while True:
            s,r,term,trunc,_=env.step(agent.act(s)); total+=r
            if term or trunc: break
        values.append(total)
    return values

def run(name,c,seed,root,smoke):
    path=root/name/f'seed_{seed}'; path.mkdir(parents=True,exist_ok=True)
    if (path/'summary.json').exists():
        old=json.loads((path/'config.json').read_text())
        if old != c: raise ValueError('Configuración cambiada: usa otra carpeta --out')
        print('Ya completado:',path,flush=True); return
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.set_num_threads(1)
    env=gym.make(c['env']); ev=gym.make(c['env']); env.action_space.seed(seed)
    agent=(Tabular if c['algorithm']=='tabular' else DQN)(env.observation_space.shape[0],env.action_space.n,c)
    (path/'config.json').write_text(json.dumps(c,indent=2))
    start=time.perf_counter(); eval_seconds=0; s,_=env.reset(seed=seed); episode_return=0; episode=0
    threshold=475 if c['env']=='CartPole-v1' else -100
    first=None
    with (path/'training.csv').open('w',newline='') as tf, (path/'evaluation.csv').open('w',newline='') as ef:
        tw=csv.writer(tf); ew=csv.writer(ef)
        tw.writerow(['step','episode','return','epsilon']); ew.writerow(['step','return_mean','return_std'])
        for step in range(c['steps']+1):
            if step % c['eval_every']==0 or step==c['steps']:
                t=time.perf_counter(); vals=evaluate(agent,ev,100000+seed*1000,2 if smoke else 10)
                eval_seconds+=time.perf_counter()-t
                ew.writerow([step,np.mean(vals),np.std(vals,ddof=1)]); ef.flush(); os.fsync(ef.fileno())
                if np.mean(vals)>=threshold and first is None: first=step
                print(name,seed,step,'eval',round(float(np.mean(vals)),1),flush=True)
            if step==c['steps']: break
            epsilon=1-.95*min(step/(c['steps']*c['decay']),1)
            a=env.action_space.sample() if random.random()<epsilon else agent.act(s)
            nxt,r,term,trunc,_=env.step(a)
            # Solo terminal real bloquea bootstrap; un límite temporal no lo bloquea.
            agent.learn(s,a,r,nxt,float(term),step+1)
            s=nxt; episode_return+=r
            if term or trunc:
                tw.writerow([step+1,episode,episode_return,epsilon]); tf.flush(); os.fsync(tf.fileno())
                episode+=1; episode_return=0; s,_=env.reset()
    agent.save(path)
    t=time.perf_counter(); final=evaluate(agent,ev,900000+seed*1000,2 if smoke else 100)
    eval_seconds+=time.perf_counter()-t
    total=time.perf_counter()-start
    summary=dict(final_mean=float(np.mean(final)),final_std=float(np.std(final,ddof=1)),
                 final_returns=final,threshold=threshold,success=bool(np.mean(final)>=threshold),
                 first_threshold_10ep=first,wall_seconds=total,eval_seconds=eval_seconds,
                 train_seconds=total-eval_seconds,smoke=smoke,seed=seed)
    (path/'summary.json').write_text(json.dumps(summary,indent=2))
    env.close(); ev.close()

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--suite',choices=['all','tier1','tier2'],default='all')
    p.add_argument('--experiment',choices=list(configs())); p.add_argument('--seeds',nargs='+',type=int,default=[11,22,33])
    p.add_argument('--smoke',action='store_true'); p.add_argument('--out'); p.add_argument('--device',default='cpu')
    a=p.parse_args(); root=Path(a.out or ('results_smoke' if a.smoke else 'results')); root.mkdir(exist_ok=True,parents=True)
    meta=dict(platform=platform.platform(),processor=platform.processor(),python=sys.version,
              torch=torch.__version__,gymnasium=gym.__version__,device=a.device,
              gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
    (root/'hardware.json').write_text(json.dumps(meta,indent=2))
    (root/'requirements_lock.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
    for name,c in configs().items():
        if a.experiment and name!=a.experiment: continue
        if not a.experiment and a.suite=='tier1' and not name.startswith('cart'): continue
        if not a.experiment and a.suite=='tier2' and not name.startswith('ac'): continue
        c=dict(c,device=a.device)
        if a.smoke: c.update(steps=1200,eval_every=600,warmup=100)
        for seed in a.seeds: run(name,c,seed,root,a.smoke)
