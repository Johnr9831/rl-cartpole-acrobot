import argparse,json,csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser(); p.add_argument('--root',default='results'); a=p.parse_args()
root=Path(a.root); dest=root/'figures'; dest.mkdir(exist_ok=True)
groups={'tier1':['cart_bins6','cart_bins10','cart_bins20','cart_dqn'],
        'learning_rate':['ac_lr_low','ac_base','ac_lr_high'],
        'exploration':['ac_decay_short','ac_base','ac_decay_long'],
        'ablation':['ac_base','ac_no_target']}
rows=[]
for title,names in groups.items():
    fig,ax=plt.subplots(figsize=(8,5))
    for name in names:
        paths=sorted((root/name).glob('seed_*/summary.json'))
        if len(paths)<3: print('Omitido (requiere 3 semillas completas):',name); continue
        data=[np.genfromtxt(x.parent/'evaluation.csv',delimiter=',',names=True) for x in paths]
        xs=data[0]['step']
        if any(not np.array_equal(xs,d['step']) for d in data): raise ValueError('Presupuestos desiguales')
        ys=np.array([d['return_mean'] for d in data]); mean=ys.mean(0); sd=ys.std(0,ddof=1)
        line,=ax.plot(xs,mean,label=name); ax.fill_between(xs,mean-sd,mean+sd,alpha=.15,color=line.get_color())
    ax.set(xlabel='Pasos de entrenamiento',ylabel='Retorno de evaluación (media ± DE entre semillas)',title=title)
    ax.axhline(475 if title=='tier1' else -100,color='gray',ls='--'); ax.legend(); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(dest/f'{title}.png',dpi=180); plt.close(fig)
for folder in sorted(root.iterdir()):
    paths=sorted(folder.glob('seed_*/summary.json')) if folder.is_dir() else []
    if not paths: continue
    ss=[json.loads(p.read_text()) for p in paths]
    v=[s['final_mean'] for s in ss]
    rows.append([folder.name,len(ss),np.mean(v),np.std(v,ddof=1) if len(v)>1 else '',
                 np.mean([s['wall_seconds'] for s in ss]),sum(s['success'] for s in ss),
                 ';'.join(str(s['first_threshold_10ep']) for s in ss)])
with (root/'summary.csv').open('w',newline='') as f:
    w=csv.writer(f); w.writerow(['experiment','seeds','final_mean','std_between_seeds','mean_wall_seconds','successful_seeds','first_threshold_10ep_per_seed']); w.writerows(rows)
print('Gráficos y tabla:',root)
