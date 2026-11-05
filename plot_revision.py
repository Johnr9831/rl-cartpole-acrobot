"""Ejecutar junto a results/ y results_diagnostico/: python plot_revision.py."""
from pathlib import Path
import csv,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path('results'); diag=Path('results_diagnostico'); out=Path('resultados_revisados');out.mkdir(exist_ok=True)
groups={
 'cartpole_comparacion':[(root,f'cart_bins{b}',f'Q-learning {b} divisiones') for b in [6,10,20]]+[(root,'cart_dqn','DQN original'),(diag,'cart_dqn_lr0001_target10000','DQN ajustado')],
 'cartpole_diagnostico':[(root,'cart_dqn','Original (3 semillas)'),(diag,'cart_dqn_lr0001','Solo lr reducido (1 semilla)'),(diag,'cart_dqn_lr0001_target10000','lr reducido + objetivo cada 10000 (3 semillas)')],
 'acrobot_tasa':[(root,n,l) for n,l in [('ac_lr_low','lr=0.0003'),('ac_base','lr=0.001'),('ac_lr_high','lr=0.003')]],
 'acrobot_exploracion':[(root,n,l) for n,l in [('ac_decay_short','75 000 pasos'),('ac_base','150 000 pasos'),('ac_decay_long','225 000 pasos')]],
 'acrobot_ablacion':[(root,'ac_base','Con red objetivo'),(root,'ac_no_target','Sin red objetivo')]}
for title,items in groups.items():
 fig,ax=plt.subplots(figsize=(9,5.5))
 for base,name,label in items:
  paths=sorted((base/name).glob('seed_*/summary.json'))
  expected=1 if name=='cart_dqn_lr0001' else 3
  if len(paths)!=expected:raise ValueError(f'{name}: se esperaban {expected} semillas')
  data=[np.genfromtxt(p.parent/'evaluation.csv',delimiter=',',names=True) for p in paths]
  xs=data[0]['step']
  if any(not np.array_equal(xs,d['step']) for d in data):raise ValueError('Pasos desiguales')
  ys=np.array([d['return_mean'] for d in data]);mean=ys.mean(0)
  line,=ax.plot(xs,mean,label=label)
  if len(paths)>1:
   sd=ys.std(0,ddof=1);ax.fill_between(xs,mean-sd,mean+sd,color=line.get_color(),alpha=.12)
 ax.axhline(475 if title.startswith('cartpole') else -100,color='gray',ls='--',label='Umbral de referencia')
 ax.set(xlabel='Pasos de entrenamiento',ylabel='Retorno de evaluación',title=title.replace('_',' ').capitalize())
 ax.legend(fontsize=8);ax.grid(alpha=.2)
 fig.text(.5,.012,'Media ± desviación estándar entre semillas; sin banda para diagnóstico de una semilla.',ha='center',fontsize=8)
 fig.tight_layout(rect=(0,.035,1,1));fig.savefig(out/f'{title}.png',dpi=180);plt.close(fig)
with (out/'resumen_completo.csv').open('w',newline='',encoding='utf-8') as f:
 w=csv.writer(f);w.writerow(['experiment','seeds','final_mean','std_between_seeds','successful_seeds','mean_wall_seconds','scope'])
 for base in [root,diag]:
  for folder in sorted(base.iterdir()):
   if not folder.is_dir():continue
   paths=sorted(folder.glob('seed_*/summary.json'))
   if not paths:continue
   ds=[json.loads(p.read_text()) for p in paths];v=[d['final_mean'] for d in ds]
   w.writerow([folder.name,len(ds),np.mean(v),np.std(v,ddof=1) if len(v)>1 else '',sum(d['success'] for d in ds),np.mean([d['wall_seconds'] for d in ds]),'diagnostico_una_semilla' if len(v)==1 else 'tres_semillas'])
print('Gráficos y tabla: resultados_revisados')
