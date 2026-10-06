import json
from collections import defaultdict
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent
m=json.loads((HERE/'ukk_mineralization_mapped.json').read_text());pipe_dir={p:z['direction'] for z in m for p in z['pipes']}
observed=defaultdict(set)
for z in m:observed[z['direction']].add(z['date'])
a=pd.read_csv(HERE/'daily_min_active_patch.csv',dtype={'id':str,'date':str})
assert len(a)==71460 and a.id.nunique()==66 and a.Min.gt(0).all()
rows=[]
for pid,g in a.groupby('id',sort=True):
 direction=pipe_dir.get(pid);obs=observed.get(direction,set())
 rows.append({'ID простого участка':pid,'Источник':'УКК своего направления' if direction else 'Суточная медиана 12 труб',
              'ID направления с замерами':direction or '',
              'Рабочих дат':len(g),'Дат прямых замеров направления':len(obs),
              'Min минимум, г/л':round(g.Min.min(),6),'Min медиана, г/л':round(g.Min.median(),6),
              'Min максимум, г/л':round(g.Min.max(),6),'Уникальных значений Min (6 знаков)':g.Min.round(6).nunique()})
pd.DataFrame(rows).to_csv(HERE/'Минерализация_по_трубам.csv',index=False,encoding='utf-8-sig')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11})
fig,ax=plt.subplots(1,2,figsize=(13,4.8),facecolor='#f8f7f3')
for q in ax:q.set_facecolor('#f8f7f3');q.spines[['top','right']].set_visible(False)
ax[0].bar([22.6627],[len(a)],width=.8,color='#153546');ax[0].set_xlim(10,50);ax[0].set_title('До: одно значение на все даты',loc='left',fontweight='bold');ax[0].set_xlabel('Min, г/л')
ax[1].hist(a.Min,bins=np.linspace(10,50,81),color='#157b7d',edgecolor='white',linewidth=.2);ax[1].set_title('После: УКК + интерполяция + суточная медиана',loc='left',fontweight='bold');ax[1].set_xlabel('Min, г/л')
for q in ax:q.set_ylabel('Число рабочих трубо-дат')
fig.suptitle('Мегион: распределение минерализации до и после исправления',x=.03,ha='left',fontsize=16,fontweight='bold',color='#153546')
fig.tight_layout(rect=(0,0,1,.9));fig.savefig(HERE/'min_before_after_active_days.png',dpi=165,facecolor='#f8f7f3');plt.close(fig)
print('PASS',len(rows),'pipes',len(a),'active pipe-days; chart created')
