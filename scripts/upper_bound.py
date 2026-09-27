"""Upper bound on macro F0.5 from records that carry no address and whose owner's name is shared by
other Source 1 entities of the same country: nothing in such a record says which entity owns it.

    python scripts/upper_bound.py --train-dir <student_resource/dataset/train>
"""
import argparse
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--train-dir", required=True)
D = ap.parse_args().train_dir.rstrip("/") + "/"
rd=lambda f: pd.read_csv(D+f,sep='\t',dtype=str,keep_default_na=False,quoting=3)
s1=rd('train_source1.tsv'); rec=pd.concat([rd('train_source2.tsv'),rd('train_source3.tsv')],ignore_index=True); gt=rd('train_ground_truth.tsv')
pairs=gt.assign(m=gt.matched_entity_ids.str.split(',')).explode('m'); pairs=pairs[pairs.m!='']
owner=pd.Series(pairs.source1_entity_id.values,index=pairs.m.values)
rec['owner']=owner.reindex(rec.entity_id).values
simp=lambda s: s.str.lower().str.replace(r'[^0-9a-z]+','',regex=True)
s1['sn']=simp(s1.business_name)+'|'+s1.country
cnt=s1.sn.value_counts()
S=s1.set_index('entity_id')
m=rec[rec.owner.notna() & (rec.business_address.str.strip()=='')].copy()
m['sn_rec']=simp(m.business_name)+'|'+m.country
m['sn_own']=S.sn.reindex(m.owner).values
m['k']=m.sn_own.map(cnt)
m['exact']=m.sn_rec==m.sn_own
print('addr-less matched recs', len(m))
print('owner name shared (k>1):', (m.k>1).mean(), 'exact copies:', m.exact.mean(), 'exact & k>1:', (m.exact&(m.k>1)).mean())
# oracle upper bound: drop ambiguous (exact name & k>1); also for non-exact ones with k>1 the record is at least as ambiguous
amb=m[(m.k>1)]
nt=gt.set_index('source1_entity_id').matched_entity_ids.str.count('S')
lost=amb.groupby('owner').size()
T=nt.reindex(lost.index).to_numpy(); L=lost.to_numpy()
tp=T-L; f=np.where(tp>0,1.25*tp/(0.25*T+tp),0)
loss=(1-f).sum()/len(gt)
print('upper-bound loss from addr-less records with shared owner name: %.5f -> max macro F0.5 %.5f'%(loss,1-loss))
amb2=m[(m.k>1)&m.exact]; lost=amb2.groupby('owner').size(); T=nt.reindex(lost.index).to_numpy(); L=lost.to_numpy(); tp=T-L
f=np.where(tp>0,1.25*tp/(0.25*T+tp),0); print('  only exact-name copies: loss %.5f'%((1-f).sum()/len(gt)))
