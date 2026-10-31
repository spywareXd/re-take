import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import random, re, json
from difflib import SequenceMatcher
import openpyxl
from openpyxl.styles import Font
from synth_groups import G as G0
from synth_n3 import N3, EXTRA
G=[g+(N3[i],) for i,g in enumerate(G0)]+EXTRA
assert len(N3)==len(G0)
def norm(t):  # same normalization as the pipeline's lexical stage
    t = t.lower().replace("let's", "let us")
    t = re.sub(r"'s\b", " is", t)
    t = re.sub(r"'re\b", " are", t)
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    return t.split()
def feats(a,b):
    ta,tb=norm(a),norm(b); sa,sb=set(ta),set(tb)
    seq=SequenceMatcher(None,ta,tb).ratio(); jac=len(sa&sb)/len(sa|sb); con=len(sa&sb)/min(len(sa),len(sb))
    ba=set(zip(ta,ta[1:])); bb=set(zip(tb,tb[1:])); big=len(ba&bb)/len(ba|bb) if (ba|bb) else 0.0
    return seq,jac,con,big,0.30*seq+0.25*jac+0.25*con+0.20*big
rng=random.Random(20261024)
FILL=["So ","Um, ","Okay so ","Alright, ","Like, "]
def noisy(t):
    r=rng.random()
    if r<0.18: return re.sub(r"[.,?!]","",t).lower()
    if r<0.33: return rng.choice(FILL)+t[0].lower()+t[1:]
    return t
HEAD=["pair_id","video_id","source_type","speaker","unit_a_id","unit_b_id","text_a","text_b","start_a","end_a","start_b","end_b","label","notes","is_hard_negative","sequence","jaccard","containment","bigram","lexical_score","semantic_score","recency_weight","restart_phrase","disagreement","passes_stage1_default","pipeline_class","split","is_synthetic","feature_version"]
rows=[]; dropped=0
for gi,(A,P1,P2,N1,N2,N3_) in enumerate(G,1):
    vid=f"synth_{gi:02d}"
    units=[(A,'pos',None),(P1,'pos',None),(P2,'pos',None),(N1[0],'neg',N1[1]),(N2[0],'neg',N2[1]),(N3_[0],'neg',N3_[1])]
    rng.shuffle(units)
    units=[(f"s{gi:02d}:{i}",noisy(t),k,ty) for i,(t,k,ty) in enumerate(units)]
    for i in range(6):
        for j in range(i+1,6):
            ua,ta,ka,tya=units[i]; ub,tb,kb,tyb=units[j]
            if ka=='pos' and kb=='pos': label,hard,note=1,"",f"synthetic|g{gi:02d}|paraphrase"
            elif ka=='neg' and kb=='neg':
                label,hard,note=0,1,f"synthetic|g{gi:02d}|hard_neg:neg_neg"
            else:
                label,hard=0,1; ty=tya or tyb; note=f"synthetic|g{gi:02d}|hard_neg:{ty}"
            seq,jac,con,big,lex=feats(ta,tb)
            if lex<0.10: dropped+=1; continue
            rows.append([f"{vid}__{ua.replace(':','-')}__{ub.replace(':','-')}",vid,"synthetic","speaker_0",ua,ub,ta,tb,"","","","",label,note,hard,
                round(seq,3),round(jac,3),round(con,3),round(big,3),round(lex,3),"","","","",1 if lex>=0.35 else 0,"","train",1,"v1"])
# trim to ~500 rows: keep all positives and every negative with lexical>=0.25; randomly drop from the easier negatives (neg_neg first)
neg_all=[r for r in rows if r[12]==0]
easy=[r for r in neg_all if r[19]<0.25]
easy.sort(key=lambda r:(0 if r[13].endswith("neg_neg") else 1, rng.random()))
drop=set(id(r) for r in easy[:len(rows)-501])
rows=[r for r in rows if id(r) not in drop]
for r in rows:
    if r[12]==0:
        r[14]=1 if (r[19]>=0.30 or r[13].endswith("opposite")) else ""
print(len(G),"groups",len(rows),"rows",dropped,"neg dropped <0.10")
pos=[r for r in rows if r[12]==1]; neg=[r for r in rows if r[12]==0]
print("pos",len(pos),"neg",len(neg))
import statistics as st
print("pos lex quartiles",[round(x,2) for x in st.quantiles([r[19] for r in pos],n=4)],"pos<0.35:",sum(r[19]<.35 for r in pos))
print("neg lex quartiles",[round(x,2) for x in st.quantiles([r[19] for r in neg],n=4)],"neg>=0.35:",sum(r[19]>=.35 for r in neg))
print("dup pair_ids",len(rows)-len({r[0] for r in rows}))
wb=openpyxl.Workbook(); ws=wb.active; ws.title="synthetic_pairs"
ws.append(HEAD)
for c in ws[1]: c.font=Font(bold=True)
for r in rows:
    ws.append(r)
    for c in ws[ws.max_row]:
        if c.column_letter in "EF" or c.column_letter in ("A","B"): c.number_format="@"
        if isinstance(c.value,str): c.data_type='s'
ws.freeze_panes="A2"
for col,w in zip("ABGHN",[26,10,60,60,30]): ws.column_dimensions[col].width=w
wb.save(os.path.join(os.path.dirname(os.path.abspath(__file__)), "synthetic_pairs_v1.xlsx"))
# show a few low-lex positives and high-lex negatives for sanity
for r in sorted(pos,key=lambda r:r[19])[:5]: print("LOWPOS",r[19],"|",r[6],"|",r[7])
for r in sorted(neg,key=lambda r:-r[19])[:5]: print("HINEG",r[19],"|",r[6],"|",r[7])
