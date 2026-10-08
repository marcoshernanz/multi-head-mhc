import json,glob,re,collections,os
RAW=os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","raw")
files=sorted(glob.glob(RAW+"/s2_cites_*.json"))
percited=collections.Counter(); papers={}
for f in files:
    seed=re.search(r"s2_cites_([\d.]+)_off",f).group(1)
    data=json.load(open(f)).get("data",[])
    percited[seed]+=len(data)
    for it in data:
        p=it.get("citingPaper") or {}
        k=p.get("paperId")
        if not k: continue
        r=papers.setdefault(k,{"t":p.get("title") or "","a":p.get("abstract") or "",
          "x":(p.get("externalIds") or {}).get("ArXiv"),"d":p.get("publicationDate"),"s":set()})
        r["s"].add(seed)
print("== citations retrieved per seed ==")
for k,v in sorted(percited.items()): print(f"  arXiv:{k}  {v}")
print("TOTAL rows:",sum(percited.values()),"| UNIQUE citing papers:",len(papers))
print("missing abstract:",sum(1 for r in papers.values() if not r["a"]))
MH=re.compile(r"multi-?head|per-head|head-?wise|channel[- ]?wise|channel group|group of channels|per-channel|block[- ]?diagonal|kronecker|grouped|partition|frac-?connection|multi-?way|per-stream|stream-specific|split",re.I)
hits=[r for r in papers.values() if MH.search(r["t"]+" "+r["a"])]
print(f"\n== multi-head-axis keyword hits: {len(hits)} ==")
for r in sorted(hits,key=lambda r:r["d"] or ""):
    print(f"  {r['d']} {r['x'] or '-':<12} {r['t'][:80]}")
