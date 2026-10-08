import json, glob, re, collections, os

RAW=os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","raw")
SEEDS={"2409.19606":"HC","2512.24880":"mHC","2503.14125":"Frac","2502.12170":"MUDD"}

papers={}  # key -> record
for f in sorted(glob.glob(RAW+"/s2_cites_*.json")):
    seed=re.search(r"s2_cites_([\d.]+)_off",f).group(1)
    d=json.load(open(f))
    for it in d.get("data",[]):
        p=it.get("citingPaper") or {}
        pid=p.get("paperId")
        if not pid: continue
        r=papers.setdefault(pid,{"paperId":pid,"seeds":set(),
            "title":p.get("title") or "","abstract":p.get("abstract") or "",
            "year":p.get("year"),"date":p.get("publicationDate"),
            "arxiv":(p.get("externalIds") or {}).get("ArXiv"),
            "url":p.get("url"),"venue":p.get("venue"),
            "cites":p.get("citationCount")})
        r["seeds"].add(SEEDS[seed])

KW=["head","group","channel","split","frac","partition","block","kronecker",
    "multiway","multi-way","stream","sinkhorn","doubly stochastic","birkhoff",
    "manifold","residual","mhc","hyper-connection","hyperconnection","hyper connection",
    "connection","width","expansion","depth","skip"]
STRONG=["hyper-connection","hyperconnection","mhc","sinkhorn","doubly stochastic",
        "birkhoff","frac-connection","multiway","multi-way","residual stream",
        "residual connection","dense connection","kronecker"]

rows=[]
for r in papers.values():
    txt=((r["title"]or"")+" "+(r["abstract"]or"")).lower()
    hits=[k for k in KW if k in txt]
    strong=[k for k in STRONG if k in txt]
    r["hits"]=hits; r["strong"]=strong
    r["score"]=len(strong)*3+len(hits)
    rows.append(r)

rows.sort(key=lambda r:(-r["score"], r["date"] or ""))
print("TOTAL UNIQUE CITING PAPERS:",len(rows))
print("by seed:",collections.Counter(s for r in rows for s in r["seeds"]))
print("no abstract:",sum(1 for r in rows if not r["abstract"]))
print()
for r in rows:
    print(f"[{r['score']:>3}] {r['date'] or r['year']} | {'+'.join(sorted(r['seeds'])):<18} | {r['arxiv'] or '-':<12} | {r['title'][:95]}")
    if r["strong"]: print(f"        STRONG: {r['strong']}")
json.dump([{k:(sorted(v) if isinstance(v,set) else v) for k,v in r.items()} for r in rows],
          open("screened.json","w"),indent=1)
