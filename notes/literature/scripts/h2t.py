import sys,re,html
src=open(sys.argv[1],encoding="utf-8",errors="replace").read()
src=re.sub(r"<(script|style|nav|footer)\b.*?</\1>","",src,flags=re.S|re.I)
# math: keep latex from alttext
src=re.sub(r'<math[^>]*alttext="([^"]*)"[^>]*>.*?</math>',lambda m:" $"+html.unescape(m.group(1))+"$ ",src,flags=re.S)
src=re.sub(r"<math[^>]*>.*?</math>"," [math] ",src,flags=re.S)
src=re.sub(r"</(p|div|li|tr|h1|h2|h3|h4|section|table|figure|caption)>","\n",src,flags=re.I)
src=re.sub(r"<br[^>]*>","\n",src,flags=re.I)
src=re.sub(r"</t[dh]>"," | ",src,flags=re.I)
src=re.sub(r"<[^>]+>","",src)
src=html.unescape(src)
src=re.sub(r"[ \t\xa0]+"," ",src)
src=re.sub(r"\n\s*\n+","\n",src)
print(src.strip())
