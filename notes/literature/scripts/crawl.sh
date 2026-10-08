#!/bin/bash
RAW="$(dirname "$0")/../raw"
FIELDS="title,abstract,year,externalIds,publicationDate,url,venue,authors,citationCount"

fetch() {  # $1=url $2=outfile
  local tries=0
  while [ $tries -lt 8 ]; do
    curl -s "$1" -o "$2.tmp"
    if grep -q '"code": *"429"' "$2.tmp" 2>/dev/null || grep -q 'Too Many Requests' "$2.tmp" 2>/dev/null; then
      tries=$((tries+1)); echo "   429, sleep $((tries*12))s" >&2; sleep $((tries*12)); continue
    fi
    mv "$2.tmp" "$2"; return 0
  done
  echo "   FAILED: $1" >&2; return 1
}

for id in "$@"; do
  off=0
  while : ; do
    out="$RAW/s2_cites_${id}_off${off}.json"
    echo "-- $id offset $off"
    fetch "https://api.semanticscholar.org/graph/v1/paper/arXiv:$id/citations?fields=$FIELDS&limit=1000&offset=$off" "$out" || break
    n=$(python3 -c "import json,sys; d=json.load(open('$out')); print(len(d.get('data',[])))" 2>/dev/null || echo 0)
    echo "   got $n"
    [ "$n" -lt 1000 ] && break
    off=$((off+1000)); sleep 5
  done
  sleep 6
done
