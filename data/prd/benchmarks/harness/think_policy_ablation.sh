#!/bin/bash
cd /home/paula/projects/searchbox
PIBIN=/home/paula/.nvm/versions/node/v24.13.1/bin/pi
RESULTS=runs/think_bench/results.jsonl; : > $RESULTS
declare -A Q
Q[q01_factoid]="What is the battery life of the Atlas-7?"
Q[q11_literal]="What error code did incident INC-2041 report?"
Q[q20_dispersed]="What is the total capital Meridian has raised to date?"
Q[q26_multihop]="Who led the team that shipped Atlas-7 v3.0?"
for qid in q01_factoid q11_literal q20_dispersed q26_multihop; do
 for cfg in on gate; do
  if [ "$cfg" = "on" ]; then POL=off; else POL=gate_all; fi
  for rep in 1 2 3; do
   OUT=runs/think_bench/${qid}_${cfg}_r${rep}; rm -rf $OUT
   env LLAMA_URL=http://127.0.0.1:8090 PI_BIN="$PIBIN" PI_SKIP_VERSION_CHECK=1 EMBED_BACKEND=local      CONTEXT_WINDOW=4096 MAX_OUTPUT_TOKENS=1024 THINKING_LEVEL=high THINKING_POLICY=$POL      SEARCHBOX_TOOLS=search_dataroom,answer_question TURN_TIMEOUT=300 DATAROOM_BOOT_TIMEOUT=180 MAX_SECONDS=600      PATH="$(dirname $PIBIN):$PWD/.venv/bin:$PATH"      ./.venv/bin/python -m server.run_searchbox --query "${Q[$qid]}"      --dataroom data/default-dataroom.zip --budget 1 --out $OUT > $OUT.log 2>&1
   ./.venv/bin/python -c "
import json,os
m=json.load(open('$OUT/run_meta.json')); t=json.load(open('$OUT/timing.json'))
ans=open('$OUT/work/ANSWER.md').read() if os.path.exists('$OUT/work/ANSWER.md') else ''
row={'qid':'$qid','cfg':'$cfg','rep':$rep,'output':m['tokens'].get('output'),'llm_ms':t['llm_ms'],'elapsed':m['elapsed_seconds'],'done':m['done'],'answer':ans[:300]}
open('$RESULTS','a').write(json.dumps(row)+chr(10))
"
  done
 done
done
echo "BENCH_DONE" >> $RESULTS
