#!/usr/bin/env python3
"""Read-only audit for Turbo refresher PM2 processes and contention risk."""
from __future__ import annotations
import argparse,json,shutil,subprocess,re
from datetime import datetime,timezone
from pathlib import Path
from typing import Any
OUT_DIR=Path('/home/jasan/Develop')
FILES=['aegis_alpha/tools/refresh_turbo_snapshots.py','ecosystem.config.js','pm2.config.js']
def stamp(): return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
def run(cmd,timeout=6):
    if shutil.which(cmd[0]) is None: return {'available':False,'stdout':'','stderr':'not found'}
    try:
        p=subprocess.run(cmd,capture_output=True,text=True,timeout=timeout,check=False)
        return {'available':True,'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
    except Exception as e: return {'available':True,'stdout':'','stderr':str(e)}
def safe(v):
    if isinstance(v,dict): return {str(k):safe(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)): return [safe(x) for x in v]
    if isinstance(v,Path): return str(v)
    return v
def inspect_code():
    rows=[]
    pats={'while_loop':r'while True|while\s+1','sleep':r'sleep\(','sqlite':r'sqlite3|binance_candles\.db','write_jsonl':r'jsonl|write\(','process_pool':r'ProcessPool|multiprocessing|ThreadPool'}
    for file in FILES:
        p=Path(file)
        if not p.exists(): continue
        text=p.read_text(errors='ignore')
        for name,pat in pats.items():
            m=re.findall(pat,text)
            if m: rows.append({'file':file,'pattern':name,'count':len(m)})
    return rows
def classify(pm2,ps,logs,code):
    text='\n'.join([pm2.get('stdout',''),ps.get('stdout',''),logs.get('stdout',''),logs.get('stderr','')]).lower()
    if 'errored' in text: return 'REFRESHER_ERRORED'
    if re.search(r'refresh.*100\.0|100%', text): return 'REFRESHER_CPU_HOT_LOOP'
    if 'database is locked' in text or 'sqlite' in text and any(r.get('pattern')=='sqlite' for r in code): return 'REFRESHER_SQLITE_CONTENTION_RISK'
    return 'REFRESHERS_OK' if 'online' in text else 'REFRESHER_UNKNOWN'
def audit():
    pm2=run(['pm2','status'],timeout=8)
    ps=run(['ps','aux','--sort=-%cpu'],timeout=4)
    logs={}
    for name in ['04-Aegis-Turbo-Refresh-A','05-Aegis-Turbo-Refresh-B','06-Aegis-Turbo-Refresh-C']:
        logs[name]=run(['pm2','logs',name,'--lines','120','--nostream'],timeout=8)
    code=inspect_code(); classification=classify(pm2,ps,logs.get('06-Aegis-Turbo-Refresh-C',{}),code)
    return {'schema_version':'aegis_turbo_refreshers_health_v1','created_at':datetime.now(timezone.utc).isoformat(),'mode':'READ_ONLY','classification':classification,'pm2_status':pm2,'top_cpu':ps,'logs':logs,'code_findings':code,'notes':['No PM2 restart, no process kill.']}
def write(payload,out):
    out.mkdir(parents=True,exist_ok=True); ts=stamp(); paths={'md':out/f'aegis_long_perf_b_refreshers_{ts}.md','json':out/f'aegis_long_perf_b_refreshers_{ts}.json'}
    paths['json'].write_text(json.dumps(safe(payload),indent=2,sort_keys=True)+'\n')
    lines=['# Aegis Turbo Refreshers Health','','## Safety','- read-only','- no PM2 restart','- no process kill','',f"## Classification\n- `{payload['classification']}`",'','## PM2','```text',payload['pm2_status'].get('stdout','')[:5000],'```','','## Code Findings']
    for r in payload['code_findings']: lines.append(f"- `{r['file']}` `{r['pattern']}` count={r['count']}")
    paths['md'].write_text('\n'.join(lines)+'\n')
    return {k:str(v) for k,v in paths.items()}
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out-dir',default=str(OUT_DIR)); args=ap.parse_args(); payload=audit(); paths=write(payload,Path(args.out_dir)); print(json.dumps({'reports':paths,'classification':payload['classification']},indent=2))
if __name__=='__main__': main()
