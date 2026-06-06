#!/usr/bin/env python3
"""Read-only audit for Aegis API memory pressure."""
from __future__ import annotations

import argparse, csv, json, os, re, shutil, subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

OUT_DIR = Path('/home/jasan/Develop')
FILES = [
    'aegis_alpha/inference/server.py',
    'aegis_alpha/turbo/turbo_signal.py',
    'aegis_alpha/turbo/snapshot_utils.py',
    'aegis_alpha/turbo/phase_o_overlay.py',
    'aegis_alpha/turbo/turbo_risk.py',
]
PATTERNS = {
    'global_cache_or_dict': r'(^|\n)\s*[A-Z_a-z0-9]*(CACHE|cache|_MODELS|models|manifest)[A-Za-z0-9_]*\s*=\s*(\{|\[|dict\(|list\()',
    'joblib_load': r'joblib\.load|load\(',
    'lru_cache': r'lru_cache\(|@cache',
    'global_dataframe': r'pd\.DataFrame|DataFrame\(',
    'append_accumulator': r'\.append\(',
}

def stamp(): return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
def run(cmd, timeout=5):
    if shutil.which(cmd[0]) is None: return {'available': False, 'stdout':'', 'stderr':'not found'}
    try:
        p=subprocess.run(cmd,capture_output=True,text=True,timeout=timeout,check=False)
        return {'available': True, 'returncode': p.returncode, 'stdout': p.stdout, 'stderr': p.stderr}
    except Exception as e: return {'available': True, 'stdout':'', 'stderr':str(e)}
def safe(v):
    if isinstance(v, dict): return {str(k): safe(x) for k,x in v.items()}
    if isinstance(v, (list, tuple)): return [safe(x) for x in v]
    if isinstance(v, Path): return str(v)
    return v

def parse_ps(out):
    rows=[]
    for line in out.splitlines()[1:]:
        parts=line.split(None,10)
        if len(parts)<11: continue
        user,pid,pcpu,pmem,vsz,rss,tty,stat,start,time,cmd=parts
        if 'inference/server.py' in cmd or 'Aegis-API' in cmd:
            rows.append({'user':user,'pid':pid,'pcpu':pcpu,'pmem':pmem,'vsz_kb':vsz,'rss_kb':rss,'stat':stat,'cmd':cmd})
    return rows

def inspect_code():
    findings=[]
    for file in FILES:
        p=Path(file)
        if not p.exists():
            findings.append({'file': file, 'exists': False})
            continue
        text=p.read_text(errors='ignore')
        for name,pattern in PATTERNS.items():
            matches=list(re.finditer(pattern,text,re.MULTILINE))
            if matches:
                findings.append({'file': file, 'exists': True, 'pattern': name, 'count': len(matches), 'examples': [text[max(0,m.start()-80):m.end()+120].replace('\n',' ')[:220] for m in matches[:5]]})
    return findings

def classify(processes, findings):
    rss_mb=sum(int(p.get('rss_kb') or 0) for p in processes)/1024.0
    if rss_mb < 4096: return 'API_MEMORY_OK'
    pats={f.get('pattern') for f in findings}
    if 'global_cache_or_dict' in pats or 'lru_cache' in pats: return 'API_MEMORY_HIGH_MODEL_CACHE'
    if 'joblib_load' in pats and rss_mb > 16000: return 'API_MEMORY_POSSIBLE_RELOAD_LOOP'
    if rss_mb > 16000: return 'API_MEMORY_POSSIBLE_LEAK'
    return 'API_MEMORY_UNKNOWN'

def audit():
    ps=run(['ps','aux','--sort=-%mem'],timeout=5)
    processes=parse_ps(ps['stdout'])
    pid=processes[0]['pid'] if processes else None
    proc_detail={}
    if pid:
        for name,path in [('status',f'/proc/{pid}/status'),('limits',f'/proc/{pid}/limits'),('cmdline',f'/proc/{pid}/cmdline')]:
            try: proc_detail[name]=Path(path).read_text(errors='ignore').replace('\x00',' ')
            except Exception as e: proc_detail[name]=str(e)
        fd_dir=Path(f'/proc/{pid}/fd')
        try: proc_detail['fd_count']=len(list(fd_dir.iterdir()))
        except Exception as e: proc_detail['fd_count_error']=str(e)
    pm2_logs=run(['pm2','logs','02-Aegis-API','--lines','180','--nostream'],timeout=8)
    findings=inspect_code()
    classification=classify(processes,findings)
    return {'schema_version':'aegis_api_memory_pressure_v1','created_at':datetime.now(timezone.utc).isoformat(),'mode':'READ_ONLY','classification':classification,'processes':processes,'process_detail':proc_detail,'code_findings':findings,'pm2_logs_tail':pm2_logs,'notes':['No runtime changes, no PM2 restart, no process kill.']}

def write(payload,out):
    out.mkdir(parents=True,exist_ok=True); ts=stamp()
    paths={'md':out/f'aegis_long_perf_b_api_memory_{ts}.md','json':out/f'aegis_long_perf_b_api_memory_{ts}.json','processes':out/f'aegis_api_memory_processes_{ts}.csv'}
    paths['json'].write_text(json.dumps(safe(payload),indent=2,sort_keys=True)+'\n')
    keys=[]
    for r in payload['processes']:
        for k in r:
            if k not in keys: keys.append(k)
    with paths['processes'].open('w',newline='') as h:
        w=csv.DictWriter(h,fieldnames=keys or ['empty']); w.writeheader(); [w.writerow(r) for r in payload['processes']]
    lines=['# Aegis API Memory Pressure Audit','','## Safety','- read-only','- no PM2 restart','- no process kill','- no live changes','',f"## Classification\n- `{payload['classification']}`",'', '## Processes','| pid | rss_mb | cpu | mem | cmd |','|---:|---:|---:|---:|---|']
    for p in payload['processes']:
        lines.append(f"| {p.get('pid')} | {int(p.get('rss_kb') or 0)/1024:.1f} | {p.get('pcpu')} | {p.get('pmem')} | `{p.get('cmd','')[:120]}` |")
    lines += ['', '## Code Findings']
    for f in payload['code_findings'][:20]: lines.append(f"- `{f.get('file')}` `{f.get('pattern')}` count={f.get('count')}")
    paths['md'].write_text('\n'.join(lines)+'\n')
    return {k:str(v) for k,v in paths.items()}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out-dir',default=str(OUT_DIR)); args=ap.parse_args()
    payload=audit(); paths=write(payload,Path(args.out_dir)); print(json.dumps({'reports':paths,'classification':payload['classification']},indent=2))
if __name__=='__main__': main()
