#!/usr/bin/env python3
"""Download complete public Amazon 2023 categories and project every review."""
from __future__ import annotations
import argparse
import concurrent.futures
import csv
import gzip
import hashlib
import json
from pathlib import Path
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
BASE = 'https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/'
CATEGORIES = ('Digital_Music', 'CDs_and_Vinyl')

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()

def save(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)

def download(raw,category,kind):
    name=f'meta_{category}.jsonl.gz' if kind=='metadata' else f'{category}.jsonl.gz'
    url=BASE+('meta_categories/' if kind=='metadata' else 'review_categories/')+name
    dest=raw/name;record=raw/(name+'.download.json')
    if dest.exists() and record.exists():
        old=json.loads(record.read_text())
        if old['sha256']==digest(dest) and old['bytes']==dest.stat().st_size:
            print(f'reuse verified {name}',flush=True);return old
    part=dest.with_suffix(dest.suffix+'.partial')
    for attempt in range(3):
        try:
            request=urllib.request.Request(url,headers={'User-Agent':'SAET-Reconstruction/0.1'})
            with urllib.request.urlopen(request,timeout=60) as response,part.open('wb') as output:
                expected=response.headers.get('Content-Length');n=0;announced=0;h=hashlib.sha256()
                actual_url=response.geturl()
                while True:
                    block=response.read(8<<20)
                    if not block:break
                    output.write(block);h.update(block);n+=len(block)
                    if n-announced>=64<<20:
                        print(f'download {name} {n/1048576:.1f} MiB',flush=True);announced=n
            if expected is not None and n!=int(expected):raise IOError(f'incomplete length {n}/{expected}')
            part.replace(dest)
            result={'category':category,'kind':kind,'url':url,'resolved_url':actual_url,'bytes':n,
                    'sha256':h.hexdigest(),'complete_http_response':True,'downloaded_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
            save(record,result);print(f'complete {name} {n} bytes',flush=True);return result
        except Exception as error:
            print(f'retry {name} {attempt+1}/3: {error}',flush=True)
            if attempt==2:raise
            time.sleep(2)

def project(raw,category):
    source=raw/f'{category}.jsonl.gz';dest=raw/f'{category}.csv';record=raw/f'{category}.projection.json'
    source_hash=digest(source)
    if dest.exists() and record.exists():
        old=json.loads(record.read_text())
        if old['source_sha256']==source_hash and old['sha256']==digest(dest):return old
    temp=dest.with_suffix('.csv.partial');count=0
    with gzip.open(source,'rt',encoding='utf-8') as incoming,temp.open('w',newline='') as output:
        writer=csv.writer(output);writer.writerow(['user_id','parent_asin','rating','timestamp'])
        for line in incoming:
            row=json.loads(line)
            writer.writerow([row['user_id'],row['parent_asin'],row['rating'],row['timestamp']]);count+=1
            if count%1000000==0:print(f'project {category}: {count} rows',flush=True)
    temp.replace(dest)
    result={'category':category,'rows':count,'source_file':source.name,'source_sha256':source_hash,
            'bytes':dest.stat().st_size,'sha256':digest(dest),'gzip_eof_verified':True,
            'scope':'every review row; no prefix, k-core, time or population filtering'}
    save(record,result);print(f'project complete {category}: {count} rows',flush=True);return result

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--raw-dir',type=Path,default=ROOT/'data/raw')
    parser.add_argument('--workers',type=int,default=2);args=parser.parse_args();raw=args.raw_dir;raw.mkdir(parents=True,exist_ok=True)
    jobs=[(category,kind) for category in CATEGORIES for kind in ('reviews','metadata')]
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        records=list(pool.map(lambda pair:download(raw,*pair),jobs))
    projected=[project(raw,c) for c in CATEGORIES]
    save(raw/'download_manifest.json',{'source':'Amazon Reviews 2023 official complete categories','files':records,'projections':projected})
    print('All downloads and complete-stream projections finished.',flush=True)

if __name__=='__main__':main()
