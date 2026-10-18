# Independent artifact audit

Audited 2026-10-18 UTC while the target, collaborative and utility runners were still active. This is a read-only audit of the reconstructed experiment, not verification of unavailable historical code/checkpoints or certification that the full execution matrix has finished. No model was fitted. Only six item score rows were recomputed; saved Top-50 arrays supplied the remaining metric checks.

The audit used the supplied `docs/inputs/*txt`, `IMPLEMENTATION_CONTRACT.md`, and the five protocol documents. Independent NumPy/pandas arithmetic, rather than the production data builder or evaluator, reconstructed the data and metrics. The executable checks are preserved below. No raw user IDs are printed.

## Data and temporal alignment: passed

All four downloaded review/metadata archives and both complete review CSV projections matched their recorded byte sizes/hashes where recorded. Every processed NPZ matched its manifest, data-code hash, and raw CSV/metadata dependency hashes. Hash agreement establishes consistency with these local manifests; it does not independently establish the publisher's original bytes.

| Domain | Complete CSV rows | Rows before 2017-01-31 UTC | Earliest-pair rows before that boundary |
|---|---:|---:|---:|
| target | 4,827,273 | 3,458,917 | 3,417,380 |
| source | 130,434 | 69,594 | 68,677 |

The independent CSV pass read every projected row and counted it. It retained events before the latest query end before stable earliest-pair deduplication: later timestamps cannot displace an earlier event, so this optimization preserves every audited prefix's earliest-pair semantics.

All 18 prefixes passed exact comparison of hash-selected users, lexical item catalogs, train/observed/positive rows, warm-event arrays, fit-only all-rating arrays, fit degrees, tail masks, and raw activity. Standardized activity matched to `1e-6`; means/scales matched exactly. In particular, nonwarm fit ratings are preserved for activity, all prior ratings exclude candidates, and no retained event crosses its prefix's query end. The fit/query lengths are exactly 365/30 days. Target and source embeddings have 64 float32 columns, finite entries, and either zero or unit norm within `1e-5`.

Every target cohort contains 6,000 users. `Eligible` below is the independently counted target population before hash capping. `Query items` counts warm target items with at least one eligible next-month positive; `Tail80 query` is its fixed fit-degree tail subset. Source counts retain all eligible source users.

| Prefix | Eligible | Target items | Fit positives | Query items | Tail80 query | Source users/items |
|---|---:|---:|---:|---:|---:|---:|
| cds_2015_d270 | 22,336 | 24,171 | 33,040 | 357 | 140 | 157/813 |
| cds_2015_d180 | 21,826 | 24,657 | 33,529 | 444 | 178 | 165/849 |
| cds_2015_d90 | 26,550 | 25,538 | 35,263 | 435 | 179 | 211/1,066 |
| cds_2015_d60 | 28,133 | 25,771 | 35,645 | 537 | 221 | 231/1,142 |
| cds_2015_d30 | 29,875 | 26,076 | 36,287 | 593 | 225 | 248/1,203 |
| cds_2015_d0 | 31,848 | 25,917 | 36,734 | 520 | 195 | 267/1,273 |
| cds_2016_d270 | 37,931 | 26,931 | 38,426 | 448 | 170 | 329/1,605 |
| cds_2016_d180 | 42,106 | 27,549 | 39,453 | 269 | 104 | 357/1,737 |
| cds_2016_d90 | 40,683 | 27,124 | 38,617 | 269 | 119 | 336/1,718 |
| cds_2016_d60 | 40,001 | 27,323 | 38,774 | 322 | 120 | 338/1,702 |
| cds_2016_d30 | 39,280 | 27,198 | 38,373 | 212 | 84 | 327/1,676 |
| cds_2016_d0 | 36,956 | 27,218 | 38,036 | 295 | 127 | 320/1,630 |
| cds_2017_d270 | 32,782 | 26,658 | 37,086 | 267 | 98 | 299/1,432 |
| cds_2017_d180 | 30,400 | 25,577 | 35,609 | 318 | 130 | 294/1,371 |
| cds_2017_d90 | 28,851 | 25,694 | 35,636 | 322 | 134 | 275/1,214 |
| cds_2017_d60 | 28,295 | 25,496 | 35,323 | 277 | 112 | 269/1,214 |
| cds_2017_d30 | 27,698 | 25,576 | 35,318 | 388 | 138 | 270/1,211 |
| cds_2017_d0 | 28,897 | 25,517 | 35,309 | 413 | 161 | 276/1,256 |

Across all target catalog rows, candidate counts range from 5,827 to 5,999 after prior-rating exclusion. Each prefix retains 1,239–1,741 nonwarm all-rating activity events. All three d0 cohort/catalog/fit-positive/source counts and Tail80 query-item counts match the supplied R3 report (section 2) exactly. Matching these counts is not evidence that reconstructed learned parameters or metrics match historical artifacts.

## Completed artifact checks: passed

Two representative final artifacts, both `cds_2015_d0/seed_11`, passed:

- All 520 evaluated items' hits, recall, precision and NDCG at 10/20/50 independently recomputed from saved Top-50 and raw-aligned labels, within `1e-12`. All/tail80/tail50 macro means and full-catalog hit coverage matched JSON summaries.
- Catalog IDs, candidate/positive counts, masks and undefined rows matched. Saved rankings contained no duplicates or prior-rated users; unscored metrics were NaN and unscored Top-50 entries were -1.
- Checkpoint bytes and code/data provenance matched their recorded hashes. Three scored items per checkpoint (first/middle/last evaluated catalog position) were rescored and independently sorted over all eligible candidates with lexical user-ID tie order; all six Top-50 arrays matched exactly. The production scorer was used for these score rows, so this is a checkpoint/ranking consistency check, not an independent model implementation.
- The target log records six epochs visiting all 25,917 items each; MF-BPR records six epochs traversing all 36,734 positive edges each.

| Method | Tail80 NDCG@20 | Checkpoint SHA-256 prefix | Test NPZ SHA-256 prefix |
|---|---:|---|---|
| activity_mean | 0.033748342430 | `81dcbed09aaa5f35` | `efca3e58cbb14f8d` |
| mf_bpr | 0.008223003036 | `409df962dea97ea2` | `c965e906e6097099` |

The target test hash above is an audit observation, not a hash stored by its runner. MF-BPR's test hash also matches its completion record. Both completed first-family HPO searches contain 12 trials and independently select trial 10 by maximum Tail80 NDCG@20, resolving ties by trial order; the selected target checkpoint is byte-identical to that trial.

All twelve utility historical folds passed byte-hash checks for target model/history, source activity checkpoint and labels, including their code/data/base/source dependencies. All three ridge groups passed checkpoint/validation-mixture hashes and four-fold label dependency hashes. This verifies recorded artifact linkage; forward-loss label values and all final utility policies were not independently rerun in this audit.

## Limits and resume finding

The target runner's completed-training fast path checks file existence and frozen identity but does **not** rehash the stored checkpoint or evaluation before reuse. Its selected-HPO fast path likewise checks identity/existence, and deterministic-baseline resume checks existence only. Therefore the general phrase “everything can resume” must not be read as a uniform corruption-detection guarantee. The representative target checkpoint matched when independently checked here; that does not repair the runner's missing validation. No core code was changed during live training. CF and utility have explicit completion hash checks, and their reviewed artifacts passed them.

The audit did not refit the 18 title encoders. Target-only vocabulary/projection fitting is established by code inspection and existing tests, while this audit verifies catalog inputs and embedding shape/norm. Present-day metadata cannot establish historical title availability. The unspecified historical preference feature map remains a reconstruction assumption. The run matrix was incomplete at this snapshot, so these results support the audited subsets only.

## Reproduce

From the repository root, the following executes the two preserved audit programs sequentially with single-thread numerical libraries. They read the repository and write only summary JSON under `/tmp`; they perform no fitting and rescore only the six representative item rows described above. Completed inputs must still be present. Expect a few minutes alongside active runners.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python - <<'PY'
from pathlib import Path
import re, subprocess, sys
source = Path('docs/ARTIFACT_AUDIT.md').read_text()
for code in re.findall(r'```python\n(.*?)```', source, re.S):
    subprocess.run([sys.executable, '-c', code], check=True)
PY
```

<details>
<summary>Independent raw-data and prefix audit program</summary>

```python
from pathlib import Path
import hashlib, json, sys
import numpy as np
import pandas as pd
ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))
DAY = 86400000

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''): h.update(b)
    return h.hexdigest()

def eq(a, b, msg):
    assert np.array_equal(a, b), msg

def close(a, b, msg):
    assert np.allclose(a, b, atol=1e-6, rtol=1e-6, equal_nan=True), msg

def npz(p):
    with np.load(p, allow_pickle=False) as z: return dict(z)

manifest = json.loads((ROOT/'data/raw/download_manifest.json').read_text())
raw_hashes = {}
for f in manifest['files']:
    name = ('meta_' if f['kind']=='metadata' else '') + f['category'] + '.jsonl.gz'
    p = ROOT/'data/raw'/name
    raw_hashes[name] = sha(p)
    assert raw_hashes[name] == f['sha256'] and p.stat().st_size == f['bytes']
print('HASH: all four downloaded archives match manifest', flush=True)
latest = int(pd.Timestamp('2017-01-31', tz='UTC').timestamp()*1000)
frames = {}
raw_counts = {}
for domain, name in [('target','CDs_and_Vinyl'),('source','Digital_Music')]:
    p = ROOT/'data/raw'/f'{name}.csv'
    record = next(f for f in manifest['projections'] if f['category']==name)
    raw_hashes[name+'.csv'] = sha(p)
    assert raw_hashes[name+'.csv'] == record['sha256']
    pieces, count, retained = [], 0, 0
    for c in pd.read_csv(p, chunksize=500000, dtype={'user_id':str, 'parent_asin':str, 'timestamp':'int64', 'rating':'float32'}):
        count += len(c)
        assert not c.isna().any().any()
        c = c.loc[c.timestamp < latest]
        retained += len(c)
        pieces.append(c)
    assert count == record['rows']
    f = pd.concat(pieces, ignore_index=True)
    del pieces
    for col in ['user_id','parent_asin']: f[col] = f[col].astype('category')
    f = f.sort_values('timestamp',kind='stable').drop_duplicates(['user_id','parent_asin'], keep='first').reset_index(drop=True)
    frames[domain] = f
    raw_counts[domain] = {'csv_rows':count,'before_latest':retained,'deduplicated_before_latest':len(f)}
print('RAW:',json.dumps(raw_counts),flush=True)
rows=[]
for p in sorted((ROOT/'data/processed').glob('cds_*.npz')):
    d=npz(p); m=json.loads(p.with_suffix('.json').read_text())
    assert sha(p)==m['data_sha256']
    assert sha(ROOT/'saet_repro/data.py')==m['code_sha256']
    for key, name in [('target','CDs_and_Vinyl.csv'),('source','Digital_Music.csv'),('target_meta','meta_CDs_and_Vinyl.jsonl.gz'),('source_meta','meta_Digital_Music.jsonl.gz')]:
        assert m['raw_sha256'][key]==raw_hashes[name]
    end=int(pd.Timestamp(f"{m['origin_year']}-01-01",tz='UTC').timestamp()*1000)-m['offset_days']*DAY
    start=end-365*DAY; stop=end+30*DAY
    eq([d[k].item() for k in ('fit_start','fit_end','query_end')],[start,end,stop],p.stem+' bounds')
    eq(d['cutoffs'],end+np.array([0,30,60,90])*DAY,p.stem+' cutoffs')
    out={'prefix':p.stem}
    for domain,f in frames.items():
        g=lambda s:d[domain+'_'+s]
        fit=f.loc[(f.timestamp>=start)&(f.timestamp<end)&(f.rating>=4)]
        counts=fit.groupby('user_id',observed=True).size()
        eligible=list(counts.index[counts>=3].astype(str))
        users=eligible
        if domain=='target': users=sorted(users,key=lambda u:(hashlib.sha256(('2026|cds:users|'+u).encode()).digest(),u))[:6000]
        users=np.array(sorted(users)); fit=fit.loc[fit.user_id.isin(users)]
        items=np.array(sorted(fit.parent_asin.unique().astype(str)))
        eq(g('user_ids'),users,p.stem+domain+' users');eq(g('item_ids'),items,p.stem+domain+' items')
        selected=f.loc[(f.timestamp<stop)&f.user_id.isin(users)]
        ui=pd.Index(users).get_indexer(selected.user_id);ii=pd.Index(items).get_indexer(selected.parent_asin)
        tt=selected.timestamp.to_numpy();rr=selected.rating.to_numpy()
        warm=ii>=0;order=np.lexsort((ii[warm],ui[warm],tt[warm]))
        eu,ei,et,er=[a[warm][order] for a in (ui,ii,tt,rr)]
        for key,a in zip(('event_users','event_items','event_times','event_ratings'),(eu,ei,et,er)):eq(g(key),a,p.stem+domain+key)
        observed=np.column_stack([eu,ei,np.where(et<start,-1,np.where(et<end,0,1))])
        positive=observed[er>=4];train=positive[positive[:,2]==0,:2]
        for key,a in [('observed',observed),('positive',positive),('train',train)]:eq(g(key),a,p.stem+domain+key)
        activity_fit=(tt>=start)&(tt<end)
        au,ai,at,ar=[a[activity_fit] for a in (ui,ii,tt,rr)]
        for key,a in zip(('all_rating_users','all_rating_items','all_rating_times','all_rating_ratings'),(au,ai,at,ar)):eq(g(key),a,p.stem+domain+key)
        deg=np.bincount(train[:,1],minlength=len(items));eq(g('degree'),deg,p.stem+' degree')
        for name, fraction in [('tail80',.8),('tail50',.5)]:
            mask=np.zeros(len(items),bool);mask[np.lexsort((items,deg))[:int(np.ceil(len(items)*fraction))]]=True
            eq(g(name),mask,p.stem+domain+name)
            if domain=='target':eq(d[name],mask,p.stem+' alias')
        act=np.zeros((len(users),5));age=(end-at)/DAY
        for j,days in enumerate([30,90,365]):act[:,j]=np.bincount(au[age<=days],minlength=len(users))
        rec=np.full(len(users),365.);np.minimum.at(rec,au,age);act[:,3]=rec
        pairs=np.unique(np.column_stack([au,(end-1-at)//(7*DAY)]),axis=0)
        act[:,4]=np.bincount(pairs[:,0],minlength=len(users))
        act=np.log1p(act).astype(np.float32);eq(g('activity_raw'),act,p.stem+domain+' activity')
        close(g('activity'),(act-d['activity_mean'])/d['activity_scale'],p.stem+domain+' standardized')
        norm=np.linalg.norm(g('x'),axis=1)
        assert g('x').shape==(len(items),64) and g('x').dtype==np.float32
        assert np.all(np.isfinite(g('x'))) and np.all((norm==0)|(abs(norm-1)<1e-5))
        prior=observed[observed[:,2]<1,:2];future=positive[positive[:,2]==1,:2]
        prior_codes=np.unique(prior[:,1]*len(users)+prior[:,0]);query_codes=np.unique(future[:,1]*len(users)+future[:,0])
        query_codes=np.setdiff1d(query_codes,prior_codes)
        nc=len(users)-np.bincount(prior_codes//len(users),minlength=len(items))
        nq=np.bincount(query_codes//len(users),minlength=len(items))
        out[domain]={'eligible_users':len(eligible),'users':len(users),'items':len(items),'fit_positive':len(train),'query_positive':int(nq.sum()),'query_items':int((nq>0).sum()),'tail80_query_items':int(((nq>0)&g('tail80')).sum()),'candidates_min':int(nc.min()),'candidates_max':int(nc.max()),'nonwarm_activity':int((ai<0).sum()),'zero_titles':int((norm==0).sum())}
        assert all(out[domain][k]==m['counts'][domain][k] for k in m['counts'][domain])
    eq(d['degree'],d['target_degree'],p.stem+' degree alias')
    eq(d['activity_mean'],d['target_activity_raw'].mean(0,dtype=np.float64).astype(np.float32),p.stem+' mean')
    eq(d['activity_scale'],np.maximum(d['target_activity_raw'].std(0,dtype=np.float64),1e-6).astype(np.float32),p.stem+' scale')
    rows.append(out);print('PREFIX:',json.dumps(out),flush=True)
assert len(rows)==18
Path('/tmp/saet_artifact_audit_counts.json').write_text(json.dumps({'raw_counts':raw_counts,'prefixes':rows},indent=2))
print('PASS: 18 raw-to-NPZ prefix audits',flush=True)
```

</details>

<details>
<summary>Independent metrics, checkpoint and utility dependency audit program</summary>

```python
from pathlib import Path
import hashlib,json,sys
import numpy as np
ROOT=Path.cwd();sys.path.insert(0,str(ROOT))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def npz(p):
    with np.load(p,allow_pickle=False) as z:return dict(z)
def eq(a,b):assert np.array_equal(a,b)
def close(a,b):assert np.allclose(a,b,atol=1e-12,rtol=1e-12,equal_nan=True)
D=ROOT/'data/processed/cds_2015_d0.npz';d=npz(D)
u=d['target_user_ids'];it=d['target_item_ids'];nu=len(u);ni=len(it)
obs=d['target_observed'];pos=d['target_positive']
prior=obs[obs[:,2]<1,:2];query=pos[pos[:,2]==1,:2]
prior_codes=np.unique(prior[:,1]*nu+prior[:,0]);query_codes=np.setdiff1d(np.unique(query[:,1]*nu+query[:,0]),prior_codes)
nc=nu-np.bincount(prior_codes//nu,minlength=ni);nq=np.bincount(query_codes//nu,minlength=ni)
results=[]
for family,method in [('targets','activity_mean'),('baselines','mf_bpr')]:
    folder=ROOT/f'results/{family}/cds_2015_d0/{method}/seed_11'
    r=npz(folder/'test.npz');j=json.loads((folder/'result.json').read_text())
    assert sha(folder/'model.npz')==j['model_sha256']
    if family=='targets':
        frozen=j['frozen'];assert frozen==json.loads((folder/'result.frozen.json').read_text())
        for key,name in [('target_code_sha256','saet_repro/target.py'),('runner_code_sha256','scripts/run_targets.py'),('evaluator_code_sha256','saet_repro/evaluation.py')]:assert frozen[key]==sha(ROOT/name)
        assert frozen['data_sha256']==sha(D)
        assert [h['epoch'] for h in j['history']]==list(range(1,7))
        assert all(h['visited_items']==ni for h in j['history'])
        from saet_repro.target import Model
        model=Model.load(folder/'model.npz',d)
    else:
        assert sha(folder/'test.npz')==j['evaluation_sha256']
        frozen=json.loads((folder/'frozen.json').read_text())
        assert frozen['model_sha256']==j['model_sha256']
        for path,h in frozen['identity']['code_sha256'].items():assert sha(ROOT/path)==h
        assert frozen['identity']['data_sha256']==sha(D)
        assert [h['epoch'] for h in frozen['training']]==list(range(1,7))
        assert all(h['positive_edges']==len(d['target_train']) for h in frozen['training'])
        import torch
        torch.set_num_threads(1)
        from saet_repro.collaborative import CollaborativeModel
        model=CollaborativeModel.load(folder/'model.npz',d)
    eq(r['item_id'],it);eq(r['n_candidates'],nc);eq(r['n_positive'],nq)
    assert str(r['candidate_mode'])=='full_eligible_users' and int(r['stage'])==1
    for key in ['requested','eligible_catalog']:eq(r[key],np.ones(ni,bool))
    for key in ['evaluated','scored','ranked']:eq(r[key],nq>0)
    assert np.all(r['top_users'][nq==0]==-1)
    expected={f'{metric}{k}':np.full(ni,np.nan) for metric in ['hits','ndcg','recall','precision'] for k in [10,20,50]}
    for i in np.flatnonzero(nq):
        top=r['top_users'][i];valid=top[top>=0]
        assert len(np.unique(valid))==len(valid) and len(valid)==min(50,nc[i])
        assert np.all(valid<nu) and not np.isin(i*nu+valid,prior_codes).any()
        rel=np.isin(i*nu+valid,query_codes)
        for k in [10,20,50]:
            slots=min(k,nc[i]);hits=int(rel[:slots].sum())
            expected[f'hits{k}'][i]=hits
            expected[f'recall{k}'][i]=hits/nq[i]
            expected[f'precision{k}'][i]=hits/slots
            expected[f'ndcg{k}'][i]=np.sum(rel[:slots]/np.log2(np.arange(slots)+2))/np.sum(1/np.log2(np.arange(min(k,nq[i]))+2))
    for key,val in expected.items():close(r[key],val)
    for stratum,mask in [('all',np.ones(ni,bool)),('tail80',d['target_tail80']),('tail50',d['target_tail50'])]:
        s=j['summary']['1'][stratum];ev=mask&(nq>0);n=int(mask.sum());e=int(ev.sum())
        assert s['items_catalog']==n and s['items_evaluated']==e and s['items_eligible']==n
        for key,val in expected.items():close(s[key],np.mean(val[ev]))
        for k in [10,20,50]:
            hit=int((expected[f'hits{k}'][ev]>0).sum())
            assert s[f'items_hit{k}']==hit
            close(s[f'coverage{k}'],hit/n);close(s[f'conditional_hit_rate{k}'],hit/e)
    indices=np.flatnonzero(nq);sample=indices[[0,len(indices)//2,-1]]
    scores=model.score_all(sample)
    for i,scores_i in zip(sample,scores):
        candidates=np.flatnonzero(~np.isin(i*nu+np.arange(nu),prior_codes))
        order=candidates[np.lexsort((u[candidates],-scores_i[candidates]))]
        eq(order[:50],r['top_users'][i])
    item={'family':family,'method':method,'seed':11,'prefix':'cds_2015_d0','metric_rows':int((nq>0).sum()),'ranked_score_rows':len(sample),'model_sha256':sha(folder/'model.npz'),'evaluation_sha256':sha(folder/'test.npz'),'tail80_ndcg20':j['summary']['1']['tail80']['ndcg20']}
    results.append(item);print('RESULT:',json.dumps(item),flush=True)
# Independent byte/dependency checks for all twelve completed historical utility folds.
for year in [2015,2016,2017]:
    for offset in [270,180,90,60]:
        folder=ROOT/f'results/utility/cds_{year}/historical/d{offset}'
        for name,artifacts in [('training.complete.json',{'model':'model.npz','history':'training.json'}),('source_activity.json',{'source_activity':'source_activity.npz'}),('result.json',{'labels':'labels.npz'})]:
            j=json.loads((folder/name).read_text())
            assert j['artifacts']=={k:sha(folder/v) for k,v in artifacts.items()}
            f=j['frozen'];assert f['data_sha256']==sha(ROOT/f'data/processed/cds_{year}_d{offset}.npz')
            for path,h in f['code_sha256'].items():assert sha(ROOT/path)==h
            if 'base_model_sha256' in f:assert sha(folder/'model.npz')==f['base_model_sha256']
            if 'source_activity_sha256' in f:assert sha(folder/'source_activity.npz')==f['source_activity_sha256']
    folder=ROOT/f'results/utility/cds_{year}/ridges';j=json.loads((folder/'selected.json').read_text())
    for k,h in j['artifacts'].items():assert sha(folder/('validation_mixtures.json' if k=='validation_mixtures' else f'{k}.npz'))==h
    for offset,h in j['frozen']['fold_hashes'].items():assert sha(ROOT/f'results/utility/cds_{year}/historical/d{offset}/labels.npz')==h
print('PASS: 12 utility historical-fold checkpoint, label, source checkpoint, and three ridge-group hashes',flush=True)
# Completed first-family searches: independently recompute selected trial and target copy hash.
for family,method in [('targets','activity_mean'),('baselines','mf_bpr')]:
    folder=ROOT/f'results/{family}/cds_2015_d30/{method}'
    s=json.loads((folder/'selected.json').read_text())
    trials=[]
    for trial in range(12):
        j=json.loads((folder/f'search/trial_{trial:02d}.json').read_text())
        trials.append(j['summary']['1']['tail80']['ndcg20'])
    best=max(range(12),key=lambda i:(trials[i],-i))
    assert s.get('trial',s.get('selected_trial'))==best
    if family=='targets':assert sha(folder/'selected_model.npz')==sha(folder/f'search/trial_{best:02d}.npz')
    print('SEARCH:',family,method,'12 trials; selected',best,flush=True)
Path('/tmp/saet_result_audit.json').write_text(json.dumps(results,indent=2))
```

</details>
