# Reconstruction contract (2026-10-17 Asia/Shanghai)

This is an independent reconstruction from three supplied DOCX documents and recorded protocol details. Original source/checkpoints are unavailable. Historical report metrics are references, never new observations. No claim of byte-identical historical reproduction.

## Shared data API
`saet_repro.data.load_data(path) -> dict[str,np.ndarray]` and `save_data(path,data)`.
Each NPZ has `fit_start`, `fit_end`, `query_end` Unix milliseconds; `cutoffs=[fit_end,fit_end+30d,fit_end+60d,fit_end+90d]` only for compatibility. Actual query is stage1 next30d.
For `target` and `source` prefixes:
- `x`: [n_items,64] normalized float32 title embeddings, zero vectors retained.
- `user_ids`,`item_ids`: lexically sorted raw IDs, unicode arrays.
- `train`: [n_positive_fit,2] local user,item integer pairs.
- `observed`: [n_events,3] user,item,stage with stage -1 prefit,0 fit,1 query, any rating.
- `positive`: same schema restricted rating>=4.
- `event_users/items/times/ratings`: aligned full warm catalog events, only time<query_end.
- `all_rating_users/items/times/ratings`: aligned fit-only any-rating events from selected users across ALL items; nonwarm item index=-1.
- `degree`, `tail80`,`tail50`: fit positive degree / lowest ceil(0.8*n),ceil(0.5*n), tie rawitemID.
- `activity_raw`: [n_users,5] log1p counts30/90/365d, recency fractional days,active7d bins (cutoff-1-t)//7d.
- `activity`: standardized by target prefix's `activity_mean`,`activity_scale` (std clamp1e-6).
Globals `tail80`,`tail50`,`degree` alias target arrays; `activity_mean`,`activity_scale`.
All cohort, encoder and activity statistics use strictly earlier events; earliest global user/item event retained, rating>=4 eligibleusers>=3fitpositiveitems. Target6000 deterministic lowest SHA256('2026|cds:users|'+uid), source all eligible. Prefix role offsets270/180/90 train utility;60 validate ridge;30 tune deployment;0 finaldevelopment. Years2015/2016/2017. Each fit365days query30days.

## Model API owned by targets agent
`Config(method='activity_mean',lr=.001,rho=.0001,tau=.1,history_cap=50,decay_days=90,history_temperature=.2,batch_size=32,max_positive=5,max_negative=5)`.
`FeatureStore(data,config,domain='target')`: `.all_features(item)` returns anchor[n_users],phi[n_users,p],activity[n_users,5]; `.pair_batch(item,pos_users,neg_users)` dict pos_anchor,pos_phi,pos_activity,neg_*; `.history(users,item,exclude_focal=False)` returns [users,64].
`Model(config,store,theta=None,beta=None,alpha=1.)`: `.theta`[p],`.beta`[5],`.alpha`; `.score_features(anchor,phi,activity)`; `.score_all(item_ids)` shape[n_items,n_users]; `.save(path)`; `Model.load(path,data,domain='target')`.
`train_target(data,config,seed,epochs=6,validation_callback=None)->(model,history)`; callback reporting only, NEVER selects final epoch. `pair_loss_and_gradient` document signature. Feature formula must be written explicitly since historical phi definition missing; normalized norm<=1. Every warm item per epoch. Training positive focal removed from preference and activity before historycap. Uniform negatives exclude any prior rating, no future labels.

## Evaluation API owned by evaluation agent
`evaluate(data,stage,score_all,ks=(10,20,50),save_topk=True,batch_size=8,item_mask=None)` dict fullcatalog arrays item_id,n_candidates,n_positive,eligible_catalog,requested,scored,evaluated,ranked,ndcgK,recallK,precisionK,hitsK,top_users,stage,candidate_mode. Stable user-ID tie order, all users except any observed stage<stage. Score only requested items with query positive, others NaN/-1, not zeros. `stratified_summaries(data,result)` keys all,tail80,tail50; `summarize(data,result,mask=None)`. Utilities `io.write_json`, `io.sha256` owned evaluation agent.

## Frozen budgets
Targets5families unbounded_mean,activity_mean,activity_decay,activity_conditional,activity_both. HPO12=lr[.001,.003,.01]×rho[0,.0001]×tau[.1,.3], seed101,2015d30,6epochs. Final3origins×5seeds11/22/33/44/55,6fixedepochs. Baselinescosine/activity deterministic eachorigin.
CF MF-BPR(32/64dims ×lr.001/.005/.01 ×l2 1e-5/1e-3); LightGCN(layers1/2 ×same lr/l2,64dims); SimpleX(lr.001/.005 ×historymix.25/.5/.75 ×negativeweight10/50,64dims,10neg),12each. Native user-item objective, transpose score, fixed6epochs; no future callbacks/selection.
Utility fixed independent historical activity_mean lr.001 rho.0001 tau.1 seed1016epochs everyfold; ridge alphas1e-4/1e-3/.01/.1/1/10 using first3folds and d60 validation then refit4. 16 semantic donors sourcecap16 targetsupport5 neg5. Onlytheta sourcecorrection. label radius.1 eta.1. Deployment7policieszero/uniform/semantic/global/behavior/full/predicted_single. GridR0/.01/.05/.1 ×temperature.1/1/10 on2015d30,freeze final3origins5seedsbaseactivity_mean; exactzero. Matchednon-nullmass1/equalnorm.1 diagnostic2×6policies.

## Ownership and outputs
Data: saet_repro/data.py scripts/download_data.py scripts/prepare_data.py tests/test_data.py.
Target: saet_repro/target.py scripts/run_targets.py tests/test_target.py.
Utility: saet_repro/utility.py scripts/run_utility.py tests/test_utility.py.
CF: saet_repro/collaborative.py scripts/run_baselines.py tests/test_collaborative.py.
Eval: saet_repro/evaluation.py io.py statistics.py tests/test_evaluation.py tests/test_statistics.py.
Root: integration/reporting/config/provenance/GitHub.
Result layout targets or baselines/cds_YEAR_d0/METHOD/seed_N/{model.npz,result.json,test.npz}; result summary['1'][stratum]. HPO cds_2015_d30/METHOD/search/trial_N.json,npz + selected.json. Utility coordinates target canonical d30 activity_mean/selected_model.npz and final15models. Everything can resume from completed artifacts. Frozen code/config/data hashes recorded before scoring; never use mainquery for selection.
