"""Bounded, serial, mode-checked research campaign. No live trading authority."""
import argparse, ctypes, datetime, hashlib, json, os, pathlib, re, runpy
import shutil, subprocess, sys, time, yaml

HERE=pathlib.Path(__file__).resolve().parent
REPO=HERE.parent
BASE=pathlib.Path.home()/'AppData/Local/neoethos/experiments'
ROOT=BASE/'codex-takeover-20261026'
ITEMS=runpy.run_path(str(HERE/'research_artifacts.py'))['array_items']
GIB=1024**3

def atomic(path, value):
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf-8')
    tmp.replace(path)

def sha(path):
    with pathlib.Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def emit(state, **data):
    row=dict(time=datetime.datetime.now(datetime.timezone.utc).isoformat(),state=state,**data)
    with (ROOT/'events.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row)+'\n')

def validate_profile(s, profile):
    small=profile['mode']=='risky';r=s['risk'];d=s['models']['discovery_runtime']
    assert (r['initial_balance']<=1000)==small,'Balance/mode mismatch'
    assert s['system']['trading_mode']==profile['mode']
    assert s['models']['discovery_mode']==''
    assert s['models']['search_runtime']['novelty_weight']==0
    assert d['weekly_cadence']==(not small)
    assert d['funnel_stage1_pct']==1.0
    assert s['models']['gene_stop_bounds']['atr_scaled'] is True
    assert s['models']['backtest_runtime']['initial_equity']==r['initial_balance']
    if small:
        assert s['system']['risky_start_balance_usd']==r['initial_balance']
        assert s['system']['risky_target_balance_usd']==50000
        assert s['system']['risky_horizon_days']==180
        assert s['models']['prop_firm_min_pass_rate']==0
        assert d['prop_firm_gate']['pass_rate']==0
        assert (r['risky_min_risk_per_trade'],r['risky_max_risk_per_trade'])==tuple(profile['band'])
    else:
        assert s['models']['prop_firm_min_pass_rate']==0.1
        assert d['prop_firm_gate']['pass_rate']==0.1
        assert d['prop_firm_gate']['max_daily_loss_pct']==0.05
        assert d['prop_firm_gate']['max_overall_drawdown_pct']==0.1
        assert d['prop_firm_gate']['profit_target_pct']==0.08
        assert d['prop_firm_gate']['window_days']==60
        assert r['monthly_profit_target_pct']==0.04
        assert (r['prop_firm_min_risk_per_trade'],r['prop_firm_max_risk_per_trade'])==tuple(profile['band'])

def fnv(raw):
    # Gene JSON contains only numeric/bool fields and whitespace-free identifiers.
    # Preserve the Rust writer's exact number spelling and struct field order.
    h=14695981039346656037
    for b in re.sub(r'\s+','',raw).encode():h=((h^b)*1099511628211)&((1<<64)-1)
    return f'fnv64:{h:016x}'

def selectors(path, wanted):
    found=[]
    for i,(g,raw) in enumerate(ITEMS(path,'candidates',True)):
        if g['strategy_id'] in wanted:found.append(f"{i}:{g['strategy_id']}:{fnv(raw)}")
    assert len(found)==len(wanted),('Missing frozen gene',wanted,found)
    return ','.join(found)

def plan():
    assert not (ROOT/'plan.json').exists(),'Existing campaign plan must not be overwritten'
    original=json.loads((BASE/'eurusd-m15-weekly-20261026o-real-1k-noprop-cons/run-arguments.json').read_text())
    common=['canonical-research']
    for flag in ['--authority-root','--data-root','--plan-sha256','--matrix-sha256','--broker-symbol-contract','--broker-account-snapshot','--symbol','--base-timeframe','--higher']:
        common.extend([flag,original[flag]])
    profiles=[]
    def add(name,balance,band,fitness='monthly_upside_v1',rr=(.8,1.6),hold=128,seed=20261021):
        profiles.append(dict(name=name,mode='risky' if balance<=1000 else 'prop_firm',balance=balance,band=band,fitness=fitness,rr=rr,hold=hold,seed=seed))
    for balance in [100,1000]:
        band=(.02,.10) if balance==100 else (.01,.016)
        tag=f'risky-{balance}'
        add(tag+'-b9-baseline',balance,band)
        add(tag+'-b9-seed2',balance,band,seed=20261029)
        add(tag+'-goal-fitness',balance,band,'ga_fitness')
        add(tag+'-rr1',balance,band,rr=(1.,1.))
        add(tag+'-rr15',balance,band,rr=(1.5,1.5))
        add(tag+'-hold64',balance,band,hold=64)
    add('risky-100-risk10-20',100,(.10,.20))
    add('risky-100-risk30',100,(.30,.30))
    add('risky-1000-risk1-5',1000,(.01,.05))
    add('risky-1000-seed3',1000,(.01,.016),seed=20261103)
    # These replace the previous Codex pair with a Risky-appropriate objective.
    add('codex-risky-1000-risk1-3-rr1',1000,(.01,.03),rr=(1.,1.))
    add('codex-risky-1000-risk1-3-rr15',1000,(.01,.03),rr=(1.5,1.5))
    for label,band in [('risk025-05',(.0025,.005)),('risk05-1',(.005,.01))]:
        for fitness in ['ga_fitness','monthly_upside_v1','monthly_upside_dd_averse_v1']:
            add('prop-100k-'+label+'-'+fitness,100000,band,fitness)
    add('prop-100k-risk1-16-b9',100000,(.01,.016))
    add('prop-100k-risk1-16-ddaverse',100000,(.01,.016),'monthly_upside_dd_averse_v1')
    for p in profiles:
        source='real-1k-noprop-cons-settings.yaml' if p['mode']=='risky' else 'propfirm-100k-settings.yaml'
        s=yaml.safe_load((BASE/'v6-config'/source).read_text())
        r=s['risk'];m=s['models'];d=m['discovery_runtime'];system=s['system']
        system.update(trading_mode=p['mode'],risky_start_balance_usd=float(p['balance']),risky_target_balance_usd=50000.,risky_horizon_days=180)
        system.update(base_timeframe='M15',higher_timeframes=['H1','H4'])
        r.update(initial_balance=float(p['balance']),monthly_profit_target_pct=.04)
        m['backtest_runtime']['initial_equity']=float(p['balance'])
        m['search_runtime'].update(seed=p['seed'],novelty_weight=0.0)
        m['prop_search_max_hours']=0.0 # external per-job deadline bounds the full 200-generation comparison
        m['discovery_ledger']['enabled']=False
        m['seen_signature_runtime']['file_path']=None
        m['discovery_mode']='' # explicit Strict override is absent; the master switch decides
        m['exit_policy'].update(max_hold_bars=p['hold'],trailing_enabled=False)
        m['gene_stop_bounds'].update(atr_scaled=True,sl_min_atr=1.,sl_max_atr=4.,rr_min=p['rr'][0],rr_max=p['rr'][1])
        d.update(weekly_cadence=p['mode']=='prop_firm',funnel_stage1_pct=1.,prefilter_top_k=240,label_cost_symmetric=True)
        gate=d['prop_firm_gate']
        if p['mode']=='risky':
            r.update(risky_min_risk_per_trade=p['band'][0],risky_max_risk_per_trade=p['band'][1])
            m['prop_firm_min_pass_rate']=0.;gate['pass_rate']=0.
        else:
            r.update(prop_firm_min_risk_per_trade=p['band'][0],prop_firm_max_risk_per_trade=p['band'][1],daily_drawdown_limit=.05,total_drawdown_limit=.10)
            m['prop_firm_min_pass_rate']=.1
            gate.update(max_daily_loss_pct=.05,max_overall_drawdown_pct=.10,profit_target_pct=.08,min_trading_days=4,window_days=60,pass_rate=.1)
        validate_profile(s,p)
        dest=ROOT/'configs'/(p['name']+'.yaml');dest.parent.mkdir(exist_ok=True)
        dest.write_text(yaml.safe_dump(s,sort_keys=False),encoding='utf-8')
        p.update(settings=str(dest),settings_sha256=sha(dest))
    frozen=[]
    for name,run,ids in [
        ('frozen-risky-20-label', 'eurusd-m15-weekly-20261026f-risky-20pip-512',['gene_421158_194','gene_331828_97']),
        ('frozen-risky-10-label', 'eurusd-m15-weekly-20261026j-risky-10pip-512',['gene_289252_158'])]:
        path=BASE/run/'research.json'
        frozen.append(dict(name=name,source=str(path),source_sha256=sha(path),selectors=selectors(path,set(ids)),settings=str(BASE/run/'settings.yaml'),
            quarantine='All requested genes use unlagged centered MinMax. Preserve exact genomes; historical IS/OOS is contaminated. Do not replay against a different feature alphabet.'))
    result=dict(schema=1,created=datetime.datetime.now(datetime.timezone.utc).isoformat(),repo=str(REPO),
        exe=str(REPO/'target-dd/debug/neoethos-cli.exe'),exe_sha256=sha(REPO/'target-dd/debug/neoethos-cli.exe'),
        common=common,profiles=profiles,frozen=frozen,population=200,generations=200,candidates=512,cpu_threads=4,
        max_hours=72,max_job_hours=2.5,start_free_gib=18,abort_free_gib=3,job_limit_gib=20,min_disk_gib=18,feature_headroom_bytes=8*GIB,
        holdout_notice='Chronological reserved tail has been inspected in older campaigns. Reuse is diagnostic OOS, not new independent confirmation. No outcome-driven tuning of this predeclared matrix.',
        objectives={'risky':'100 or1000 ->50000 in180days, actual confidence-sized compounding; report floor10 and drawdown',
        'prop_firm':'100000; 4% monthly target;8% operator research target/60days;5% daily,10% total loss limits; min4days; no challenge passed claim from whole-span diagnostics'},
        is_oos_screen={'min_trades':60,'min_profit_factor':1.15,'net_profit_positive':True,'max_frozen_candidates_per_run':4})
    result['feature_change']='All four centered MinMax outputs excluded until a separately versioned causal producer exists. Shared legacy cache disabled.'
    atomic(ROOT/'plan.json',result)
    print('PLANNED',len(profiles),'fresh searches and',len(frozen),'preserved-gene replays')

def available():
    class Memory(ctypes.Structure):
        _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(k,ctypes.c_ulonglong) for k in ['physical','free','page','freepage','virtual','freevirtual','extended']]
    m=Memory();m.length=ctypes.sizeof(m)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):raise ctypes.WinError()
    return m.free/GIB

def competitors(exclude=()):
    command="Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^(neoethos-cli|cargo|rustc)\\.exe$' } | Select-Object -ExpandProperty ProcessId"
    r=subprocess.run(['pwsh','-NoProfile','-Command',command],capture_output=True,text=True,timeout=20)
    if r.returncode:raise RuntimeError('Cannot inspect competing processes')
    return [int(x) for x in r.stdout.split() if int(x) not in exclude]

class Job:
    def __init__(self,limit):
        from ctypes import wintypes as w
        class Basic(ctypes.Structure):
            _fields_=[('a',ctypes.c_longlong),('b',ctypes.c_longlong),('flags',w.DWORD),('minimum',ctypes.c_size_t),('maximum',ctypes.c_size_t),('count',w.DWORD),('affinity',ctypes.c_size_t),('priority',w.DWORD),('scheduling',w.DWORD)]
        class Extended(ctypes.Structure):
            _fields_=[('basic',Basic),('io',ctypes.c_ulonglong*6),('processmem',ctypes.c_size_t),('jobmem',ctypes.c_size_t),('peakprocess',ctypes.c_size_t),('peakjob',ctypes.c_size_t)]
        self.k=ctypes.WinDLL('kernel32',use_last_error=True)
        self.k.CreateJobObjectW.restype=w.HANDLE
        self.k.CreateJobObjectW.argtypes=[ctypes.c_void_p,w.LPCWSTR]
        self.k.SetInformationJobObject.argtypes=[w.HANDLE,ctypes.c_int,ctypes.c_void_p,w.DWORD]
        self.k.AssignProcessToJobObject.argtypes=[w.HANDLE,w.HANDLE]
        self.k.CloseHandle.argtypes=[w.HANDLE]
        self.handle=self.k.CreateJobObjectW(None,None)
        info=Extended();info.basic.flags=0x2200;info.jobmem=int(limit*GIB)
        if not self.handle or not self.k.SetInformationJobObject(self.handle,9,ctypes.byref(info),ctypes.sizeof(info)):raise ctypes.WinError(ctypes.get_last_error())
    def attach(self,p):
        if not self.k.AssignProcessToJobObject(self.handle,int(p._handle)):raise ctypes.WinError(ctypes.get_last_error())
    def close(self):self.k.CloseHandle(self.handle)

def wait_resources(plan,deadline):
    quiet=None
    while time.time()<deadline:
        if (ROOT/'STOP').exists():raise RuntimeError('STOP requested')
        ready=available()>=plan['start_free_gib'] and shutil.disk_usage(ROOT).free>=plan['min_disk_gib']*GIB and not competitors()
        if ready:
            quiet=quiet or time.time()
            if time.time()-quiet>=20:return
        else:quiet=None
        time.sleep(10)
    raise RuntimeError('Campaign time budget reached')

def run_job(plan,name,args,deadline):
    wait_resources(plan,deadline)
    if sha(plan['exe'])!=plan['exe_sha256']:raise RuntimeError('Pinned binary changed; refusing new launches')
    env=os.environ.copy();env['NEOETHOS_PINNED_FEATURE_BUDGET_HEADROOM_BYTES']=str(plan['feature_headroom_bytes'])
    # This campaign owns no shared-cache mutation or eviction policy.
    env.pop('NEOETHOS_FEATURE_FRAME_CACHE_DIR',None);env.pop('NEOETHOS_FEATURE_FRAME_CACHE_MAX_BYTES',None)
    logs=ROOT/'logs';logs.mkdir(exist_ok=True)
    job=Job(plan['job_limit_gib']);p=None
    try:
        with (logs/(name+'.stdout.log')).open('wb') as stdout,(logs/(name+'.stderr.log')).open('wb') as stderr:
            p=subprocess.Popen([plan['exe']]+args,cwd=REPO,env=env,stdout=stdout,stderr=stderr,creationflags=subprocess.CREATE_NO_WINDOW|subprocess.BELOW_NORMAL_PRIORITY_CLASS)
            job.attach(p);emit('running',job=name,pid=p.pid)
            end=min(deadline,time.time()+plan['max_job_hours']*3600);check=0
            while p.poll() is None:
                if available()<plan['abort_free_gib'] or time.time()>end or (ROOT/'STOP').exists():
                    p.kill();raise RuntimeError('Owned job aborted: memory/time/STOP guard')
                if time.time()>check:
                    if competitors((p.pid,)):p.kill();raise RuntimeError('Other heavy work appeared; owned job stopped')
                    check=time.time()+30
                time.sleep(2)
            emit('job_finished',job=name,exit_code=p.returncode)
            return p.returncode
    finally:
        if p and p.poll() is None:p.kill();p.wait()
        job.close()

def screen(path):
    assert not any(re.search(r'(^|_)minmax_',name) for name in ITEMS(path,'effective_feature_names')), 'Noncausal MinMax entered the new feature alphabet'
    rows=[]
    for q in ITEMS(path,'quality_metrics'):
        if q.get('total_trades',0)>=60 and q.get('profit_factor',0)>=1.15 and q.get('net_profit',0)>0:
            a=q.get('monthly_breakdown') or []
            positive=sum(m['net_profit']>0 for m in a)/len(a) if a else 0
            rows.append((positive,q.get('net_expectancy_t_stat',0),q['strategy_id']))
    rows.sort(reverse=True)
    return [x[2] for x in rows[:4]]

def summarize_oos(path, profile, notice):
    import statistics
    doc=json.loads(path.read_text())
    expected={'risky':'Risky','prop_firm':'PropFirm'}[profile['mode']]
    assert doc['mode']==expected,('OOS mode mismatch',doc.get('mode'),expected)
    rows=[]
    for strategy in doc['strategies']:
        metrics=strategy['forward_test'];months=strategy['monthly_breakdown']
        returns=[m['return_pct'] for m in months if m['return_pct'] is not None]
        windows=strategy['goal_windows']
        if profile['mode']=='risky':
            assert windows['mode']=='risky'
            outcomes=windows['windows']
            goal={'window_count':len(outcomes),'target_hits':sum(w['target_reached'] for w in outcomes),
                'floor_hits':sum(w['floor_hit'] for w in outcomes),'best_stopping_balance':max((w['balance_at_stop'] for w in outcomes),default=None)}
        else:
            assert windows['mode']=='prop_firm' and windows['rules']['require_profit_target']
            goal={'window_count':windows['windows'],'challenge_passes':windows['passed'],'empirical_pass_fraction':windows['pass_rate']}
        rows.append(dict(strategy_id=strategy['strategy_id'],exact_gene_hash=strategy['exact_gene_hash'],
            mode=profile['mode'],initial_balance=doc['initial_balance'],net_profit=metrics['net_profit'],
            final_balance=doc['initial_balance']+metrics['net_profit'],max_drawdown_fraction=metrics['max_drawdown'],
            trade_count=metrics['trade_count'],profit_factor=metrics['profit_factor'],
            calendar_months=len(months),positive_months=sum(m['net_profit']>0 for m in months),
            negative_months=sum(m['net_profit']<0 for m in months),zero_months=sum(m['net_profit']==0 for m in months),
            median_month_pct=statistics.median(returns) if returns else None,
            worst_month_pct=min(returns) if returns else None,goal=goal))
    atomic(path.parent/'goal-evaluation.json',dict(profile=profile,scope_notice=notice,strategies=rows))
    emit('financial_review_written',job=profile['name'],strategies=rows)

def run():
    plan=json.loads((ROOT/'plan.json').read_text());statepath=ROOT/'state.json'
    state=json.loads(statepath.read_text()) if statepath.exists() else {'started':time.time(),'completed':[],'failed':[]}
    deadline=state['started']+plan['max_hours']*3600
    emit('campaign_started',fresh_runs=len(plan['profiles']))
    def finish(name,code):
        state['completed' if code==0 else 'failed'].append(name);atomic(statepath,state)
    for frozen in plan['frozen']:
        name=frozen['name']
        if name in state['completed'] or name in state['failed']:continue
        assert sha(frozen['source'])==frozen['source_sha256'],'Original research changed'
        if frozen.get('quarantine'):
            atomic(ROOT/(name+'-quarantine.json'),frozen)
            emit('preserved_gene_quarantined',job=name,reason=frozen['quarantine'])
            finish(name,0);continue
        out=ROOT/'runs'/name
        args=plan['common']+['--settings-source',frozen['settings'],'--out-dir',str(out),'--max-batches','1','--cpu-threads','4','--resume-research',frozen['source'],'--resume-research-sha256',frozen['source_sha256'],'--diagnose-candidate-selectors',frozen['selectors'],'--diagnose-candidate-oos-out',str(out/'frozen-oos.json')]
        finish(name,run_job(plan,name,args,deadline))
    for profile in plan['profiles']:
        name=profile['name']
        if name in state['completed'] or name in state['failed']:continue
        assert sha(profile['settings'])==profile['settings_sha256']
        validate_profile(yaml.safe_load(pathlib.Path(profile['settings']).read_text()),profile)
        out=ROOT/'runs'/name
        args=plan['common']+['--settings-source',profile['settings'],'--out-dir',str(out),'--max-batches','1','--cpu-threads','4','--population','200','--generations','200','--candidates','512','--max-indicators','5','--fitness-table',profile['fitness']]
        code=run_job(plan,name,args,deadline)
        if code:
            finish(name,code)
            if len(state['failed'])>=3:raise RuntimeError('Three failed jobs; stop rather than repeat an unresolved failure')
            continue
        source=out/'research.json';wanted=screen(source)
        if wanted:
            # Freeze using selection-only metrics BEFORE reading any OOS output.
            frozen=dict(source=str(source),sha256=sha(source),ids=wanted,selectors=selectors(source,set(wanted)),profile=profile,scope_notice=plan['holdout_notice'])
            atomic(out/'frozen-before-oos.json',frozen)
            replayout=ROOT/'runs'/(name+'-oos')
            replay=plan['common']+['--settings-source',profile['settings'],'--out-dir',str(replayout),'--max-batches','1','--cpu-threads','4','--resume-research',str(source),'--resume-research-sha256',frozen['sha256'],'--diagnose-candidate-selectors',frozen['selectors'],'--diagnose-candidate-oos-out',str(replayout/'frozen-oos.json')]
            ooscode=run_job(plan,name+'-oos',replay,deadline)
            emit('oos_completed',job=name,exit_code=ooscode,genes=wanted)
            if ooscode:state['failed'].append(name+'-oos')
            else:summarize_oos(replayout/'frozen-oos.json',profile,plan['holdout_notice'])
            redundant=replayout/'research.json'
            if ooscode==0 and redundant.exists() and sha(redundant)==frozen['sha256']:
                assert redundant.resolve().is_relative_to((ROOT/'runs').resolve())
                redundant.unlink() # identical duplicate only; the original IS evidence stays frozen
        else:emit('no_is_candidate',job=name)
        finish(name,0)
    emit('campaign_complete',completed=state['completed'],failed=state['failed'])

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['plan','run','selftest']);args=parser.parse_args()
    if args.action=='plan':plan()
    elif args.action=='selftest':
        job=Job(.125);p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(1)'],creationflags=subprocess.CREATE_NO_WINDOW)
        try:job.attach(p);assert p.wait()==0;print('PASS job attach, free GiB',available())
        finally:job.close()
    else:
        lock=ROOT/'worker.lock'
        with lock.open('x') as f:f.write(str(os.getpid()))
        try:
            run()
        except Exception as e:
            emit('campaign_blocked',error=str(e));raise
        finally:
            if lock.read_text()==str(os.getpid()):lock.unlink()
