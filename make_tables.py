#!/usr/bin/env python3
"""Regenerates every table of the paper from data/*.csv  (python make_tables.py)."""
import pandas as pd, numpy as np
from pathlib import Path
D, T = Path('data'), Path('tables'); T.mkdir(exist_ok=True)
LAB = {'pid_dob':'PID+DOB','physlag_mpc_dob':'Gray-box MPC+DOB','koop_mpc_dob':'Koopman MPC+DOB',
       'koopstab_mpc_dob':'Koopman-stable MPC+DOB','lqr_koopstab_dob':'Koopman-stable+LQR+DOB',
       'lqr_koop_dob':'Koopman(raw)+LQR+DOB'}
def clean(n): return (n.replace('Koopman(brut)','Koopman(raw)').replace('PID+DOB (tuned)','PID+DOB')
                       .replace('(no DOB)','(no DOB)').replace('_','\\_'))
def tab(path, caption, label, header, rows, colspec, small=True):
    L = [r'\begin{table}[H]', r'\centering', r'\small' if small else '', f'\\caption{{{caption}}}', f'\\label{{{label}}}',
         r'\resizebox{\linewidth}{!}{%', f'\\begin{{tabular}}{{{colspec}}}', r'\toprule', ' & '.join(header)+r' \\', r'\midrule']
    L += [' & '.join(r)+r' \\' for r in rows]
    L += [r'\bottomrule', r'\end{tabular}}', r'\end{table}']
    Path(path).write_text('\n'.join(x for x in L if x)+'\n')
f3 = lambda v: '--' if pd.isna(v) else f'{v:.3f}'
def summary_rows(s):
    return [[clean(r.controller), str(int(r.n_fail))+f'/{int(r.n)}', f3(r.RMSE_median),
             f'[{f3(r.RMSE_q25)}, {f3(r.RMSE_q75)}]', f3(r.RMSE_pen_mean), f'{r.CV_med:.1f}', f'{r.CPU_ms:.2f}'] for r in s.itertuples()]
HS = ['Controller','Fail','Median RMSE [m]','IQR [m]',r'Mean RMSE$_\mathrm{pen}$ [m]','CV (median)','CPU [ms/step]']
CS = 'lcccccc'
# ---- main tables ----
for rg, cap in [('mismatch','Held-out test set, \\texttt{mismatch} regime ($100$ trials). Median RMSE and IQR are over successful trials; failed trials enter the penalized mean with $1$\\,m.'),
                ('ideal','Held-out test set, \\texttt{ideal} regime ($32$ trials); no controller fails.')]:
    tab(T/f'table_main_{rg}.tex', cap, f'tab:main_{rg}', HS, summary_rows(pd.read_csv(D/f'summary_{rg}.csv')), CS)
# ---- ablation ----
a = pd.read_csv(D/'summary_ablation.csv')
tab(T/'table_ablation.tex', 'Ablations on the \\texttt{abl} split (\\texttt{mismatch}, $32$ trials): observer and dictionary.',
    'tab:ablation', HS, summary_rows(a), CS)
# ---- noise x2 ----
n2 = pd.read_csv(D/'summary_noise2.csv')
tab(T/'table_noise2.tex', 'Measurement noise doubled (\\texttt{mismatch}, $32$ trials, same trials as the ablation).',
    'tab:noise2', HS, summary_rows(n2), CS)
# ---- open loop ----
v = pd.read_csv(D/'validation_open_loop.csv'); H = [1,5,10,20]
names = ['phys','physlag','koop-lin','koop-drag','koop-lag','koop-full','koopstab']
disp = {'phys':'Physics (no lag)','physlag':'Physics + nominal lag','koop-lin':'Koopman lin','koop-drag':'Koopman drag',
        'koop-lag':'Koopman lag','koop-full':'Koopman full','koopstab':'Koopman full, clipped'}
rows = []
for m in names:
    r = [disp[m]]
    for rg in ['mismatch','ideal']:
        g = v[(v.regime==rg)&(v.model==m)].groupby('horizon').pos_rmse.min()
        r += [f'{g[h]:.4f}' if g[h] < 0.1 else f'{g[h]:.3f}' for h in H]
    rows.append(r)
tab(T/'table_openloop.tex', 'Open-loop free-run position RMSE [m] on validation episodes (plants outside the identification set), '
    'versus prediction horizon $H$ (steps). For the Koopman dictionaries, the best ridge parameter on the validation split is shown.',
    'tab:openloop', ['Model']+[f'$H{{=}}{h}$' for h in H]*2, rows, 'lcccccccc')
# add grouped header
p = T/'table_openloop.tex'; s = p.read_text()
s = s.replace(r'\toprule', r'\toprule'+'\n'+r' & \multicolumn{4}{c}{\texttt{mismatch}} & \multicolumn{4}{c}{\texttt{ideal}} \\'+'\n'+r'\cmidrule(lr){2-5}\cmidrule(lr){6-9}',1)
s = s.replace('Model & ', ' & ',1); p.write_text(s)
# ---- LQR isolation ----
tr = pd.read_csv(D/'test_results.csv')
order = ['koop_mpc_dob','lqr_koop_dob','koopstab_mpc_dob','lqr_koopstab_dob']
rows = []
for c in order:
    mm = tr[(tr.regime=='mismatch')&(tr.ctrl==c)]; ide = tr[(tr.regime=='ideal')&(tr.ctrl==c)]
    ag = ide[ide.family=='agile'].RMSE_pen.mean()
    rows.append([LAB[c], 'MPC' if 'mpc' in c else 'LQR', 'raw' if c in order[:2] else 'clipped',
                 f'{int(mm.fail.sum())}/100', f3(mm[mm.fail==0].RMSE.median()), f3(mm.RMSE_pen.mean()), f'{mm.CPU_mean_ms.mean():.2f}', f'{mm.ControlVariation.median():.1f}',
                 f3(ide.RMSE_pen.mean()), f3(ag)])
pp = pd.read_csv(D/'paired_tests.csv')
pm = pp[(pp.regime=='mismatch')&(pp.a=='koop_mpc_dob')&(pp.b=='lqr_koop_dob')].p_holm.iloc[0]
tab(T/'table_lqr_isolation.tex', 'Isolating the control law from the model: MPC vs.\\ LQR on the raw and on the clipped Koopman model. '
    '\\texttt{mismatch} ($100$ trials): failures, median RMSE over successful trials [m], mean penalized RMSE [m], CPU [ms/step], median CV; '
    '\\texttt{ideal} ($32$ trials): mean penalized RMSE [m], overall and on \\texttt{agile}. '
    f'On the raw model, MPC and LQR are statistically indistinguishable under mismatch (paired Wilcoxon, Holm-adjusted $p={pm:.3f}$), '
    'while LQR is about $10\\times$ cheaper and $3\\times$ smoother; MPC keeps a significant edge in the mismatch-free regime, where receding-horizon preview matters more than disturbance rejection.',
    'tab:lqr_isolation', ['Controller','Law','Model','Fail','Med.\\ RMSE','RMSE$_\\mathrm{pen}$','CPU','CV','Ideal: all','Ideal: agile'], rows, 'lllccccccc')
# ---- paired tests ----
pt = pd.read_csv(D/'paired_tests.csv')
for rg in ['mismatch','ideal']:
    g = pt[pt.regime==rg]; rows = []
    for r in g.itertuples():
        ph = ('$<10^{-4}$' if r.p_holm < 1e-4 else f'{r.p_holm:.3f}') + ('$^{*}$' if r.p_holm < 0.05 else '')
        rows.append([f'{LAB[r.a]} vs.\\ {LAB[r.b]}', f'{r.median_diff:+.3f}', f'[{r.ci_lo:+.3f}, {r.ci_hi:+.3f}]', f'{r.win_a_pct:.0f}', ph])
    tab(T/f'table_paired_{rg}.tex', f'Paired comparisons, \\texttt{{{rg}}} regime ($n={int(g.n.iloc[0])}$): median difference of RMSE$_\\mathrm{{pen}}$ (a$-$b, negative favours a), '
        f'bootstrap $95\\%$ CI, percentage of trials won by a, Holm-adjusted Wilcoxon $p$ ($15$ tests); $^{{*}}p<0.05$ after Holm correction.',
        f'tab:paired_{rg}', ['Comparison (a vs.\\ b)','Median diff. [m]','95\\% CI [m]','a wins [\\%]','$p_\\mathrm{Holm}$'], rows, 'lcccc', small=True)
print('done', sorted(x.name for x in T.glob('*.tex')))

# =====================================================================
# Motivation section: limits of classical control (preliminary study + main study)
# =====================================================================
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
F = Path('figures')
cm = pd.read_csv(D/'prelim'/'canonical_metrics.csv'); cm = cm[cm.controller.isin(['pid','fixed'])]
scn = {'S1':'S1 nominal (circle)','S2':'S2 wind 8--17\\,s (circle)','S3':'S3 $+25\\%$ mass at 14\\,s (fig.-8)'}
cl = {'pid':'Geometric PID','fixed':'Koopman MPC'}
rows = []
for sc in ['S1','S2','S3']:
    for c in ['pid','fixed']:
        r = cm[(cm.scenario==sc)&(cm.controller==c)].iloc[0]
        rows.append([scn[sc] if c=='pid' else '', cl[c], f3(r.RMSE), f3(r.MaxError), f'{r.IAE:.2f}', f'{r.ControlVariation:.2f}'])
tab(T/'table_prelim_scenarios.tex',
    'Preliminary experiment (single noise-free run per cell, $30$\\,s): tracking error and command activity of the classical geometric PID and of a Koopman MPC '
    '(model identified on nominal data). CV: control variation $\\sum\\norm{\\Delta u}^2\\Delta t$.',
    'tab:prelim_scenarios', ['Scenario','Controller','RMSE [m]','Max error [m]','IAE [m\\,s]','CV'], rows, 'llcccc')

z = np.load(D/'prelim'/'canonical_trajectories.npz')
def win(sc, c, a, b):
    e = np.linalg.norm(z[f'{sc}_{c}_p']-z[f'{sc}_{c}_pref'],axis=1); t = z[f'{sc}_{c}_t']; m=(t>=a)&(t<b)
    return float(np.sqrt(np.mean(e[m]**2)))
rows = []
for lab, sc, ws in [('S2 wind','S2',[(0,8),(8,17),(17,30)]),('S3 mass step','S3',[(0,14),(14,30)])]:
    for c in ['pid','fixed']:
        v = [win(sc,c,a,b) for a,b in ws]; v += [np.nan]*(3-len(v))
        rows.append([lab if c=='pid' else '', cl[c]] + [f3(x) for x in v])
tab(T/'table_prelim_windows.tex', 'Windowed position RMSE [m] around the disturbance (same runs as \\cref{tab:prelim_scenarios}). '
    'S2: before ($0$--$8$\\,s), during ($8$--$17$\\,s) and after ($17$--$30$\\,s) the wind; S3: before ($0$--$14$\\,s) and after ($14$--$30$\\,s) the mass step.',
    'tab:prelim_windows', ['Scenario','Controller','Before','During / after','After wind'], rows, 'llccc')

mc = pd.read_csv(D/'prelim'/'monte_carlo.csv'); mc = mc[mc.controller.isin(['pid','fixed'])]
rows = []
for sc, lab in [('S2','Wind (15 trials)'),('S3','Mass step (15 trials)')]:
    for c in ['pid','fixed']:
        g = mc[(mc.scenario==sc)&(mc.controller==c)]
        nfail = (15 - len(g)) + int((g.MaxError > 3.0).sum()); ok = g[g.MaxError <= 3.0]
        rows.append([lab if c=='pid' else '', cl[c], f'{nfail}/15', f3(ok.RMSE.median()),
                     f'[{f3(ok.RMSE.quantile(.25))}, {f3(ok.RMSE.quantile(.75))}]', f3(ok.MaxError.max()), f'{g.ControlVariation.median():.1f}'])
tab(T/'table_prelim_mc.tex',
    'Preliminary randomized trials ($15$ per scenario, measurement noise on, wind amplitude $0.6$--$2$ or mass ratio $0.8$--$1.3$). '
    'A trial is a failure if the run diverged (excluded by the preliminary validity check) or its maximum error exceeds $3$\\,m; '
    'median RMSE, IQR and worst maximum error are over the non-failed trials.',
    'tab:prelim_mc', ['Scenario','Controller','Failures','Median RMSE [m]','IQR [m]','Worst max err. [m]','CV (median)'], rows, 'llcccrc')

tr = pd.read_csv(D/'test_results.csv'); p = tr[tr.ctrl=='pid_dob']
rows = []
for rg in ['ideal','mismatch']:
    for kd in ['nominal','wind','payload','combined']:
        g = p[(p.regime==rg)&(p.kind==kd)]; ok = g[g.fail==0]
        rows.append([rg if kd=='nominal' else '', kd, f'{int(g.fail.sum())}/{len(g)}', f3(ok.RMSE.median()), f3(g.RMSE_pen.mean())])
    g = p[p.regime==rg]; ok = g[g.fail==0]
    rows.append(['', '\\emph{all}', f'{int(g.fail.sum())}/{len(g)}', f3(ok.RMSE.median()), f3(g.RMSE_pen.mean())])
tab(T/'table_pid_degradation.tex',
    'Tuned PID+DOB on the held-out test set, by scenario kind. In the \\texttt{mismatch} regime every kind includes the structural mismatch, '
    'so even the \\texttt{nominal} kind has drag, lag, delay and inertia errors. Failures are counted, never discarded.',
    'tab:pid_degradation', ['Regime','Kind','Failures','Median RMSE [m]','Mean RMSE$_\\mathrm{pen}$ [m]'], rows, 'llccc')

fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
kinds = ['nominal','wind','payload','combined']; x = np.arange(4); w = .38
for i, rg in enumerate(['ideal','mismatch']):
    d = [p[(p.regime==rg)&(p.kind==k)&(p.fail==0)].RMSE.values for k in kinds]
    ax[0].boxplot(d, positions=x+(i-.5)*w, widths=w*.9, showfliers=False, patch_artist=True,
                  boxprops=dict(facecolor=['#9ecae1','#fdae6b'][i]), medianprops=dict(color='k'))
ax[0].set_xticks(x); ax[0].set_xticklabels(kinds, fontsize=8); ax[0].set_yscale('log'); ax[0].set_ylabel('RMSE, successful trials [m]')
ax[0].set_title('(a) accuracy'); ax[0].grid(alpha=.25)
ax[0].legend([plt.Rectangle((0,0),1,1,fc='#9ecae1'), plt.Rectangle((0,0),1,1,fc='#fdae6b')], ['ideal','mismatch'], fontsize=8)
fr = [100*p[(p.regime=='mismatch')&(p.kind==k)].fail.mean() for k in kinds]
ax[1].bar(kinds, fr, color='#fdae6b'); ax[1].set_ylabel('failure rate [%]'); ax[1].set_title('(b) reliability, mismatch'); ax[1].tick_params(axis='x', labelsize=8); ax[1].grid(alpha=.25, axis='y')
fam = ['circle','figure8','agile']; ff = [100*p[(p.regime=='mismatch')&(p.family==f_)].fail.mean() for f_ in fam]
ax[2].bar(fam, ff, color='#fdae6b'); ax[2].set_ylabel('failure rate [%]'); ax[2].set_title('(c) by reference, mismatch'); ax[2].grid(alpha=.25, axis='y')
fig.tight_layout(); fig.savefig(F/'fig_pid_degradation.pdf'); plt.close(fig)
print('motivation tables done')
