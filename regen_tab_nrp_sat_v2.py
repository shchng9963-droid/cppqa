"""Regenerate tab_nrp_sat.tex from results/nrp_round_sweep_v2.csv (10 seeds).

This v2 version reports CP-PQA-only saturation (no MOQA-v0 row), to avoid
mixing budgets with the main tab_rq2_hv. The new sweep shows that on every
NRP scale, HV is already saturated at N_R=10; further rounds only fill the
curator interior (|A| grows with N_R but HV does not).
"""
import csv, os, statistics
from collections import defaultdict

CSV = os.path.join(os.path.dirname(__file__), 'results', 'nrp_round_sweep_v2.csv')
OUT = os.path.join(os.path.dirname(__file__), 'results', 'tab_nrp_sat.tex')

hv_bag  = defaultdict(list)
arc_bag = defaultdict(list)
with open(CSV) as f:
    rdr = csv.DictReader(f)
    for r in rdr:
        hv_bag [(r['problem'], r['config'])].append(float(r['HV']))
        arc_bag[(r['problem'], r['config'])].append(int(r['archive']))

PRETTY = {
    'nrp_large_v2'  : 'NRP-100',
    'nrp_xlarge_v2' : 'NRP-200',
    'nrp_xxlarge_v2': 'NRP-500',
}
CFGS = ['CP-PQA-R10','CP-PQA-R20','CP-PQA-R40','CP-PQA-R80']

def fmt(v):
    if v >= 1e6: return f"{v/1e6:.2f}M"
    if v >= 1e3: return f"{v/1e3:.0f}k"
    return f"{v:.0f}"

rows = []
for pkey in ['nrp_large_v2','nrp_xlarge_v2','nrp_xxlarge_v2']:
    cells = [PRETTY[pkey]]
    for c in CFGS:
        hvs = hv_bag[(pkey, c)]
        ars = arc_bag[(pkey, c)]
        m  = statistics.mean(hvs)
        s  = statistics.stdev(hvs) if len(hvs)>1 else 0.0
        ma = statistics.mean(ars)
        cells.append(f"{fmt(m)}\\,/\\,{ma:.0f}")
    rows.append(' & '.join(cells) + r' \\')

tex = r"""\begin{table}[t]\centering\footnotesize
\caption{RQ9 (saturation). CP-PQA hypervolume\,/\,mean archive size $|\mathcal{A}|$ across $N_R\in\{10,20,40,80\}$ rounds, $10$ seeds. HV is already saturated at $N_R\!=\!10$ on every NRP scale; additional rounds enlarge $|\mathcal{A}|$ (curator interior fills toward $\kappa\!=\!64$) but leave HV unchanged. This table reports CP-PQA saturation only and uses a uniform $200$ reads / $16$ weights with a sweep-internal reference point; the absolute CP-PQA-vs.-MOQA-v0 ranking on \textsc{nrp} is given in \cref{tab:rq2-hv}, which uses the per-instance budget reported in \cref{tab:rq3-time} and a union-over-all-baselines reference point and is therefore on a different HV scale. Because each row uses a per-instance, sweep-internal reference, absolute HV magnitudes are not comparable across rows.}
\label{tab:nrp-sat}
\begin{tabular}{l|cccc}
\toprule
Instance & R10 & R20 & R40 & R80 \\
\midrule
""" + '\n'.join(rows) + r"""
\bottomrule
\end{tabular}\end{table}
"""
with open(OUT, 'w') as f:
    f.write(tex)
print('Wrote', OUT)
