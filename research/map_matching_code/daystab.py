"""Are stop landmarks stable across recording days?"""
import sys, pickle, datetime
sys.path.insert(0, 'C:/MosTransHack/research/map_matching_code')
import numpy as np
import evaldr2 as E

day = {}
for n, c in E.C.items():
    day[n] = datetime.datetime.fromtimestamp(c['tw'][0], datetime.timezone.utc).strftime('%m-%d')
print('runs per day:', {d: sum(1 for v in day.values() if v == d) for d in sorted(set(day.values()))})
for dirn in ('AB', 'BA'):
    D = E.dbs(dirn, 'none')['stop']
    ev = [e for e in E.EV if e[1] == dirn and e[0] in day]
    print('==', dirn)
    for s0, sig, frac in D:
        if frac < 0.6:
            continue
        per = {}
        for e in ev:
            if abs(e[2] - s0) < 5:
                per.setdefault(day[e[0]], []).append(e[2])
        txt = '  '.join('%s: %+.2f (n=%d)' % (d, np.median(v) - s0, len(v)) for d, v in sorted(per.items()))
        print('  stop s=%7.1f sigma %.2f | per-day median offset: %s' % (s0, sig, txt))
