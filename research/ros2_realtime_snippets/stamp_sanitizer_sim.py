"""Per-topic stamp sanitizer v2: monotonic + bounded forward jump + resync.
accept iff last < stamp <= last + rate*(recv-last_recv) + TOL ; K consistent rejects -> resync."""
import numpy as np, glob
TOL = 0.35; K = 8; RATE = 1.0

def sanitize(recv, stamp):
    acc = np.zeros(len(recv), bool)
    last_s = last_r = None; pend = []
    for i, (r, s) in enumerate(zip(recv, stamp)):
        if s <= 0: continue
        if last_s is None:
            acc[i] = True; last_s, last_r = s, r; continue
        ok = (s > last_s) and (s - last_s <= RATE * (r - last_r) + TOL)
        if ok:
            acc[i] = True; last_s, last_r = s, r; pend = []
        else:
            # candidate resync chain: consistent among themselves
            if pend and (s > pend[-1][0]) and (s - pend[-1][0] <= RATE * (r - pend[-1][1]) + TOL):
                pend.append((s, r))
            else:
                pend = [(s, r)]
            if len(pend) >= K:
                acc[i] = True; last_s, last_r = s, r; pend = []
    return acc

tot = {}
bad_examples = []
for f in sorted(glob.glob('C:/MosTransHack/data/npz/*.npz')):
    d = np.load(f)
    for k in ['vehicle__front_bogie_velocity', 'vehicle__rear_bogie_velocity', 'vehicle__driver_position_cmd']:
        a = d[k]
        if len(a) < 20: continue
        acc = sanitize(a[:, 0], a[:, 1])
        t = tot.setdefault(k, [0, 0, 0])
        t[0] += len(a); t[1] += int((~acc).sum())
        s = a[acc, 1]; t[2] += int((np.diff(s) <= 0).sum())
        if (~acc).sum() > 0: bad_examples.append((f[-18:-4], k[9:14], int((~acc).sum())))
for k, (n, rej, nonmono) in tot.items():
    print(f'{k}: msgs={n} rejected={rej} ({100*rej/n:.3f}%) non-monotonic after sanitize={nonmono}')
print('bags with rejections:', len(bad_examples)); print(bad_examples[:30])
