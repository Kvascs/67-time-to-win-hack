"""OSM view of tram route 10 (Metro Shchukinskaya -> Ulitsa Kulakova): member ways, maxspeed profile, stops.

Data: OpenStreetMap contributors, ODbL (https://www.openstreetmap.org/copyright). Cached Overpass answers in osm_cache/:
  osm_rel_1224026.json : relation(1224026); out body; >; out skel qt;
  osm_tram.json        : tram stops / stop positions / switches / railway=tram ways / tram route relations in
                         bbox (55.790,37.375,55.816,37.475), 'out tags center qt'
Used in competitions_prior_art.md, section 1.3. The stop 's' values are coarse (nearest way node).
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'analysis', 'map_build'))
import geo  # noqa: E402

base = os.path.join(HERE, 'osm_cache')
d = json.load(open(os.path.join(base, 'osm_rel_1224026.json'), encoding='utf-8'))
t = json.load(open(os.path.join(base, 'osm_tram.json'), encoding='utf-8'))
wtags = {e['id']: e.get('tags', {}) for e in t['elements'] if e['type'] == 'way'}
stopname = {e['id']: e.get('tags', {}).get('name') for e in t['elements'] if e['type'] == 'node'}
els = d['elements']
rel = [e for e in els if e['type'] == 'relation'][0]
nodes = {e['id']: (e['lat'], e['lon']) for e in els if e['type'] == 'node'}
wayn = {e['id']: e['nodes'] for e in els if e['type'] == 'way'}
lat0, lon0 = 55.81, 37.46


def enu(ll):
    la = np.array([p[0] for p in ll])
    lo = np.array([p[1] for p in ll])
    e, n, _ = geo.geodetic_to_enu(la, lo, np.zeros_like(la), lat0, lon0, 0.0)
    return e, n


S = 0.0
prof, allpts = [], []
for m in rel['members']:
    if m['type'] != 'way':
        continue
    ns = wayn.get(m['ref'])
    if not ns:
        continue
    e, n = enu([nodes[x] for x in ns if x in nodes])
    L = float(np.sum(np.hypot(np.diff(e), np.diff(n))))
    tg = wtags.get(m['ref'], {})
    prof.append((round(S), round(S + L), tg.get('maxspeed'), tg.get('service')))
    allpts += [(x, S) for x in ns if x in nodes]
    S += L
print('route 10 (Shchukinskaya -> Kulakova) OSM length ~ %.0f m, member ways %d' % (S, len(prof)))
mp = []
for a, b, v, _ in prof:
    if mp and mp[-1][2] == v:
        mp[-1] = (mp[-1][0], b, v)
    else:
        mp.append((a, b, v))
print('maxspeed profile [s0, s1, km/h]:', mp)
for m in rel['members']:
    if m['type'] == 'node' and m.get('role', '').startswith('stop') and m['ref'] in nodes:
        la, lo = nodes[m['ref']]
        best = min(allpts, key=lambda p: (nodes[p[0]][0] - la) ** 2 + ((nodes[p[0]][1] - lo) * 0.56) ** 2)
        print('  stop %-40s approx s = %.0f m' % (stopname.get(m['ref']), best[1]))
