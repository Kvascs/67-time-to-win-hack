"""Paired statistics for the accept/reject decisions D30-D33 on per-bag results that already exist.

No replay is run. Inputs:
  build_core/eval/bl_<tag>.csv              per-bag base_link metrics (tools/replay/eval_base_link.py)
  build_core/replay_tmp/bl_<tag>/*_out.csv  saved outputs of the same runs -> speed RMSE vs GNSS doppler
                                             (same metric as tools/replay/speed_vs_doppler.py, cached in
                                             analysis/stats/speed_per_bag.csv)
For every decision: paired difference d = after - before per bag (negative = better), n better / worse / same,
median and mean of d with a bootstrap-by-bag 95 % CI (10 000 resamples), Wilcoxon signed-rank p (scipy) and a
two-sided sign test; Holm correction over all rows.

  python analysis/stats/paired_tests.py            # writes analysis/stats/paired_tests.md and paired_tests.csv
  python analysis/stats/paired_tests.py --respeed  # recompute the speed cache
"""
from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
EVAL = ROOT / 'build_core' / 'eval'
RTMP = ROOT / 'build_core' / 'replay_tmp'
OUT = Path(__file__).resolve().parent
SPEED_CACHE = OUT / 'speed_per_bag.csv'
sys.path.insert(0, str(ROOT / 'tools' / 'replay'))

N_BOOT = 10_000
SEED = 20260927
TOL = {'p3_rmse': 1e-3, 'along_rmse': 1e-3, 'v_rmse': 1e-4}   # |d| below this = "так же" (1 мм, 0.1 мм/с)
METRIC_RU = {'p3_rmse': '3-D RMSE base_link, м', 'along_rmse': 'RMSE вдоль пути, м',
             'v_rmse': 'RMSE скорости против доплера, м/с'}

# (id, label, split, before tag, after tag, note). Tags = eval_base_link.py --tag; same split, same evaluator.
PAIRS = [
    ('D30', 'k по шагу квантования', 'val', 'qk0_val', 'qk_val', 'одна сборка, quant_k_enable=0 / 1'),
    ('D30', 'k по шагу квантования', 'train', 'qk0_train', 'qk_train', 'одна сборка, quant_k_enable=0 / 1'),
    ('D30', 'k по шагу квантования', 'val без GNSS', 'qk0_val_ng', 'qk_val_ng', 'одна сборка, quant_k_enable=0 / 1'),
    ('D31', 'тупик: раннее решение', 'val', 'qk_val', 'se_val', 'соседние сборки'),
    ('D31', 'тупик: раннее решение', 'train', 'qk_train', 'se_train', 'соседние сборки'),
    ('D32', 'тупик: сдвиг дуги +0.77 м', 'val', 'se_val', 'so_val', 'соседние сборки'),
    ('D32', 'тупик: сдвиг дуги +0.77 м', 'train', 'se_train', 'so_train', 'соседние сборки'),
    ('D33', 'поправки по карте отношения тележек', 'val', 'so_val', 'rm3_val', 'соседние сборки, rm3 = сдаваемая'),
    ('D33', 'поправки по карте отношения тележек', 'train', 'so_train', 'rm3_train', 'соседние сборки, rm3 = сдаваемая'),
    ('D31–D33', 'тупик + карта тележек (без GNSS)', 'val без GNSS', 'qk_val_ng', 'rm3_val_ng',
     'прогона so_val_ng нет; D31/D32 на val не меняют ничего (строки D31/D32 val), так что это фактически D33'),
    ('D33-alt', 'отвергнутый «полный» вариант D33 (двигает k)', 'val', 'so_val', 'rm_val', 'отвергнут: 4d487b0d +11.6 м'),
    ('D33-alt', 'отвергнутый «полный» вариант D33 (двигает k)', 'train', 'so_train', 'rm_train', 'отвергнут'),
    ('D30–D33', 'всё вместе (как в plot_report.py)', 'val', 'qk0_val', 'rm3_val', 'сумма D30–D33'),
    ('D30–D33', 'всё вместе', 'train', 'qk0_train', 'rm3_train', 'сумма D30–D33'),
]


# ---------------------------------------------------------------- data
def load_bl(tag: str) -> pd.DataFrame:
    d = pd.read_csv(EVAL / f'bl_{tag}.csv')
    d = d[d['p3_rmse'].notna()].set_index('bag')
    return d[['p3_rmse', 'along_rmse']]


def _speed_one(args):
    tag, f = args
    from speed_vs_doppler import speed_rmse   # tools/replay, reads saved output + data/npz only
    bag = f.name[:-len('_out.csv')]
    return tag, bag, speed_rmse(f, bag)


def speed_table(tags, recompute=False) -> pd.DataFrame:
    cache = pd.read_csv(SPEED_CACHE) if SPEED_CACHE.exists() and not recompute else \
        pd.DataFrame(columns=['tag', 'bag', 'v_rmse'])
    have = set(zip(cache.tag, cache.bag))
    jobs = [(t, f) for t in tags for f in sorted((RTMP / f'bl_{t}').glob('*_out.csv'))
            if (t, f.name[:-len('_out.csv')]) not in have]
    if jobs:
        print(f'speed vs doppler: {len(jobs)} outputs to read ...', flush=True)
        with ProcessPoolExecutor(max_workers=4) as ex:
            rows = list(ex.map(_speed_one, jobs, chunksize=4))
        cache = pd.concat([cache, pd.DataFrame(rows, columns=['tag', 'bag', 'v_rmse'])], ignore_index=True)
        cache.sort_values(['tag', 'bag']).to_csv(SPEED_CACHE, index=False)
    return cache


# ---------------------------------------------------------------- statistics
def boot_ci(d: np.ndarray, rng) -> dict:
    idx = rng.integers(0, len(d), size=(N_BOOT, len(d)))
    s = d[idx]
    med, mean = np.median(s, axis=1), s.mean(axis=1)
    return {'med_lo': np.percentile(med, 2.5), 'med_hi': np.percentile(med, 97.5),
            'mean_lo': np.percentile(mean, 2.5), 'mean_hi': np.percentile(mean, 97.5)}


def paired(before: pd.Series, after: pd.Series, tol: float, rng) -> dict:
    j = pd.concat([before.rename('b'), after.rename('a')], axis=1, join='inner').astype(float).dropna()
    d = (j.a - j.b).to_numpy(dtype=float)
    d = np.where(np.abs(d) < tol, 0.0, d)
    nb, nw = int((d < 0).sum()), int((d > 0).sum())
    r = {'n': len(d), 'better': nb, 'worse': nw, 'same': len(d) - nb - nw,
         'before_med': float(j.b.median()), 'after_med': float(j.a.median()),
         'before_mean': float(j.b.mean()), 'after_mean': float(j.a.mean()),
         'd_med': float(np.median(d)), 'd_mean': float(d.mean()),
         'worst_bag': j.index[int(np.argmax(d))] if nw else '', 'worst_d': float(d.max()) if nw else 0.0}
    r.update(boot_ci(d, rng))
    if nb + nw == 0:
        r.update(p_wilcoxon=1.0, wilcoxon_method='все пары совпали', p_sign=1.0)
    else:
        # zeros dropped by hand (Wilcoxon's rule) so that scipy's 'auto' can use the exact null distribution
        # (n <= 50 and no tied |d|); with zeros left in, scipy falls back to the normal approximation even for n = 3
        nz = d[d != 0]
        w = stats.wilcoxon(nz, alternative='two-sided')
        ties = len(np.unique(np.abs(nz))) < len(nz)
        r['p_wilcoxon'] = float(w.pvalue)
        r['wilcoxon_method'] = 'нормальное прибл.' if (len(nz) > 50 or ties) else 'точный'
        r['p_sign'] = float(stats.binomtest(nb, nb + nw, 0.5).pvalue)
    return r


def holm(p: np.ndarray) -> np.ndarray:
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    run = 0.0
    for k, i in enumerate(order):
        run = max(run, (m - k) * p[i])
        adj[i] = min(run, 1.0)
    return adj


# ---------------------------------------------------------------- formatting
def g3(x: float) -> str:
    if x is None or not np.isfinite(x):
        return '—'
    if x == 0:
        return '0'
    s = f'{x:#.3g}'
    if s.endswith('.'):
        s = s[:-1]
    if 'e' in s:
        m, e = s.split('e')
        s = f'{m}·10^{int(e)}'
    return s.replace('-', '−')


def verdict(r) -> str:
    if r['better'] + r['worse'] == 0:
        return 'нет различий'
    if r['p_wilcoxon'] >= 0.05:
        return 'не значимо'
    better = r['d_med'] < 0 or (r['d_med'] == 0 and r['d_mean'] < 0)
    v = '**лучше' if better else '**хуже'
    v += ', значимо и с поправкой**' if r['p_holm'] < 0.05 else '**, p < 0.05 только без поправки'
    if r['d_med'] != 0 and np.sign(r['d_mean']) != np.sign(r['d_med']):
        v += '; среднее — в другую сторону (выброс)'
    return v


def main():
    rng = np.random.default_rng(SEED)
    tags = sorted({t for p in PAIRS for t in (p[3], p[4])})
    missing = [t for t in tags if not (EVAL / f'bl_{t}.csv').exists()]
    if missing:
        raise SystemExit(f'missing eval files: {missing}')
    bl = {t: load_bl(t) for t in tags}
    sp = speed_table(tags, recompute='--respeed' in sys.argv)
    for t in tags:
        v = sp[sp.tag == t].set_index('bag').v_rmse
        bl[t] = bl[t].join(v, how='left')     # speed only for bags that have an RTK row (same bag set)

    rows = []
    for dec, label, split, tb, ta, note in PAIRS:
        for m in ('p3_rmse', 'along_rmse', 'v_rmse'):
            r = paired(bl[tb][m], bl[ta][m], TOL[m], rng)
            r.update(decision=dec, label=label, split=split, before=tb, after=ta, metric=m, note=note)
            rows.append(r)
    res = pd.DataFrame(rows)
    tested = (res.better + res.worse) > 0          # rows with all-equal pairs are not tests
    res['p_holm'] = np.nan
    res.loc[tested, 'p_holm'] = holm(res.loc[tested, 'p_wilcoxon'].to_numpy())
    res.to_csv(OUT / 'paired_tests.csv', index=False)

    L = ['# Парные тесты решений D30–D33 по бэгам', '',
         'Скрипт: `analysis/stats/paired_tests.py` (новых прогонов нет — только готовые файлы). '
         'Разность Δ = после − до по каждому бэгу, **отрицательная Δ — улучшение**. '
         f'ДИ 95 % — бутстреп по бэгам ({N_BOOT} повторов, перцентильный). '
         'p Уилкоксона — `scipy.stats.wilcoxon`, двусторонний, нулевые разности отброшены (точный при n ≤ 50 без '
         'связок); знаковый тест — биномиальный, двусторонний; p Холма — поправка на множественность по всем строкам, '
         'где есть хотя бы одно различие. '
         f'«Так же» — |Δ| < {TOL["p3_rmse"] * 1000:g} мм (положение) или < {TOL["v_rmse"] * 1000:g} мм/с (скорость). '
         'Значимо — p Уилкоксона < 0.05 (без поправки; с поправкой см. столбец Холма). Числа — 3 значащие цифры.', '',
         '| Решение | Выборка | Метрика | n | лучше / хуже / так же | медиана до → после | медиана Δ [95 % ДИ] '
         '| среднее Δ [95 % ДИ] | p Уилкоксона | p знаков | p Холма | Вывод |',
         '|---|---|---|---|---|---|---|---|---|---|---|---|']
    for _, r in res.iterrows():
        L.append(f"| {r.decision} {r.label} | {r.split} | {METRIC_RU[r.metric]} | {r.n} | "
                 f"{r.better} / {r.worse} / {r.same} | {g3(r.before_med)} → {g3(r.after_med)} | "
                 f"{g3(r.d_med)} [{g3(r.med_lo)}; {g3(r.med_hi)}] | {g3(r.d_mean)} [{g3(r.mean_lo)}; {g3(r.mean_hi)}] | "
                 f"{g3(r.p_wilcoxon)} | {g3(r.p_sign)} | {g3(r.p_holm)} | {verdict(r)} |")
    L += ['', '## Какие файлы сопоставлены', '',
          '| Решение | Выборка | до | после | примечание |', '|---|---|---|---|---|']
    for dec, label, split, tb, ta, note in PAIRS:
        L.append(f'| {dec} | {split} | `build_core/eval/bl_{tb}.csv` + `replay_tmp/bl_{tb}/` '
                 f'| `build_core/eval/bl_{ta}.csv` + `replay_tmp/bl_{ta}/` | {note} |')
    L += ['', 'Все пары — один оценщик (`tools/replay/eval_base_link.py`, эталон как у судьи: base_link, выравнивание '
          'по времени), одна выборка, одни и те же бэги (val 15, train 46 с RTK). Скорость — `speed_vs_doppler.py` '
          '(как публикуется, ближайший выход в 0.05 с к меткам доплера master), кэш `analysis/stats/speed_per_bag.csv`. '
          'Бэг организаторов — один (n = 1), парный тест на нём невозможен; его числа есть в DECISIONS §4.6–4.8.',
          '', 'Не использовано: `bl_rm2_*` — промежуточный вариант D33 без описания в документах (цифры почти как rm3); '
          '`bl_qk0_val` против `bl_rm3_val` из `plot_report.py` — это сумма D30–D33 (строка «всё вместе»), а не D30.',
          '', '## Оговорки', '',
          '- D31 не меняет ни одного бэга val и train (все Δ = 0), D32 — только три тупиковых прогона train. '
          'При трёх ненулевых парах наименьшее возможное двустороннее p Уилкоксона — 0.25, знакового теста — 0.25: '
          'эти решения статистически по бэгам не проверяемы, их обоснование — бэг организаторов (n = 1) и геометрия.',
          '- Бэги не вполне независимы (один маршрут, общие места карты), бутстреп по бэгам это не учитывает; '
          'ДИ медианы при n = 15 грубый (перцентили дискретны).',
          '- D30 val: в DECISIONS §4.6 «9 лучше, 5 хуже, 1 так же» — там 30639_9c362687 (+0.0021 м) считан как «так же»; '
          'здесь порог 1 мм, поэтому 9 / 6 / 0.',
          '- Метрики уже выровнены по времени и в системе судьи (base_link); скорость — как публикуется (с упреждением).',
          '']
    (OUT / 'paired_tests.md').write_text('\n'.join(L), encoding='utf-8')
    with pd.option_context('display.width', 250, 'display.max_columns', 30):
        print(res[['decision', 'split', 'metric', 'n', 'better', 'worse', 'same', 'd_med', 'med_lo', 'med_hi',
                   'd_mean', 'mean_lo', 'mean_hi', 'p_wilcoxon', 'p_sign', 'p_holm']].round(4).to_string())


if __name__ == '__main__':
    main()
