# Карта пути и ориентиры

Файлы в этой папке — поставка для жюри, построены по **всем** бэгам с GNSS (train + val).

| Файл | Содержимое |
|---|---|
| `track_map.csv` | главный замкнутый цикл 11 053 м: `s,x,y,z,grade,curvature` в локальной ENU карты (начало координат и `cyclic` — в строке-комментарии) |
| `branch_fan_F2.csv`, `branch_fan_F3.csv`, `branch_wb_detour.csv` | ветки, вливающиеся в главный цикл (`join_s` — дуга точки слияния); нужны, если прогон начинается не на главном пути |
| `landmarks.csv` | места остановок («виртуальные балисы»): `s,sigma,p_stop,cls` |
| `cutoffs.csv` | места резкого сброса тяги (позиция ≥ 4 → 0 на ходу): `s,sigma,n` |

Как получены:

```bash
python analysis/map_build/run_all.py                      # карта по GNSS (train -> map_train, всё -> map)
python tools/map/export_core_map.py --map-dir analysis/map_build/map
python tools/map/cutoff_landmarks.py --map-dir analysis/map_build/map --split all
```

Для честной валидации на val используются карты, построенные только по train:
`analysis/validation_maps/` (их берут `tools/replay/quick_eval.py` и `tools/replay/eval_par.py`).
