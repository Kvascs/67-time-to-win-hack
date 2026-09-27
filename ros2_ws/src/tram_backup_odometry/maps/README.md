# Карта пути и ориентиры

Файлы в этой папке — поставка для жюри, построены по **всем** бэгам с GNSS (train + val).

| Файл | Содержимое |
|---|---|
| `track_map.csv` | главный замкнутый цикл 11 053 м: `s,x,y,z,grade,curvature` в локальной ENU карты (начало координат и `cyclic` — в строке-комментарии) |
| `branch_fan_F2.csv`, `branch_fan_F3.csv`, `branch_wb_detour.csv` | ветки, вливающиеся в главный цикл (`join_s` — дуга точки слияния); нужны, если прогон начинается не на главном пути |
| `landmarks.csv` | места остановок («виртуальные балисы»): `s,sigma,p_stop,cls` |
| `cutoffs.csv` | места резкого сброса тяги (позиция ≥ 4 → 0 на ходу): `s,sigma,n` |
| `stub_west_arrival_2.csv` | тупиковый путь у западной конечной (125 м), на котором кончаются некоторые записи; `join_s` — дуга главного цикла, где он начинается |
| `dfield.csv` | выученное поле невязки модели тяги по месту: `s,value` (ячейки 10 м) |
| `gl_stops.csv`, `gl_cutoffs.csv`, `speed_envelope.csv` | признаки для поиска места без GNSS: места остановок, места сброса тяги, огибающая скорости |
| `wheel_epochs.csv` | шаг квантования скорости тележек по вагонам и эпохам колёс: `vehicle,date,c_front,c_rear,n`; масштаб колеса k = q / c − 1 (6 эпох двух вагонов) |
| `ratio_map.csv` | выученная карта отношения скоростей тележек по 1 м пути главного цикла: `bin,mu,sd,rel` (z = log(v_перед/v_зад) / σ_v(v)), в заголовке — длина цикла и σ_v по скорости; `rel = 0` — поправки там не делаются (проверка leave-one-out по train) |

Как получены:

```bash
python analysis/map_build/run_all.py                      # карта по GNSS (train -> map_train, всё -> map)
python tools/map/export_core_map.py --map-dir analysis/map_build/map
python tools/map/cutoff_landmarks.py --map-dir analysis/map_build/map --split all
python tools/map/export_stub.py --map-dir analysis/map_build/map
python analysis/ideas_check/bogie_correlation/s5_quantum_k.py   # шаг квантования и k по бэгам
python tools/map/export_wheel_epochs.py                         # wheel_epochs.csv (всё) и validation_maps (train)
python analysis/ideas_check/ratio_map_correction/build_map.py   # суммы карты отношения по прогонам
python analysis/ideas_check/ratio_map_correction/reliab.py      # надёжность leave-one-out по train
python tools/map/export_ratio_map.py                            # ratio_map.csv (всё) и validation_maps (train)
```

Для честной валидации на val используются карты, построенные только по train:
`analysis/validation_maps/` (их берут `tools/replay/quick_eval.py` и `tools/replay/eval_par.py`).
