# Карты для честной валидации

Карта пути, ветки, балисы (остановки) и места отсечки тяги, построенные **только по train-бэгам**
(`analysis/map_build/map_train`, `tools/map/cutoff_landmarks.py --split train`).
Все офлайн-метрики на val (`tools/replay/quick_eval.py`, `tools/replay/eval_par.py`) считаются на этих картах,
чтобы траектории val не попадали в карту.

В пакет (`ros2_ws/src/tram_backup_odometry/maps`) для жюри кладётся карта по всем данным
(`analysis/map_build/map`, train + val), в ней есть веерный путь F3.
