# Инструкция для жюри: сборка, запуск, проверка

Пакеты ROS 2 Humble (C++ и Python) собираются стандартным `colcon build` без доступа в интернет.
Решение публикует `/result/velocity` и `/result/position` в реальном времени, пока проигрывается bag.

## 1. Состав

| Пакет | Назначение |
|---|---|
| `tram_vehicle_msgs` | сообщения организаторов (без изменений) |
| `tbo_msgs` | `EstimatorStatus` — диагностика оценщика (флаги проскальзывания/юза, режимы, сцепление, задержка) |
| `tram_backup_odometry` | нода `tbo_node` (C++), ядро оценщика, параметры, карта пути, launch |
| `tram_backup_odometry_tools` | `latency_probe` (задержка/частота/CPU/RAM), `evaluate_run` (точность против GNSS из bag) |

## 2. Сборка

Вариант А — ROS 2 Humble установлен (Ubuntu 22.04):

```bash
mkdir -p ~/tbo_ws && cp -r <repo>/ros2_ws/src ~/tbo_ws/
cd ~/tbo_ws
source /opt/ros/humble/setup.bash
colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release
source install/setup.bash
```

Зависимости только стандартные: `rclcpp`, `nav_msgs`, `sensor_msgs`, `diagnostic_msgs`, `rosidl_default_generators`,
`rosbag2_py` (всё входит в `ros-humble-ros-base`) и `python3-numpy`. `python3-psutil` необязателен (нужен только для замера CPU/RAM).

Вариант Б — Docker (из корня репозитория):

```bash
docker build -t tram_backup_odometry .
```

При сборке образа выполняются модульные тесты ядра (27 тестов). Если хоть один не пройдёт, сборка упадёт.

## 3. Запуск

Терминал 1 — нода:

```bash
ros2 launch tram_backup_odometry tram_backup_odometry.launch.py
```

Терминал 2 — bag (любым удобным способом; `--clock` не требуется, время берётся из `header.stamp` сообщений):

```bash
ros2 bag play <путь к bag>
```

Нода сама начинает с первого сообщения, никаких ручных действий не нужно. Проигрывание bag с начала
(в том числе по кругу через `--loop`) нода распознаёт по скачку времени назад и начинает новый прогон.
Так же распознаётся другой bag, если он записан позже (скачок вперёд ≥ 60 с). Если все входы замолкают
на несколько секунд, нода продолжает прогон: модель перекрывает паузу. Одиночные сообщения с мусорной меткой
времени отбрасываются.

## 4. Выходные топики

| Топик | Тип | Содержимое |
|---|---|---|
| `/result/velocity` | `tram_vehicle_msgs/VelocitySensor` | продольная скорость, м/с (`velocity`), `frame_id = base_link` |
| `/result/position` | `nav_msgs/Odometry` | положение `base_link` (центр передней тележки, уровень рельса) в системе карты Autoware **MGRS 37U CB** (x = E−300000, y = N−6100000 в UTM 37N, без переноса через 100 км; z — высота антенн по GNSS − 3.0 м); ориентация; ковариации; `twist.linear.x` = скорость |
| `/result/status` | `tbo_msgs/EstimatorStatus` | флаги (`FLAG_*_SLIP/SLIDE/DROPOUT/STUCK/...`), вероятности 5 режимов, коэффициент скольжения каждой тележки, использованное сцепление, время обработки |
| `/diagnostics` | `diagnostic_msgs/DiagnosticArray` | 1 Гц: OK/WARN со словесным статусом («обнаружено проскальзывание», «только модель», ...), счётчики |

Частота публикации ≈ 40 Гц: на каждой метке контроллера и колёс плюс сетка 50 мс. Метки строго возрастают,
`header.stamp` = время соответствующего входа из bag. Позиция публикуется после привязки по первым фиксам GNSS
(обычно первые 0.1–0.5 с). Если GNSS нет, через 4 с нода переходит в относительную одометрию от старта.

Система координат и точка задаются параметрами (`config/params.yaml`):
`output_frame` (`mgrs` по умолчанию, также `enu`, `utm`, `map`), `base_link_along_m`, `base_link_height_m`.

## 5. Логи, метрики, задержка

Статус в реальном времени:

```bash
ros2 topic echo /diagnostics
ros2 topic echo /result/status --field flags
ros2 topic hz /result/velocity
```

Задержка «вход → публикация», частота, CPU и RAM ноды (печатает сводку каждые 5 с, пишет CSV по каждому выходу):

```bash
ros2 run tram_backup_odometry_tools latency_probe --csv latency.csv
```

Точность по записанному прогону. Во время проигрывания запишите выходы, затем сравните с GNSS-эталоном из того же bag:

```bash
ros2 bag record -o run_out /result/velocity /result/position /result/status
ros2 run tram_backup_odometry_tools evaluate_run --input-bag <путь к bag> --output-bag run_out
```

`evaluate_run` сравнивает с антенной GNSS (ENU от первого фикса), поэтому для этой проверки запускайте ноду в режиме антенны:
`-p output_frame:=enu -p base_link_along_m:=0.0 -p base_link_height_m:=0.0`.

Всё сразу — один прогон с замером под ограничениями жюри (2 ядра, 0.5 ГБ):

```bash
docker run --rm --cpus=2 --memory=512m -v <каталог с bag-ами>:/bags:ro -v $(pwd)/out:/out \
    tram_backup_odometry bash -lc "ros2 run tram_backup_odometry check_run.sh <имя bag> 120"
```

## 6. Проверка без ROS (офлайн, то же ядро)

```bash
cmake -S ros2_ws/src/tram_backup_odometry/core -B build_core -DCMAKE_BUILD_TYPE=Release
cmake --build build_core -j
./build_core/tbo_core_tests                     # 27 тестов
python tools/replay/quick_eval.py               # метрики на отложенных bag-ах (нужен экспорт npz: tools/extract_bags.py)
```
