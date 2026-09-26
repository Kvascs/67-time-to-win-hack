# ROS 2 Humble: реализация, реальное время, измерение задержки, упаковка

**Кейс:** «Резервная одометрия по модели» (хакатон Московского транспорта, дедлайн 27.09.2026 23:59 МСК).
**Зона отчёта:** всё, что относится к ROS 2: структура пакетов, узлы, QoS, время и метки, публикация, `nav_msgs/Odometry`, диагностика, параметры и launch, измерение задержки, частоты и ресурсов, тесты, Docker, офлайн-реплей и инструкция для жюри. Модель тяги, динамики и проскальзывания разобрана в соседних отчётах. Здесь описан только «каркас» для неё.

**Приложения (проверенный код и скрипты):** `C:\MosTransHack\research\ros2_realtime_snippets\`
- `rt_utils.hpp`: ROS-независимые помощники без аллокаций. `StampSanitizer`, `RingBuffer`, `LatencyHistogram`, `fill_pose_covariance`. Код скомпилирован (g++ 16, `-std=c++17 -Wall -Wextra`) и прошёл тесты на реальных метках времени из bag-файлов.
- `test_rt_utils.cpp`: юнит-тесты и прогон по CSV из bag. Стоимость `check()` меньше 1 мкс, 15–25 мкс вместе с разбором CSV.
- `stamp_sanitizer_sim.py`: та же логика на Python, прогон по всем 122 bag.
- `dataset_timing_audit.py`: воспроизводит все цифры из раздела 2.

---

## 0. TL;DR: ключевые решения

1. **Язык и исполнитель.** Рабочий узел на **C++ (rclcpp)**, `SingleThreadedExecutor`, одна взаимоисключающая группа колбэков. Без потоков и мьютексов, значит без гонок. Ядро оценивателя вынесено в **ROS-независимую библиотеку** с явной передачей времени (clock injection). Одно и то же ядро работает в онлайн-узле, в офлайн-реплее (`rosbag2_cpp::Reader`, десятки часов данных за секунды) и в gtest. Python используется для инструментов: latency probe, метрики, калибровка.
2. **QoS.** Подписки `best_effort` + `KeepLast(100)`: такая подписка совместима и с reliable, и с best-effort издателем. Издатели `reliable` + `KeepLast(10)`: совместимы с подпиской судьи (best-effort) и с `ros2 bag record`.
3. **Без `message_filters::ApproximateTime`.** Эта политика добавляет задержку порядка периода, то есть около 100 мс при 10 Гц, и перестаёт выдавать данные, если один топик пропал. Вместо неё асинхронное слияние: каждое измерение обрабатывается по своей метке. Передняя и задняя тележки парятся по **точному совпадению** `header.stamp` (совпадает в 99,8 % сообщений).
4. **Модель времени (по результатам анализа данных).** Метки колёс совпадают по временной базе с метками GNSS: лучшая задержка в кросс-корреляции 0 ± 10 мс. При этом колёсные сообщения приходят в bag с «плавающей» задержкой 20–90 мс. Метки `driver_position_cmd` почти равны времени bag (лаг 1–2 мс). Отсюда схема: **фильтр работает во времени меток колёс**, `cmd` подаётся как кусочно-постоянное управление (ZOH), а **выход публикуется по каждому `cmd` (20 Гц) с меткой `cmd` и состоянием, предсказанным моделью на эту метку.** Горизонт предсказания p50 = 0,104 с, p99 = 0,207 с. Задержка «вход → выход» в этой схеме равна времени обработки, меньше 1 мс.
5. **Санитайзер меток** (проверен на всех 122 bag). Сообщение принимается, если `last < s ≤ last + r·Δrecv + 0,35 с`. После K = 8 согласованных отказов подряд происходит пересинхронизация. Отбрасывается 0,005 % сообщений, и все они «призраки» с ±1 с. Выходные метки строго монотонны.
6. **Сторожевой таймер** (`create_wall_timer`, 50 Гц, **steady clock**, не ROS time). Если выхода не было больше 75 мс, публикуется модельное предсказание с экстраполированной меткой. Экстраполяция прекращается, если входов нет дольше 3 с (bag закончился). Частота ≥ 10 Гц сохраняется при любых пропусках.
7. **`/result/position`.** `frame_id="odom"`, `child_frame_id="base_link"`, кватернион из курса (никогда не нулевой), `twist.linear.x` содержит скорость в базовой СК, ковариации честно строятся через вдоль- и поперечно-путевые дисперсии: Σ_xy = R(ψ)·diag(σ²_along, σ²_cross)·R(ψ)ᵀ. `/result/velocity`: м/с. **Сырые колёсные скорости в bag даны в км/ч** (отношение к GNSS 3,597), хотя в README написано м/с.
8. **Диагностика.** Прямая публикация `diagnostic_msgs/DiagnosticArray` в `/diagnostics` с частотой 1 Гц. `diagnostic_updater` не входит в варианты ros_core/ros_base/desktop, поэтому **на него нельзя полагаться при офлайн-сборке**. Дополнительно собственный `EstimatorDebug.msg` с флагом проскальзывания, сцеплением, весами доверия и задержкой.
9. **GNSS только для начальной выставки.** Одноразовые подписки уничтожаются сразу после выставки, и в лог пишется явное сообщение. Это жёсткая гарантия, что GNSS не используется в основном контуре.
10. **Самоизмерение.** Задержка обработки = steady(publish) − steady(recv). Сквозная задержка = system(publish) − `MessageInfo.source_timestamp` (Fast DDS заполняет оба поля, в Cyclone в Humble `received_timestamp = 0`). Гистограмма фиксированного размера даёт p50/p99/max, счётчики > 100 и > 250 мс и частоту. Для жюри есть отдельный инструмент `latency_probe`: он сопоставляет выход со входом по точному `header.stamp`.
11. **Сборка офлайн.** Используются только пакеты из ros-base/desktop. `CMAKE_BUILD_TYPE` по умолчанию `Release`: `colcon build` без аргументов собирает с `-O0`. Никаких FetchContent/pip. Offline-сборка проверяется в Docker с `--network=none`, ресурсы ограничиваются через `--cpus=2 --memory=512m`.
12. **Риски, которые стоит уточнить у организаторов:** (а) начало и ось локальной СК эталона (ENU от первого фикса? какая антенна, master или rover?); (б) что судья считает временем эталона: header GNSS или время bag. От (б) зависит, какой тип меток на выходе даёт минимальную ошибку сопоставления.

---

## 1. Критерии → инженерные решения → доказательства

| Критерий (баллы) | Требование | Решение в ROS-слое | Как доказать жюри |
|---|---|---|---|
| Реальное время (15) | задержка ≤ 100 мс, пик ≤ 250 мс | публикация прямо в колбэке входа, C++, без аллокаций в горячем пути, одна нить | `~/debug.proc_latency_ms`, `/diagnostics` (p50/p99/max), `latency_probe` (сопоставление по точной метке), `ros2 topic delay -s` |
| | ≥ 10 Гц (реком. 20–50) | 20 Гц по `cmd` + сторожевой таймер 50 Гц на пропусках; опционально режим `timer` 50 Гц | `ros2 topic hz /result/position`, счётчик в `/diagnostics` |
| | ≤ 2 ядра, ≤ 0,5 ГБ, без утечек и гонок | одна нить, только кольцевые буферы и гистограммы фиксированного размера, логирование с троттлингом | `docker run --cpus=2 --memory=512m`, `resource_monitor.py` (наклон RSS ≈ 0), ASan/LSan, valgrind massif |
| | colcon build без интернета | зависимости только из ros-base; Release по умолчанию | `docker build --network=none` |
| Устойчивость (20) | не падать, пропуски, выбросы, восстановление | `guarded()` try/catch в каждом колбэке, `isfinite`, санитайзер меток, сброс при скачке времени назад, сторожевой таймер, подписки best-effort | fuzz-gtest, launch_test с «плохим» bag, счётчики в `/diagnostics` |
| | флаг проскальзывания, сцепление, адаптация | `EstimatorDebug.slip_state`, `mu_used`, `mu_max_est`, веса доверия; DiagnosticStatus WARN/ERROR | `ros2 topic echo /tram_odometry/debug` |
| Положение (35) | корректный `Odometry` (метка, frame_id, скорость, ковариации) | см. раздел 8 | `ros2 topic echo /result/position --once`, NEES-тест |
| Скорость (30) | RMSE/MAE, отсутствие смещения в переходных режимах | предсказание моделью на метку выхода (а не «последнее значение колёс») | офлайн-реплей по всем bag с таблицей ошибок |

Из PDF кейса: жюри требует «инструкцию для жюри по проверке: как воспроизвести rosbag, какие топики ожидать, как посмотреть логи/метрики и задержку» и «замеры задержки/частоты/ресурсов». Разделы 13 и 17 закрывают эти артефакты.

---

## 2. Временные свойства датасета (собственный анализ, 122 bag, 36,7 ч)

Все цифры получены скриптом `ros2_realtime_snippets/dataset_timing_audit.py` по `data/npz/*.npz`. Используются `t` (время записи в bag, оно же время воспроизведения) и `hs` (header.stamp).

### 2.1 Частоты и лаги

| Топик | Частота | лаг `recv − stamp`, p50 | дрейф лага внутри bag (p50 / max) | макс. пропуск | немонотонных меток |
|---|---|---|---|---|---|
| `/vehicle/front_bogie_velocity` | 10,0 Гц | 0,050 с | **0,076 / 0,091 с** | 19,8 с | 25 |
| `/vehicle/rear_bogie_velocity` | 10,0 Гц | 0,050 с | **0,076 / 0,091 с** | **73,5 с** | 27 |
| `/vehicle/driver_position_cmd` | 20,0 Гц | **0,001 с** | 0,001 / 0,032 с | 1,1 с | 75 |
| `/sensing/gnss/master/fix` | 10,0 Гц | 0,045 с | 0,010 / 0,165 с | 29,4 с | 44 |
| `/sensing/gnss/master/vel` | 10,0 Гц | 0,085 с | 0,017 / 0,147 с | 33,4 с | 45 |

QoS, записанный в `metadata.yaml` для `/vehicle/*`: `reliability=RELIABLE`, `durability=VOLATILE`, `KEEP_LAST depth=1`. У `/vehicle/driver_position_cmd` **два издателя**.

### 2.2 Находки, которые влияют на ROS-реализацию

1. **Единицы.** Медианное отношение скорости колёс к GNSS-скорости равно **3,597**. Сырые значения в **км/ч**, хотя README указывает м/с. Нужен параметр `input.wheel_speed_scale = 1/3.6`, а выход обязан быть в м/с.
2. **Пакет «догоняющих» сообщений в начале каждого bag.** За первые ~0,17 с воспроизведения приходит 50–75 сообщений `cmd` и ~10–30 колёсных с **монотонными** метками, которые уходят в прошлое на 1,4–3,5 с. То же происходит с GNSS. Это легитимные, но устаревшие данные. Их нужно обработать по порядку. Вывод: подписка с глубиной ≥ 100 и узел, который уже готов к моменту запуска `ros2 bag play`. GNSS-фикс приходит в первые ~10 мс, поэтому выставка фактически мгновенная.
3. **«Призраки» ±1 с.** В 4 уникальных bag (7 файлах: `30618_2255aade`, `…40ffd323`, `…7bfbb5ed`, `…efb92709`, `30639_9c362687` и др.) встречаются дубликаты сообщений, у которых `header.stamp` сдвинут ровно на +1 с (иногда на −1 с). Время приёма у них то же, что у нормальной пары. В `2255aade` в районе 196,8–199 с поток идёт *только* со сдвигом +1 с. В GNSS `30639_3b3d9eb8` метки master и rover **сдвинуты на +1 с на 4 минуты** (600–840 с). Если метрики считаются по header GNSS, эталон в этом bag на этом участке будет сдвинут на 1 с. Об этом стоит сообщить организаторам.
4. **Лаг колёс дрейфует, а метки колёс согласованы с GNSS.** В одном bag лаг `recv − stamp` колёс плавно уходит с 86 до 20 мс, при этом лаг `cmd` остаётся на уровне 1 мс. Кросс-корреляция скорости колёс со скоростью GNSS даёт лучшую задержку **0,000–0,010 с при сопоставлении header колёс с header GNSS** в начале и в конце bag. При сопоставлении по времени приёма колёс задержка плавает от 0,070 до 0,025 с. Значит, **header колёс — это истинное время измерения**, а время прихода в bag содержит переменную транспортную задержку.
5. **Передняя и задняя тележки:** в 99,83 % сообщений у пары *совпадают метки*, а приходят они с разницей ≤ 1,6 мс (p99). Удобно объединять их в пару по точному совпадению метки и делать совместное обновление фильтра с признаком «перед − зад».
6. **Горизонт предсказания** `stamp_cmd − max(stamp_колёс, полученных до этого cmd)`: p1 = 0,013, p50 = 0,104, p95 = 0,179, p99 = 0,207 с. Отрицательный горизонт встречается в 0,1 % случаев, его надо обрезать до 0.
7. **Дубликаты bag:** уникальных только **97 из 122**, 25 пар идентичны (например, `30618_2255aade` = `30618_7bfbb5ed`). Для кросс-валидации при калибровке пары надо держать в одном фолде, иначе будет утечка.

### 2.3 Следствия для выбора метки выхода

- Судья сопоставляет выход с эталоном по ближайшей метке с допуском ~0,05 с. При 20 Гц ближайшая метка выхода всегда не дальше 25 мс.
- **Метка `cmd` ≈ время bag** (лаг 1–2 мс). Поэтому обе трактовки фразы README «временем соответствующего входа (временем из bag)» дают одно и то же с точностью до 2 мс, а `ros2 topic delay -s` покажет около 1 мс плюс квантование `/clock`. Если ставить на выход метки колёс, «задержка по часам» составит 20–90 мс, в опасной близости от 100 мс. Кроме того, метки из разных доменов при чередовании дают немонотонный поток.
- **Рекомендация по умолчанию:** `timing.output_stamp_source: cmd`. Выход публикуется по каждому `cmd` с его меткой, и состояние предсказывается на неё. Альтернативы `wheel` и `bag_clock` оставить параметрами. После ответа организаторов выбор проверяется офлайн-реплеем. Если эталон строится по header GNSS, а часы записи и сенсоров совпадают, `cmd` корректен. Если часы расходятся на смещение δ, вдольпутевая ошибка ≈ v·δ, то есть ≤ 1,3 м при 15 м/с и δ = 90 мс. Это мелочь на фоне дрейфа, но её легко измерить.

---

## 3. Рекомендуемая структура пакетов

```
tram_ws/src/
├── tram_vehicle_msgs/            # из датасета, копия без изменений (нужна для сборки; если у жюри уже есть
│                                 #   в underlay — overlay с идентичными .msg безопасен)
├── tram_odometry_msgs/           # собственные интерфейсы (ament_cmake + rosidl)
│   ├── msg/EstimatorDebug.msg
│   ├── CMakeLists.txt, package.xml
├── tram_odometry/                # основной пакет (ament_cmake, C++17)
│   ├── include/tram_odometry/
│   │   ├── rt_utils.hpp          # StampSanitizer, RingBuffer, LatencyHistogram, covariance (см. приложение)
│   │   ├── time_keeper.hpp       # отображение steady → время bag, оценка скорости воспроизведения
│   │   ├── estimator.hpp         # ROS-FREE ядро: модель тяги/динамики, слип, фильтр, predict()
│   │   ├── traction_model.hpp, geodesy.hpp, track_map.hpp
│   ├── src/core/*.cpp            # ядро (без ROS; собирается и нативно на Windows)
│   ├── src/tram_odometry_node.cpp        # вся ROS-обвязка (компонент + исполняемый файл)
│   ├── src/tools/bag_replay.cpp          # офлайн-реплей через rosbag2_cpp::Reader тем же ядром
│   ├── scripts/latency_probe.py          # инструмент жюри: задержка/частота по точной метке
│   ├── scripts/resource_monitor.py       # CPU/RSS (psutil), наклон RSS
│   ├── scripts/run_eval.sh               # launch + bag play + record + метрики
│   ├── config/tram_odometry.yaml         # параметры (+ vehicle_30618.yaml, vehicle_30639.yaml)
│   ├── maps/                             # карта пути (если строите pathgraph из обучающих GNSS)
│   ├── launch/tram_odometry.launch.py
│   ├── test/test_core.cpp, test/test_rt_utils.cpp, test/test_realtime.launch.py, test/data/excerpt_60s/
│   ├── CMakeLists.txt, package.xml
└── tram_odometry_tools/          # (опц.) ament_python: evaluate.py (метрики как у судьи), plots, калибровка
```

Почему именно так:
- **Интерфейсы вынесены в отдельный пакет.** Для rosidl это обязательно (`member_of_group rosidl_interface_packages`). Кроме того, так проще не тащить его в чужие окружения.
- **Ядро не зависит от ROS.** Его можно гонять миллионами шагов в оптимизаторе калибровки, покрыть gtest без DDS и собрать на Windows без Docker (на этой машине есть g++ 16.1 MinGW и CMake 4.4, см. раздел 16).
- **Компонент** (`rclcpp_components_register_node`) даёт одновременно `.so` для `component_container` (возможна внутрипроцессная передача) и обычный исполняемый файл `tram_odometry_node`.

---

## 4. Язык, исполнитель, модель потоков

### 4.1 rclcpp или rclpy

- Узел обрабатывает около 60 маленьких сообщений в секунду (10 + 10 + 20 + GNSS в начале), на старте всплеск до ~400 сообщений в секунду. По пропускной способности справятся **оба** клиента.
- Сообщения маленькие (< 100 Б), поэтому известная проблема «rclpy в 30–100 раз медленнее на публикации больших сообщений» ([rclpy#763](https://github.com/ros2/rclpy/issues/763), [karl-schulz/ros2_latency](https://github.com/karl-schulz/ros2_latency)) нас напрямую не задевает. Решающими оказываются **детерминизм и хвосты** распределения задержки: в Python это GIL, сборщик мусора, накладные расходы исполнителя rclpy и больший RSS. В C++ задержка обработки стабильно укладывается в десятки микросекунд, и её легко доказать гистограммой.
- **Рекомендация:** рабочий узел на C++. Python оставить для инструментов, метрик, калибровки и launch. Модель, придуманную в Python/numpy, переносить в C++-ядро, а совпадение сверять офлайн-реплеем: одинаковые входы должны давать одинаковый выход с точностью до 1e-9.

### 4.2 Исполнитель и группы колбэков

- В Humble есть `SingleThreadedExecutor`, `StaticSingleThreadedExecutor` и `MultiThreadedExecutor`. `EventsExecutor` в ядре Humble **отсутствует**: он существует как внешний пакет iRobot, а в rclcpp появился позже. Сторонние зависимости добавлять не стоит.
- Семантика стандартного исполнителя: колбэки выбираются из wait set **не в порядке FIFO**, приоритет у таймеров, затем идут подписки. Исполнение не вытесняющее. За один цикл берётся не больше одного экземпляра каждого колбэка ([ROS 2 docs: Executors](https://github.com/ros2/ros2_documentation/blob/humble/source/Concepts/Intermediate/About-Executors.rst); Casini et al., ECRTS 2019). **Следствие:** глубина очереди подписки должна покрывать всплески. При `depth=1`, как в записанном QoS, часть стартового всплеска потеряется.
- **Рекомендация:** `rclcpp::spin(node)`, то есть `SingleThreadedExecutor`, и все колбэки в группе по умолчанию (MutuallyExclusive). Тогда состояние меняется только из одной нити, мьютексы не нужны, и гонок нет *по построению*. Это сильный аргумент для пункта «нет гонок». Вся работа в колбэке занимает меньше 50 мкс, поэтому параллелизм ничего не даст. `MultiThreadedExecutor` с Reentrant-группой выигрыша не приносит и требует блокировок; по данным обзоров, в нём появляются дополнительные задержки из-за блокировки wait set ([обзор RT в ROS 2, 2026](https://arxiv.org/html/2601.10722v1)).
- `StaticSingleThreadedExecutor` снижает накладные расходы процессора, но при нашей нагрузке это не важно. Можно оставить как опцию.
- WaitSet вместо исполнителя даёт полностью детерминированный порядок. Для нас это лишняя сложность.

### 4.3 Накладные расходы связи (ориентиры из литературы)

- Для маленьких сообщений задержка ROS 2 поверх DDS до +50 % относительно «голого» DDS. Большая часть приходится на категории «DDS» и «rclcpp notification delay». Reliable добавляет не больше 15 % задержки относительно best-effort ([Kronauer et al., 2021, arXiv:2101.02074](https://arxiv.org/abs/2101.02074)). В абсолютных цифрах на одной машине это доли миллисекунды, то есть на 2–3 порядка ниже бюджета в 100 мс.
- Композиция с внутрипроцессной передачей снижает CPU до 10 раз и задержку до ~70 раз на *больших* сообщениях. На стеке AMR композиция экономит 28 % CPU и 33 % RAM ([Macenski et al., RA-L 2023, arXiv:2305.09933](https://arxiv.org/abs/2305.09933)). Нам это не критично, но компонент стоит иметь как опцию.

---

## 5. Подписки, QoS, синхронизация и время входов

### 5.1 QoS

Совместимость ([ROS 2 docs: QoS](https://github.com/ros2/ros2_documentation/blob/humble/source/Concepts/Intermediate/About-Quality-of-Service-Settings.rst)):

| Издатель \ Подписка | Best effort | Reliable |
|---|---|---|
| Best effort | ✔ | ✘ (не соединятся) |
| Reliable | ✔ | ✔ |

- Профиль по умолчанию: KEEP_LAST 10, RELIABLE, VOLATILE. `SensorDataQoS`: KEEP_LAST **5**, BEST_EFFORT, VOLATILE ([rclcpp qos.hpp](https://github.com/ros2/rclcpp/blob/humble/rclcpp/include/rclcpp/qos.hpp)).
- `ros2 bag play` в Humble публикует с QoS, **записанным в bag**, если все издатели топика предлагали одинаковый профиль (история приводится к default). Если профили различаются, берётся профиль Rosbag2 по умолчанию и выводится предупреждение ([rosbag2 qos.cpp, humble](https://github.com/ros2/rosbag2/blob/humble/rosbag2_transport/src/rosbag2_transport/qos.cpp)). Жюри может переопределить QoS через `--qos-profile-overrides-path`.
- **Входы:** `rclcpp::QoS(rclcpp::KeepLast(100)).best_effort().durability_volatile()`. Такая подписка совместима с любым издателем. Глубина 100 покрывает всплеск на старте. Потерь на localhost при 60 сообщениях в секунду практически нет.
- **Выходы:** `rclcpp::QoS(rclcpp::KeepLast(10)).reliable()`. README подтверждает, что судья подписывается best-effort и совместим с любым издателем. Reliable-издатель к тому же совместим с `ros2 bag record` и rviz.
- **Durability VOLATILE:** узел, запущенный позже воспроизведения, пропустит начало, включая GNSS. В инструкции надо требовать запускать узел до `ros2 bag play`. Кроме того, в узле нужен фолбэк на случай, когда GNSS так и не пришёл: тогда публикуется относительная одометрия от стартовой точки, что допустимо по тексту кейса.

### 5.2 Почему не `message_filters`

`ApproximateTime` выдаёт набор только тогда, когда уверен в его оптимальности. При среднем интервале T он вносит задержку **порядка T**, то есть ~100 мс для колёс ([message_filters docs](https://docs.ros.org/en/jazzy/p/message_filters/doc/index.html)). Если один топик замолчит (реальные пропуски колёс до 73,5 с), выхода не будет вовсе. Кроме того, входы асинхронны по природе: `cmd` 20 Гц, колёса 10 Гц. **Правильная схема — асинхронное слияние:**
- каждый `cmd` записывается в буфер управления с меткой (ZOH);
- колёса объединяются в пары по точному совпадению метки. Ожидание партнёра ≤ 20 мс по steady clock, затем одиночное обновление. Фильтр обновляется по метке пары;
- выход — **чистое предсказание** состояния фильтра на метку выхода (`predict(t)` помечен `const` и не меняет состояние).

### 5.3 Измерения не по порядку (OOSM)

Выход строится по свежим меткам `cmd`, а колёса отстают на 20–200 мс. Есть два варианта:
- **(рекомендуется) «Фильтр во времени измерений, выход как предсказание».** Состояние фильтра хранится на момент последнего колёсного измерения, а выход получается прогнозом по модели тяги и динамики на горизонт h (p99 = 0,21 с). Проблема OOSM не возникает, потому что колёсные метки монотонны после санитайзера.
- **«Откат и повторная фильтрация».** Кольцевой буфер состояний и управлений примерно на 1 с (`RingBuffer<State,64>`): откатиться к метке измерения, выполнить обновление и заново прогнать прогноз. Так работает `robot_localization` (`history_length`, `smooth_lagged_data`). Точное решение OOSM дано в [Bar-Shalom, IEEE TAES 38(3), 2002](https://doi.org/10.1109/TAES.2002.1039398). Этот вариант понадобится, если появятся измерения с метками «из будущего» относительно фильтра.

### 5.4 Метаданные сообщения: `rclcpp::MessageInfo`

В Humble поддерживается сигнатура `void(std::shared_ptr<const MsgT>, const rclcpp::MessageInfo &)`. Сигнатуры с неконстантным `shared_ptr` помечены deprecated ([rclcpp#2972](https://github.com/ros2/rclcpp/issues/2972)). В `rmw_message_info_t` есть поля `source_timestamp` (время публикации у издателя, нс), `received_timestamp`, `publication_sequence_number`, `reception_sequence_number`, `publisher_gid` и `from_intra_process` ([rmw/types.h, humble](https://github.com/ros2/rmw/blob/humble/rmw/include/rmw/types.h)).
- **rmw_fastrtps (по умолчанию в Humble)** заполняет `source_timestamp`, `received_timestamp` и `publication_sequence_number`. По разрывам в номерах можно считать потери сообщений.
- **rmw_cyclonedds (Humble):** `received_timestamp = 0`, номера последовательности не поддерживаются ([rmw_node.cpp](https://github.com/ros2/rmw_cyclonedds/blob/humble/rmw_cyclonedds_cpp/src/rmw_node.cpp)). Поэтому время приёма надо **всегда брать самим** (`steady_clock::now()` в начале колбэка), а `source_timestamp` использовать только если он не равен нулю.

### 5.5 Санитайзер меток (проверен на данных)

```
accept(s, r)  ⇔  s > s_last  ∧  s − s_last ≤ rate · (r − r_last) + tol          (tol = 0.35 s)
иначе: цепочка «кандидатов» (взаимно согласованных по тому же правилу, без принятых между ними);
       K = 8 кандидатов подряд → пересинхронизация (реальный скачок часов / --loop)
```
- Результат на всех 122 bag: отброшено 57 + 57 + 107 сообщений (0,005 %). Все они «призраки» ±1 с. Принятые метки строго монотонны. Всплеск на старте (устаревшие, но упорядоченные метки) **принимается** и обрабатывается по порядку.
- Одна пересинхронизация случилась в `2255aade` (196,8 с), где поток на ~2 с остался только с меткой +1 с. Это поведение по замыслу.
- Реализация: `rt_utils.hpp::StampSanitizer`, свой экземпляр на каждый входной топик. `rate` задаётся параметром (1,0) или оценивается `TimeKeeper`-ом, если жюри играет bag с `--rate`.
- Более продвинутая альтернатива — пассивная оценка смещения и дрейфа часов (E. Olson, «A passive solution to the sensor synchronization problem», IROS 2010). Её имеет смысл применять, если понадобится «чинить» метки вместо отбрасывания.

### 5.6 Прочие проверки входов (модуль предобработки из PDF кейса)
- `std::isfinite`, диапазоны: скорость в [−1; 30] м/с после пересчёта, позиция ручки в [−15; 15] (обрезка и счётчик).
- Свежесть каждого входа: `age = steady_now − steady_last_msg`. Пороги WARN > 0,3 с и STALE > 1 с попадают в `/diagnostics`.
- Скачок времени назад больше 5 с (например, `--loop`) → полный сброс оценивателя, выставка заново или продолжение от последней позиции (параметр).
- Всё это учитывается в счётчиках `rejected_stamps`, `outliers`, `dropouts` и публикуется в `EstimatorDebug` и `/diagnostics`.

---

## 6. Время в узле: `use_sim_time`, `/clock`, таймеры

- `ros2 bag play --clock` публикует `/clock` с частотой **40 Гц по умолчанию** (`nargs='?', const=40`). Без флага `/clock` нет ([ros2bag play.py, humble](https://github.com/ros2/rosbag2/blob/humble/ros2bag/ros2bag/verb/play.py)). Другие полезные флаги: `--rate`, `--start-offset`, `--delay`, `--read-ahead-queue-size` (1000), `--disable-keyboard-controls`, `--qos-profile-overrides-path`, `--wait-for-all-acked`.
- При `use_sim_time=true` до первого `/clock` ROS time **равно 0** («time value of zero should be considered an error») ([design: Clock and Time](https://design.ros2.org/articles/clock_and_time.html)). Таймеры на ROS time (`rclcpp::create_timer(node, clock, …)`) при этом **не срабатывают**, а с `/clock` на 40 Гц они квантуются с шагом 25 мс.
- **Правила:**
  1. Метки выходов **никогда не берутся из `now()`**. Источник меток — header входов (раздел 2.3).
  2. Таймеры создаются через `create_wall_timer` (steady clock). Узел работает одинаково с `--clock` и без него, с `use_sim_time` и без.
  3. `use_sim_time` принимается как параметр (жюри может его задать), но логика от него не зависит. Если `/clock` есть, `TimeKeeper` может использовать `now()` как проверку.
  4. Внутри узла время хранится как `double` секунды или `int64` нс. `rclcpp::Time(msg.header.stamp)` имеет тип `RCL_ROS_TIME`. Операции с `rclcpp::Time` другого типа часов (например, `RCL_STEADY_TIME`) **бросают исключение**.
  5. Для `RCLCPP_*_THROTTLE` передаётся отдельный `rclcpp::Clock steady_clock_{RCL_STEADY_TIME}`. Под sim time с `now() = 0` троттлинг по ROS-часам ведёт себя непредсказуемо.
- **`TimeKeeper`** (отображение steady → время bag). По приходу `cmd`: `bag(t_steady) = s_cmd + r·(t_steady − r_cmd)`. `r` оценивается как Δstamp/Δrecv на окне ~2 с и обрезается до [0,2; 20]. Если `cmd` не свежий (> 0,5 с), используются пары колёс со смещением, выученным, пока `cmd` был доступен. Нужен для меток сторожевого таймера и для санитайзера при `--rate ≠ 1`.

---

## 7. Публикация: по событию или по таймеру

| Режим | Как | Плюсы | Минусы |
|---|---|---|---|
| **event (по умолч.)** | публикация в колбэке каждого `cmd` (20 Гц), метка = метка `cmd`, состояние = `predict(stamp)` | задержка = обработка (< 1 мс); связь вход–выход 1:1 по точной метке, задержку легко проверить; не зависит от часов и `--rate` | 20 Гц (нижняя граница рекомендованного диапазона) |
| **гэп-филлер** (всегда включён) | wall-таймер 50 Гц: если выхода не было > 75 мс, публикуется предсказание на метку `TimeKeeper.bag(now)`; стоп через 3 с без входов | ≥ 10 Гц при любых пропусках `cmd`/колёс, модель «ведёт» трамвай | метки экстраполированы |
| **timer 50 Гц** (опция) | wall-таймер 20 мс, метка = `TimeKeeper.bag(now)` | 50 Гц, плотнее сопоставление (≤ 10 мс) | зависит от оценки скорости воспроизведения, нет связи 1:1 со входом |

Инварианты, которые соблюдаются в любом режиме:
- **Строго монотонные метки выхода.** Если `stamp ≤ last_pub_stamp + 1 мс`, публикация пропускается. Причина: у топиков разные домены лага (1 мс против 50 мс), и при чередовании поток стал бы немонотонным. Судья может, например, делать `np.interp` без сортировки.
- `/result/velocity` и `/result/position` публикуются **парой с одной меткой**.
- На пропуск NaN или Inf не попадает: при нефинитном значении делается мягкий сброс и выход пропускается (счётчик `nan_resets`).
- Стартовый всплеск (метки на 1,4–3,5 с в прошлом) по умолчанию тоже публикуется: трамвай стоит, поэтому точки точные. В инструкции для жюри стоит отметить, что задержка по `/clock` в первые ~0,2 с бессмысленна, это артефакт записи. Параметр `timing.publish_catchup_burst: true|false` оставить.

Средняя ошибка сопоставления «ближайшей меткой» при 20 Гц составляет ≤ 12,5 мс, то есть ≤ 0,19 м при 15 м/с. При 50 Гц это ≤ 5 мс и 0,08 м. Эффект мал по сравнению с дрейфом, поэтому режим `timer` имеет смысл включать, только если офлайн-оценка покажет выигрыш.

---

## 8. Правильное заполнение `nav_msgs/Odometry` и `VelocitySensor`

Определения ([Odometry.msg](https://github.com/ros2/common_interfaces/blob/humble/nav_msgs/msg/Odometry.msg), [PoseWithCovariance.msg](https://github.com/ros2/common_interfaces/blob/humble/geometry_msgs/msg/PoseWithCovariance.msg)): поза задаётся в `header.frame_id`, скорость в `child_frame_id`. Ковариация — матрица 6×6, построчно, порядок (x, y, z, rotX, rotY, rotZ).

| Поле | Значение | Обоснование |
|---|---|---|
| `header.stamp` | метка входа (`cmd`) после санитайзера | README, раздел 2.3 |
| `header.frame_id` | `"odom"` (параметр) | [REP-105](https://github.com/ros-infrastructure/rep/blob/master/rep-0105.rst): odom — непрерывная СК без скачков с неограниченным дрейфом. Это ровно наш случай (dead reckoning). В начале она совпадает с локальной ENU (x = Восток, y = Север, начало в точке выставки). Если используется привязка к карте pathgraph с коррекциями и скачками, правильнее `"map"` |
| `child_frame_id` | `"base_link"` | REP-105 |
| `pose.pose.position` | x, y (ENU, м), z = 0 или высота по карте | [REP-103](https://github.com/ros-infrastructure/rep/blob/master/rep-0103.rst): СИ, ENU, x вперёд, y влево, z вверх |
| `pose.pose.orientation` | `(0, 0, sin ψ/2, cos ψ/2)`, ψ — курс ENU (0 = Восток, против часовой) | **никогда не нулевой кватернион**, это невалидно |
| `pose.covariance` | формула ниже | не нули: нулевая ковариация означает «бесконечно точно» |
| `twist.twist.linear.x` | v (м/с) в `base_link` | продольная скорость |
| `twist.twist.angular.z` | v·κ (кривизна по карте), иначе 0 | |
| `twist.covariance` | [0] = σ²_v; [7] = [14] = 1e-4 (рельсовое ТС: поперечная и вертикальная скорость ≈ 0); [21] = [28] = 1e-4; [35] = σ²_ωz | |

Ковариация позы через вдоль- и поперечно-путевые дисперсии:

```
Σ_xy = R(ψ) · diag(σ²_along, σ²_cross) · R(ψ)ᵀ
c[0]  = cos²ψ·σ²_a + sin²ψ·σ²_c      c[1] = c[6] = cosψ·sinψ·(σ²_a − σ²_c)
c[7]  = sin²ψ·σ²_a + cos²ψ·σ²_c      c[14] = σ²_z,  c[21] = c[28] = σ²_roll/pitch,  c[35] = σ²_ψ
```
- Вдоль пути: σ²_along(s) = σ²_0 + (ε_s·s)² + q_s·s. Здесь ε_s — остаточная масштабная ошибка одометрии (подбирается по распределению дрейфа на обучающих bag, ожидаемо 0,3–1 %), q_s — случайное блуждание. Во время проскальзывания и работы только по модели σ² растёт быстрее (дисперсия из KF).
- Поперёк пути: при привязке к карте σ_cross ≈ точность карты (0,3–1 м). Без карты σ²_cross = σ²_c0 + (s·σ_ψ)², и её честно указывать большой.
- **Проверка честности ковариаций:** NEES = eᵀΣ⁻¹e по всем bag. Для 2D среднее должно быть около 2, а доля выходов за 95 % χ²(2) = 5,99 — около 5 % (Bar-Shalom, Li, Kirubarajan, *Estimation with Applications to Tracking and Navigation*, 2001). Это хороший слайд для питча.
- Позиция «неизмеряемых» осей: z/roll/pitch задаются небольшими честными значениями (например, σ_z = 1–2 м, σ_roll = σ_pitch = 0,01 рад). Раздувать до 1e3 и больше не нужно, в документации robot_localization это прямо не рекомендуется ([robot_localization: preparing sensor data](http://docs.ros.org/en/noetic/api/robot_localization/html/preparing_sensor_data.html)).

`/result/velocity` (`VelocitySensor`): та же метка, `frame_id="base_link"`, `velocity` в **м/с**, неотрицательная для трамвая. Выход ≥ 0 можно обрезать, если модель допускает малые отрицательные значения при остановке, но это надо согласовать с оценкой смещения на остановках.

---

## 9. Диагностика, отладка, TF

### 9.1 `/diagnostics` без `diagnostic_updater`
`diagnostic_updater` входит в репозиторий ros/diagnostics, но **не входит** в метапакеты `ros_core`, `ros_base` и `desktop` ([variants](https://github.com/ros2/variants/tree/humble)). В окружении жюри его может не быть. `diagnostic_msgs` входит в `common_interfaces` (ros_core), поэтому `DiagnosticArray` публикуется вручную раз в секунду. Уровни: `OK = 0`, `WARN = 1`, `ERROR = 2`, `STALE = 3` ([DiagnosticStatus.msg](https://github.com/ros2/common_interfaces/blob/humble/diagnostic_msgs/msg/DiagnosticStatus.msg)).

| name | level | values (KeyValue) |
|---|---|---|
| `tram_odometry: inputs` | OK / WARN (возраст входа > 0,3 с) / STALE (> 1 с) | `age_front_ms`, `age_rear_ms`, `age_cmd_ms`, `rejected_stamps`, `outliers`, `seq_gaps` |
| `tram_odometry: slip` | OK / WARN (подозрение) / ERROR (обе тележки недостоверны, работа только по модели) | `slip_state`, `slip_front`, `slip_rear`, `mu_used`, `mu_max_est`, `w_front`, `w_rear` |
| `tram_odometry: timing` | OK / WARN (p99 > 50 мс или частота < 15 Гц) / ERROR (> 100 мс или < 10 Гц) | `proc_p50_ms`, `proc_p99_ms`, `proc_max_ms`, `e2e_p99_ms`, `over100`, `over250`, `rate_hz` |
| `tram_odometry: estimator` | OK / WARN (нет GNSS-выставки: относительная одометрия) / ERROR (сбросы из-за NaN) | `initialized`, `nan_resets`, `exceptions`, `sigma_along_m`, `distance_m` |

`header.stamp` у `DiagnosticArray` — последняя метка выхода (домен времени bag).

### 9.2 Собственное отладочное сообщение
`tram_odometry_msgs/msg/EstimatorDebug.msg`, публикуется в `~/debug`, то есть `/tram_odometry/debug`, с той же частотой и меткой:
```
std_msgs/Header header
float64 v_est            # м/с
float64 a_est            # м/с^2
float64 s_along          # м, пройденный путь
float64 v_front_raw      # м/с (после пересчёта из км/ч)
float64 v_rear_raw
float64 v_model          # чисто модельная скорость
float64 w_front          # доверие 0..1
float64 w_rear
float64 w_model
float64 slip_front       # (v_wheel - v_est)/max(v_est, v_min)
float64 slip_rear
float64 mu_used          # F_x/(m g) — используемое сцепление
float64 mu_max_est       # оценка доступного сцепления
float64 f_traction       # Н
float64 f_brake
float64 f_resist
float64 mass_est         # кг (адаптивная)
float64 grade_est        # рад
uint8 SLIP_NONE=0
uint8 SLIP_SUSPECTED=1
uint8 SLIP_WHEELSPIN=2
uint8 SLIP_SLIDE=3
uint8 slip_state
bool front_valid
bool rear_valid
bool cmd_valid
float32 horizon_s        # горизонт предсказания от последнего колёсного измерения
float32 proc_latency_ms
uint32 rejected_stamps
uint32 outliers
```
`EstimatorDebug` удобен и для жюри (`ros2 topic echo`), и для `ros2 bag record` с последующими графиками.

### 9.3 TF
`tf2_ros::TransformBroadcaster`: `odom → base_link` с меткой выхода. Включается параметром `frames.publish_tf` и нужен для rviz. Статическое `map → odom` = identity (система выставлена в ENU) задаётся только при необходимости. В Humble заголовок `tf2_ros/transform_broadcaster.h` присутствует, `.hpp` тоже. `tf2_ros` входит в geometry2 (ros_base).

---

## 10. Параметры, YAML, launch, lifecycle

### 10.1 YAML (`config/tram_odometry.yaml`)
```yaml
tram_odometry:
  ros__parameters:
    input:
      wheel_speed_scale: 0.2777777777777778   # км/ч -> м/с (подтверждено: отношение к GNSS = 3.597)
      max_speed_mps: 30.0
      pair_wait_s: 0.02                        # ожидание партнёрской тележки
    timing:
      output_stamp_source: "cmd"               # cmd | wheel | bag_clock
      publish_mode: "event"                    # event | timer
      timer_period_s: 0.02                     # для publish_mode=timer
      gap_fill_after_s: 0.075
      max_extrapolation_s: 3.0
      stamp_tolerance_s: 0.35
      stamp_resync_k: 8
      play_rate: 1.0                           # 0 = оценивать онлайн
      publish_catchup_burst: true
    frames:
      world: "odom"
      base: "base_link"
      publish_tf: true
    init:
      use_gnss: true
      gnss_window_s: 3.0
      origin_mode: "first_fix"                 # first_fix | fixed | utm
      origin_lat: 0.0
      origin_lon: 0.0
      origin_alt: 0.0
      reference_antenna: "master"              # точка, положение которой публикуем
      heading_source: "dual_antenna"           # dual_antenna | track_map | param
    vehicle:
      id: "auto"                               # auto | 30618 | 30639 (параметры по вагону)
    # model.*, slip.*, filter.*, map.* — от команды моделирования
```
В rclcpp вложенные ключи объявляются через точку: `declare_parameter<double>("input.wheel_speed_scale", 1.0/3.6)`. Подстановки по маске (`/**:` вместо имени узла) работают для любого имени и пространства имён.

### 10.2 Launch (`launch/tram_odometry.launch.py`)
```python
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg = FindPackageShare('tram_odometry')
    params = LaunchConfiguration('params_file')
    use_sim_time = LaunchConfiguration('use_sim_time')
    return LaunchDescription([
        DeclareLaunchArgument('params_file',
                              default_value=PathJoinSubstitution([pkg, 'config', 'tram_odometry.yaml'])),
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        DeclareLaunchArgument('log_level', default_value='info'),
        Node(
            package='tram_odometry', executable='tram_odometry_node', name='tram_odometry',
            output='screen', emulate_tty=True,
            parameters=[params, {'use_sim_time': ParameterValue(use_sim_time, value_type=bool)}],
            arguments=['--ros-args', '--log-level', LaunchConfiguration('log_level')],
            respawn=True, respawn_delay=0.5,   # страховка «нода не падает»; штатно не срабатывает
        ),
    ])
```
- `ParameterValue(..., value_type=bool)` нужен, чтобы строку `'false'` не передать как string. Иначе будет ошибка типа параметра `use_sim_time`.
- `respawn=True` — последняя линия обороны. После перезапуска положение теряется, поэтому основная защита — `guarded()` (раздел 12).

### 10.3 Lifecycle-узлы
`rclcpp_lifecycle` входит в ros_core. Если узел не активирован, он **ничего не публикует**. Судья, запустивший `ros2 run`, получит 0 баллов. Рекомендация: **обычный `rclcpp::Node`**. Если нужна «управляемость», можно сделать lifecycle с самоактивацией (`configure` и `activate` в конце конструктора через таймер) или `launch_ros.actions.LifecycleNode` плюс `EmitEvent(ChangeState)`. Выигрыша по критериям это не даёт.

---

## 11. Начальная выставка по GNSS в ROS-слое

- Подписки: `/sensing/gnss/master/fix`, `/sensing/gnss/rover/fix` и при необходимости `/sensing/gnss/master/vel` (QoS как у входов).
- Фикс считается валидным при `status.status ≥ 0` (`STATUS_NO_FIX = −1`), конечных lat/lon и ковариации ниже порога. Выставка заканчивается, когда у master и rover накопилось N ≥ 5 валидных фиксов или истекло `init.gnss_window_s` по меткам.
- Начало СК: геодезические координаты → ECEF → ENU (WGS-84, a = 6378137 м, f = 1/298,257223563, e² = f(2 − f)):
  - N(φ) = a/√(1 − e² sin²φ); X = (N + h)cosφ·cosλ; Y = (N + h)cosφ·sinλ; Z = (N(1 − e²) + h)sinφ;
  - [e, n, u]ᵀ = R(φ₀, λ₀)·([X, Y, Z] − [X₀, Y₀, Z₀]), где R = [[−sinλ₀, cosλ₀, 0], [−sinφ₀cosλ₀, −sinφ₀sinλ₀, cosφ₀], [cosφ₀cosλ₀, cosφ₀sinλ₀, sinφ₀]].
- Курс по двум антеннам: ψ = atan2(n_rover − n_master, e_rover − e_master) плюс калибруемая поправка монтажа. Курс определён только при известном направлении базы, это уточнить по данным. Запасные варианты: касательная карты пути, направление движения по первым метрам.
- **После выставки подписки уничтожаются** (`sub_fix_.reset()` в следующем тике сторожевого таймера, а не внутри собственного колбэка) и в лог пишется `GNSS init done at t=…; GNSS subscriptions destroyed`. Это наглядное доказательство, что GNSS не используется в основном контуре.
- Если GNSS не пришёл за окно выставки, узел стартует из (0, 0, 0) с курсом из параметра, в `/diagnostics` ставится WARN «relative odometry» и публикация продолжается. PDF кейса такое допускает.
- **Открытые вопросы к организаторам:** начало эталонной СК (первый фикс master? UTM?) и точка трамвая, которой соответствует эталон (антенна master или rover). Трамвай длиной около 30 м, поэтому неверная точка даст постоянное смещение до 10–30 м.

---

## 12. Надёжность: «ноды не падают, ошибки не взрываются»

1. **`guarded()` вокруг каждого колбэка:** `try { … } catch (const std::exception&) { ++exceptions_; est_.soft_reset(); } catch (...) { … }`. Лог через `RCLCPP_ERROR_THROTTLE(get_logger(), steady_clock_, 2000, …)`.
2. **Мягкий сброс** сохраняет позицию и путь, а скорость переинициализирует по последней достоверной паре колёс или по модели, ковариацию увеличивает. **Жёсткий сброс** выполняется только при скачке времени назад больше 5 с.
3. **Проверка конечности** на входе (`std::isfinite`, диапазоны) и на выходе (`EstimatorOutput::finite()`). NaN никогда не публикуется.
4. **Ограниченная память:** только `std::array`, `RingBuffer<T,N>`, `LatencyHistogram` (10 000 бинов по 0,05 мс, 40 КБ). Никаких `push_back` без ограничения, никаких `std::map` по меткам. Карта пути загружается один раз при старте (≤ несколько МБ).
5. **Никаких аллокаций в горячем пути, кроме сообщения на публикацию.** Сообщения можно держать членами класса и переиспользовать. `publish(const MsgT&)` копирует сообщение в DDS.
6. **Логирование:** в колбэках только `*_THROTTLE` или `*_ONCE`. Раз в 30 с выводится сводная INFO-строка (частота, задержки, счётчики).
7. **Порядок запуска:** карта и параметры загружаются в конструкторе *до* создания подписок. Подписки создаются последними, и тогда первый колбэк не «ждёт» загрузки.
8. **Выход из процесса:** SIGINT/SIGTERM штатно обрабатываются в `rclcpp::init`. При завершении в stderr (не в ROS-логгер, он может быть уже закрыт) и в CSV `~/.ros/tram_odometry_summary.csv` пишется сводка: число входов и выходов, p50/p99/max задержки, число пропусков.

---

## 13. Измерение задержки, частоты и ресурсов

### 13.1 Самоизмерение в узле (главный источник цифр)
- **Задержка обработки:** `L_proc = steady(после publish) − steady(в начале колбэка входа)`. Честно определённая величина «вход → публикация» внутри узла.
- **Сквозная задержка:** `L_e2e = system_clock(после publish) − MessageInfo.source_timestamp`. Включает DDS-хоп от `ros2 bag play` до узла. Оба времени берутся с часов одной машины. Считается только при `source_timestamp > 0`.
- Обе величины попадают в `LatencyHistogram` (p50/p99/max, > 100 мс, > 250 мс), в `/diagnostics` раз в 1 с, в `EstimatorDebug.proc_latency_ms` на каждый выход и в итоговый CSV.
- Частота считается как число публикаций за окно 5 с по steady clock.

### 13.2 Как проверить жюри

**Стандартными средствами:**
- `ros2 topic hz /result/position` и `ros2 topic hz /result/velocity`. По умолчанию используется ROS clock: без sim time это системное время, есть флаг `--wall-time` ([hz.py](https://github.com/ros2/ros2cli/blob/humble/ros2topic/ros2topic/verb/hz.py)).
- `ros2 topic delay /result/position` считает `now() − header.stamp` по часам узла CLI ([delay.py](https://github.com/ros2/ros2cli/blob/humble/ros2topic/ros2topic/verb/delay.py)). **Без `--clock`/`-s` это бессмысленно**: время bag 2026-08 сравнивается с текущими часами. С `ros2 bag play --clock` и `ros2 topic delay -s /result/position` (флаг `--use-sim-time` из DirectNode; проверьте `ros2 topic delay -h`) получится около 1 мс плюс квантование `/clock` в 25 мс. Возможны даже отрицательные значения, потому что `/clock` отстаёт.
- `ros2 topic echo /diagnostics` и `ros2 topic echo /tram_odometry/debug --field proc_latency_ms`.

**`latency_probe` — инструмент жюри, наш скрипт:**
```python
#!/usr/bin/env python3
"""Jury-style probe: latency input->output matched by EXACT header.stamp, output rate, gaps.
Run BEFORE `ros2 bag play`. Prints a table every 5 s and a summary on Ctrl-C."""
import collections
import time

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from tram_vehicle_msgs.msg import DriverControllerCommand, VelocitySensor

OUT = ('/result/velocity', '/result/position')


def key(st):
    return st.sec * 1_000_000_000 + st.nanosec


class Probe(Node):
    def __init__(self):
        super().__init__('latency_probe')
        qos = QoSProfile(depth=500, history=HistoryPolicy.KEEP_LAST, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.t_in = collections.OrderedDict()                      # stamp_ns -> monotonic recv
        self.lat = {t: [] for t in OUT}
        self.last_arrival = {t: None for t in OUT}
        self.max_gap = {t: 0.0 for t in OUT}
        self.n = {t: 0 for t in OUT}
        self.unmatched = {t: 0 for t in OUT}
        self.t0 = time.monotonic()
        self.create_subscription(DriverControllerCommand, '/vehicle/driver_position_cmd', self.on_in, qos)
        self.create_subscription(VelocitySensor, '/vehicle/front_bogie_velocity', self.on_in, qos)
        self.create_subscription(VelocitySensor, '/vehicle/rear_bogie_velocity', self.on_in, qos)
        self.create_subscription(VelocitySensor, OUT[0], lambda m: self.on_out(OUT[0], m), qos)
        self.create_subscription(Odometry, OUT[1], lambda m: self.on_out(OUT[1], m), qos)
        self.create_timer(5.0, self.report)

    def on_in(self, m):
        self.t_in.setdefault(key(m.header.stamp), time.monotonic())
        while len(self.t_in) > 20000:
            self.t_in.popitem(last=False)

    def on_out(self, topic, m):
        now = time.monotonic()
        self.n[topic] += 1
        if self.last_arrival[topic] is not None:
            self.max_gap[topic] = max(self.max_gap[topic], now - self.last_arrival[topic])
        self.last_arrival[topic] = now
        t = self.t_in.get(key(m.header.stamp))
        if t is None:
            self.unmatched[topic] += 1            # e.g. gap-filler outputs with extrapolated stamps
        else:
            self.lat[topic].append((now - t) * 1e3)

    def report(self):
        el = time.monotonic() - self.t0
        for t in OUT:
            l = sorted(self.lat[t])
            if not l:
                continue
            p = lambda q: l[min(len(l) - 1, int(q / 100 * len(l)))]
            print(f'{t:18s} rate {self.n[t] / el:6.1f} Hz | lat p50 {p(50):6.2f} p99 {p(99):6.2f} '
                  f'max {l[-1]:6.2f} ms | >100ms {sum(x > 100 for x in l)} >250ms {sum(x > 250 for x in l)} '
                  f'| max gap {self.max_gap[t] * 1e3:6.1f} ms | unmatched {self.unmatched[t]}', flush=True)


def main():
    rclpy.init()
    node = Probe()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.report()
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == '__main__':
    main()
```
Пояснения. Задержка здесь — это разница времени приёма выхода и входа **в одном процессе** по `time.monotonic()`, так что синхронизация часов не нужна. Цифра включает DDS-хоп выхода и накладные расходы rclpy, то есть это **верхняя оценка**. Словарь меток ограничен 20 000 элементами.

### 13.3 Ресурсы и утечки
- **Итоговые цифры:** `/usr/bin/time -v install/tram_odometry/lib/tram_odometry/tram_odometry_node --ros-args --params-file …` выдаёт «Maximum resident set size» и время CPU. Бинарник запускается напрямую, без обёртки `ros2 run`.
- **По времени:** `pidstat -r -u -p <pid> 1` или `resource_monitor.py`. Второй раз в секунду снимает `psutil.Process(pid).cpu_percent()` и `memory_info().rss` в CSV, затем считает наклон RSS (линейная регрессия после 60 с прогрева). Критерий: |наклон| < 10 КБ/мин на 30-минутном bag. Ориентир для C++-узла с Fast DDS — десятки МБ RSS. Это оценка, её надо измерить и вписать в таблицу.
- **Docker:** `docker stats --no-stream`. Прогоны с `--cpus=2 --memory=512m --memory-swap=512m`: лимит действует на весь контейнер вместе с `ros2 bag play`, то есть с запасом.
- **Утечки и гонки:**
  - ASan + LSan + UBSan: `colcon build --packages-select tram_odometry --cmake-args -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_CXX_FLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer"`. Запускать бинарник напрямую, `ASAN_OPTIONS=detect_leaks=1`. Для утечек внутри Fast DDS при выходе нужен suppressions-файл.
  - `valgrind --tool=massif <node>` плюс `ms_print`: плоский график heap после старта.
  - TSan не обязателен, потому что нить одна. Это и есть аргумент «гонок нет по построению».
- **Стресс-тест:** `ros2 bag play --rate 5` и `--rate 10`. Задержка должна остаться < 100 мс, без потерь (по счётчикам `seq_gaps`/`rejected`). Это доказывает запас примерно в 10 раз. При `--rate ≠ 1` включить `timing.play_rate: 0` (онлайн-оценка).

### 13.4 ros2_tracing и CARET (опционально, для «глубины» на питче)
- `ros2_tracing` (LTTng): накладные расходы около **3,3 мкс** на сообщение ([Bédard et al., RA-L 2022, arXiv:2201.00393](https://arxiv.org/abs/2201.00393)). **В бинарных пакетах Humble точки трассировки не включены**, нужна сборка `tracetools` из исходников с установленным LTTng (`ros2 run tracetools status` → «Tracing enabled») ([ros2_tracing README, humble](https://github.com/ros2/ros2_tracing/blob/humble/README.md)). [REP-2014](https://ros.org/reps/rep-2014.html) рекомендует серый ящик с LTTng.
- CARET (Tier IV) показывает сквозную задержку цепочек и тоже основан на LTTng ([CARET](https://tier4.github.io/caret_doc/main/)).
- Для хакатона достаточно самоизмерения и `latency_probe`. Трассировку упомянуть как план развития.

---

## 14. Тесты

### 14.1 gtest (ядро и утилиты, без DDS)
```cpp
#include <gtest/gtest.h>
#include <cmath>
#include <random>
#include "tram_odometry/rt_utils.hpp"
#include "tram_odometry/estimator.hpp"

TEST(StampSanitizer, RejectsGhostDuplicatesAndResyncsOnRealJump) {
  tram_rt::StampSanitizer s;
  using V = tram_rt::StampSanitizer::Verdict;
  EXPECT_EQ(s.check(100.0, 0.0), V::kAccepted);
  EXPECT_EQ(s.check(101.15, 0.15), V::kRejected);  // +1 s ghost
  EXPECT_EQ(s.check(100.2, 0.2), V::kAccepted);
  EXPECT_EQ(s.check(173.0, 73.2), V::kAccepted);   // long dropout, consistent
  int resync = 0;                                   // bag --loop: time back by 1000 s
  for (int i = 0; i < 10; ++i) resync += s.check(10.0 + 0.1 * i, 80.0 + 0.1 * i) == V::kResynced;
  EXPECT_EQ(resync, 1);
}

TEST(Estimator, NeverOutputsNonFiniteUnderFuzz) {
  tram_odometry::Estimator est; est.configure(tram_odometry::Params::defaults());
  std::mt19937 rng(42); std::uniform_real_distribution<double> u(-1e3, 1e3);
  const double bad[] = {NAN, INFINITY, -INFINITY, -5.0, 1e9};
  double t = 1.0;
  for (int k = 0; k < 200000; ++k) {
    t += 0.05;
    if (k % 2) est.set_control(t, static_cast<int>(u(rng)) % 16);
    double v = (k % 97 == 0) ? bad[k % 5] : std::abs(u(rng)) / 50.0;
    est.add_wheel(tram_odometry::Bogie::kFront, t, v);
    auto o = est.predict(t + 0.1);
    ASSERT_TRUE(std::isfinite(o.v) && std::isfinite(o.x) && std::isfinite(o.y));
  }
}
```

### 14.2 launch_testing (интеграция: узел + короткий bag)
Отрывок bag на 60 с (десятки–сотни КБ) кладётся в `test/data/excerpt_60s/`. Сделать его можно на Windows через `rosbags` (Writer) из любого обучающего bag, лучше из того, где есть «призраки» и пропуски.
```python
import math
import os
import time
import unittest

import launch
import launch_ros.actions
import launch_testing
import launch_testing.actions
import launch_testing.asserts
import pytest
import rclpy
from ament_index_python.packages import get_package_share_directory
from launch.actions import ExecuteProcess, TimerAction
from nav_msgs.msg import Odometry


@pytest.mark.launch_test
def generate_test_description():
    share = get_package_share_directory('tram_odometry')
    node = launch_ros.actions.Node(package='tram_odometry', executable='tram_odometry_node',
                                   parameters=[os.path.join(share, 'config', 'tram_odometry.yaml')])
    play = ExecuteProcess(cmd=['ros2', 'bag', 'play', os.path.join(share, 'test', 'data', 'excerpt_60s'),
                               '--disable-keyboard-controls'], output='screen')
    return launch.LaunchDescription([node, TimerAction(period=2.0, actions=[play]),
                                     launch_testing.actions.ReadyToTest()]), {'node': node}


class TestRealtimeContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def test_rate_stamps_finiteness(self):
        n = rclpy.create_node('contract_checker')
        got = []
        n.create_subscription(Odometry, '/result/position', lambda m: got.append((time.monotonic(), m)), 100)
        end = time.monotonic() + 30.0
        while time.monotonic() < end:
            rclpy.spin_once(n, timeout_sec=0.05)
        n.destroy_node()
        self.assertGreater(len(got), 10 * 20)                                  # >= 10 Hz for >= 20 s
        st = [m.header.stamp.sec + 1e-9 * m.header.stamp.nanosec for _, m in got]
        self.assertTrue(all(s > 0 for s in st))
        self.assertTrue(all(b > a for a, b in zip(st, st[1:])))                # strictly monotonic
        self.assertTrue(all(math.isfinite(m.pose.pose.position.x) for _, m in got))
        self.assertTrue(all(m.header.frame_id and m.child_frame_id == 'base_link' for _, m in got))
        q = got[-1][1].pose.pose.orientation
        self.assertAlmostEqual(q.x**2 + q.y**2 + q.z**2 + q.w**2, 1.0, places=6)
        gaps = [b[0] - a[0] for a, b in zip(got, got[1:])]
        self.assertLess(max(gaps), 0.25)


@launch_testing.post_shutdown_test()
class TestShutdown(unittest.TestCase):
    def test_exit_code(self, proc_info, node):
        launch_testing.asserts.assertExitCodes(proc_info, process=node, allowable_exit_codes=[0, -2, -15])
```
Запуск: `colcon test --packages-select tram_odometry --event-handlers console_direct+ && colcon test-result --verbose` ([launch_testing README](https://github.com/ros2/launch/blob/humble/launch_testing/README.md)).

---

## 15. Сборка: package.xml, CMakeLists.txt, офлайн-требования

### 15.1 Что точно есть в окружении жюри
Метапакеты Humble ([ros2/variants, humble](https://github.com/ros2/variants/tree/humble)):
- **ros_core:** `ament_cmake(_auto/_gtest/_gmock/_pytest/_ros)`, `rclcpp`, `rclcpp_lifecycle`, `rclpy`, `rosidl_default_generators/runtime`, `common_interfaces` (nav_msgs, sensor_msgs, geometry_msgs, diagnostic_msgs, std_msgs), `launch`, `launch_ros`, `launch_testing`, `launch_testing_ament_cmake`, `launch_testing_ros`, `ros2cli_common_extensions`, `class_loader`, `pluginlib`.
- **ros_base:** + `rosbag2` (включая `rosbag2_cpp`), `geometry2` (`tf2`, `tf2_ros`, `tf2_eigen` и т. д.), `kdl_parser` → `orocos_kdl_vendor` → **`eigen` и `eigen3_cmake_module`** (значит, Eigen установлен), `urdf`, `robot_state_publisher`. `message_filters` приходит вместе с `tf2_ros`.
- **desktop:** + rviz2, demos, rqt… **`diagnostic_updater` нет ни в одном варианте.**
- Образ `ros:humble-ros-base` уже содержит `build-essential`, `python3-colcon-common-extensions`, `python3-rosdep`; `osrf/ros:humble-desktop` построен поверх него ([osrf/docker_images](https://github.com/osrf/docker_images/tree/master/ros/humble/ubuntu/jammy)).

### 15.2 `package.xml`
```xml
<?xml version="1.0"?>
<package format="3">
  <name>tram_odometry</name>
  <version>1.0.0</version>
  <description>Model-based backup odometry (speed + position) for a tram without GNSS/IMU</description>
  <maintainer email="team@example.org">Team</maintainer>
  <license>MIT</license>
  <buildtool_depend>ament_cmake</buildtool_depend>
  <depend>rclcpp</depend>
  <depend>rclcpp_components</depend>
  <depend>builtin_interfaces</depend>
  <depend>nav_msgs</depend>
  <depend>geometry_msgs</depend>
  <depend>sensor_msgs</depend>
  <depend>diagnostic_msgs</depend>
  <depend>tf2_ros</depend>
  <depend>tram_vehicle_msgs</depend>
  <depend>tram_odometry_msgs</depend>
  <depend>rosbag2_cpp</depend>          <!-- только офлайн-реплей; в CMake — find_package(... QUIET) -->
  <!-- Eigen при необходимости: <depend>eigen</depend> + <build_depend>eigen3_cmake_module</build_depend> -->
  <exec_depend>launch_ros</exec_depend>
  <exec_depend>rclpy</exec_depend>      <!-- scripts/latency_probe.py -->
  <test_depend>ament_cmake_gtest</test_depend>
  <test_depend>launch_testing_ament_cmake</test_depend>
  <test_depend>launch_testing_ros</test_depend>
  <test_depend>ros2bag</test_depend>
  <export><build_type>ament_cmake</build_type></export>
</package>
```

### 15.3 `CMakeLists.txt`
```cmake
cmake_minimum_required(VERSION 3.8)
project(tram_odometry)

# colcon build без --cmake-args собирает БЕЗ оптимизации (-O0) — ставим Release по умолчанию
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)   # без -Werror: у жюри может быть другой компилятор
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(diagnostic_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tram_vehicle_msgs REQUIRED)
find_package(tram_odometry_msgs REQUIRED)

# 1) ROS-free core (собирается и нативно на Windows с -DTRAM_CORE_ONLY)
add_library(tram_odometry_core STATIC
  src/core/estimator.cpp src/core/traction_model.cpp src/core/geodesy.cpp src/core/track_map.cpp)
target_include_directories(tram_odometry_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include>)
set_target_properties(tram_odometry_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

# 2) ROS node: component (.so) + standalone executable tram_odometry_node
add_library(tram_odometry_component SHARED src/tram_odometry_node.cpp)
target_link_libraries(tram_odometry_component tram_odometry_core)
ament_target_dependencies(tram_odometry_component rclcpp rclcpp_components builtin_interfaces
  nav_msgs geometry_msgs sensor_msgs diagnostic_msgs tf2_ros tram_vehicle_msgs tram_odometry_msgs)
rclcpp_components_register_node(tram_odometry_component
  PLUGIN "tram_odometry::TramOdometryNode" EXECUTABLE tram_odometry_node)

# 3) Offline deterministic replay (same core, no DDS)
find_package(rosbag2_cpp QUIET)
if(rosbag2_cpp_FOUND)
  add_executable(bag_replay src/tools/bag_replay.cpp)
  target_link_libraries(bag_replay tram_odometry_core)
  ament_target_dependencies(bag_replay rclcpp rosbag2_cpp tram_vehicle_msgs sensor_msgs)
  install(TARGETS bag_replay DESTINATION lib/${PROJECT_NAME})
endif()

install(TARGETS tram_odometry_core tram_odometry_component
  ARCHIVE DESTINATION lib LIBRARY DESTINATION lib RUNTIME DESTINATION bin)
install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY launch config maps DESTINATION share/${PROJECT_NAME})
install(PROGRAMS scripts/latency_probe.py scripts/resource_monitor.py scripts/run_eval.sh
  DESTINATION lib/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_core test/test_core.cpp test/test_rt_utils.cpp)
  target_link_libraries(test_core tram_odometry_core)
  find_package(launch_testing_ament_cmake REQUIRED)
  add_launch_test(test/test_realtime.launch.py TIMEOUT 180)
  install(DIRECTORY test/data DESTINATION share/${PROJECT_NAME}/test)
endif()

ament_package()
```
Пакет `tram_odometry_msgs`: `find_package(rosidl_default_generators REQUIRED)`, `rosidl_generate_interfaces(${PROJECT_NAME} "msg/EstimatorDebug.msg" DEPENDENCIES std_msgs)`, в package.xml `<member_of_group>rosidl_interface_packages</member_of_group>`. Шаблон тот же, что у `tram_vehicle_msgs`.

### 15.4 Правила «сборка без интернета»
- Никаких `FetchContent`, `ExternalProject`, `pip install`, `git clone` в CMake или setup.py. gtest берётся из `ament_cmake_gtest` → `gtest_vendor`.
- Python-инструменты жюри (`latency_probe.py`) используют только `rclpy` и сообщения. `rosbags`, `scipy` и `matplotlib` нужны только в наших офлайн-скриптах и **не указываются** в package.xml, иначе `rosdep` у жюри упадёт.
- `.gitattributes`: `* text=auto eol=lf`. CRLF с Windows ломает шебанг `#!/usr/bin/env python3\r` и bash-скрипты.
- `ament_lint_auto` в тестах либо проходить (cpplint/uncrustify), либо не подключать: упавшие стиль-тесты выглядят плохо в `colcon test-result`.
- Копию `tram_vehicle_msgs` держать в `src/`. Если у жюри пакет уже в underlay, colcon выдаст только предупреждение о переопределении, а определения идентичны.

---

## 16. Docker (Windows-хост) и офлайн-реплей

### 16.1 Dockerfile
```dockerfile
FROM osrf/ros:humble-desktop
# (или ros:humble-ros-base — меньше, достаточно для узла; rviz не нужен)
SHELL ["/bin/bash", "-c"]
ENV ROS_DOMAIN_ID=42 ROS_LOCALHOST_ONLY=1
WORKDIR /ws
COPY src ./src
RUN source /opt/ros/humble/setup.bash && \
    colcon build --event-handlers console_cohesion+ && \
    echo "source /ws/install/setup.bash" >> /root/.bashrc
```
Команды в PowerShell на хосте. Docker Desktop с бэкендом WSL2 сейчас остановлен, его надо запустить. Базовый образ скачивается один раз заранее:
```powershell
docker build --network=none -t tram-odom .          # доказательство офлайн-сборки (базовый образ уже в кэше)
docker run --rm -it --cpus=2 --memory=512m --memory-swap=512m --shm-size=256m `
  -v C:\MosTransHack\data\bags:/bags:ro tram-odom bash
# внутри контейнера (tmux или три `docker exec`):
ros2 launch tram_odometry tram_odometry.launch.py
ros2 run tram_odometry latency_probe.py
ros2 bag play /bags/30618_0e41eac3 --clock --disable-keyboard-controls
```
Замечания для Windows:
- Все процессы (узел, player, probe) лучше держать **в одном контейнере**. DDS между контейнером и Windows-хостом на Docker Desktop требует unicast-настройки, потому что `--network host` относится к VM, а не к Windows ([AirSim DDS setup](https://microsoft.github.io/AirSimExtensions/ros2_dds_config_setup/), [Fast-DDS#1698](https://github.com/eProsima/Fast-DDS/issues/1698)).
- Fast DDS использует общую память, `/dev/shm` по умолчанию 64 МБ. Для маленьких сообщений этого хватает, `--shm-size=256m` берётся с запасом. При нескольких контейнерах нужен `--ipc=host`, иначе будут проблемы с SHM.
- Bind-mount с диска C: медленный (9p/DrvFs). Для воспроизведения 1× это не важно. Для массовых прогонов лучше скопировать bag в docker volume.

### 16.2 Офлайн-реплей тем же ядром (`bag_replay`)
```cpp
#include <rosbag2_cpp/reader.hpp>
#include <rclcpp/serialization.hpp>
#include "tram_vehicle_msgs/msg/velocity_sensor.hpp"
#include "tram_vehicle_msgs/msg/driver_controller_command.hpp"
// ...
rosbag2_cpp::Reader reader;
reader.open(bag_dir);                               // sqlite3, metadata.yaml
rclcpp::Serialization<tram_vehicle_msgs::msg::VelocitySensor> ser_v;
rclcpp::Serialization<tram_vehicle_msgs::msg::DriverControllerCommand> ser_c;
while (reader.has_next()) {
  auto bm = reader.read_next();                     // SerializedBagMessage{serialized_data, time_stamp, topic_name}
  const double t_recv = 1e-9 * static_cast<double>(bm->time_stamp);   // = время воспроизведения
  rclcpp::SerializedMessage sm(*bm->serialized_data);
  if (bm->topic_name == "/vehicle/driver_position_cmd") {
    tram_vehicle_msgs::msg::DriverControllerCommand m; ser_c.deserialize_message(&sm, &m);
    node_logic.on_cmd(m, t_recv);                   // тот же код, что в онлайн-узле, время инжектируется
  } else if (bm->topic_name == "/vehicle/front_bogie_velocity") { /* ... */ }
}
```
- **Главный принцип:** логика узла (санитайзер, TimeKeeper, спаривание тележек, публикация) получает время приёма **аргументом**, а не читает часы сама. Тогда реплей воспроизводит онлайн-поведение побитово, включая метки и частоту выхода, а 36,7 ч данных обрабатываются за секунды. Это и есть основа калибровки и таблиц точности.
- Ядро собирается и **нативно на Windows** (на машине есть g++ 16.1 MinGW и CMake 4.4). Вход — CSV, выгруженные из `data/npz`. **Особенность этой машины:** динамическая линковка берёт `libstdc++-6.dll` из Git for Windows и падает с segfault при конструировании `std::ifstream`. Решение — `-static`, проверено.

### 16.3 Сквозной прогон (`scripts/run_eval.sh`)
```bash
#!/usr/bin/env bash
# usage: run_eval.sh <bag_dir> <out_dir>   — реальный ROS-контур: launch + play + record + метрики
set -euo pipefail
BAG=$1; OUT=$2; mkdir -p "$OUT"
ros2 launch tram_odometry tram_odometry.launch.py > "$OUT/node.log" 2>&1 & NODE=$!
ros2 run tram_odometry latency_probe.py > "$OUT/latency.txt" 2>&1 & PROBE=$!
ros2 run tram_odometry resource_monitor.py --name tram_odometry_node --out "$OUT/resources.csv" & MON=$!
ros2 bag record -o "$OUT/result_bag" /result/velocity /result/position /tram_odometry/debug /diagnostics & REC=$!
sleep 3
ros2 bag play "$BAG" --disable-keyboard-controls
sleep 2
kill -INT $REC $PROBE $MON $NODE; wait || true
python3 "$(ros2 pkg prefix tram_odometry_tools)/lib/tram_odometry_tools/evaluate.py" \
        --input-bag "$BAG" --result-bag "$OUT/result_bag" --out "$OUT/metrics.json"
```

---

## 17. Шаблон «Инструкция для жюри» (обязательный артефакт № 2)

```markdown
## Сборка (без интернета, ROS 2 Humble ros-base/desktop)
mkdir -p ~/tram_ws/src && cp -r <repo>/src/* ~/tram_ws/src/
cd ~/tram_ws && source /opt/ros/humble/setup.bash && colcon build      # Release по умолчанию
source install/setup.bash

## Запуск (узел ДО воспроизведения bag)
T1: ros2 launch tram_odometry tram_odometry.launch.py [use_sim_time:=true если bag с --clock]
T2 (опц.): ros2 run tram_odometry latency_probe.py        # задержка/частота по точной метке
T3: ros2 bag play <bag_dir> [--clock]

## Что ожидать на выходе
/result/velocity   tram_vehicle_msgs/VelocitySensor  ~20 Гц, м/с, frame_id=base_link
/result/position   nav_msgs/Odometry  ~20 Гц, frame_id=odom (локальная ENU от точки выставки), child=base_link,
                   pose.covariance/twist.covariance заполнены; twist.linear.x — продольная скорость
/tram_odometry/debug  tram_odometry_msgs/EstimatorDebug  (слип, сцепление, веса доверия, задержка)
/diagnostics       diagnostic_msgs/DiagnosticArray 1 Гц (inputs / slip / timing / estimator)

## Проверка реального времени
ros2 topic hz /result/position
ros2 topic echo /diagnostics            # proc_p50_ms / proc_p99_ms / proc_max_ms / rate_hz
ros2 topic delay -s /result/position    # только при ros2 bag play --clock
Примечание: первые ~0,2 с каждого bag содержат «догоняющий» пакет записей с метками до 3,5 с в прошлом —
«задержка по /clock» в этот момент — артефакт записи, не узла.

## Логи и метрики
~/.ros/log/…/tram_odometry*.log, итог: ~/.ros/tram_odometry_summary.csv
GNSS используется только для выставки: в логе строка «GNSS init done …; GNSS subscriptions destroyed».
```

---

## 18. Чек-лист перед сдачей

- [ ] `colcon build` в чистом `ros:humble-ros-base` с `--network=none` проходит, `colcon test` зелёный.
- [ ] Узел стартует меньше чем за 1 с и не падает при запуске **без** bag, **без** GNSS, при `--loop`, при `--rate 10`, при удалённом топике (`--topics` без rear или cmd).
- [ ] Выходные метки строго монотонны, ненулевые, пара velocity и position с одной меткой. NaN нет.
- [ ] Частота ≥ 20 Гц при нормальных входах и ≥ 10 Гц при полном пропуске входов на время ≤ 3 с.
- [ ] p99 задержки обработки < 1 мс, p99 `latency_probe` < 10 мс, 0 случаев > 100 мс. Всё это в таблице с методикой.
- [ ] RSS стабилен (наклон ≈ 0) на 30-минутном bag, CPU < 5 % одного ядра. Прогон с `--cpus=2 --memory=512m`.
- [ ] `Odometry`: frame_id, child_frame_id, единичный кватернион, осмысленные ковариации (NEES-тест).
- [ ] `/result/velocity` в **м/с** (пересчёт из км/ч).
- [ ] GNSS-подписки уничтожены после выставки, в логе есть подтверждение.
- [ ] `.gitattributes eol=lf`, у скриптов есть шебанги.
- [ ] Уточнены у организаторов начало и точка эталонной СК и источник времени эталона.

---

## 19. Эскиз узла (C++, Humble) — вся ROS-обвязка

```cpp
// src/tram_odometry_node.cpp — ROS glue only; model math lives in the ROS-free core (estimator.hpp)
#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>
#include <string>

#include "diagnostic_msgs/msg/diagnostic_array.hpp"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp_components/register_node_macro.hpp"
#include "sensor_msgs/msg/nav_sat_fix.hpp"
#include "tf2_ros/transform_broadcaster.h"
#include "tram_odometry/estimator.hpp"     // Estimator, Params, Bogie, EstimatorOutput (ROS-free)
#include "tram_odometry/rt_utils.hpp"      // StampSanitizer, LatencyHistogram, fill_pose_covariance
#include "tram_odometry/time_keeper.hpp"   // steady -> bag-time mapping
#include "tram_odometry_msgs/msg/estimator_debug.hpp"
#include "tram_vehicle_msgs/msg/driver_controller_command.hpp"
#include "tram_vehicle_msgs/msg/velocity_sensor.hpp"

namespace tram_odometry {
using VelocitySensor = tram_vehicle_msgs::msg::VelocitySensor;
using DriverCmd = tram_vehicle_msgs::msg::DriverControllerCommand;
using Odometry = nav_msgs::msg::Odometry;
using NavSatFix = sensor_msgs::msg::NavSatFix;
using Verdict = tram_rt::StampSanitizer::Verdict;
using namespace std::chrono_literals;

namespace {
inline double to_sec(const builtin_interfaces::msg::Time & t) { return t.sec + 1e-9 * t.nanosec; }
inline builtin_interfaces::msg::Time to_stamp(double s) {
  const int64_t ns = static_cast<int64_t>(std::llround(s * 1e9));
  builtin_interfaces::msg::Time t;
  t.sec = static_cast<int32_t>(ns / 1000000000LL);
  t.nanosec = static_cast<uint32_t>(ns % 1000000000LL);
  return t;
}
inline double steady_now() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
inline int64_t system_now_ns() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
    std::chrono::system_clock::now().time_since_epoch()).count();
}
}  // namespace

class TramOdometryNode : public rclcpp::Node {
public:
  explicit TramOdometryNode(const rclcpp::NodeOptions & options)
  : Node("tram_odometry", options), steady_clock_(RCL_STEADY_TIME) {
    world_frame_ = declare_parameter<std::string>("frames.world", "odom");
    base_frame_ = declare_parameter<std::string>("frames.base", "base_link");
    const bool publish_tf = declare_parameter<bool>("frames.publish_tf", true);
    wheel_scale_ = declare_parameter<double>("input.wheel_speed_scale", 1.0 / 3.6);
    gap_fill_after_s_ = declare_parameter<double>("timing.gap_fill_after_s", 0.075);
    max_extrap_s_ = declare_parameter<double>("timing.max_extrapolation_s", 3.0);
    est_.configure(Params::from_node(*this));        // declare/read model.*, slip.*, filter.* params

    const auto in_qos = rclcpp::QoS(rclcpp::KeepLast(100)).best_effort().durability_volatile();
    const auto out_qos = rclcpp::QoS(rclcpp::KeepLast(10)).reliable().durability_volatile();
    pub_vel_ = create_publisher<VelocitySensor>("/result/velocity", out_qos);
    pub_odom_ = create_publisher<Odometry>("/result/position", out_qos);
    pub_dbg_ = create_publisher<tram_odometry_msgs::msg::EstimatorDebug>("~/debug", out_qos);
    pub_diag_ = create_publisher<diagnostic_msgs::msg::DiagnosticArray>("/diagnostics", 10);
    if (publish_tf) tf_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

    // subscriptions LAST (everything above is ready before the first callback)
    sub_front_ = create_subscription<VelocitySensor>("/vehicle/front_bogie_velocity", in_qos,
      [this](VelocitySensor::ConstSharedPtr m, const rclcpp::MessageInfo & i) {
        guarded("front", [&] { on_wheel(Bogie::kFront, *m, i); });
      });
    sub_rear_ = create_subscription<VelocitySensor>("/vehicle/rear_bogie_velocity", in_qos,
      [this](VelocitySensor::ConstSharedPtr m, const rclcpp::MessageInfo & i) {
        guarded("rear", [&] { on_wheel(Bogie::kRear, *m, i); });
      });
    sub_cmd_ = create_subscription<DriverCmd>("/vehicle/driver_position_cmd", in_qos,
      [this](DriverCmd::ConstSharedPtr m, const rclcpp::MessageInfo & i) {
        guarded("cmd", [&] { on_cmd(*m, i); });
      });
    // GNSS: initial alignment ONLY; destroyed after init (see on_watchdog)
    sub_fix_master_ = create_subscription<NavSatFix>("/sensing/gnss/master/fix", in_qos,
      [this](NavSatFix::ConstSharedPtr m) { guarded("gnss", [&] { on_fix(*m, true); }); });
    sub_fix_rover_ = create_subscription<NavSatFix>("/sensing/gnss/rover/fix", in_qos,
      [this](NavSatFix::ConstSharedPtr m) { guarded("gnss", [&] { on_fix(*m, false); }); });

    watchdog_ = create_wall_timer(20ms, [this] { guarded("watchdog", [&] { on_watchdog(); }); });
    diag_timer_ = create_wall_timer(1s, [this] { guarded("diag", [&] { publish_diagnostics(); }); });
  }

private:
  template <class F>
  void guarded(const char * where, F && f) {
    try {
      f();
    } catch (const std::exception & e) {
      ++exceptions_;
      RCLCPP_ERROR_THROTTLE(get_logger(), steady_clock_, 2000, "[%s] %s -> soft reset", where, e.what());
      est_.soft_reset();
    } catch (...) {
      ++exceptions_;
      est_.soft_reset();
    }
  }

  void on_cmd(const DriverCmd & m, const rclcpp::MessageInfo & info) {
    const double t_recv = steady_now();
    last_input_steady_ = t_recv;
    const double st = to_sec(m.header.stamp);
    if (san_cmd_.check(st, t_recv) == Verdict::kRejected) return;
    est_.set_control(st, std::clamp<int>(m.position, -15, 15));     // zero-order hold
    tk_.observe_master(st, t_recv);                                  // cmd stamp ~ bag time (lag ~1 ms)
    last_cmd_steady_ = t_recv;
    publish_at(st, t_recv, info.get_rmw_message_info().source_timestamp);
  }

  void on_wheel(Bogie b, const VelocitySensor & m, const rclcpp::MessageInfo & info) {
    const double t_recv = steady_now();
    last_input_steady_ = t_recv;
    const double st = to_sec(m.header.stamp);
    auto & san = (b == Bogie::kFront) ? san_front_ : san_rear_;
    if (san.check(st, t_recv) == Verdict::kRejected) return;
    const double v = m.velocity * wheel_scale_;                      // km/h -> m/s
    if (!std::isfinite(v) || v < -1.0 || v > 30.0) { ++outliers_; return; }
    est_.add_wheel(b, st, v, t_recv);            // pairs front/rear by exact stamp, updates filter
    tk_.observe_slave(st, t_recv);
    if (t_recv - last_cmd_steady_ > 0.5) {       // cmd stream missing -> wheel events drive output
      publish_at(tk_.bag_time(t_recv), t_recv, info.get_rmw_message_info().source_timestamp);
    }
  }

  void on_fix(const NavSatFix & m, bool master) {
    if (gnss_done_ || m.status.status < 0 || !std::isfinite(m.latitude) || !std::isfinite(m.longitude)) return;
    if (est_.add_init_fix(to_sec(m.header.stamp), m.latitude, m.longitude, m.altitude, master)) {
      gnss_done_ = true;                          // subscriptions destroyed in the next watchdog tick
    }
  }

  void publish_at(double stamp, double t_recv, int64_t src_ns) {
    if (!(stamp > last_pub_stamp_ + 1e-3)) return;                   // strictly monotonic output
    const EstimatorOutput o = est_.predict(stamp);                   // const: no state change
    if (!o.finite()) { ++nan_resets_; est_.soft_reset(); return; }
    const auto st = to_stamp(stamp);

    vel_msg_.header.stamp = st;
    vel_msg_.header.frame_id = base_frame_;
    vel_msg_.velocity = o.v;

    odom_msg_.header.stamp = st;
    odom_msg_.header.frame_id = world_frame_;
    odom_msg_.child_frame_id = base_frame_;
    odom_msg_.pose.pose.position.x = o.x;
    odom_msg_.pose.pose.position.y = o.y;
    odom_msg_.pose.pose.position.z = o.z;
    odom_msg_.pose.pose.orientation.x = 0.0;
    odom_msg_.pose.pose.orientation.y = 0.0;
    odom_msg_.pose.pose.orientation.z = std::sin(0.5 * o.yaw);
    odom_msg_.pose.pose.orientation.w = std::cos(0.5 * o.yaw);
    tram_rt::fill_pose_covariance(odom_msg_.pose.covariance, o.yaw, o.var_along, o.var_cross, o.var_z,
                                  1e-4, o.var_yaw);
    odom_msg_.twist.twist.linear.x = o.v;
    odom_msg_.twist.twist.angular.z = o.v * o.curvature;
    odom_msg_.twist.covariance.fill(0.0);
    odom_msg_.twist.covariance[0] = o.var_v;
    odom_msg_.twist.covariance[7] = odom_msg_.twist.covariance[14] = 1e-4;
    odom_msg_.twist.covariance[21] = odom_msg_.twist.covariance[28] = 1e-4;
    odom_msg_.twist.covariance[35] = o.var_wz;

    pub_vel_->publish(vel_msg_);
    pub_odom_->publish(odom_msg_);
    if (tf_) {
      geometry_msgs::msg::TransformStamped tf;
      tf.header = odom_msg_.header;
      tf.child_frame_id = base_frame_;
      tf.transform.translation.x = o.x;
      tf.transform.translation.y = o.y;
      tf.transform.translation.z = o.z;
      tf.transform.rotation = odom_msg_.pose.pose.orientation;
      tf_->sendTransform(tf);
    }
    last_pub_stamp_ = stamp;
    last_pub_steady_ = steady_now();
    const double proc_ms = (last_pub_steady_ - t_recv) * 1e3;
    lat_proc_.add(proc_ms);
    if (src_ns > 0) lat_e2e_.add(1e-6 * static_cast<double>(system_now_ns() - src_ns));
    ++n_pub_;
    publish_debug(o, stamp, proc_ms);            // EstimatorDebug (same stamp)
  }

  void on_watchdog() {
    if (gnss_done_ && sub_fix_master_) {
      sub_fix_master_.reset();
      sub_fix_rover_.reset();
      RCLCPP_INFO(get_logger(), "GNSS init done; GNSS subscriptions destroyed (not used in main loop)");
    }
    const double now = steady_now();
    if (!tk_.valid() || now - last_pub_steady_ < gap_fill_after_s_) return;   // outputs are flowing
    if (now - last_input_steady_ > max_extrap_s_) return;                      // bag ended/paused
    publish_at(tk_.bag_time(now), now, 0);                                     // model-only prediction
  }

  void publish_diagnostics();                     // DiagnosticArray, see section 9.1
  void publish_debug(const EstimatorOutput & o, double stamp, double proc_ms);

  rclcpp::Clock steady_clock_;
  Estimator est_;
  TimeKeeper tk_;
  tram_rt::StampSanitizer san_front_, san_rear_, san_cmd_;
  tram_rt::LatencyHistogram lat_proc_, lat_e2e_;
  VelocitySensor vel_msg_;
  Odometry odom_msg_;
  std::string world_frame_, base_frame_;
  double wheel_scale_{1.0 / 3.6}, gap_fill_after_s_{0.075}, max_extrap_s_{3.0};
  double last_pub_stamp_{0.0}, last_pub_steady_{0.0}, last_input_steady_{0.0}, last_cmd_steady_{-1e9};
  bool gnss_done_{false};
  uint64_t n_pub_{0}, outliers_{0}, nan_resets_{0}, exceptions_{0};
  rclcpp::Publisher<VelocitySensor>::SharedPtr pub_vel_;
  rclcpp::Publisher<Odometry>::SharedPtr pub_odom_;
  rclcpp::Publisher<tram_odometry_msgs::msg::EstimatorDebug>::SharedPtr pub_dbg_;
  rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr pub_diag_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_;
  rclcpp::Subscription<VelocitySensor>::SharedPtr sub_front_, sub_rear_;
  rclcpp::Subscription<DriverCmd>::SharedPtr sub_cmd_;
  rclcpp::Subscription<NavSatFix>::SharedPtr sub_fix_master_, sub_fix_rover_;
  rclcpp::TimerBase::SharedPtr watchdog_, diag_timer_;
};
}  // namespace tram_odometry

RCLCPP_COMPONENTS_REGISTER_NODE(tram_odometry::TramOdometryNode)
```
Для полной детерминированности реплея (раздел 16.2) вынесите тела `on_cmd`/`on_wheel`/`publish_at` в класс `NodeLogic` без ROS-зависимостей, принимающий `t_recv` аргументом и возвращающий «что опубликовать». Узел тогда — тонкая обёртка.

---

## 20. Источники

**ROS 2 (первоисточники, код Humble):**
- ROS 2 docs, Executors (humble): https://github.com/ros2/ros2_documentation/blob/humble/source/Concepts/Intermediate/About-Executors.rst
- ROS 2 docs, QoS (humble): https://github.com/ros2/ros2_documentation/blob/humble/source/Concepts/Intermediate/About-Quality-of-Service-Settings.rst
- rclcpp qos.hpp (humble): https://github.com/ros2/rclcpp/blob/humble/rclcpp/include/rclcpp/qos.hpp
- rclcpp message_info.hpp (humble): https://github.com/ros2/rclcpp/blob/humble/rclcpp/include/rclcpp/message_info.hpp ; устаревшие сигнатуры колбэков: https://github.com/ros2/rclcpp/issues/2972
- rmw types.h (humble), `rmw_message_info_t`: https://github.com/ros2/rmw/blob/humble/rmw/include/rmw/types.h
- rmw_fastrtps rmw_take.cpp (humble): https://github.com/ros2/rmw_fastrtps/blob/humble/rmw_fastrtps_shared_cpp/src/rmw_take.cpp ; rmw_cyclonedds rmw_node.cpp (humble): https://github.com/ros2/rmw_cyclonedds/blob/humble/rmw_cyclonedds_cpp/src/rmw_node.cpp
- rosbag2 play CLI (humble): https://github.com/ros2/rosbag2/blob/humble/ros2bag/ros2bag/verb/play.py ; выбор QoS при воспроизведении: https://github.com/ros2/rosbag2/blob/humble/rosbag2_transport/src/rosbag2_transport/qos.cpp ; переопределение QoS: https://docs.ros.org/en/humble/How-To-Guides/Overriding-QoS-Policies-For-Recording-And-Playback.html ; rosbag2_cpp Reader: https://github.com/ros2/rosbag2/blob/humble/rosbag2_cpp/include/rosbag2_cpp/reader.hpp
- ros2topic hz/delay (humble): https://github.com/ros2/ros2cli/blob/humble/ros2topic/ros2topic/verb/hz.py , https://github.com/ros2/ros2cli/blob/humble/ros2topic/ros2topic/verb/delay.py , DirectNode `--use-sim-time`: https://github.com/ros2/ros2cli/blob/humble/ros2cli/ros2cli/node/direct.py
- Design: Clock and Time: https://design.ros2.org/articles/clock_and_time.html
- Варианты Humble (ros_core/ros_base/desktop): https://github.com/ros2/variants/tree/humble ; geometry2: https://github.com/ros2/geometry2/tree/humble ; orocos_kdl_vendor (eigen, eigen3_cmake_module): https://github.com/ros2/orocos_kdl_vendor
- Образы Docker: https://github.com/osrf/docker_images/tree/master/ros/humble/ubuntu/jammy
- nav_msgs/Odometry: https://github.com/ros2/common_interfaces/blob/humble/nav_msgs/msg/Odometry.msg ; PoseWithCovariance: https://github.com/ros2/common_interfaces/blob/humble/geometry_msgs/msg/PoseWithCovariance.msg ; DiagnosticStatus: https://github.com/ros2/common_interfaces/blob/humble/diagnostic_msgs/msg/DiagnosticStatus.msg
- REP-103: https://github.com/ros-infrastructure/rep/blob/master/rep-0103.rst ; REP-105: https://github.com/ros-infrastructure/rep/blob/master/rep-0105.rst ; REP-2014 (бенчмаркинг): https://ros.org/reps/rep-2014.html
- launch_testing README (humble): https://github.com/ros2/launch/blob/humble/launch_testing/README.md
- ros2_tracing README (humble): https://github.com/ros2/ros2_tracing/blob/humble/README.md
- message_filters (ApproximateTime): https://docs.ros.org/en/jazzy/p/message_filters/doc/index.html
- diagnostics (ros2-humble): https://github.com/ros/diagnostics/tree/ros2-humble
- robot_localization, подготовка данных: http://docs.ros.org/en/noetic/api/robot_localization/html/preparing_sensor_data.html

**Научные работы:**
- T. Kronauer et al., «Latency Analysis of ROS2 Multi-Node Systems», 2021, arXiv:2101.02074: https://arxiv.org/abs/2101.02074
- C. Bédard, I. Lütkebohle, M. Dagenais, «ros2_tracing: Multipurpose Low-Overhead Framework for Real-Time Tracing of ROS 2», IEEE RA-L 7(3), 2022, doi:10.1109/LRA.2022.3174346, arXiv:2201.00393
- S. Macenski, A. Soragna, M. Carroll, «Impact of ROS 2 Node Composition in Robotic Systems», IEEE RA-L 2023, arXiv:2305.09933
- D. Casini, T. Blaß, I. Lütkebohle, B. Brandenburg, «Response-Time Analysis of ROS 2 Processing Chains Under Reservation-Based Scheduling», ECRTS 2019, LIPIcs 133: https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.ECRTS.2019.6
- «A Survey of Real-Time Support, Analysis, and Advancements in ROS 2», 2026, arXiv:2601.10722: https://arxiv.org/html/2601.10722v1
- Y. Bar-Shalom, «Update with out-of-sequence measurements in tracking: exact solution», IEEE TAES 38(3):769–777, 2002, doi:10.1109/TAES.2002.1039398
- Y. Bar-Shalom, X.R. Li, T. Kirubarajan, *Estimation with Applications to Tracking and Navigation*, Wiley, 2001 (NEES/NIS-тесты согласованности)
- E. Olson, «A passive solution to the sensor synchronization problem», IROS 2010
- CARET (Tier IV): https://tier4.github.io/caret_doc/main/ , IEEE: https://ieeexplore.ieee.org/document/10086380/

**Практика и сообщество:**
- rclpy vs rclcpp: https://github.com/ros2/rclpy/issues/763 , https://github.com/karl-schulz/ros2_latency
- EventsExecutor (iRobot): https://github.com/irobot-ros/events-executor
- DDS в Docker на Windows: https://microsoft.github.io/AirSimExtensions/ros2_dds_config_setup/ , https://github.com/eProsima/Fast-DDS/issues/1698
- Страница хакатона: https://mt-hackathon.ru/ ; PDF кейса «Резервная одометрия по модели» и README датасета (локально).
