# Индекс GitHub-проектов: нейросети и ML для League of Legends

Собран 2026-10-08. В основной части 196 проверенных репозиториев: 179 в 13 категориях и 17 дополнений из веб-поиска в разделе 14. В приложении ещё 270 однотипных учебных проектов.

## Как собрано

- **Поиск GitHub** (`gh search repos`): около 140 запросов по темам (`league-of-legends` × `deep-learning`/`yolo`/`reinforcement-learning`/…) и ключевым фразам (minimap, OCR, draft, replay, LLM, coach, TFT, Wild Rift…). Получилось 1043 уникальных репозитория. Фильтр «LoL + ML» оставил 600, затем шум убран вручную: Rocket League, Discord-боты без ML, читы, спам.
- **Второй круг:** форки и соседние репозитории ключевых авторов (Oleffa, Maknee, MiscellaneousStuff, davidweatherall), прямые проверки известных проектов. Так нашёлся `farzaa/DeepLeague` (1,2k★), которого поиск по ключевым словам не выдал.
- **Веб-поиск:** awesome-списки, статьи arXiv/AAAI/CVPR/IEEE со ссылками на код, страницы GitHub Topics.
- **Проверка:** каждая запись основной части сверена через GitHub GraphQL. Все 179 существуют; звёзды, дата последнего push, архивность и лицензия взяты оттуда. Описания взяты из README и сниппетов. Код и заявленная точность не проверялись.

## Главное для lol-game-state-vision

1. **Прямого аналога нашей задачи не найдено.** Никто не распознаёт область карты по основному виду с замаскированной миникартой. Почти вся работа по CV сделана по миникарте. Основной экран разбирают только LeagueAI (детекция объектов) и spectral-sight.
2. **Миникарта может служить бесплатной разметкой.** Детекторы рамки камеры и иконок (DeeperLeague, pyLoL, DeepLeague, `may850608-glitch/lol-minimap-yolo`) дают координаты, которые видит основной кадр. Метку можно снять с немаскированного кадра, а обучать модель на маскированном.
3. **Самый сильный рецепт сообщества — синтетика.** DeeperLeague (~300k кадров), pyLoL (YOLOv12 на ассетах DataDragon), LeagueAI (3D-модели), AAAI-26 (sim-to-real для миникарты). Для основного вида синтетику получить труднее, но этот путь проверен.
4. **Истинные позиции из реплеев.** TLoL и датасеты `maknee` на HuggingFace содержат около 1,4M декодированных игр, `Mowokuma/ROFL` даёт позиции раз в секунду. Если связать реплей с видео, получится разметка «где камера/игрок» без ручного труда.
5. **Около двух третей экосистемы — предсказание исхода по табличным данным Riot API** (Kaggle, первые 10 минут). Для зрения эти проекты бесполезны.

## Статьи без найденного кода (для ориентира)

- Kim et al., IEEE ToG 2024, «Real-Time Player Tracking Framework on MOBA Game Video Through Object Detection» (doi 10.1109/TG.2024.3515140).
- Mutsvanga et al., IEEE SAS 2024, «Data Pipelines for Real-Time, Custom Object Detection and Tracking in LoL» ([запись](https://iris.unitn.it/handle/11572/475010)).
- [Maymin, MIT Sloan: Open-Sourced Optical Tracking … for LoL](https://www.sloansportsconference.com/research-papers/an-open-sourced-optical-tracking-and-advanced-esports-analytics-platform-for-league-of-legends).
- [Game-MUG, arXiv 2404.19175](https://arxiv.org/abs/2404.19175): мультимодальный датасет комментария.
- [LoL-V2T, CVPRW 2021](https://openaccess.thecvf.com/content/CVPR2021W/CVSports/html/Tanaka_LoL-V2T_Large-Scale_Esports_Video_Description_Dataset_CVPRW_2021_paper.html): видео → описание.
- Статьи с кодом из индекса: [AAAI-26 minimap sim-to-real](https://ojs.aaai.org/index.php/AAAI/article/view/42235) → `eidus/AAAI26_LoL_MinimapDetection` (код, данные, результаты), [LeagueAI arXiv 1905.13546](https://arxiv.org/abs/1905.13546), [DraftRec arXiv 2204.12750](https://arxiv.org/abs/2204.12750), [AIIDE 2020 bot](https://ojs.aaai.org/index.php/AIIDE/article/view/7449), [GCN-WP arXiv 2207.13191](https://arxiv.org/abs/2207.13191), [toxicity arXiv 2604.10175](https://arxiv.org/abs/2604.10175), [data-to-text arXiv 2212.10935](https://arxiv.org/abs/2212.10935).
- Датасеты вне GitHub: [maknee на HuggingFace](https://huggingface.co/datasets/maknee) (декодированные реплеи); модели Wild Rift есть только на Roboflow Universe.

«Последний push» означает месяц последнего коммита, а не обновления звёзд. Лицензия указана, если GitHub её распознал. Её отсутствие означает «все права защищены», поэтому код без лицензии нельзя заимствовать.

## 1. Зрение: миникарта

Позиции чемпионов, вардов и рамки камеры по миникарте. Для нашей задачи это источник разметки: рамка камеры на миникарте показывает, какую область карты видит основной кадр.

| Репозиторий | ★ | Последний push | Язык | Что делает | Пометки |
|---|--:|---|---|---|---|
| [farzaa/DeepLeague](https://github.com/farzaa/DeepLeague) | 1211 | 2019-12 | Python | Трекинг чемпионов на миникарте по VOD LCS, датасет 100k размеченных кадров; YOLO (YAD2K/Keras) |  |
| [league-of-legends-replay-extractor/pyLoL](https://github.com/league-of-legends-replay-extractor/pyLoL) | 77 | 2026-05 | Python | Позиции игроков и вардов из реплеев/видео (YOLOv12 на синтетике DataDragon) + OCR KDA/CS | GPL-3.0 |
| [davidweatherall/DeeperLeague](https://github.com/davidweatherall/DeeperLeague) | 31 | 2024-01 | Python | YOLOv8n на ~300k синтетических кадров миникарты; ядро Replays.lol; GPL-3.0 или закрытая лицензия |  |
| [Steeeephen/LAVA](https://github.com/Steeeephen/LAVA) | 27 | 2023-11 | Python | Анализ VOD: позиции по миникарте, хитмапы | Apache-2.0 |
| [dcheng728/League-X](https://github.com/dcheng728/League-X) | 24 | 2020-03 | Python | CNN (Keras) + OpenCV, позиции врагов на миникарте | GPL-3.0 |
| [Maknee/LeagueMinimapDetectionOpenCV](https://github.com/Maknee/LeagueMinimapDetectionOpenCV) | 23 | 2021-09 | Python | То же на классическом OpenCV (патч 10.7) | GPL-3.0 |
| [Maknee/LeagueMinimapDetectionCNN](https://github.com/Maknee/LeagueMinimapDetectionCNN) | 20 | 2021-09 | Jupyter Notebook | Детекция чемпионов на миникарте CNN |  |
| [dcheng728/League-Minimap-Scanner](https://github.com/dcheng728/League-Minimap-Scanner) | 14 | 2019-12 | Python | CNN, распознавание врагов на миникарте |  |
| [BrandonClapp/league-of-legends-opencv](https://github.com/BrandonClapp/league-of-legends-opencv) | 7 | 2024-05 | Python | OpenCV: чемпионы на миникарте, алерты |  |
| [jparedesDS/lol-map-tracking-object-detection](https://github.com/jparedesDS/lol-map-tracking-object-detection) | 3 | 2026-02 |  | Детекция и трекинг объектов на карте |  |
| [Baseult/League-of-Legends-YoloV5-Training-Images-Generator](https://github.com/Baseult/League-of-Legends-YoloV5-Training-Images-Generator) | 2 | 2022-08 | C# | Генератор синтетики для YOLOv5 (миникарта) |  |
| [Quinntana/LOL_Minimap_Tracker](https://github.com/Quinntana/LOL_Minimap_Tracker) | 1 | 2026-07 | Python | Трекинг врагов на миникарте в реальном времени | GPL-3.0 |
| [realr4an/LeagueOfLegendsMinimapTracker](https://github.com/realr4an/LeagueOfLegendsMinimapTracker) | 1 | 2025-08 | Python | Трекер миникарты с REST и оверлеем |  |
| [Deuzwood/minimap-analyser](https://github.com/Deuzwood/minimap-analyser) | 1 | 2021-06 | Python | Анализ миникарты |  |
| [BotzillaX/League-of-Legends-Jungle-Tracker](https://github.com/BotzillaX/League-of-Legends-Jungle-Tracker) | 1 | 2023-07 | Python | Трекер лесника |  |
| [Beomi/DeeperLeague](https://github.com/Beomi/DeeperLeague) | 0 | 2025-05 | Python | Форк DeeperLeague, обновлён под патч 25.10 |  |
| [Maknee/DeepLeague](https://github.com/Maknee/DeepLeague) | 0 | 2020-04 | Python | Вариант DeepLeague | форк |
| [sdtran14/Minimap_Detector](https://github.com/sdtran14/Minimap_Detector) | 0 | 2023-10 | Python | Детектор 169 классов на миникарте + генератор синтетики |  |
| [may850608-glitch/lol-minimap-yolo](https://github.com/may850608-glitch/lol-minimap-yolo) | 0 | 2026-07 | Python | YOLO11 по миникарте трансляций; синтетика и sim2real без ручной разметки |  |
| [eidus/AAAI26_LoL_MinimapDetection](https://github.com/eidus/AAAI26_LoL_MinimapDetection) | 0 | 2026-05 | Python | Репозиторий к AAAI-26 student abstract: синтетика → реальность для миникарты | MIT |
| [Evanjbraun/map-detection-league-of-legends](https://github.com/Evanjbraun/map-detection-league-of-legends) | 0 | 2026-02 | Python | CV по миникарте |  |
| [JanithCB/LOL-map-control-clustering](https://github.com/JanithCB/LOL-map-control-clustering) | 0 | 2026-05 | Python | CV-признаки миникарты + кластеризация контроля карты | MIT |
| [thesmartwon/lol-minimap-tracking](https://github.com/thesmartwon/lol-minimap-tracking) | 0 | 2018-12 |  | Трекинг по миникарте |  |
| [MADLionsEC/jungle_pathing_tracker](https://github.com/MADLionsEC/jungle_pathing_tracker) | 0 | 2018-05 |  | Трекер путей лесника на основе DeepLeague (MAD Lions) |  |
| [fadimeland-republic/LeagueIconDatasetGenerator](https://github.com/fadimeland-republic/LeagueIconDatasetGenerator) | 0 | 2025-10 | Python | Генератор датасета: иконки чемпионов + шум на пустой карте |  |

## 2. Зрение: основной игровой вид

Детекция объектов и извлечение состояния из основного кадра. Ближе всего к текущей задаче репозитория.

| Репозиторий | ★ | Последний push | Язык | Что делает | Пометки |
|---|--:|---|---|---|---|
| [Oleffa/LeagueAI](https://github.com/Oleffa/LeagueAI) | 248 | 2019-11 | TeX | YOLOv3 по основному экрану, синтетика из 3D-моделей; arXiv 1905.13546 | GPL-3.0 |
| [RoboNuke/LeagueAI2.0](https://github.com/RoboNuke/LeagueAI2.0) | 14 | 2021-05 | Python | Продолжение LeagueAI: CV-извлечение состояния игры |  |
| [Sahcim/HealthBarDetector](https://github.com/Sahcim/HealthBarDetector) | 5 | 2021-01 | Python | Детектор полосок HP | архив, MIT |
| [Shinobu-Kazahana/lol-vision](https://github.com/Shinobu-Kazahana/lol-vision) | 4 | 2024-09 | TypeScript | Детекция и трекинг персонажей на экране |  |
| [ryancole/spectral-sight](https://github.com/ryancole/spectral-sight) | 2 | 2026-10 | Python | Vision-only извлечение состояния из реплеев: пиксели → структурированное состояние |  |
| [austinmartratt/League-of-Legends-Image-Classifiers](https://github.com/austinmartratt/League-of-Legends-Image-Classifiers) | 2 | 2017-11 | Python | InceptionV3: классификация чемпионов, предметов, UI | GPL-3.0 |
| [Stefano314/LoLTrainer](https://github.com/Stefano314/LoLTrainer) | 1 | 2023-04 | Python | Распознавание предметов, dense NN | MIT |
| [Brendanhhh/LOL-Vision](https://github.com/Brendanhhh/LOL-Vision) | 1 | 2024-04 | Jupyter Notebook | Сбор данных (vision) |  |
| [Toasteee/LolAI](https://github.com/Toasteee/LolAI) | 0 | 2020-08 | Python | YOLOv3 + OpenCV/PyTorch по мотивам LeagueAI | GPL-3.0 |
| [DaniRuizPerez/PlayerImageRecognizerLeagueOfLegends](https://github.com/DaniRuizPerez/PlayerImageRecognizerLeagueOfLegends) | 0 | 2017-11 | MATLAB | MATLAB, распознавание чемпионов на изображении | GPL-3.0 |
| [JuniusYou/lol-analysis-toolkit](https://github.com/JuniusYou/lol-analysis-toolkit) | 0 | 2026-06 | Python | Захват Live Client + анализ Vision/LLM | MIT |
| [MariuszMackowski/LOL-Vision-challange](https://github.com/MariuszMackowski/LOL-Vision-challange) | 0 | 2021-02 | Python | Решение CV-челленджа esportsLabgg |  |

## 3. OCR и чтение HUD

Чтение чисел, таблиц и текста с экрана.

| Репозиторий | ★ | Последний push | Язык | Что делает | Пометки |
|---|--:|---|---|---|---|
| [Nyx0ra/lol-aram-mayhem-hextech-helper](https://github.com/Nyx0ra/lol-aram-mayhem-hextech-helper) | 47 | 2026-09 | Python | RapidOCR: распознавание аугментов ARAM Mayhem | MIT |
| [floh22/LeagueOCR](https://github.com/floh22/LeagueOCR) | 12 | 2021-04 | C# | OCR-сбор данных из режима наблюдателя (C#) | MIT |
| [teovoinea/LoLOCR](https://github.com/teovoinea/LoLOCR) | 3 | 2014-07 | C# | OCR по LoL |  |
| [themaxdavitt/league-of-legends-ocr](https://github.com/themaxdavitt/league-of-legends-ocr) | 1 | 2021-03 | TypeScript | Tesseract/Leptonica: значения из spectator mode | архив, MIT |
| [vrevolverrr/league-cs-overlay](https://github.com/vrevolverrr/league-cs-overlay) | 0 | 2021-11 | Python | OCR CS/мин, оверлей |  |
| [Pundeng/lol-inhouse-ocr](https://github.com/Pundeng/lol-inhouse-ocr) | 0 | 2026-08 | Python | OpenCV + OCR статистики со скриншотов |  |
| [yimfeng6/aram-augment-advisor](https://github.com/yimfeng6/aram-augment-advisor) | 0 | 2026-09 | Python | OpenCV + OCR аугментов ARAM | MIT |
| [Clarinet-koiyun/hex-companion](https://github.com/Clarinet-koiyun/hex-companion) | 0 | 2026-09 | JavaScript | Распознавание чемпионов + OCR аугментов | MIT |
| [Nixz0824/RiftLens-LoL-Korean-Translator-Purely-Visual](https://github.com/Nixz0824/RiftLens-LoL-Korean-Translator-Purely-Visual) | 0 | 2026-08 | C# | OCR-переводчик корейского чата по экрану | MIT |

## 4. Видео: хайлайты, описание, анализ VOD

Модели, работающие с последовательностями кадров.

| Репозиторий | ★ | Последний push | Язык | Что делает | Пометки |
|---|--:|---|---|---|---|
| [Flowtter/crispy](https://github.com/Flowtter/crispy) | 158 | 2025-03 | Python | Нейросеть детектит хайлайты в кадрах, автомонтаж | MIT |
| [tsunehiko/lol-v2t](https://github.com/tsunehiko/lol-v2t) | 5 | 2022-12 | Ruby | LoL-V2T: видео → описание, 9.7k клипов (CVPRW 2021) |  |
| [PepeTapia/WildAI](https://github.com/PepeTapia/WildAI) | 4 | 2023-02 | Python | Wild Rift: статистика из видео |  |
| [JackXu2333/teamfight-tactics-ai](https://github.com/JackXu2333/teamfight-tactics-ai) | 2 | 2023-03 | Jupyter Notebook | TFT: исход по визуальным данным (CNN+DNN) | архив |
| [londogard/lol_highlight_detection](https://github.com/londogard/lol_highlight_detection) | 1 | 2024-02 | Jupyter Notebook | Детекция хайлайтов | MIT |
| [Johnnyr612/LoL-Clip-Pipeline](https://github.com/Johnnyr612/LoL-Clip-Pipeline) | 0 | 2026-09 | Python | VideoMAE fine-tune, нарезка хайлайтов |  |

## 5. RL, агенты и игровые среды

Среды, боты на RL/BC и датасеты для обучения агентов.

| Репозиторий | ★ | Последний push | Язык | Что делает | Пометки |
|---|--:|---|---|---|---|
| [LeagueSandbox/GameServer](https://github.com/LeagueSandbox/GameServer) | 1172 | 2022-09 | C# | Эмулятор сервера v4.20 (база для RL-сред, сам не ML) | архив, AGPL-3.0 |
| [csci-599-applied-ml-for-games/league-of-legends-bot](https://github.com/csci-599-applied-ml-for-games/league-of-legends-bot) | 45 | 2023-03 | Python | YOLOv3 + LSTM + PPO в упрощённой LoL (AIIDE 2020) |  |
| [MiscellaneousStuff/tlol](https://github.com/MiscellaneousStuff/tlol) | 34 | 2023-12 | Jupyter Notebook | TLoL: датасеты реплеев S11–13 для RL/BC, центр экосистемы |  |
| [MiscellaneousStuff/lolgym](https://github.com/MiscellaneousStuff/lolgym) | 29 | 2021-10 | Python | Gym-среды поверх PyLoL | MIT |
| [MiscellaneousStuff/pylol](https://github.com/MiscellaneousStuff/pylol) | 23 | 2025-02 | Python | RL-среда LoL v4.20 (по образцу pysc2) | MIT |
| [MiscellaneousStuff/tlol-py](https://github.com/MiscellaneousStuff/tlol-py) | 21 | 2023-12 | Python | Python-модуль TLoL | MIT |
| [MiscellaneousStuff/tlol-rl](https://github.com/MiscellaneousStuff/tlol-rl) | 20 | 2023-02 | Python | Gym-интерфейс к живому клиенту | MIT |
| [Maknee/league-of-legends-decoded-replay-packets-gym](https://github.com/Maknee/league-of-legends-decoded-replay-packets-gym) | 16 | 2025-09 | Python | Gymnasium на декодированных пакетах про-реплеев (HF maknee), пример OpenLeague5 | Apache-2.0 |
| [MiscellaneousStuff/LeagueSandbox-RL-Learning](https://github.com/MiscellaneousStuff/LeagueSandbox-RL-Learning) | 14 | 2025-02 | C# | LeagueSandbox + Redis: наблюдения/действия для RL | AGPL-3.0 |
| [adrian27513/MOBA-AI-Gamer](https://github.com/adrian27513/MOBA-AI-Gamer) | 5 | 2022-12 | Python | CV + NN + RL по экрану |  |
| [MiscellaneousStuff/pylol-demo](https://github.com/MiscellaneousStuff/pylol-demo) | 2 | 2026-03 | Jupyter Notebook | Ноутбук-демо pylol | MIT |
| [MiscellaneousStuff/LoLRLE](https://github.com/MiscellaneousStuff/LoLRLE) | 2 | 2021-10 | Python | PPO, распределённое обучение, сценарии 1v1 | MIT |
| [MiscellaneousStuff/riftgym](https://github.com/MiscellaneousStuff/riftgym) | 2 | 2026-05 | Python | RL-библиотека для LoL Season 1 | MIT |
| [Ulyssse31/LeagueOfLegendsReinforcementLearning](https://github.com/Ulyssse31/LeagueOfLegendsReinforcementLearning) | 2 | 2025-04 |  | Попытка RL-бота |  |
| [MiscellaneousStuff/tlol-agent](https://github.com/MiscellaneousStuff/tlol-agent) | 1 | 2026-02 | Python | Агент на ~1.4M игр из датасетов maknee |  |
| [MiscellaneousStuff/tlol-ml](https://github.com/MiscellaneousStuff/tlol-ml) | 1 | 2023-02 | Jupyter Notebook | Обучение с учителем (833-JinxML) | MIT |
| [MiscellaneousStuff/tlol-general](https://github.com/MiscellaneousStuff/tlol-general) | 1 | 2025-02 |  | Sample-efficient агент на foundation/reasoning-моделях | MIT |
| [jerryqin1/lol-skillshot-dodger](https://github.com/jerryqin1/lol-skillshot-dodger) | 1 | 2022-05 | Python | Deep RL: уклонение от скиллшотов |  |
| [sunho/simple-lol-ai](https://github.com/sunho/simple-lol-ai) | 1 | 2018-06 | C++ | DL-агент для 2D-клона LoL |  |
| [sean-qin-usa/League-of-Legends-AI-Coach](https://github.com/sean-qin-usa/League-of-Legends-AI-Coach) | 1 | 2026-09 | Python | Коуч лесника: supervised + unsupervised + offline RL |  |
| [YAHIAG13/League-RL-Agent](https://github.com/YAHIAG13/League-RL-Agent) | 0 | 2024-06 | Python | Deep RL агент |  |
| [iagxferreira/riftlab](https://github.com/iagxferreira/riftlab) | 0 | 2026-09 | Python | DS + RL-эксперименты по принятию решений |  |

## 6. Реплеи, датасеты и сбор данных для ML

Парсеры .rofl, краулеры Riot API, генераторы датасетов.

| Репозиторий | ★ | Последний push | Язык | Что делает | Пометки |
|---|--:|---|---|---|---|
| [simoncos/lola](https://github.com/simoncos/lola) | 142 | 2026-08 | Python | Аналитика игровых данных |  |
| [Mowokuma/ROFL](https://github.com/Mowokuma/ROFL) | 41 | 2025-03 | Rust | Парсер .rofl: позиции раз в 1 с, варды | архив |
| [fraxiinus/roflxd](https://github.com/fraxiinus/roflxd) | 16 | 2024-06 |  | Каталог и состояние парсеров .rofl | GPL-3.0 |
| [JohnsonJDDJ/zilean](https://github.com/JohnsonJDDJ/zilean) | 15 | 2022-08 | Python | Краулер таймлайнов, фичи 10/15 мин для ML | MIT |
| [MiscellaneousStuff/tlol-scraper](https://github.com/MiscellaneousStuff/tlol-scraper) | 14 | 2023-02 | C++ | Экстрактор реплеев и генератор датасетов (форк LViewLoL) | форк |
| [yerich/LOLReplayAnalyser](https://github.com/yerich/LOLReplayAnalyser) | 13 | 2019-07 | HTML | Экспериментальный анализ реплеев | GPL-2.0 |
| [fattorib/LeagueMatchScraper](https://github.com/fattorib/LeagueMatchScraper) | 12 | 2021-02 | Python | Скрапер матчей и таймлайнов |  |
| [Allan-Cao/lol-voice-lines](https://github.com/Allan-Cao/lol-voice-lines) | 3 | 2023-05 | Jupyter Notebook | Датасет голосовых реплик | MIT |
| [MiscellaneousStuff/tlol-analysis](https://github.com/MiscellaneousStuff/tlol-analysis) | 2 | 2023-12 | Jupyter Notebook | Анализ датасета 60k игр Ezreal | MIT |
| [MiscellaneousStuff/tlol-explorer](https://github.com/MiscellaneousStuff/tlol-explorer) | 2 | 2022-06 | JavaScript | Визуальный менеджер датасетов TLoL | MIT |
| [mehdbenguiza/lol-dataset-generator](https://github.com/mehdbenguiza/lol-dataset-generator) | 1 | 2026-08 | Python | Генератор датасета 800k+ high-elo матчей |  |
| [rsanandres/jaxstats](https://github.com/rsanandres/jaxstats) | 1 | 2026-02 | Python | Анализатор реплеев + ML-скоринг | MIT |
| [mbonaker/lol-generator](https://github.com/mbonaker/lol-generator) | 0 | 2019-05 | Python | NN генерирует внутриигровые данные из предматчевых |  |

## 7. Предсказание исхода (отобранные)

Самые заметные или методически интересные. Ещё ~250 однотипных учебных проектов перечислены в приложении.

| Репозиторий | ★ | Последний push | Язык | Что делает | Пометки |
|---|--:|---|---|---|---|
| [oracle-devrel/leagueoflegends-optimizer](https://github.com/oracle-devrel/leagueoflegends-optimizer) | 158 | 2026-02 | Jupyter Notebook | Учебный проект Oracle | UPL-1.0 |
| [reneleogp/ML-Prediction-LoL](https://github.com/reneleogp/ML-Prediction-LoL) | 53 | 2022-08 | Jupyter Notebook | Два ML-алгоритма, опыт игрока на чемпионе | архив |
| [minihat/LoL-Match-Prediction](https://github.com/minihat/LoL-Match-Prediction) | 51 | 2020-07 | Python | MLP по истории игроков |  |
| [tongtzeho/LOLPrediction](https://github.com/tongtzeho/LOLPrediction) | 35 | 2021-12 | Python | Предсказание исхода |  |
| [1FarZ1/league-of-legends-win-predector](https://github.com/1FarZ1/league-of-legends-win-predector) | 30 | 2024-06 | Jupyter Notebook | Вероятность победы синих/красных |  |
| [MRittinghouse/ProjektZero-LoL-Model](https://github.com/MRittinghouse/ProjektZero-LoL-Model) | 22 | 2022-02 | Python | Модели про-матчей по Oracle’s Elixir | AGPL-3.0 |
| [mvpeng/TeamCompML](https://github.com/mvpeng/TeamCompML) | 21 | 2015-08 | TeX | CS229: исход по составам |  |
| [Flames1217/LOL-DeepWinPredictor](https://github.com/Flames1217/LOL-DeepWinPredictor) | 17 | 2026-07 | TypeScript | BiLSTM + attention |  |
| [aliciusschroeder/LeagueOfPredictions](https://github.com/aliciusschroeder/LeagueOfPredictions) | 8 | 2024-10 | Python | Предматчевый прогноз | GPL-3.0 |
| [kyfuse/league-win-predictor](https://github.com/kyfuse/league-win-predictor) | 8 | 2023-04 | Python | Live-прогноз | MIT |
| [Komorebi660/LoL-Winner-Predict](https://github.com/Komorebi660/LoL-Winner-Predict) | 7 | 2023-12 | Python | Решение соревнования PaddlePaddle (0.8595) |  |
| [dwang733/lol-winrate](https://github.com/dwang733/lol-winrate) | 5 | 2018-09 | Jupyter Notebook | fast.ai, шанс победы из чемп-селекта |  |
| [RobinHCK/ML4LoL](https://github.com/RobinHCK/ML4LoL) | 3 | 2024-03 | Python | Код к обзору «ML for LoL Match Outcome Prediction: A Review» | MIT |
| [chakrakan/lol-esports-predictions](https://github.com/chakrakan/lol-esports-predictions) | 3 | 2023-11 | Jupyter Notebook | Победитель хакатона Riot × AWS Global Power Rankings 2023 |  |
| [guilherme-deschamps/Predicting-LeagueOfLegends-Games-With-LSTM](https://github.com/guilherme-deschamps/Predicting-LeagueOfLegends-Games-With-LSTM) | 2 | 2021-08 | Python | LSTM |  |
| [JKang025/lol-ai](https://github.com/JKang025/lol-ai) | 2 | 2024-01 | Python | Набор моделей исхода по составам |  |
| [stephenjayakar/LoL-tensorflow](https://github.com/stephenjayakar/LoL-tensorflow) | 1 | 2018-04 | Python | DNN по составам |  |
| [quiet98k/LOL-Win-Prediction](https://github.com/quiet98k/LOL-Win-Prediction) | 1 | 2026-01 | Jupyter Notebook | DNN mid-game с проверкой робастности (PGD) | MIT |
| [thomaszhou01/leagueAI](https://github.com/thomaszhou01/leagueAI) | 1 | 2023-05 | Python | Live-прогноз через Live Client API |  |
| [EricBriscoe/lol-genius](https://github.com/EricBriscoe/lol-genius) | 1 | 2026-08 | Python | XGBoost + SHAP, Electron live | MIT |
| [Hab5/league-machine-learning](https://github.com/Hab5/league-machine-learning) | 1 | 2021-07 | Python | 15 мин, 80k матчей |  |
| [aerrowfar/ML-ESports-Match-Prediction](https://github.com/aerrowfar/ML-ESports-Match-Prediction) | 1 | 2021-04 | Python | Contextual bandits для про-матчей |  |
| [loubbrad/dodge](https://github.com/loubbrad/dodge) | 0 | 2023-01 | Jupyter Notebook | Transformer для high-elo |  |
| [ajbisberg/gcn](https://github.com/ajbisberg/gcn) | 0 | 2021-05 | Jupyter Notebook | GCN-WP: графовая сеть по лигам (arXiv 2207.13191) | форк, MIT |
| [ShahinHussain/lol-temporal-prediction-model](https://github.com/ShahinHussain/lol-temporal-prediction-model) | 0 | 2026-09 | Python | Временная модель исхода |  |
| [pabloChantada/LeaguePredictor](https://github.com/pabloChantada/LeaguePredictor) | 0 | 2026-08 | Python | Live Client API + калиброванная модель |  |
| [L1nom/LOL-Game-Prediction](https://github.com/L1nom/LOL-Game-Prediction) | 0 | 2022-02 | Jupyter Notebook | ANN 8k игр |  |
| [Ra1nForest/LOL_Predictions](https://github.com/Ra1nForest/LOL_Predictions) | 0 | 2026-10 | Python | XGBoost live-борд для lolesports |  |
| [Felichz/LoL-Impact](https://github.com/Felichz/LoL-Impact) | 0 | 2026-09 | Svelte | Логрегрессия на каждую минуту | MIT |

## 8. Драфт и рекомендации

Пики, баны, сборки, рекомендации чемпионов.

| Репозиторий | ★ | Последний push | Язык | Что делает | Пометки |
|---|--:|---|---|---|---|
| [VRichardJP/LoLAnalyzer](https://github.com/VRichardJP/LoLAnalyzer) | 45 | 2019-04 | Python | NN: лучший пик | архив, MIT |
| [lightd22/swainBot](https://github.com/lightd22/swainBot) | 20 | 2022-11 | Python | DQN, драфт как MDP | Apache-2.0 |
| [dojoteef/loltorch](https://github.com/dojoteef/loltorch) | 19 | 2017-03 | Lua | Torch7: оптимальные сборки | MIT |
| [dojeon-ai/DraftRec](https://github.com/dojeon-ai/DraftRec) | 18 | 2023-08 | Jupyter Notebook | Иерархический Transformer, персональные рекомендации пиков (WWW'22, arXiv 2204.12750) |  |
| [JonahFarc/League-Champion-Recommender](https://github.com/JonahFarc/League-Champion-Recommender) | 4 | 2020-05 | Python | Рекомендации чемпионов |  |
| [lipeeeee/league-draft-analyzer](https://github.com/lipeeeee/league-draft-analyzer) | 1 | 2024-11 | Python | LDANet: эмбеддинги + attention |  |
| [Maelian25/lol-draft-prediction](https://github.com/Maelian25/lol-draft-prediction) | 1 | 2025-12 | Python | Бот предсказания драфта |  |
| [ryanthtra/lol-ai-draftpick](https://github.com/ryanthtra/lol-ai-draftpick) | 1 | 2018-05 | R | Движок драфта на R |  |
| [paper0205/GARENRec](https://github.com/paper0205/GARENRec) | 0 | 2025-05 | Python | Graph-attention Transformer для рекомендаций драфта |  |
| [lgestin/loldrafter](https://github.com/lgestin/loldrafter) | 0 | 2023-06 | Python | Transformer, симуляция драфта |  |
| [Sorixon/esports-draft-predictor](https://github.com/Sorixon/esports-draft-predictor) | 0 | 2026-08 | Jupyter Notebook | MLP (PyTorch) по про-драфтам |  |
| [ckjasonlee0722/Intro_of_AI_Final_Project](https://github.com/ckjasonlee0722/Intro_of_AI_Final_Project) | 0 | 2026-09 | Python | LightGBM + Optuna ban/pick | форк |
| [alex-p-d/league-of-legends-recommender-system](https://github.com/alex-p-d/league-of-legends-recommender-system) | 0 | 2025-05 | Python | LightFM-рекомендации |  |
| [AsperaDesu/LoL-AI-Draft-Picker](https://github.com/AsperaDesu/LoL-AI-Draft-Picker) | 0 | 2025-09 | Jupyter Notebook | ML-ассистент пиков | MIT |
| [jacaballero1241/smartpick-showcase](https://github.com/jacaballero1241/smartpick-showcase) | 0 | 2026-10 |  | DL-ассистент драфта (только кейс, без кода) |  |
| [PedroGiu13/lol_gmm_archetype](https://github.com/PedroGiu13/lol_gmm_archetype) | 0 | 2026-07 | Jupyter Notebook | GMM: архетипы драфтов | MIT |

## 9. LLM-коучи, ассистенты, MCP

Новая волна 2025–2026: RAG, агенты, MCP-серверы поверх Riot API и Live Client.

| Репозиторий | ★ | Последний push | Язык | Что делает | Пометки |
|---|--:|---|---|---|---|
| [valkia/aramgg_client](https://github.com/valkia/aramgg_client) | 464 | 2026-09 | TypeScript | ARAM Mayhem компаньон, рекомендации (ML не основной) |  |
| [123Cx330Yrx/riftcoach-agent](https://github.com/123Cx330Yrx/riftcoach-agent) | 105 | 2026-10 | Python | Пост-игровой коуч: детерминированные данные + RAG + LLM-eval | MIT |
| [rumi-chan/league-client-mcp](https://github.com/rumi-chan/league-client-mcp) | 21 | 2026-03 | TypeScript | MCP-мост к League Client | MIT |
| [sorena-ai/LeagueAiCoach](https://github.com/sorena-ai/LeagueAiCoach) | 10 | 2026-10 | Python | Голосовой ИИ-ассистент | Apache-2.0 |
| [MiscellaneousStuff/tlol-llm](https://github.com/MiscellaneousStuff/tlol-llm) | 6 | 2023-04 | Jupyter Notebook | LLM для анализа и игры | MIT |
| [Remus3/Amberstone](https://github.com/Remus3/Amberstone) | 5 | 2026-10 | Python | Локальный real-time коучинг LoL/TFT | Apache-2.0 |
| [fqhd/LoLytics](https://github.com/fqhd/LoLytics) | 4 | 2026-10 | Python | ML-оценка ошибок для разбора игр | MIT |
| [JPClow3/League-AI-Oracle](https://github.com/JPClow3/League-AI-Oracle) | 3 | 2025-11 | TypeScript | ИИ-копилот драфта | архив |
| [MiscellaneousStuff/LeagueLLM](https://github.com/MiscellaneousStuff/LeagueLLM) | 2 | 2023-04 |  | LLM применительно к LoL | MIT |
| [Mattbusel/lolcoach](https://github.com/Mattbusel/lolcoach) | 2 | 2026-09 | Python | Локальный коуч: волны, пути лесника, причины смертей | MIT |
| [jinayoon/analyze-lol-match](https://github.com/jinayoon/analyze-lol-match) | 2 | 2026-05 | Python | Claude Code skill для разбора матча |  |
| [vincntlaw/aram-love-bot](https://github.com/vincntlaw/aram-love-bot) | 2 | 2026-09 | Python | Локальная LLM в чате игры | MIT |
| [Sliv3er/DraftCoach](https://github.com/Sliv3er/DraftCoach) | 2 | 2026-10 | TypeScript | Gemini-оптимизатор сборок | MIT |
| [Giggitycountless/LOL-AI-Intelligence](https://github.com/Giggitycountless/LOL-AI-Intelligence) | 1 | 2026-08 | Rust | Desktop-компаньон с LLM-анализом | AGPL-3.0 |
| [rexlManu/lol-mcp-server](https://github.com/rexlManu/lol-mcp-server) | 1 | 2026-10 | TypeScript | MCP-сервер: анализ игрока через Riot API | MIT |
| [EzzatEsam/leagueOfLegendsChatbot](https://github.com/EzzatEsam/leagueOfLegendsChatbot) | 1 | 2024-09 | Python | RAG LangChain чат-бот | Apache-2.0 |
| [lowlune/RiftSense](https://github.com/lowlune/RiftSense) | 1 | 2026-09 | Python | Live Client API + агент, live-коучинг | MIT |
| [cmiao104/lol-ai-matchmaking-agent](https://github.com/cmiao104/lol-ai-matchmaking-agent) | 1 | 2026-05 | Python | Gemini-агент + модель синергии дуо | MIT |
| [kielbas1729/esport-text-to-sql](https://github.com/kielbas1729/esport-text-to-sql) | 1 | 2026-08 | Python | Text-to-SQL агент по киберспорту |  |
| [Kun-AzureLotus/LOL-AI-Commentary](https://github.com/Kun-AzureLotus/LOL-AI-Commentary) | 1 | 2026-08 | Rust | Real-time ИИ-комментарий |  |
| [jasonbdt/nexus-iq](https://github.com/jasonbdt/nexus-iq) | 1 | 2026-05 | Python | ИИ-коуч макро/микро ошибок |  |
| [eliasxurri/coachmcp](https://github.com/eliasxurri/coachmcp) | 0 | 2026-09 | Python | MCP + AWS аналитика | AGPL-3.0 |
| [ayoubnajjout/a-simple-league-of-legends-rag-chatbot](https://github.com/ayoubnajjout/a-simple-league-of-legends-rag-chatbot) | 0 | 2026-02 | Python | RAG на локальных LLM |  |
| [ezequielmaranda/ward-eye](https://github.com/ezequielmaranda/ward-eye) | 0 | 2026-10 | TypeScript | dbt/DuckDB → LLM-отчёты |  |

## 10. NLP: чат, токсичность, комментарий

Тексты вокруг игры.

| Репозиторий | ★ | Последний push | Язык | Что делает | Пометки |
|---|--:|---|---|---|---|
| [googleforgames/clean-chat](https://github.com/googleforgames/clean-chat) | 55 | 2026-07 | Python | Google: фреймворк против токсичности, данные Tribunal | Apache-2.0 |
| [ArnoZWang/esports-data-to-text](https://github.com/ArnoZWang/esports-data-to-text) | 7 | 2024-07 |  | Генерация комментария из данных матча (NAACL-SRW'24) |  |
| [junj1ehx/CS-lol](https://github.com/junj1ehx/CS-lol) | 2 | 2023-04 | Jupyter Notebook | Комментарии зрителей, привязанные к сценам (arXiv 2301.06876) |  |
| [CChriz/Esports-Event-to-Commentary-Generation-LoL](https://github.com/CChriz/Esports-Event-to-Commentary-Generation-LoL) | 1 | 2026-06 | Python | Нейрогенерация комментария по логам событий |  |
| [irdin-pekaric/esorics26_toxicity](https://github.com/irdin-pekaric/esorics26_toxicity) | 0 | 2026-05 | Python | Размеченный чат и детекторы токсичности (ESORICS'26) | MIT |
| [Yashtech-714/Sentiment-Analysis-of-PvP-Game-Communications](https://github.com/Yashtech-714/Sentiment-Analysis-of-PvP-Game-Communications) | 0 | 2025-10 | Jupyter Notebook | Тональность игрового чата |  |

## 11. TFT и Wild Rift

Смежные игры Riot.

| Репозиторий | ★ | Последний push | Язык | Что делает | Пометки |
|---|--:|---|---|---|---|
| [silverlight6/TFTMuZeroAgent](https://github.com/silverlight6/TFTMuZeroAgent) | 226 | 2026-10 | Python | TFT: симулятор Set 4 + MuZero/PPO | Apache-2.0 |
| [utilForever/AlphaTFT](https://github.com/utilForever/AlphaTFT) | 13 | 2020-04 | CMake | TFT: симулятор на C++ + RL | MIT |
| [MielPopsssssss/TFT_GOAT](https://github.com/MielPopsssssss/TFT_GOAT) | 5 | 2026-06 | Python | TFT Set 17: self-play PPO + нейросуррогат боя |  |
| [hubkrieb/tft-round-prediction](https://github.com/hubkrieb/tft-round-prediction) | 1 | 2026-07 | Python | TFT: исход раунда (XGBoost, CNN, ViT) |  |
| [Lobotuerk/TFTMuZeroAgent](https://github.com/Lobotuerk/TFTMuZeroAgent) | 0 | 2026-09 | Python | TFT: MuZero, PettingZoo | Apache-2.0 |
| [calebyhan/tft_rl](https://github.com/calebyhan/tft_rl) | 0 | 2026-10 | Python | TFT: симулятор и RL-среда | MIT |
| [aristo6253/TFT-Reinforcement-Learning](https://github.com/aristo6253/TFT-Reinforcement-Learning) | 0 | 2023-06 | Python | TFT RL |  |
| [ericlin11354/Teamfight-Tactics-AI-Planner](https://github.com/ericlin11354/Teamfight-Tactics-AI-Planner) | 0 | 2024-08 | Python | TFT: генетический алгоритм составов |  |
| [vinxieezihuang/wild-rift-churn-prediction](https://github.com/vinxieezihuang/wild-rift-churn-prediction) | 0 | 2026-06 | Jupyter Notebook | Wild Rift: отток (XGBoost/LightGBM) |  |

## 12. Боты по экрану без нейросетей или с минимальным CV

Для полноты: автоматизация через пиксели и шаблоны. Нарушают ToS Riot.

| Репозиторий | ★ | Последний push | Язык | Что делает | Пометки |
|---|--:|---|---|---|---|
| [iholston/lol-bot](https://github.com/iholston/lol-bot) | 373 | 2026-03 | Python | Бот прокачки аккаунта, без нейросетей | MIT |
| [Skinz3/League-of-Legends-Bot](https://github.com/Skinz3/League-of-Legends-Bot) | 155 | 2024-03 | C# | Пиксельный бот, обработка изображений (C#) |  |
| [DorskFR/LeaguePyBot](https://github.com/DorskFR/LeaguePyBot) | 62 | 2024-11 | Python | CV-бот | архив, MIT |
| [ARTIRL/LevelingBOT_LOL](https://github.com/ARTIRL/LevelingBOT_LOL) | 20 | 2026-05 |  | Бот прокачки на CV | MIT |
| [SvetoslavDoychinov/yuumi-bot](https://github.com/SvetoslavDoychinov/yuumi-bot) | 4 | 2024-04 | Python | Бот-Юми, фреймворк для ботов | MIT |
| [alexandre-buisset/Bot-League-of-Legends](https://github.com/alexandre-buisset/Bot-League-of-Legends) | 0 | 2022-06 | Python | Автономный бот |  |

## 13. Подборки

| Репозиторий | ★ | Последний push | Язык | Что делает | Пометки |
|---|--:|---|---|---|---|
| [CommunityDragon/awesome-league](https://github.com/CommunityDragon/awesome-league) | 664 | 2026-07 |  | Awesome-список экосистемы LoL (ML мало) | CC0-1.0 |
| [FloPrm/lol_analytics](https://github.com/FloPrm/lol_analytics) | 93 | 2025-03 |  | Лучшая подборка по аналитике LoL (CV по VOD, данные, реплеи) |  |

## 14. Дополнения из веб-поиска

Найдены 2026-10-08 веб-поиском (`site:github.com …`, статьи, подборки); в поиске GitHub не всплыли. Сверены через GraphQL: дубль переименованного `kyfuse/league-win-predictor` и несуществующий `Neetre/LoL` убраны.

| Репозиторий | ★ | Последний push | Категория | Что делает | Пометки |
|---|--:|---|---|---|---|
| [Ali213000/LOLHelper](https://github.com/Ali213000/LOLHelper) | 0 | 2026-10 | LLM-коуч | Коучинг по Riot API; LLM-обвязка есть, но по README не подключена |  |
| [Laksh-Goyal/tft-agent](https://github.com/Laksh-Goyal/tft-agent) | 0 | 2026-09 | TFT, RL | Сравнение стратегий обучения RL-агента в упрощённом TFT (PyTorch и порт на JAX) |  |
| [Junnie13/wildrift-vs-mobilelegends](https://github.com/Junnie13/wildrift-vs-mobilelegends) | 1 | 2024-06 | Wild Rift, NLP | Тональность отзывов Play Store: Naive Bayes, LDA |  |
| [jjzhao05/LoL-Win-Probability](https://github.com/jjzhao05/LoL-Win-Probability) | 0 | 2026-09 | исход | Вероятность победы по событиям; AUC от 0,63 (1–5 мин) до 0,91 (26–30 мин) |  |
| [Gigrise/Live_League_Outcome_Prediction_with_ML](https://github.com/Gigrise/Live_League_Outcome_Prediction_with_ML) | 0 | 2024-08 | исход | Live-прогноз на отметке 15 мин, Random Forest, Dash | MIT |
| [hellman-zhao/lol-neural-net](https://github.com/hellman-zhao/lol-neural-net) | 2 | 2022-05 | исход | NN + скрапер, заявлено ~80% |  |
| [LioussSuperDev/HeimerHelper](https://github.com/LioussSuperDev/HeimerHelper) | 3 | 2023-11 | исход | DL-прогноз шанса победы | MIT |
| [NicolasAlbiges/research_esport_lol](https://github.com/NicolasAlbiges/research_esport_lol) | 0 | 2021-08 | исход | Предматчевый прогноз по стилям и профилям игроков |  |
| [ThalesRod/lol-pro-match-prediction](https://github.com/ThalesRod/lol-pro-match-prediction) | 1 | 2021-09 | исход | Про-матчи, прогноз по данным до момента t |  |
| [jadenoca/LolEsportsData](https://github.com/jadenoca/LolEsportsData) | 0 | 2022-03 | исход | Сравнение моделей по ранней игре (Oracle's Elixir) |  |
| [eoinpinaqui/machine-learning](https://github.com/eoinpinaqui/machine-learning) | 0 | 2022-08 | исход, датасет | 13 803 матча, срезы на 5/10/15/20 мин | архив |
| [cjw612/League_of_Legends_Game_Outcome_Classification](https://github.com/cjw612/League_of_Legends_Game_Outcome_Classification) | 0 | 2025-02 | исход | Срез на 15 мин, четыре модели, >78% |  |
| [lbh199711/game-prediction-ML](https://github.com/lbh199711/game-prediction-ML) | 0 | 2021-08 | исход | Kaggle, срез на 10 мин: LR и однослойная NN | MIT |
| [hhshhd/LeagueOfLegendGamePrediction](https://github.com/hhshhd/LeagueOfLegendGamePrediction) | 2 | 2023-06 | исход | Модель предсказания исхода |  |
| [mataiscat/STAT301-3-final-project](https://github.com/mataiscat/STAT301-3-final-project) | 0 | 2020-08 | исход | Первые 10 минут, Random Forest |  |
| [arilato/ranked_prediction](https://github.com/arilato/ranked_prediction) | 26 | 2018-09 | исход, драфт | Прогноз по данным чемп-селекта (не закончен) |  |
| [Voling/lolpredictor](https://github.com/Voling/lolpredictor) | 2 | 2026-10 | исход | Совместимость и динамика пятёрки | GPL-3.0 |

Ещё статьи без найденного кода: [Do et al., arXiv 2108.02799](https://arxiv.org/abs/2108.02799) (исход по опыту игрока на чемпионе, 75,1%), [Junior & Campelo, arXiv 2309.02449](https://arxiv.org/abs/2309.02449) (real-time прогноз, LightGBM), [Fu et al., EMNLP 2017, arXiv 1707.08559](https://arxiv.org/abs/1707.08559) (хайлайты по чату Twitch), [Vinot & Perez, arXiv 2411.19793](https://arxiv.org/abs/2411.19793) (голосовая коммуникация в LoL), [Action2Score, arXiv 2207.10297](https://arxiv.org/abs/2207.10297), [Silva & Lopes, SBGames: temporal/spatial attribution](https://sol.sbc.org.br/index.php/sbgames/article/view/45416).

## Приложение: остальные учебные и однотипные проекты (270)

В основном это Jupyter-ноутбуки: предсказание исхода по статистике 10–15 минуты (логистическая регрессия, Random Forest, XGBoost, MLP), EDA курсов DSC80 и CS229, разбор Kaggle-датасетов. Звёзд почти нет. Список нужен для полноты, методически они повторяют раздел 7.

[ffaheroes/LeagueOfLegends_Prediction](https://github.com/ffaheroes/LeagueOfLegends_Prediction) ★12 · [vojay-dev/airflow-riot](https://github.com/vojay-dev/airflow-riot) ★12 · [artoria-dev/data-science-lol](https://github.com/artoria-dev/data-science-lol) ★9 · [jimjimliu/LOL_Rank_Game_Predictor](https://github.com/jimjimliu/LOL_Rank_Game_Predictor) ★8 · [SamuelAitamaa/lolesports-predictor](https://github.com/SamuelAitamaa/lolesports-predictor) ★6 · [bobbyroylee/smiteless](https://github.com/bobbyroylee/smiteless) ★5 · [Blizzeq/league-of-legends-data-collector](https://github.com/Blizzeq/league-of-legends-data-collector) ★4 · [nicholaspun/ARAMNet](https://github.com/nicholaspun/ARAMNet) ★4 · [AReyH/league_of_legends_win_prediction](https://github.com/AReyH/league_of_legends_win_prediction) ★3 · [NotGAspegic/lol-dashboard](https://github.com/NotGAspegic/lol-dashboard) ★3 · [protesq/League-of-Legends-win-prediction](https://github.com/protesq/League-of-Legends-win-prediction) ★3 · [Stan370/League-mbti-analytics](https://github.com/Stan370/League-mbti-analytics) ★3 · [teto-ri/League-of-Legends-Match-Analysis-and-Predict](https://github.com/teto-ri/League-of-Legends-Match-Analysis-and-Predict) ★3 · [Voltether/League-of-Legends-Win-Prediction-At-Minute-10-LAN-APEX-SoloQ](https://github.com/Voltether/League-of-Legends-Win-Prediction-At-Minute-10-LAN-APEX-SoloQ) ★3 · [a-huk/lol_esports-predictions](https://github.com/a-huk/lol_esports-predictions) ★2 · [Andrew-Fojas/lol-esports-prediction](https://github.com/Andrew-Fojas/lol-esports-prediction) ★2 · [bonem97/league-of-machine-learning](https://github.com/bonem97/league-of-machine-learning) ★2 · [bushal01/league-ml2](https://github.com/bushal01/league-ml2) ★2 · [fisher60/League_of_Legends_Analysis](https://github.com/fisher60/League_of_Legends_Analysis) ★2 · [juan230500/LeagueReplaysAnalyzer](https://github.com/juan230500/LeagueReplaysAnalyzer) ★2 · [LiamQuinoNeff/LoL-Match-Predictor-ML](https://github.com/LiamQuinoNeff/LoL-Match-Predictor-ML) ★2 · [luisgon-dev/Transcendence](https://github.com/luisgon-dev/Transcendence) ★2 · [oneonlee/Prediction-of-LoL-Tier-with-Summoner-Name](https://github.com/oneonlee/Prediction-of-LoL-Tier-with-Summoner-Name) ★2 · [Plutokekz/League-of-Legends-Predict-Lane](https://github.com/Plutokekz/League-of-Legends-Predict-Lane) ★2 · [RKirlew/League-of-Legends-Champion-Classifier-using-K-Nearest-Neighbors-KNN-](https://github.com/RKirlew/League-of-Legends-Champion-Classifier-using-K-Nearest-Neighbors-KNN-) ★2 · [saikaryekar/lol-match-prediction](https://github.com/saikaryekar/lol-match-prediction) ★2 · [1bryanvalenzuela/lol-simple-analysis](https://github.com/1bryanvalenzuela/lol-simple-analysis) ★1 · [417-devops/lol_player_stats](https://github.com/417-devops/lol_player_stats) ★1 · [A17PRO/Lane-Dominance-Tracker](https://github.com/A17PRO/Lane-Dominance-Tracker) ★1 · [ace-racer/KE5107_LeagueofLegends](https://github.com/ace-racer/KE5107_LeagueofLegends) ★1 · [Aimee-Shin/Kaggle---LOL-Match-Prediction](https://github.com/Aimee-Shin/Kaggle---LOL-Match-Prediction) ★1 · [alavrouk/champions-queue-model](https://github.com/alavrouk/champions-queue-model) ★1 · [andrewinston/leaguePredictor](https://github.com/andrewinston/leaguePredictor) ★1 · [arsyraihan/League-of-Legends-Machine-Learning](https://github.com/arsyraihan/League-of-Legends-Machine-Learning) ★1 · [aymaan-kj/League-of-Legends-Match-Analysis](https://github.com/aymaan-kj/League-of-Legends-Match-Analysis) ★1 · [bananannn/League-of-Legends-Winner-prediction](https://github.com/bananannn/League-of-Legends-Winner-prediction) ★1 · [Bubble0421/rise-to-challenger](https://github.com/Bubble0421/rise-to-challenger) ★1 · [Cintaa1223/Statistics--Multiple-Linear-Regression](https://github.com/Cintaa1223/Statistics--Multiple-Linear-Regression) ★1 · [codeup-germain/Esports-Analysis](https://github.com/codeup-germain/Esports-Analysis) ★1 · [Crrisp/lol-esports-prediction-model](https://github.com/Crrisp/lol-esports-prediction-model) ★1 · [Danmoreng/league-replay-analyzer](https://github.com/Danmoreng/league-replay-analyzer) ★1 · [emily-yu/league-player-classifier](https://github.com/emily-yu/league-player-classifier) ★1 · [ikhsaniorahman/League-of-Legends-Ranked-Match-Prediction](https://github.com/ikhsaniorahman/League-of-Legends-Ranked-Match-Prediction) ★1 · [IvanBeke/TFM](https://github.com/IvanBeke/TFM) ★1 · [janajar/league-of-legends-predictor](https://github.com/janajar/league-of-legends-predictor) ★1 · [jmontag21/LeagueOfLegends-Predictor](https://github.com/jmontag21/LeagueOfLegends-Predictor) ★1 · [Joseph-Ano/League-of-Legends-dataset-visualization](https://github.com/Joseph-Ano/League-of-Legends-dataset-visualization) ★1 · [JP-sDEV/league_of_legends_model](https://github.com/JP-sDEV/league_of_legends_model) ★1 · [lekshanapriya2003/League-of-legends-Esports-analytics](https://github.com/lekshanapriya2003/League-of-legends-Esports-analytics) ★1 · [links1234/lol-esports-win-predictor](https://github.com/links1234/lol-esports-win-predictor) ★1 · [MADLionsEC/lol-analyst-source-with-R](https://github.com/MADLionsEC/lol-analyst-source-with-R) ★1 · [matejkadlec/league-analysis](https://github.com/matejkadlec/league-analysis) ★1 · [MenLine1/Lol_stats](https://github.com/MenLine1/Lol_stats) ★1 · [MoritzPalm/LeagueOfLegendsWinPrediction](https://github.com/MoritzPalm/LeagueOfLegendsWinPrediction) ★1 · [moyu03/lol-win-prediction](https://github.com/moyu03/lol-win-prediction) ★1 · [Nikhil-Mathur13/League_of_Legends-Prediction-](https://github.com/Nikhil-Mathur13/League_of_Legends-Prediction-) ★1 · [ogescalante/lolanalysis](https://github.com/ogescalante/lolanalysis) ★1 · [Phabibi/league_of_legends_ML](https://github.com/Phabibi/league_of_legends_ML) ★1 · [QuennieZeng/League-Of-Legends-Early-Stats-Analysis](https://github.com/QuennieZeng/League-Of-Legends-Early-Stats-Analysis) ★1 · [samika2000/League_of_Legends_Position_Prediction](https://github.com/samika2000/League_of_Legends_Position_Prediction) ★1 · [sirreajohn/League_of_legends_win_prediction](https://github.com/sirreajohn/League_of_legends_win_prediction) ★1 · [suhyeonnnnn/lol-match-prediction](https://github.com/suhyeonnnnn/lol-match-prediction) ★1 · [tim1234ltp/lol-predict](https://github.com/tim1234ltp/lol-predict) ★1 · [tonumayworkspace-creator/league-of-legends-match-predictor](https://github.com/tonumayworkspace-creator/league-of-legends-match-predictor) ★1 · [urattems/lol-analytics](https://github.com/urattems/lol-analytics) ★1 · [XerNNoS/DataViz-LeagueOfLegends](https://github.com/XerNNoS/DataViz-LeagueOfLegends) ★1 · [yi-ye-zhi-qiu/liam.gg](https://github.com/yi-ye-zhi-qiu/liam.gg) ★1 · [Zajicek-Adam/ZPredict](https://github.com/Zajicek-Adam/ZPredict) ★1 · [zoeludena/League-Of-Legends-Soul-Analysis](https://github.com/zoeludena/League-Of-Legends-Soul-Analysis) ★1 · [01zhas/LOL-Tier-Prediction](https://github.com/01zhas/LOL-Tier-Prediction) · [A-Waters/LeagueAI](https://github.com/A-Waters/LeagueAI) · [aaronnmach/League-of-Legends-Prediction-Model](https://github.com/aaronnmach/League-of-Legends-Prediction-Model) · [aastopher/League_of_Legends_XGBoost](https://github.com/aastopher/League_of_Legends_XGBoost) · [AdamWentworth/WinRift](https://github.com/AdamWentworth/WinRift) · [adoria93/League-of-Legends-Predictions](https://github.com/adoria93/League-of-Legends-Predictions) · [Afhrodite/League-of-Legends-match-outcome-prediction](https://github.com/Afhrodite/League-of-Legends-match-outcome-prediction) · [ahg223/DeepLeague_Data_Creator](https://github.com/ahg223/DeepLeague_Data_Creator) · [albertoalagon0bot-collab/lolHighlights](https://github.com/albertoalagon0bot-collab/lolHighlights) · [AlexandreAkyra/lol-ranked-analysis](https://github.com/AlexandreAkyra/lol-ranked-analysis) · [alexandregg1/esports-predictive-analysis](https://github.com/alexandregg1/esports-predictive-analysis) · [AlexAuragan/TAL-AI](https://github.com/AlexAuragan/TAL-AI) · [alexbarkovitch/League-of-Legends-Win-Prediction](https://github.com/alexbarkovitch/League-of-Legends-Win-Prediction) · [AlisaDavle/League-of-Legends-Winner-Prediction-Using-Machine-Learning](https://github.com/AlisaDavle/League-of-Legends-Winner-Prediction-Using-Machine-Learning) · [AljoschaDataAnalyst/Finales-Abschlussprojekt](https://github.com/AljoschaDataAnalyst/Finales-Abschlussprojekt) · [alpha4s/LoLDraftLogReg](https://github.com/alpha4s/LoLDraftLogReg) · [AlvocadoToast/ophelia-prediction-engine](https://github.com/AlvocadoToast/ophelia-prediction-engine) · [AndreaBosi96/League_Of_Legends_Predictor](https://github.com/AndreaBosi96/League_Of_Legends_Predictor) · [andrewshandy/League-Of-Legends-win-prediction](https://github.com/andrewshandy/League-Of-Legends-win-prediction) · [AnitaLiu98/LOL-winning-rate-prediction](https://github.com/AnitaLiu98/LOL-winning-rate-prediction) · [AnkitKolhe149/LeagueOracle](https://github.com/AnkitKolhe149/LeagueOracle) · [anthony6605/Lol-Airflow-Pulse](https://github.com/anthony6605/Lol-Airflow-Pulse) · [AReyH/LoL_Game_Predictor](https://github.com/AReyH/LoL_Game_Predictor) · [ariful59/League-of-Legends-Win-Prediction](https://github.com/ariful59/League-of-Legends-Win-Prediction) · [Arthur1511/lol-match-prediction](https://github.com/Arthur1511/lol-match-prediction) · [austinchinn/prophet](https://github.com/austinchinn/prophet) · [Axllogan/League_of_Legends_Prediction](https://github.com/Axllogan/League_of_Legends_Prediction) · [AyeshaMoghis/league-of-legends-winner-prediction-model](https://github.com/AyeshaMoghis/league-of-legends-winner-prediction-model) · [b17w1z4rd/league-of-legends-win-prediction](https://github.com/b17w1z4rd/league-of-legends-win-prediction) · [baljor3/League-of-Legends-Dataset](https://github.com/baljor3/League-of-Legends-Dataset) · [BBennemann/RiftAnalytics](https://github.com/BBennemann/RiftAnalytics) · [bgalkows/MetaMind](https://github.com/bgalkows/MetaMind) · [Blimabru/league-of-legends-predictor](https://github.com/Blimabru/league-of-legends-predictor) · [borisyue1/League-of-Legends-Predictor](https://github.com/borisyue1/League-of-Legends-Predictor) · [Bowen-n/LOL_Analysis](https://github.com/Bowen-n/LOL_Analysis) · [BreezyInterwebs/Analyzing-League](https://github.com/BreezyInterwebs/Analyzing-League) · [buildwithmehul/pytorch-lol-match-prediction](https://github.com/buildwithmehul/pytorch-lol-match-prediction) · [carralas/league-of-legends-prediction-project](https://github.com/carralas/league-of-legends-prediction-project) · [ChinmayKrishna1/Identifying-High-Ranked-Players](https://github.com/ChinmayKrishna1/Identifying-High-Ranked-Players) · [chriscrawfordAL/League-of-Legends-AI-Powered-Coach](https://github.com/chriscrawfordAL/League-of-Legends-AI-Powered-Coach) · [ChristopherHsu07/league-of-legends-predictor](https://github.com/ChristopherHsu07/league-of-legends-predictor) · [chrisvu1007/lol_match_prediction](https://github.com/chrisvu1007/lol_match_prediction) · [ClareYan0308/lol-match-prediction](https://github.com/ClareYan0308/lol-match-prediction) · [ColinLi33/CSE158-Assignment-2](https://github.com/ColinLi33/CSE158-Assignment-2) · [comus3/league_of_legends_dataset](https://github.com/comus3/league_of_legends_dataset) · [cpak123123/lolproject](https://github.com/cpak123123/lolproject) · [cuppolattos/Drift-Analysis-Patch-League-of-Legends-Win-Prediction-Research](https://github.com/cuppolattos/Drift-Analysis-Patch-League-of-Legends-Win-Prediction-Research) · [CYF-Grace/lol-match-prediction](https://github.com/CYF-Grace/lol-match-prediction) · [dahyun0225/lol-esports-prediction](https://github.com/dahyun0225/lol-esports-prediction) · [Dalso13/lol_match_prediction](https://github.com/Dalso13/lol_match_prediction) · [DanielJu0925/League_of_Legends_prediction](https://github.com/DanielJu0925/League_of_Legends_prediction) · [DanielMilesGH/LoL-VisionScoreAnalysis](https://github.com/DanielMilesGH/LoL-VisionScoreAnalysis) · [dat091234/LOL_Win-rate_Prediction](https://github.com/dat091234/LOL_Win-rate_Prediction) · [davidleonardi418/LeagueofLegends_Matches_Analysis](https://github.com/davidleonardi418/LeagueofLegends_Matches_Analysis) · [davidvalorwork/league-of-legends-predict-win](https://github.com/davidvalorwork/league-of-legends-predict-win) · [Dayvon618/League-of-Legends-Pytorch-Classification-Project](https://github.com/Dayvon618/League-of-Legends-Pytorch-Classification-Project) · [dennis20413/esports-win-predictor-LoL](https://github.com/dennis20413/esports-win-predictor-LoL) · [DiegoFernandoLojanTenesaca/xyra](https://github.com/DiegoFernandoLojanTenesaca/xyra) · [dongkyunk/LOL-Win-Prediction](https://github.com/dongkyunk/LOL-Win-Prediction) · [dudehacker/LoL-Ranked-2020-Win-Prediction](https://github.com/dudehacker/LoL-Ranked-2020-Win-Prediction) · [dudehacker/LoL-Win-Prediction](https://github.com/dudehacker/LoL-Win-Prediction) · [EdanMizrahi/League_of_Legends_Logistic_Regression](https://github.com/EdanMizrahi/League_of_Legends_Logistic_Regression) · [efastovsky/lol-esports-prediction](https://github.com/efastovsky/lol-esports-prediction) · [eidus/2022-Machine-Learning](https://github.com/eidus/2022-Machine-Learning) · [ejeong0915/farzaa-DeepLeague](https://github.com/ejeong0915/farzaa-DeepLeague) · [El-Wally/Leagueoflegends-Analysis](https://github.com/El-Wally/Leagueoflegends-Analysis) · [elisaias-se/lol_match_prediction_model](https://github.com/elisaias-se/lol_match_prediction_model) · [emilprzygonski/League-of-Legends-Classifier](https://github.com/emilprzygonski/League-of-Legends-Classifier) · [EthanCota/LOL-Match-Prediction](https://github.com/EthanCota/LOL-Match-Prediction) · [ethandam3/lol-match-prediction](https://github.com/ethandam3/lol-match-prediction) · [Eueheb/League-of-Legends-Win-Prediction](https://github.com/Eueheb/League-of-Legends-Win-Prediction) · [Fabrig0s/League-of-Legends-Predictor-Pipeline-de-Dados-Machine-Learning](https://github.com/Fabrig0s/League-of-Legends-Predictor-Pipeline-de-Dados-Machine-Learning) · [fatihhozkoc/League-Of-Legends-Win-Prediction](https://github.com/fatihhozkoc/League-Of-Legends-Win-Prediction) · [felipefrm/lol-win-prediction](https://github.com/felipefrm/lol-win-prediction) · [foalem/Kaggle_League_of_Legends_dataset](https://github.com/foalem/Kaggle_League_of_Legends_dataset) · [francourbi/lol_match_predictions_rf](https://github.com/francourbi/lol_match_predictions_rf) · [Friebay/LeagueOfLegendsPredictions](https://github.com/Friebay/LeagueOfLegendsPredictions) · [ftcister/Lol-AI-Predict](https://github.com/ftcister/Lol-AI-Predict) · [Geghi/League-of-Legends-Strategic-Analysis-in-R](https://github.com/Geghi/League-of-Legends-Strategic-Analysis-in-R) · [GiladGecht/League-of-Legends---Ranked-Matches-Analysis](https://github.com/GiladGecht/League-of-Legends---Ranked-Matches-Analysis) · [gooriiie/LOL-League-Of-Legends-Prediction-Not-in-game-](https://github.com/gooriiie/LOL-League-Of-Legends-Prediction-Not-in-game-) · [gregor-ovsenjak/League_of_Legends_predictor](https://github.com/gregor-ovsenjak/League_of_Legends_predictor) · [grniles/League_of_Legends_wins](https://github.com/grniles/League_of_Legends_wins) · [Gute-Git/League-of-Legends-Ranked-Gameplay-Analysis](https://github.com/Gute-Git/League-of-Legends-Ranked-Gameplay-Analysis) · [Hget15/lol-esports-predictor](https://github.com/Hget15/lol-esports-predictor) · [honux/league-replay-decoder](https://github.com/honux/league-replay-decoder) · [HowardHowonYu/lol_win_rate_prediction](https://github.com/HowardHowonYu/lol_win_rate_prediction) · [hrgarber/League_of_Legends_RF_Classifier](https://github.com/hrgarber/League_of_Legends_RF_Classifier) · [hyaoang/LoL-specific-damage-dealt](https://github.com/hyaoang/LoL-specific-damage-dealt) · [ianshimabukuro/clustering-league-of-legends-classes](https://github.com/ianshimabukuro/clustering-league-of-legends-classes) · [IasonKyriakopoulos/lol-melee-vs-ranged](https://github.com/IasonKyriakopoulos/lol-melee-vs-ranged) · [IbrahimBasit5802/DeepLeague-master](https://github.com/IbrahimBasit5802/DeepLeague-master) · [infiniti888/LOL-Data-analysis](https://github.com/infiniti888/LOL-Data-analysis) · [IsmailEB/league-of-legends-machine-learning](https://github.com/IsmailEB/league-of-legends-machine-learning) · [jacky3331/League-of-Legends-Predictive-Analysis](https://github.com/jacky3331/League-of-Legends-Predictive-Analysis) · [janampatel15/PySpark_LOL](https://github.com/janampatel15/PySpark_LOL) · [JaydannG/lolpredict](https://github.com/JaydannG/lolpredict) · [jmai321/league-of-legends-predictor](https://github.com/jmai321/league-of-legends-predictor) · [jnhuang02/League-of-Legends-Win-prediction](https://github.com/jnhuang02/League-of-Legends-Win-prediction) · [joshuanp/League-of-Legends-Machine-Learning-Data-Analysis-Project](https://github.com/joshuanp/League-of-Legends-Machine-Learning-Data-Analysis-Project) · [josipcatic/league-of-legends-win-predictions](https://github.com/josipcatic/league-of-legends-win-predictions) · [jsuyin/league-of-legends-win-prediction](https://github.com/jsuyin/league-of-legends-win-prediction) · [JulianefAlves/league-of-legends-win-prediction](https://github.com/JulianefAlves/league-of-legends-win-prediction) · [khaliloualdchaib/Pro-League-of-Legends-Prediction-Tool](https://github.com/khaliloualdchaib/Pro-League-of-Legends-Prediction-Tool) · [khiemtranngoc/LOL_match_prediction](https://github.com/khiemtranngoc/LOL_match_prediction) · [kmollard/lol-match-prediction](https://github.com/kmollard/lol-match-prediction) · [krosellCage/league-match-analysis](https://github.com/krosellCage/league-match-analysis) · [krzysztof-siedlecki/Leageoflegends_analysis](https://github.com/krzysztof-siedlecki/Leageoflegends_analysis) · [kvbc/lol-alistar-matchups](https://github.com/kvbc/lol-alistar-matchups) · [LaucoTec/lol-match-prediction](https://github.com/LaucoTec/lol-match-prediction) · [laura-wiesner/League-of-Legends-Win-Prediction-Using-Deep-Learning](https://github.com/laura-wiesner/League-of-Legends-Win-Prediction-Using-Deep-Learning) · [LeSingh1/edge-lol](https://github.com/LeSingh1/edge-lol) · [lol-wiki/league-items-spells](https://github.com/lol-wiki/league-items-spells) · [lucasedglima/lol-match-prediction](https://github.com/lucasedglima/lol-match-prediction) · [lucasquintino/LeagueOfLegendsPredictApp](https://github.com/lucasquintino/LeagueOfLegendsPredictApp) · [lucienqchen/lol-match-prediction](https://github.com/lucienqchen/lol-match-prediction) · [luizmathias1/League-of-Legends-AI-Predictor](https://github.com/luizmathias1/League-of-Legends-AI-Predictor) · [M-Borsuk/LeagueOfLegendsWinPrediction](https://github.com/M-Borsuk/LeagueOfLegendsWinPrediction) · [ManasNagesh01/League-of-Legends-Predictor](https://github.com/ManasNagesh01/League-of-Legends-Predictor) · [Marissa-the-Analyst/NA-Dragon-Souls-Summer](https://github.com/Marissa-the-Analyst/NA-Dragon-Souls-Summer) · [Marsian2020/League-of-Legends-Dataset](https://github.com/Marsian2020/League-of-Legends-Dataset) · [mast0w/Cloud-App-League-of-Legends](https://github.com/mast0w/Cloud-App-League-of-Legends) · [MatheusMAssis/League-of-Legends-Analysis](https://github.com/MatheusMAssis/League-of-Legends-Analysis) · [MattBuiles/Prediccion_Roles_League_of_Legends-Machine_Learning](https://github.com/MattBuiles/Prediccion_Roles_League_of_Legends-Machine_Learning) · [Mazzz69/lol-match-prediction](https://github.com/Mazzz69/lol-match-prediction) · [mdu2017/league-of-legends-predictor](https://github.com/mdu2017/league-of-legends-predictor) · [medalha01/GameStatsCleaner](https://github.com/medalha01/GameStatsCleaner) · [MichaelM64/LeagueOfLegends_NeuralNetworksShowcase](https://github.com/MichaelM64/LeagueOfLegends_NeuralNetworksShowcase) · [MichalRosowski/lol-match-prediction-ml](https://github.com/MichalRosowski/lol-match-prediction-ml) · [Micik24/League_of_Legends_predictor](https://github.com/Micik24/League_of_Legends_predictor) · [MirandaDataScience/PCA-Gustavo-Miranda](https://github.com/MirandaDataScience/PCA-Gustavo-Miranda) · [Mixtrue/lol_ai_analysis](https://github.com/Mixtrue/lol_ai_analysis) · [MonsMali/lol-match-prediction](https://github.com/MonsMali/lol-match-prediction) · [mykelbengineer/League-of-Legends-Win-Prediction](https://github.com/mykelbengineer/League-of-Legends-Win-Prediction) · [Myriam-Thameri/League_Of_Legends_Win_Prediction](https://github.com/Myriam-Thameri/League_Of_Legends_Win_Prediction) · [n0ab/Mining-Large-LoL-Datasets-BarnesBradshawTreu](https://github.com/n0ab/Mining-Large-LoL-Datasets-BarnesBradshawTreu) · [NathanSmallcalder/LeagueLiveMatchTracker](https://github.com/NathanSmallcalder/LeagueLiveMatchTracker) · [NeuralBlau/league-knowledge-platform](https://github.com/NeuralBlau/league-knowledge-platform) · [NevineAKF/League-of-Legends-Logistic-Regression-Classifier](https://github.com/NevineAKF/League-of-Legends-Logistic-Regression-Classifier) · [nhdoan0412/LoL-dsc-prj](https://github.com/nhdoan0412/LoL-dsc-prj) · [NJurquet/neuralol](https://github.com/NJurquet/neuralol) · [nnrs/LOL-Esports-Prediction-Tool](https://github.com/nnrs/LOL-Esports-Prediction-Tool) · [NoahDanan/League-of-Legends-Project](https://github.com/NoahDanan/League-of-Legends-Project) · [nonoye206/lol-player-composition-updates](https://github.com/nonoye206/lol-player-composition-updates) · [Ojas6987/LeagueOfLegendsWinPrediction](https://github.com/Ojas6987/LeagueOfLegendsWinPrediction) · [okanolgun/League-of-Legends-Win-Prediction-machineLearning](https://github.com/okanolgun/League-of-Legends-Win-Prediction-machineLearning) · [oscartong0723/lol-rank-analysis](https://github.com/oscartong0723/lol-rank-analysis) · [otsoweckstrom/AI-Win-Prediction-League](https://github.com/otsoweckstrom/AI-Win-Prediction-League) · [pasmud/League-of-Legends-Win-Prediction](https://github.com/pasmud/League-of-Legends-Win-Prediction) · [PatrekurTh/LoL-ML](https://github.com/PatrekurTh/LoL-ML) · [patrick-fitzs/League_Of_Legends_Predictor](https://github.com/patrick-fitzs/League_Of_Legends_Predictor) · [Peter-Shamoun/league-of-legends-win-prediction](https://github.com/Peter-Shamoun/league-of-legends-win-prediction) · [PeterEng3/LoLVisualAnalysis](https://github.com/PeterEng3/LoLVisualAnalysis) · [petrosmp/LeagueRANT](https://github.com/petrosmp/LeagueRANT) · [phamann000/league-of-legends-predictive-dragons](https://github.com/phamann000/league-of-legends-predictive-dragons) · [Pmtal3122/League-of-Legends-Classifier](https://github.com/Pmtal3122/League-of-Legends-Classifier) · [rachelwsakamoto/League-of-Legends-Win-Prediction](https://github.com/rachelwsakamoto/League-of-Legends-Win-Prediction) · [rayyguo/lol-match-prediction](https://github.com/rayyguo/lol-match-prediction) · [refuna/League-of-Legends-Prediction](https://github.com/refuna/League-of-Legends-Prediction) · [renersonreninho-collab/projeto-final-ciencia-de-dados-lol-ebac](https://github.com/renersonreninho-collab/projeto-final-ciencia-de-dados-lol-ebac) · [robarreola77-alt/Game-predictor](https://github.com/robarreola77-alt/Game-predictor) · [robertwyb/League-of-Legends-winrate-prediction](https://github.com/robertwyb/League-of-Legends-winrate-prediction) · [Rr2410/League-of-Legends-Predict](https://github.com/Rr2410/League-of-Legends-Predict) · [rubemalmeida/ml-lol-match-prediction](https://github.com/rubemalmeida/ml-lol-match-prediction) · [ryan3222/League-of-Legends-WinPrediction](https://github.com/ryan3222/League-of-Legends-WinPrediction) · [RyanKirwan/Exploring-LoL-Datasets](https://github.com/RyanKirwan/Exploring-LoL-Datasets) · [Saitamacode/League-of-Legos-](https://github.com/Saitamacode/League-of-Legos-) · [samgeng14/League-of-Legends-Winrate-Prediction](https://github.com/samgeng14/League-of-Legends-Winrate-Prediction) · [SamuelBegley/League-Of-Legends-Machine-Learning](https://github.com/SamuelBegley/League-Of-Legends-Machine-Learning) · [samuraixcode/loladviser](https://github.com/samuraixcode/loladviser) · [SharnSingh/LeagueOfLegends_Diamond_PredictiveAnalysis](https://github.com/SharnSingh/LeagueOfLegends_Diamond_PredictiveAnalysis) · [ShiqiLu77/MachinLearning-LOL_Gaming](https://github.com/ShiqiLu77/MachinLearning-LOL_Gaming) · [sigunthedeer/lol-match-prediction](https://github.com/sigunthedeer/lol-match-prediction) · [SinuheV1/League-of-Legends-Win-Prediction](https://github.com/SinuheV1/League-of-Legends-Win-Prediction) · [slwalley/league_of_legends_predictions](https://github.com/slwalley/league_of_legends_predictions) · [sohailarizk/League-of-legends-predictor](https://github.com/sohailarizk/League-of-legends-predictor) · [Sz-Adrian/AI_Ecamania](https://github.com/Sz-Adrian/AI_Ecamania) · [taliyuh/lol-match-predictor](https://github.com/taliyuh/lol-match-predictor) · [tanishkadharmaraj/LeagueOfLegends-NeuralNetwork](https://github.com/tanishkadharmaraj/LeagueOfLegends-NeuralNetwork) · [tarnowsky/LoL-EDA](https://github.com/tarnowsky/LoL-EDA) · [tdighe2001/LoL-Comparing-Action](https://github.com/tdighe2001/LoL-Comparing-Action) · [TechTusker/League-Of-Legends-Predictor](https://github.com/TechTusker/League-Of-Legends-Predictor) · [teozz17/league-of-legends-dataset](https://github.com/teozz17/league-of-legends-dataset) · [thealper2/League-of-Legends-Win-Prediction](https://github.com/thealper2/League-of-Legends-Win-Prediction) · [thomasGR9/League_of_Legends_winner_prediction](https://github.com/thomasGR9/League_of_Legends_winner_prediction) · [tungdnguyen/league_of_legends_predict](https://github.com/tungdnguyen/league_of_legends_predict) · [tyemalshara/Datenanalyse](https://github.com/tyemalshara/Datenanalyse) · [usffish/lol-match-prediction](https://github.com/usffish/lol-match-prediction) · [Vakhshoori101/League-of-Legends-Predictor](https://github.com/Vakhshoori101/League-of-Legends-Predictor) · [VincentC1120/LoL-Match-Prediction-vlpl](https://github.com/VincentC1120/LoL-Match-Prediction-vlpl) · [vinxieezihuang/lol-player-analytics](https://github.com/vinxieezihuang/lol-player-analytics) · [vishudhshah/league-analysis](https://github.com/vishudhshah/league-analysis) · [vladiseki/League_of_Legends-Predicting_Wins_and_Losses](https://github.com/vladiseki/League_of_Legends-Predicting_Wins_and_Losses) · [vpabba03/League-of-Legends-Classifier](https://github.com/vpabba03/League-of-Legends-Classifier) · [Wang-Shinan/lol-match-prediction](https://github.com/Wang-Shinan/lol-match-prediction) · [William-Zhan-bot/LEC_2024Summer_Winner_Prediction](https://github.com/William-Zhan-bot/LEC_2024Summer_Winner_Prediction) · [wpalswpa/lol-win-prediction](https://github.com/wpalswpa/lol-win-prediction) · [xiaocheng05/LOL-vision-analysis](https://github.com/xiaocheng05/LOL-vision-analysis) · [Yoshi353/League-of-Legends-Win-Prediction](https://github.com/Yoshi353/League-of-Legends-Win-Prediction) · [yungtristxn/lol-bot-leveling-free](https://github.com/yungtristxn/lol-bot-leveling-free) · [Zidmanito/League-of-Legends-Winner-Prediction](https://github.com/Zidmanito/League-of-Legends-Winner-Prediction)
