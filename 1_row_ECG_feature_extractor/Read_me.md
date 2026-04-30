ФАЙЛ МОЖЕТ БЫТЬ НЕМНОГО УСТАРЕВШИМ!



Проект предназначен для пакетной обработки записей ЭКГ, ручной/автоматической валидации R-пиков и морфологии, а также извлечения признаков ВСР и морфологических признаков ЭКГ.



Примерная структура:



Feature\_extractor/

│

├── run\_extract.py

├── protocol.yaml

├── config.py

│

├── protocol.py

├── pipeline.py

├── review.py

├── features.py

│

├── records/

│   ├── 01.04.2026/

│   │   ├── before/

│   │   │   ├── sit.csv

│   │   │   ├── stand.csv

│   │   │   ├── squat.csv

│   │   │   ├── breath\_in.csv

│   │   │   ├── breath\_out.csv

│   │   │   └── long.csv

│   │   │

│   │   ├── after/

│   │   │   └── ...

│   │   │

│   │   ├── before2/

│   │   │   └── ...

│   │   │

│   │   └── after2/

│   │       └── ...

│   │

│   ├── \_reviews/

│   │   └── ...

│   │

│   └── features\_protocol.csv

Основные файлы

run\_extract.py



Точка запуска обработки. Обычно в нём задаются пути:



ROOT\_DIR = Path(r"...\\records")

PROTOCOL\_CONFIG = ROOT\_DIR / "protocol.yaml"

OUTPUT\_CSV = ROOT\_DIR / "features\_protocol.csv"

REVIEW\_DIR = ROOT\_DIR / "\_reviews"



и вызывается:



process\_all\_records\_with\_protocol(

&#x20;   root\_dir=ROOT\_DIR,

&#x20;   protocol\_config\_path=PROTOCOL\_CONFIG,

&#x20;   output\_csv=OUTPUT\_CSV,

&#x20;   review\_dir=REVIEW\_DIR,

&#x20;   interactive\_mode=None,

)



Если interactive\_mode=None, режим review берётся из protocol.yaml.



protocol.py



Модуль работы с протоколом исследования.



Отвечает за:



загрузку protocol.yaml;

проверку структуры конфига;

подстановку значений по умолчанию;

построение runtime-конфига для конкретной записи;

поиск файла записи;

оценку частоты дискретизации;

разбиение записи на сегменты.



Ключевые функции:



load\_protocol\_config()

validate\_protocol\_config()

normalize\_protocol\_config()

build\_record\_runtime\_config()

resolve\_existing\_file()

estimate\_sampling\_rate\_from\_runtime()

resolve\_segments()

pipeline.py



Основной модуль пакетной обработки.



Отвечает за:



обход папок по датам;

обход фаз и типов записей из protocol.yaml;

запуск review;

применение signal-QC и morphology-QC;

извлечение признаков;

формирование строк итоговой таблицы;

сохранение features\_protocol.csv.



Поддерживаемые режимы обработки:



static       — запись обрабатывается целиком или по заданным сегментам

windowed     — запись режется на равные окна

breath\_hold  — запись с кастомными сегментами, например full/start/end

review.py



Модуль ручной и автоматической проверки.



Отвечает за:



отображение сигнала с R-пиками;

ручное добавление, удаление и коррекцию R-пиков;

автоматический gate качества для ВСР;

построение median beat;

отображение морфологии;

gate качества морфологии.



Основная идея:



HRV-validity и Morphology-validity оцениваются отдельно.



Запись может быть пригодна для ВСР, но непригодна для морфологии, и наоборот.



features.py



Модуль извлечения признаков.



Содержит:



фильтрацию ЭКГ;

детекцию R-пиков;

расчёт RR-интервалов;

признаки ВСР во временной области;

частотные признаки ВСР;

нелинейные признаки;

построение median beat;

морфологические признаки QRS/P/T.

Инструкция по написанию protocol.yaml



Файл protocol.yaml описывает структуру исследования: какие записи есть, в каких фазах они выполняются, как их обрабатывать, как проверять качество и как нарезать на сегменты.



Минимальная логика такая:



protocol.yaml -> protocol.py -> pipeline.py -> features\_protocol.csv

1\. Общая информация о протоколе

protocol:

&#x20; name: "fatigue\_monitoring\_v1"

&#x20; version: "1.0"

&#x20; description: "Protocol for individual fatigue assessment using single-lead ECG"



Этот блок служит для описания протокола. На обработку напрямую почти не влияет.



2\. Настройки испытуемого и записи

subject\_defaults:

&#x20; sampling\_rate\_mode: "fixed"

&#x20; sampling\_rate\_hz: 234.45

&#x20; adc:

&#x20;   min: 0

&#x20;   max: 675

sampling\_rate\_mode



Поддерживаемые варианты:



sampling\_rate\_mode: "fixed"



Используется фиксированная частота дискретизации из sampling\_rate\_hz.



sampling\_rate\_mode: "from\_expected\_duration"



Частота рассчитывается как:



fs = число\_отсчётов / expected\_duration\_sec



Для текущего проекта чаще всего используется "fixed".



3\. Структура хранения файлов

storage:

&#x20; date\_format: "%d.%m.%Y"

&#x20; phases\_as\_directories: true

&#x20; default\_signal\_extension: ".csv"



Если:



phases\_as\_directories: true



то ожидается структура:



records/

&#x20; 01.04.2026/

&#x20;   before/

&#x20;     sit.csv

&#x20;   after/

&#x20;     sit.csv



То есть фазы — это отдельные папки внутри папки дня.



4\. Значения по умолчанию

defaults:

&#x20; processing\_profile: "static\_default"

&#x20; segmentation\_profile: "full\_record"

&#x20; feature\_groups: \["hrv\_time", "hrv\_extended", "morphology"]

&#x20; signal\_qc\_profile: "default\_signal"

&#x20; morphology\_qc\_profile: "default\_morphology"

&#x20; review\_mode: "bad\_only"

&#x20; review\_scope: "record"



Эти значения применяются ко всем записям, если в конкретной записи не указано иное.



review\_mode



Поддерживаемые варианты:



review\_mode: "auto"



Ручной review не запускается.



review\_mode: "manual"



Все подходящие записи проходят ручной review.



review\_mode: "bad\_only"



Ручной review запускается только если автоматический gate качества не прошёл.



review\_scope

review\_scope: "record"



Сначала выполняется review всей записи, потом она режется на сегменты.



review\_scope: "segment"



Сначала запись режется на сегменты, затем каждый сегмент проходит review отдельно.



Для задержек дыхания обычно лучше:



review\_scope: "segment"

5\. Фазы исследования

phases:

&#x20; - id: "before"

&#x20;   label: "Before training"



&#x20; - id: "after"

&#x20;   label: "After training"



&#x20; - id: "before2"

&#x20;   label: "Before second training"



&#x20; - id: "after2"

&#x20;   label: "After second training"



id должен совпадать с названием папки фазы, если phases\_as\_directories: true.



Например:



01.04.2026/

&#x20; before/

&#x20; after/

&#x20; before2/

&#x20; after2/



Если в записи явно указан список фаз:



phases: \["before", "after"]



то она будет обрабатываться только в этих фазах.



Если поле phases у записи не указано, запись применяется ко всем фазам из верхнего блока phases.



6\. Профили качества для ВСР

signal\_qc\_profiles:

&#x20; default\_signal:

&#x20;   clipping\_ratio\_max: 0.02

&#x20;   rr\_phys\_bad\_ratio\_max: 0.10

&#x20;   suspicious\_ratio\_max: 0.15

&#x20;   min\_rpeaks: 3

&#x20;   edge\_guard\_sec: 0.5



&#x20;   rpeak\_amp\_ratio\_low: 0.40

&#x20;   rpeak\_amp\_ratio\_high: 2.50

&#x20;   rpeak\_amp\_bad\_ratio\_max: 0.15

&#x20;   rpeak\_amp\_median\_min: null



Этот профиль отвечает за пригодность записи для анализа ВСР.



Проверяются:



клиппинг;

число R-пиков;

физиологичность RR-интервалов;

локальные выбросы RR;

согласованность амплитуды R-пиков.



Странная форма комплекса сама по себе не является причиной исключения записи из ВСР, если R-пики детектируются корректно.



7\. Профили качества морфологии

morphology\_qc\_profiles:

&#x20; default\_morphology:

&#x20;   min\_beats\_extracted: 3

&#x20;   min\_beats\_good: 3

&#x20;   good\_beats\_ratio\_min: 0.40

&#x20;   corr\_median\_min: 0.70

&#x20;   require\_qrs: true

&#x20;   qrs\_duration\_min\_ms: 20

&#x20;   qrs\_duration\_max\_ms: 180

&#x20;   require\_p: false

&#x20;   require\_t: false



Этот профиль отвечает за пригодность записи для морфологического анализа.



Проверяются:



количество извлечённых комплексов;

количество хороших комплексов;

корреляция комплексов с медианным шаблоном;

наличие QRS;

ожидаемый диапазон длительности QRS;

при необходимости наличие P/T.



Единичные артефактные комплексы не обязательно портят морфологию, потому что анализируется медианный комплекс.



8\. Профили сегментации

Полная запись

segmentation\_profiles:

&#x20; full\_record:

&#x20;   mode: "full"



Вся запись обрабатывается как один сегмент.



Равные окна

&#x20; fixed\_windows\_5s:

&#x20;   mode: "fixed\_windows"

&#x20;   window\_sec: 5.0

&#x20;   step\_sec: 5.0

&#x20;   min\_window\_sec: 2.0

&#x20;   drop\_last\_short\_window: true



Используется, например, для записи после приседаний.



Именованные сегменты

&#x20; breath\_hold\_default:

&#x20;   mode: "named\_subsegments"

&#x20;   segments:

&#x20;     - label: "full"

&#x20;       start\_sec: 0.0

&#x20;       end\_sec: "full"



&#x20;     - label: "start"

&#x20;       start\_sec: 0.0

&#x20;       end\_sec: 5.0



&#x20;     - label: "end"

&#x20;       relative\_to: "end"

&#x20;       duration\_sec: 5.0



Такой профиль создаёт три сегмента:



full  — вся запись

start — первые 5 секунд

end   — последние 5 секунд



Если для записи стоит:



review\_scope: "segment"



то каждый из этих сегментов будет отдельно выводиться на review.



9\. Профили обработки сигнала

processing\_profiles:

&#x20; static\_default:

&#x20;   detector\_method: "neurokit"



&#x20;   hrv\_filter:

&#x20;     lowcut: 0.5

&#x20;     highcut: 30

&#x20;     order: 2

&#x20;     trim\_seconds: 0.5



&#x20;   morphology\_filter:

&#x20;     lowcut: 0.3

&#x20;     highcut: 35

&#x20;     order: 2

&#x20;     trim\_seconds: 0.0



&#x20;   morphology:

&#x20;     pre\_ms: 200

&#x20;     post\_ms: 400

&#x20;     corr\_threshold: 0.7

&#x20;     smooth\_ms: 15

hrv\_filter



Фильтр для детекции R-пиков и анализа ВСР.



morphology\_filter



Фильтр для построения комплексов и медианного удара.



morphology



Настройки нарезки ударов вокруг R-пика:



pre\_ms: 200

post\_ms: 400



означает, что вокруг каждого R-пика берётся окно:



200 мс до R-пика

400 мс после R-пика

10\. Описание записей



Пример записи в покое:



recordings:

&#x20; - id: "sit"

&#x20;   file\_names: \["sit.csv"]

&#x20;   expected\_duration\_sec: 60

&#x20;   processing\_mode: "static"

&#x20;   processing\_profile: "static\_default"

&#x20;   segmentation\_profile: "full\_record"

&#x20;   signal\_qc\_profile: "strict\_signal"

&#x20;   morphology\_qc\_profile: "strict\_morphology"

&#x20;   review\_mode: "bad\_only"

&#x20;   review\_scope: "record"

&#x20;   aggregation\_role: "rest\_reference"



Пример записи после приседаний:



&#x20; - id: "squat"

&#x20;   file\_names: \["squat.csv"]

&#x20;   expected\_duration\_sec: 60

&#x20;   processing\_mode: "windowed"

&#x20;   processing\_profile: "windowed\_default"

&#x20;   segmentation\_profile: "fixed\_windows\_5s"

&#x20;   feature\_groups: \["hrv\_time", "hrv\_extended"]

&#x20;   signal\_qc\_profile: "default\_signal"

&#x20;   morphology\_qc\_profile: null

&#x20;   review\_mode: "bad\_only"

&#x20;   review\_scope: "segment"

&#x20;   aggregation\_role: "functional\_load\_test"



Для squat морфология отключена:



morphology\_qc\_profile: null

feature\_groups: \["hrv\_time", "hrv\_extended"]



Пример задержки дыхания:



&#x20; - id: "breath\_in"

&#x20;   file\_names: \["breath\_in.csv"]

&#x20;   expected\_duration\_sec: null

&#x20;   processing\_mode: "breath\_hold"

&#x20;   processing\_profile: "breath\_hold\_default"

&#x20;   segmentation\_profile: "breath\_hold\_default"

&#x20;   signal\_qc\_profile: "strict\_signal"

&#x20;   morphology\_qc\_profile: "strict\_morphology"

&#x20;   review\_mode: "bad\_only"

&#x20;   review\_scope: "segment"

&#x20;   aggregation\_role: "respiratory\_test"



Здесь важно:



review\_scope: "segment"



Тогда на review отдельно попадут:



breath\_in / full

breath\_in / start

breath\_in / end

11\. Группы признаков

feature\_groups: \["hrv\_time", "hrv\_extended", "morphology"]



Поддерживаемые группы:



hrv\_time      — временные признаки ВСР

hrv\_extended  — частотные и нелинейные признаки ВСР

morphology    — морфологические признаки ЭКГ



Если морфология для записи не нужна:



feature\_groups: \["hrv\_time", "hrv\_extended"]

morphology\_qc\_profile: null

12\. Частые ошибки при написании протокола

Новая фаза добавлена, но не обрабатывается



Причина: в recordings у записи явно указан старый список фаз:



phases: \["before", "after"]



Решение: либо добавить новые фазы:



phases: \["before", "after", "before2", "after2"]



либо убрать phases из записи, чтобы она применялась ко всем фазам.



Файл не найден



Проверь:



storage:

&#x20; phases\_as\_directories: true



Тогда структура должна быть:



records/

&#x20; 01.04.2026/

&#x20;   before/

&#x20;     sit.csv



А не:



records/

&#x20; 01.04.2026/

&#x20;   before\_sit.csv

Review старого формата ломает обработку



Если структура review изменилась, старые файлы в \_reviews могут быть несовместимы.



Самый простой способ:



удалить папку \_reviews



После этого review будет создан заново.



Минимальный пример protocol.yaml

protocol:

&#x20; name: "fatigue\_monitoring\_v1"

&#x20; version: "1.0"



subject\_defaults:

&#x20; sampling\_rate\_mode: "fixed"

&#x20; sampling\_rate\_hz: 234.45

&#x20; adc:

&#x20;   min: 0

&#x20;   max: 675



storage:

&#x20; date\_format: "%d.%m.%Y"

&#x20; phases\_as\_directories: true

&#x20; default\_signal\_extension: ".csv"



defaults:

&#x20; processing\_profile: "static\_default"

&#x20; segmentation\_profile: "full\_record"

&#x20; feature\_groups: \["hrv\_time", "hrv\_extended", "morphology"]

&#x20; signal\_qc\_profile: "default\_signal"

&#x20; morphology\_qc\_profile: "default\_morphology"

&#x20; review\_mode: "bad\_only"

&#x20; review\_scope: "record"



phases:

&#x20; - id: "before"

&#x20;   label: "Before training"

&#x20; - id: "after"

&#x20;   label: "After training"



signal\_qc\_profiles:

&#x20; default\_signal:

&#x20;   clipping\_ratio\_max: 0.02

&#x20;   rr\_phys\_bad\_ratio\_max: 0.10

&#x20;   suspicious\_ratio\_max: 0.15

&#x20;   min\_rpeaks: 3

&#x20;   edge\_guard\_sec: 0.5

&#x20;   rpeak\_amp\_ratio\_low: 0.40

&#x20;   rpeak\_amp\_ratio\_high: 2.50

&#x20;   rpeak\_amp\_bad\_ratio\_max: 0.15

&#x20;   rpeak\_amp\_median\_min: null



morphology\_qc\_profiles:

&#x20; default\_morphology:

&#x20;   min\_beats\_extracted: 3

&#x20;   min\_beats\_good: 3

&#x20;   good\_beats\_ratio\_min: 0.40

&#x20;   corr\_median\_min: 0.70

&#x20;   require\_qrs: true

&#x20;   qrs\_duration\_min\_ms: 20

&#x20;   qrs\_duration\_max\_ms: 180

&#x20;   require\_p: false

&#x20;   require\_t: false



segmentation\_profiles:

&#x20; full\_record:

&#x20;   mode: "full"



&#x20; fixed\_windows\_5s:

&#x20;   mode: "fixed\_windows"

&#x20;   window\_sec: 5.0

&#x20;   step\_sec: 5.0

&#x20;   min\_window\_sec: 2.0

&#x20;   drop\_last\_short\_window: true



&#x20; breath\_hold\_default:

&#x20;   mode: "named\_subsegments"

&#x20;   segments:

&#x20;     - label: "full"

&#x20;       start\_sec: 0.0

&#x20;       end\_sec: "full"

&#x20;     - label: "start"

&#x20;       start\_sec: 0.0

&#x20;       end\_sec: 5.0

&#x20;     - label: "end"

&#x20;       relative\_to: "end"

&#x20;       duration\_sec: 5.0



processing\_profiles:

&#x20; static\_default:

&#x20;   detector\_method: "neurokit"

&#x20;   hrv\_filter:

&#x20;     lowcut: 0.5

&#x20;     highcut: 30

&#x20;     order: 2

&#x20;     trim\_seconds: 0.5

&#x20;   morphology\_filter:

&#x20;     lowcut: 0.3

&#x20;     highcut: 35

&#x20;     order: 2

&#x20;     trim\_seconds: 0.0

&#x20;   morphology:

&#x20;     pre\_ms: 200

&#x20;     post\_ms: 400

&#x20;     corr\_threshold: 0.7

&#x20;     smooth\_ms: 15



&#x20; windowed\_default:

&#x20;   detector\_method: "neurokit"

&#x20;   hrv\_filter:

&#x20;     lowcut: 0.5

&#x20;     highcut: 30

&#x20;     order: 2

&#x20;     trim\_seconds: 0.0

&#x20;   morphology\_filter:

&#x20;     lowcut: 0.3

&#x20;     highcut: 35

&#x20;     order: 2

&#x20;     trim\_seconds: 0.0

&#x20;   morphology:

&#x20;     pre\_ms: 200

&#x20;     post\_ms: 400

&#x20;     corr\_threshold: 0.8

&#x20;     smooth\_ms: 15



recordings:

&#x20; - id: "sit"

&#x20;   file\_names: \["sit.csv"]

&#x20;   expected\_duration\_sec: 60

&#x20;   processing\_mode: "static"

&#x20;   processing\_profile: "static\_default"

&#x20;   segmentation\_profile: "full\_record"

&#x20;   aggregation\_role: "rest\_reference"



&#x20; - id: "squat"

&#x20;   file\_names: \["squat.csv"]

&#x20;   expected\_duration\_sec: 60

&#x20;   processing\_mode: "windowed"

&#x20;   processing\_profile: "windowed\_default"

&#x20;   segmentation\_profile: "fixed\_windows\_5s"

&#x20;   feature\_groups: \["hrv\_time", "hrv\_extended"]

&#x20;   morphology\_qc\_profile: null

&#x20;   review\_scope: "segment"

&#x20;   aggregation\_role: "functional\_load\_test"



&#x20; - id: "breath\_in"

&#x20;   file\_names: \["breath\_in.csv"]

&#x20;   expected\_duration\_sec: null

&#x20;   processing\_mode: "breath\_hold"

&#x20;   processing\_profile: "static\_default"

&#x20;   segmentation\_profile: "breath\_hold\_default"

&#x20;   review\_scope: "segment"

&#x20;   aggregation\_role: "respiratory\_test"

