# Surzhyk ASR  
Це система автоматичного розпізнавання мовлення для суржику на базі fine-tuned Whisper. Проєкт охоплює повний ML-цикл: збір і розмітку аудіодатасету, версіонування даних (DVC + MinIO), тренування моделі та оцінку якості за метриками WER/CER. 

## Dataset & Data Versioning
Датасет суржику для fine-tuning Whisper. 

## Структура проєкту

```
surzhyk-asr/
├── data/
│   ├── raw/          # оригінальне аудіо
│   ├── processed/    # нарізані кліпи ~10 c, wav 16kHz mono (форматовано для Whisper)
│   └── labeled/      # розмітка з Label Studio
├── scripts/
│   └── preprocess.py # нарізка raw > processed (ffmpeg)
├── training/         
├── inference/        
├── monitoring/       
├── models/           
├── docker-compose.yml
└── README.md
```

**Лінія даних:**
`data/raw` > `scripts/preprocess.py` > `data/processed` > розмітка в
Label Studio > `data/labeled` > версія (git tag + DVC push у MinIO).

---
## Data Management
## 1. Який інструмент використано для розмітки

**Label Studio** (запускається з `docker-compose.yml`), шаблон
**Automatic Speech Recognition**: аудіоплеєр + текстове поле транскрипції.

Конфігурація інтерфейсу розмітки:

```xml
<View>
  <Audio name="audio" value="$audio"/>
  <TextArea name="transcription" toName="audio"
            rows="4" placeholder="Транскрипція суржиком..."
            editable="true" maxSubmissions="1"/>
</View>
```

**Правила анотації:**

- **Пишемо як чуємо.** Суржикові форми зберігаються дослівно
  («шо ти дєлаєш», не «що ти робиш»).
- Мінімальна пунктуація, малі літери.
- Числа написані словами («двадцять п'ять», не «25»).
- Нерозбірливі фрагменти позначкені `[нерозбірливо]`.
- Кліпи, що не містять мовлення (музика, тиша), пропускаємо.

## 2. Як запустити розмітку

**Передумови:** Docker Desktop, git, python 3.10+, ffmpeg.

**Крок 1 — клонувати та підняти середовище:**

```bash
git clone git@github.com:ol-nikolaieva/surzhyk-asr.git
cd surzhyk-asr
docker compose up -d
```

**Крок 2 — доступи до сервісів:**

| Сервіс | URL | Доступ |
|---|---|---|
| Label Studio (розмітка) | http://localhost:8080 | створити акаунт при першому вході |
| MinIO console (сховище) | http://localhost:9001 | `minioadmin` / `minioadmin` |
| MinIO S3 API | http://localhost:9000 | використовується DVC, не браузером |

При першому запуску в MinIO console створіть bucket **`surzhyk-dataset`**
(Buckets → Create Bucket).

**Крок 3 — підготувати кліпи з власного сирого аудіо:**

```bash
python3 -m venv .venv && source .venv/bin/activate
# покласти аудіо (mp3/wav/m4a/ogg/mp4/webm) у data/raw/
python scripts/preprocess.py
# нарізані кліпи з'являться у data/processed/
```

**Крок 4 — відкрити розмітку в Label Studio:**

1. http://localhost:8080 > **Create Project** > назва проєкту.
2. Labeling Setup > Audio/Speech Processing > **Automatic Speech
   Recognition** (або вставити XML-конфіг вище через вкладку Code) > Save.
3. **Import** > завантаження wav-файлів з `data/processed/`
   (обмеження ~100 файлів за один імпорт, але можна заливати кількома підходами у той самий проєкт).
4. **Label All Tasks** > прослухати > надрукувати транскрипцію > Submit.
5. Після розмітки: **Export > JSON** > зберегти як `data/labeled/annotations.json`.

## 3. Як працює версіонування датасету

**Інструменти: DVC + MinIO**

Принцип поділу обов'язків: git погано працює з гігабайтами, тому:

- **git** зберігає маленькі текстові `.dvc`-файли (md5-хеш + розмір + шлях);
- **DVC** кладе/забирає реальні файли у bucket `surzhyk-dataset` в MinIO;
- **git tag** (`v1`, `v2`, …) фіксує іменовану версію датасету.

**Створити нову версію датасету:**

```bash
dvc add data/raw data/processed data/labeled   
git add data/*.dvc
git commit -m "data: v2 — +15 labeled clips"
git tag v2
dvc push                                       
git push && git push --tags                    
```

**Перемкнутися між версіями:**

```bash
git checkout v1 && dvc checkout   # весь датасет у стані v1
git checkout main && dvc checkout # назад до актуальної версії
```

**Наявні версії:** `v1` — перша розмічена вибірка;
`v2` — розширена розмітка (додаткові кліпи).

> **Примітка про відтворюваність:** DVC remote у цьому проєкті — локальний
> MinIO (`localhost:9000`), тому `dvc pull` на вашій машині даних не
> отримає. Для відтворення сетапу: `docker compose up -d` треба створити bucket, додати власні дані (або скрпіювати нарізані - дивіться в Google Drive) > `dvc push`. Міграція на хмарний S3 `dvc remote modify`.

## 4. Для яких задач ці дані використовуються надалі
Тренування моделі на парах «аудіо ↔ транскрипція» з `annotations.json` для розпізнавання суржика. 
Створення сервісу транскрипції поверх натренованої моделі.

---

## Джерело даних

Аудіокнига «Шахмати для дебілів», взята з YouTube. Використовується виключно в навчальних цілях.

## Прийняті рішення

- **pydub несумісний з Python 3.13+** (зі stdlib видалено модуль
  `audioop`). Скрипт нарізки переписано на чистий ffmpeg.
- Drag&drop-імпорт Label Studio обмежений ~100 файлами за раз —
  задокументовано в інструкції запуску, партії імпортуються послідовно.

## Training + Tracking

Fine-tuning Whisper із повним трекінгом експериментів через **MLflow**.

### Запуск тренування

Підняти інфраструктуру (MLflow + MinIO + Label Studio) і активувати venv:

```bash
docker compose up -d
source .venv/bin/activate
```

Запустити тренування:

```bash
python training/train.py --model openai/whisper-small --epochs 3 --lr 1e-5 --batch 8 --dataset-version v2
```

Прапорці:

- `--model` — базова модель Whisper (`openai/whisper-tiny` | `openai/whisper-small`);
- `--epochs` — кількість епох;
- `--lr` — learning rate;
- `--batch` — розмір батча;
- `--dataset-version` — версія датасету (DVC tag), логується в MLflow для lineage.

Скрипт `train.py` логує в MLflow параметри й метрики та реєструє натреновану модель у Model Registry під іменем `surzhyk-whisper`.

### Результати в MLflow

MLflow UI: **http://localhost:5001** > експеримент **surzhyk-whisper**. Там доступні параметри кожного запуску, метрики (WER/CER, loss), порівняння запусків і Model Registry з версіями моделі.


### Порівняння запусків

Дві ітерації: v2 (один диктор, 141 кліп) > v3 (два диктори, 258 кліпів).

| Запуск | Модель | Дані | lr | WER | CER |
|---|---|---|---|---|---|
| 1 | whisper-tiny | v2 | 1e-5 | 0.638 | 0.197 |
| 2 | whisper-tiny | v2 | 5e-6 | 0.684 | 0.211 |
| 3 | whisper-small | v2 | 1e-5 | 0.383 | 0.117 |
| 4 | whisper-tiny | v3 | 1e-5 | 0.719 | 0.283 |
| 5 | **whisper-small** | **v3** | 1e-5 | **0.440** | **0.205** |

**Висновки:**

- **Розмір моделі** найбільше впливає на результат. На обох версіях даних small обганяє tiny майже вдвічі по WER. Зменшення lr на tiny результат не покращило.
- **WER на v3 вищий, ніж на v2** - додавання другого диктора зробило задачу складнішою, модель показує чеснішу метрику на різноманітніших даних.
- Кожен запуск логує `dataset_version`, тож видно lineage: яка модель на якій версії даних тренувалась.

### Champion-модель

Найкраща версія (whisper-small, WER 0.383) позначена в Model Registry alias'ом **`champion`**. 

### Обмеження

Eval-сет — 22 кліпи з одного джерела. Метрика надійна для порівняння моделей між собою, але не є загальною точністю розпізнавання суржику. Наступний крок — розширити датасет кількома дикторами/джерелами.

---

## Inference

FastAPI-сервіс для транскрипції суржику. Модель завантажується з MLflow Model Registry за alias'ом `champion` (не захардкоджена в репозиторії).

### Як підняти сервіс

Переконатись, що інфраструктура працює:

```bash
docker compose up -d
source .venv/bin/activate
```

Запустити сервіс:

```bash
uvicorn inference.app:app --host 0.0.0.0 --port 8000
```

При старті сервіс завантажує champion-модель з MLflow Registry. Це займає кілька секунд — модель вантажиться один раз і живе в пам'яті.

### Як надіслати тестовий запит

```bash
curl -X POST http://localhost:8000/transcribe \
  -F "file=@data/processed/Andrii Nikolaiev - record_000.wav"
```

Відповідь:

```json
{"text": "транскрипція суржиком", "duration_sec": 10.0}
```

Ендпоінт приймає аудіо будь-якого формату (wav, mp3, m4a, ogg) — на вході файл автоматично конвертується у wav 16kHz mono через ffmpeg (та сама нормалізація, що й при підготовці тренувальних даних, щоб уникнути training/serving skew).

### Перевірка здоров'я

```bash
curl http://localhost:8000/health
```

```json
{"status": "ok", "model": "surzhyk-whisper@champion"}
```

### Архітектура інференсу

```
Клієнт (curl/браузер)
    ↓ POST /transcribe + аудіофайл
FastAPI (inference/app.py)
    ↓ ffmpeg: будь-який формат → wav 16kHz mono
    ↓ Whisper pipeline (завантажений з MLflow Registry)
    ↓ {"text": "...", "duration_sec": ...}
Клієнт
```

Модель тягнеться з реєстру за URI `models:/surzhyk-whisper@champion`. Коли в реєстрі з'явиться краща модель — достатньо перевісити alias `champion` і перезапустити сервіс. Код змін не потребує.

---

## Monitoring

Збір метрик із працюючого сервісу інференсу через **Prometheus** і візуалізація в **Grafana**.

### Які метрики збираються

- `transcribe_requests_total` — загальна кількість запитів на транскрипцію (Counter)
- `transcribe_latency_seconds` — час обробки кожного запиту (Histogram, бакети від 0.5с до 120с)
- `transcribe_in_progress` — кількість запитів, що обробляються прямо зараз (Gauge)

Метрики експортуються ендпоінтом `GET /metrics` у форматі Prometheus.

### Як підняти моніторинг

Prometheus і Grafana піднімаються разом з рештою інфраструктури:

```bash
docker compose up -d
source .venv/bin/activate
uvicorn inference.app:app --host 0.0.0.0 --port 8000
```

| Сервіс | URL | Доступ |
|---|---|---|
| Grafana (дашборд) | http://localhost:3000 | `admin` / `admin` |
| Prometheus (метрики) | http://localhost:9090 | — |
| FastAPI /metrics | http://localhost:8000/metrics | — |

### Де дивитися дашборд

Grafana: http://localhost:3000 → дашборд **Surzhyk ASR Monitoring**.

Три панелі:
- **Requests per minute** — кількість транскрипцій за хвилину
- **Average latency** — середній час обробки запиту
- **P95 latency** — час, за який встигають 95% запитів

### Як згенерувати навантаження

```bash
python monitoring/load_test.py
```

Скрипт послідовно відправляє аудіокліпи з датасету на `/transcribe` з паузою між запитами, щоб на дашборді було видно рівномірне навантаження.