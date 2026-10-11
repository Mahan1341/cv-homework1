# ДЗ 1 «Фотолаборатория» — отчёт

**Автор:** Гаранин Владимир
**Группа / трек:** MFAI-02
**Дата сдачи:** 10.10.2026

## 1. Числа

### 1.1. Гамма (`data/dark.jpg`)

| γ | среднее | медиана | p1 | p99 | обрезано чёрных, % | обрезано белых, % |
|---|---:|---:|---:|---:|---:|---:|
| исходник | 7.7 | 7 | 4 | 16 | 0.00 | 0.00 |
| 0.5 | 43.9 | 43 | 33 | 63 | 0.00 | 0.00 |
| 2.0 | 0.1 | 0 | 0 | 1 | 94.31 | 0.00 |

Файлы: `out/dark_g05.jpg`, `out/dark_g05_hist.png`, `out/dark_g20.jpg`, `out/dark_g20_hist.png`.

### 1.2. Баланс белого (`data/bright.jpg`)

| Метод | усиление B, G, R | средние до B, G, R | средние после B, G, R |
|---|---|---|---|
| grey-world | 1.518, 0.894, 0.818 | 69.4, 117.8, 128.8 | 99.3, 105.3, 105.4 |
| white-patch (p99) | 1.000, 1.000, 1.000 | 69.4, 117.8, 128.8 | 69.4, 117.8, 128.8 |

Файлы: `out/bright_gw.jpg`, `out/bright_gw_hist.png`, `out/bright_wp.jpg`, `out/bright_wp_hist.png`.

### 1.3. `equalizeHist` против CLAHE (`data/dark.jpg`, clip = 2, tiles = 8)

| Кадр | σ в тенях | среднее в тенях | среднее в светах |
|---|---:|---:|---:|
| исходный | 1.0 | 4.4 | 8.1 |
| equalizeHist | 52.1 | 120.5 | 237.2 |
| CLAHE | 2.0 | 10.5 | 19.1 |

Файлы: `out/dark_clahe.jpg`, `out/dark_clahe_hist.png`, `out/dark_cmp.jpg`.

### 1.4. Хромакей (`data/object.jpg`, `--sample 410,500,180,260`)

Для финального варианта использован ярко-зелёный однотонный объект на светлом фоне.

| Что | Значение |
|---|---|
| границы lo / hi | `(30, 113, 40)` / `(57, 255, 255)` |
| доля маски | 8.02 % |
| компонент до / после обработки | 59 / 1 |
| площадь крупнейшей компоненты | 141883 px |
| bbox | x=375, y=453, w=330, h=478 |
| дыр | 0 |
| `ball-lamp.jpg` курса: recall / precision | 94.6 % / 99.9 % — эталон из `sample_data/README.md` |

Файлы: `out/object_on_bg.jpg`, `out/object_mask.png`.

### 1.5. Геометрия

| Что | Значение |
|---|---|
| поворот 30° | 865×1536 → 1518×1763 |
| пустой холст после поворота | 50.4 % |
| уменьшение ×0.25: PSNR naive / blur / area к Lanczos | 23.3 / 32.7 / 33.0 дБ |
| `iu-building.jpg`: PSNR naive / blur / area | 27.9 / 36.0 / 37.2 дБ — эталон из `sample_data/README.md` |
| стороны `poster.jpg` в кадре | top 497.4, right 571.8, bottom 706.9, left 565.3 px |
| rectify | `--aspect 0.707`, выход 707×1000 |
| `poster-angle.jpg` против `poster-gt.png` | 99.1 % — эталон из `sample_data/README.md` |

Файлы: `out/poster_rot30.jpg`, `out/text_x025.jpg`, `out/poster_front.jpg`.

### 1.6. Контактный лист

`N = 12`, сетка `4 × 3`, размер листа `1320 × 830` — `out/sheet.jpg`.

## 2. Три вопроса

### 2.1. Какое цветовое пространство выбрано для хромакея и почему? Что было бы в RGB?

Используется HSV, потому что оттенок `H` и насыщенность `S` отделены от яркости `V`. Это позволяет одному диапазону лучше переживать блики и тени, чем прямоугольный диапазон в RGB, где изменение освещения одновременно меняет три канала. На финальном зелёном объекте после морфологии осталась одна компонента без дыр, а маска заняла 8.02 % кадра.

### 2.2. Что происходит с гистограммой при γ < 1 и при γ > 1? Куда деваются обрезанные пиксели?

При γ = 0.5 тёмные значения растягиваются вправо: средняя яркость выросла с 7.7 до 43.9, а `p1` — с 4 до 33. При γ = 2 значения сжались к нулю: среднее стало 0.1, и 94.31 % пикселей после округления получили значение 0. Перед переводом обратно в `uint8` значения ограничиваются диапазоном 0…255, поэтому вышедшие за диапазон значения были бы прижаты к 0 или 255.

### 2.3. Почему CLAHE лучше `equalizeHist` на вашем фото?

На этом очень тёмном фото глобальный `equalizeHist` оказался слишком агрессивным: σ в тенях выросла с 1.0 до 52.1, а среднее в светах — с 8.1 до 237.2, поэтому вместе с деталями резко усилился шум. CLAHE поднял σ в тенях только до 2.0 и среднее в светах до 19.1. Визуально он сохраняет естественный тёмный вид кадра и вытаскивает локальные детали без такого сильного усиления шума.

## 3. Где сломалось

Неудачным оказался `white-patch` на `data/bright.jpg`. Из-за сильного пересвета 99-й процентиль каждого канала уже равен 255, поэтому коэффициенты усиления получились `(1.000, 1.000, 1.000)` и метод вообще не исправил выраженный жёлтый оттенок: средние каналов остались `69.4 / 117.8 / 128.8`. Здесь помог бы кадр без выбитых светов или выбор менее экстремальной белой области; `grey-world` на этом же снимке сработал заметно лучше.

## 4. Команды

Все файлы в `out/` получены следующими командами:

```bash
python photolab.py gamma data/dark.jpg --gamma 0.5 -o out/dark_g05.jpg --hist out/dark_g05_hist.png
python photolab.py gamma data/dark.jpg --gamma 2.0 -o out/dark_g20.jpg --hist out/dark_g20_hist.png

python photolab.py wb data/bright.jpg --method grey-world -o out/bright_gw.jpg --hist out/bright_gw_hist.png
python photolab.py wb data/bright.jpg --method white-patch -o out/bright_wp.jpg --hist out/bright_wp_hist.png

python photolab.py clahe data/dark.jpg --clip 2 --tiles 8 -o out/dark_clahe.jpg --compare out/dark_cmp.jpg --hist out/dark_clahe_hist.png

python photolab.py chromakey data/object.jpg --bg data/bg.jpg --sample 410,500,180,260 -o out/object_on_bg.jpg --mask out/object_mask.png

python photolab.py rotate data/poster.jpg --deg 30 -o out/poster_rot30.jpg

python photolab.py resize data/text.jpg --scale 0.25 --report -o out/text_x025.jpg

python photolab.py rectify data/poster.jpg --corners 292,410,789,430,744,1000,42,917 --aspect 0.707 -o out/poster_front.jpg

python photolab.py sheet data/dark.jpg data/bright.jpg data/object.jpg data/bg.jpg data/poster.jpg data/text.jpg out/dark_g05.jpg out/bright_gw.jpg out/dark_clahe.jpg out/poster_front.jpg out/text_x025.jpg out/object_on_bg.jpg --cols 4 -o out/sheet.jpg
```

## 5. Инструменты

При выполнении ДЗ использовался ChatGPT. С его помощью были скорректированы TODO-функции, подобраны параметры для собственных фотографий, проверены результаты и подготовлен отчёт. Реализация проверена командой `pytest tests -q`: все 11 тестов домашнего задания проходят; результаты на собственных изображениях дополнительно проверены визуально.
