"""
Улучшенный сборщик аннотаций с нормализацией имен артистов
Работает с данными из russian_song_lyrics.csv
"""
import os
import requests
import json
import time
import re
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional
import logging
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data/collection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ВАЖНО: Вставь свой токен
GENIUS_TOKEN = os.environ.get("GENIUS_TOKEN")


class AnnotationCollector:
    def __init__(self, token: str, timeout: int = 20, max_retries: int = 3):
        if not token:
            raise ValueError("GENIUS_TOKEN не задан. Установи переменную окружения GENIUS_TOKEN.")
        self.token = token
        self.base_url = "https://api.genius.com"
        self.headers = {'Authorization': f'Bearer {token}'}
        self.collected = 0
        self.failed = 0
        self.cache = {}  # Кэш для уже собранных песен
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = requests.Session()

    def _get(self, url: str, params: Optional[Dict] = None) -> Optional[requests.Response]:
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.session.get(url, headers=self.headers, params=params, timeout=self.timeout)
                if response.status_code == 429:
                    try:
                        payload = response.json()
                    except Exception:
                        payload = response.text
                    logger.error(f"HTTP 429: {payload}")
                    return response
                return response
            except requests.RequestException as e:
                if attempt == self.max_retries:
                    logger.warning(f"HTTP ошибка: {e}")
                    return None
                time.sleep(0.5 * attempt)
        return None

    def normalize_artist_name(self, artist: str) -> str:
        """Нормализует имя артиста"""
        # Убираем лишние пробелы
        artist = artist.strip()

        # Убираем скобки и содержимое
        artist = re.sub(r'\s*\([^)]*\)', '', artist)

        # Убираем множественные пробелы
        artist = re.sub(r'\s+', ' ', artist)

        # Убираем префиксы типа " & "
        artist = re.sub(r'^\s*&\s*', '', artist)

        return artist.strip()

    def normalize_title(self, title: str) -> str:
        """Нормализует название песни"""
        # Убираем английские переводы в скобках
        title = re.sub(r'\s+[A-Z][a-z\s]+$', '', title)

        # Убираем (feat. ...), (prod. ...), и т.д.
        title = re.sub(r'\s*\(feat\..*?\)', '', title, flags=re.IGNORECASE)
        title = re.sub(r'\s*\(prod\..*?\)', '', title, flags=re.IGNORECASE)

        # Убираем множественные пробелы
        title = re.sub(r'\s+', ' ', title)

        return title.strip()

    def search_song(self, artist: str, title: str) -> Optional[Dict]:
        """Ищет песню на Genius"""
        # Нормализуем
        artist_norm = self.normalize_artist_name(artist)
        title_norm = self.normalize_title(title)

        # Проверяем кэш
        cache_key = f"{artist_norm}::{title_norm}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        # Пробуем разные варианты поиска
        queries = [
            f"{artist_norm} {title_norm}",
            f"{artist} {title}",  # Оригинальные имена
            title_norm,  # Только название
        ]

        for query in queries:
            try:
                url = f"{self.base_url}/search"
                params = {'q': query}
                response = self._get(url, params=params)
                if response and response.status_code == 200:
                    hits = response.json()['response']['hits']

                    if hits:
                        # Проверяем первые 5 результатов
                        for hit in hits[:5]:
                            result = hit['result']
                            result_artist = result['primary_artist']['name']
                            result_title = result['title']

                            # Проверяем совпадение (нечувствительно к регистру)
                            if (artist_norm.lower() in result_artist.lower() and
                                any(word.lower() in result_title.lower()
                                    for word in title_norm.split()[:3])):  # Первые 3 слова
                                self.cache[cache_key] = result
                                return result

                time.sleep(0.5)  # Rate limiting

            except Exception as e:
                logger.warning(f"Ошибка поиска '{query}': {e}")
                time.sleep(1)

        self.cache[cache_key] = None
        return None

    def get_song_details(self, song_id: int) -> Optional[Dict]:
        """Получает базовые метаданные песни по song_id"""
        try:
            url = f"{self.base_url}/songs/{song_id}"
            response = self._get(url)
            if response and response.status_code == 200:
                return response.json()['response']['song']
            return None
        except Exception as e:
            logger.error(f"Ошибка получения данных песни song_id={song_id}: {e}")
            return None

    def get_song_annotations(self, song_id: int, min_votes: int = 0) -> List[Dict]:
        """Получает аннотации для песни"""
        try:
            url = f"{self.base_url}/referents"
            params = {
                'song_id': song_id,
                'text_format': 'plain',
                'per_page': 50
            }

            annotations = []
            page = 1
            max_pages = 5

            while page <= max_pages:
                params['page'] = page
                response = self._get(url, params=params)
                if not response or response.status_code != 200:
                    break

                data = response.json()['response']
                referents = data.get('referents', [])
                if not referents:
                    break

                for ref in referents:
                    fragment = str(ref.get('fragment', '')).strip()
                    if not fragment:
                        continue
                    for ann in ref.get('annotations', []):
                        votes = ann.get('votes_total', 0)
                        if votes < min_votes:
                            continue
                        body = ann.get('body', {})
                        if isinstance(body, dict):
                            text = body.get('plain', '')
                        else:
                            text = str(body)
                        text = str(text).strip()
                        if not text:
                            continue
                        annotations.append({
                            'fragment': fragment,
                            'annotation': text,
                            'votes': votes
                        })

                if not data.get('next_page'):
                    break

                page += 1
                time.sleep(0.2)

            return annotations

        except Exception as e:
            logger.error(f"Ошибка получения аннотаций для song_id={song_id}: {e}")
            return []

    def collect_for_artist(self, artist: str, songs: List[str],
                          max_songs: int = 20) -> List[Dict]:
        """Собирает аннотации для артиста"""
        logger.info(f"Сбор для артиста: {artist} ({len(songs)} песен)")

        collected_songs = []
        processed = 0

        for title in songs[:max_songs]:
            processed += 1

            # Ищем песню
            song_result = self.search_song(artist, title)

            if not song_result:
                logger.debug(f"  Не найдено: {title}")
                self.failed += 1
                continue

            song_id = song_result['id']
            song_title = song_result['title']
            song_artist = song_result['primary_artist']['name']

            # Получаем аннотации
            annotations = self.get_song_annotations(song_id, min_votes=3)

            if annotations:
                collected_songs.append({
                    'artist': song_artist,
                    'title': song_title,
                    'url': song_result['url'],
                    'annotations': annotations
                })
                self.collected += 1
                logger.info(f"  {song_title} - {len(annotations)} аннотаций")
            else:
                logger.debug(f"  {song_title} - нет аннотаций")
                self.failed += 1

            # Rate limiting
            time.sleep(1)

            # Прогресс
            if processed % 5 == 0:
                logger.info(f"  Обработано {processed}/{min(len(songs), max_songs)}")

        return collected_songs


def collect_from_csv_ids(csv_path: str,
                         output_path: str = 'data/annotations_dataset_new.json',
                         tag_filter: Optional[str] = None,
                         language_filter: Optional[str] = None,
                         max_songs: Optional[int] = None,
                         min_votes: int = 0,
                         max_workers: int = 4,
                         flush_every: int = 25,
                         output_jsonl_path: Optional[str] = None,
                         resume: bool = True,
                         validate_sample: int = 10,
                         fallback_to_search: bool = True) -> List[Dict]:
    """Сбор аннотаций по song_id из CSV (Genius ID)"""
    logger.info(f"Загрузка CSV: {csv_path}")
    df = pd.read_csv(csv_path)

    # Фильтры по языку и тегу
    if language_filter and 'language' in df.columns:
        df = df[df['language'] == language_filter]
    if tag_filter and 'tag' in df.columns:
        df = df[df['tag'] == tag_filter]

    if 'id' not in df.columns:
        raise KeyError("CSV не содержит колонку 'id' с song_id")

    # Оставляем уникальные song_id
    df = df.dropna(subset=['id']).drop_duplicates(subset=['id'])
    if 'views' in df.columns:
        df = df.sort_values(by='views', ascending=False)

    if max_songs:
        df = df.head(max_songs)

    logger.info(f"Кандидатов для сбора: {len(df)}")

    collector = AnnotationCollector(GENIUS_TOKEN, timeout=20, max_retries=3)
    collected_songs = []
    lock = Lock()

    if output_jsonl_path is None:
        if output_path.endswith('.json'):
            output_jsonl_path = output_path[:-5] + '.jsonl'
        else:
            output_jsonl_path = output_path + '.jsonl'

    seen_ids = set()
    if resume and output_jsonl_path and Path(output_jsonl_path).exists():
        try:
            with open(output_jsonl_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    item = json.loads(line)
                    song_id = item.get('song_id')
                    if song_id is not None:
                        seen_ids.add(int(song_id))
            logger.info(f"Найдено уже собранных: {len(seen_ids)}")
        except Exception as e:
            logger.warning(f"Не удалось прочитать {output_jsonl_path}: {e}")
    elif output_jsonl_path:
        Path(output_jsonl_path).touch()

    if validate_sample:
        valid_count = 0
        sample = df.head(validate_sample)
        for _, row in sample.iterrows():
            try:
                song_id = int(row['id'])
            except Exception:
                continue
            if collector.get_song_details(song_id):
                valid_count += 1
        logger.info(f"Проверка song_id: {valid_count}/{len(sample)} валидны")

    def process_row(row: pd.Series) -> Optional[Dict]:
        song_id = int(row['id'])
        if song_id in seen_ids:
            return None

        artist = str(row.get('artist', '')).strip()
        title = str(row.get('title', '')).strip()

        annotations = collector.get_song_annotations(song_id, min_votes=min_votes)
        song_details = None
        if annotations:
            song_details = collector.get_song_details(song_id)
        elif fallback_to_search:
            search_result = collector.search_song(artist, title)
            if search_result:
                song_id = int(search_result.get('id', song_id))
                annotations = collector.get_song_annotations(song_id, min_votes=min_votes)
                song_details = {
                    'primary_artist': {'name': search_result.get('primary_artist', {}).get('name', artist)},
                    'title': search_result.get('title', title),
                    'url': search_result.get('url', '')
                }

        if not annotations:
            with lock:
                collector.failed += 1
            return None

        if song_details:
            artist = song_details.get('primary_artist', {}).get('name', artist)
            title = song_details.get('title', title)
            url = song_details.get('url', '')
        else:
            url = ''

        return {
            'song_id': song_id,
            'artist': artist,
            'title': title,
            'url': url,
            'annotations': annotations
        }

    rows = [row for _, row in df.iterrows()]
    processed = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_row, row) for row in rows]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Collecting", unit="song"):
            processed += 1
            result = future.result()
            if not result:
                if processed % flush_every == 0:
                    with open(output_path, 'w', encoding='utf-8') as f:
                        json.dump(collected_songs, f, ensure_ascii=False, indent=2)
                    logger.info(f"  Промежуточное сохранение: {len(collected_songs)} песен")
                continue

            with lock:
                collected_songs.append(result)
                collector.collected += 1

                if output_jsonl_path:
                    with open(output_jsonl_path, 'a', encoding='utf-8') as f:
                        f.write(json.dumps(result, ensure_ascii=False) + "\n")

                if processed % flush_every == 0:
                    with open(output_path, 'w', encoding='utf-8') as f:
                        json.dump(collected_songs, f, ensure_ascii=False, indent=2)
                    logger.info(f"  Промежуточное сохранение: {len(collected_songs)} песен")

    # Финальное сохранение
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(collected_songs, f, ensure_ascii=False, indent=2)

    logger.info("Сбор завершен")
    logger.info(f"Собрано песен: {len(collected_songs)}")
    logger.info(f"Успешно: {collector.collected}")
    logger.info(f"Неудачно: {collector.failed}")
    logger.info(f"Сохранено в: {output_path}")

    return collected_songs


def main():
    """Основная функция сбора"""
    logger.info("=" * 60)
    logger.info("Запуск улучшенного сборщика аннотаций")
    logger.info("=" * 60)

    # Режим: сбор по song_id из CSV
    csv_path = 'data/russian_song_lyrics.csv'
    collect_from_csv_ids(
        csv_path=csv_path,
        output_path='data/annotations_dataset_new.json',
        tag_filter=None,
        language_filter=None,
        max_songs=None,
        min_votes=0,
        validate_sample=10,
        fallback_to_search=True,
        max_workers=2,
        flush_every=25
    )
    return

if __name__ == "__main__":
    main()
