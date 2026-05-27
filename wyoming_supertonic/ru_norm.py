import logging
import re

from num2words import num2words

log = logging.getLogger(__name__)

try:
    from silero_stress import load_accentor
    SILERO_AVAILABLE = True
except ImportError:
    SILERO_AVAILABLE = False


class RussianTextNormalizer:

    # Множество для быстрого поиска гласных (O(1))
    _VOWELS_SET = set('аеёиоуыэюяАЕЁИОУЫЭЮЯ')

    # Эмодзи и спецсимволы
    _emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF"
        "\U0001F1E0-\U0001F1FF\u2600-\u26FF\u2700-\u27BF"
        "\U0001F900-\U0001F9FF\u200D\uFE0F"
        "]+",
        flags=re.UNICODE
    )

    _chars_to_delete = "=#$“”„«»*\"‘’‚‹›'"
    _map_from = "—–−\xa0"
    _map_to = "--- "
    _translation_table = str.maketrans(_map_from, _map_to, _chars_to_delete)

    _FINAL_CLEANUP_PATTERN = re.compile(r'[^а-яА-ЯёЁa-zA-Z?!.,<>\/\+\u0300-\u036f -]+')

    _TAGS_PATTERN = re.compile(r'(<[a-z]+>)')

    def __init__(self):
        """
        Предзагрузка модели Silero при старте сервера.
        Она загружается один раз и хранится в памяти.
        """
        self.accentor = None
        if SILERO_AVAILABLE:
            try:
                log.info("Loading Silero stress model...")
                self.accentor = load_accentor()
                log.info("Silero stress model loaded successfully.")
            except Exception as e:
                log.error(f"Failed to load Silero stress model: {e}")
        else:
            log.warning(
                "silero_stress package not installed. "
                "Automatic stress placement is disabled."
            )

    def normalize(self, text: str) -> str:
        """
        Публичная точка входа. Никогда не вызывает исключений — 
        при любой ошибке возвращает исходный текст без изменений.
        """
        try:
            parts = self._TAGS_PATTERN.split(text)
            normalized_parts = []

            for part in parts:
                if not part:
                    continue
                if self._TAGS_PATTERN.match(part):
                    normalized_parts.append(part)
                else:
                    normalized_parts.append(self._normalize_pipeline(part))

            result = " ".join(normalized_parts)
            return re.sub(r'\s+', ' ', result).strip()
            
        except Exception as e:
            log.warning("ru_norm failed, passing text through unmodified: %s", e)
            return text

    def _normalize_pipeline(self, text: str) -> str:
        if not text.strip():
            return text

        # 1. Проценты
        text = self._normalize_percentages(text)
        # 2. Спецсимволы и эмодзи
        text = self._normalize_special_chars(text)
        # 3. Плюс перед цифрами (+7 -> плюс семь)
        text = self._normalize_plus_before_number(text)
        # 4. Цифры в слова (100 -> сто)
        text = self._normalize_numbers(text)

        # 5. Расстановка ударений Silero (только после перевода цифр в слова!)
        text = self._apply_stress(text)

        # 6. Финальная очистка
        text = self._cleanup_final_text(text).strip()
        return text

    def _apply_stress(self, text: str) -> str:
        """
        Прогоняет текст через Silero, сохраняет ручные ударения 
        и переводит формат "з+амок" в "за\u0301мок".
        """
        if not self.accentor:
            return self._convert_plus_to_unicode(text)

        try:
            # Silero расставляет знаки `+` перед гласными
            stressed_text = self.accentor(text)
        except Exception as e:
            log.debug(f"Silero stress error: {e}")
            return self._convert_plus_to_unicode(text)

        # Защита ручной расстановки: разбиваем текст на слова и пунктуацию
        split_pattern = r'([^а-яА-ЯёЁa-zA-Z\+\u0300-\u036f]+)'
        orig_parts = re.split(split_pattern, text)
        silero_parts = re.split(split_pattern, stressed_text)

        # Если структура текста не сломалась, объединяем умно
        if len(orig_parts) == len(silero_parts):
            final_parts = []
            for orig_part, silero_part in zip(orig_parts, silero_parts):
                # Если пользователь уже поставил + или \u0301, берем его версию
                if '+' in orig_part or re.search(r'[\u0300-\u036f]', orig_part):
                    final_parts.append(orig_part)
                else:
                    final_parts.append(silero_part)
            text = "".join(final_parts)
        else:
            # Если длины не совпали (редкость), доверяем результату Silero
            log.debug("Silero parts mismatch, falling back to raw Silero output")
            text = stressed_text

        # Превращаем все + (ручные и от Silero) в юникод-акцент
        return self._convert_plus_to_unicode(text)

    def _convert_plus_to_unicode(self, text: str) -> str:
        """
        Преобразует + в юникод-ударение.
        Очищает односложные слова, окончания -его/-ого и букву ё.
        """
        # 1. Функция подсчета гласных (убираем +, если гласных 1 или 0)
        def _remove_if_single_vowel(match):
            word = match.group(0)
            # Быстрый подсчет гласных через пересечение со множеством
            vowels_count = sum(1 for char in word if char in self._VOWELS_SET)
            if vowels_count <= 1:
                return word.replace('+', '')
            return word

        # Ищем последовательности из букв и плюсов (без \b, чтобы ловилось "+У" и "о+")
        text = re.sub(r'[а-яА-ЯёЁ\+]+', _remove_if_single_vowel, text)

        # 2. Убираем ударения на последнюю "о" в словах типа "него", "синего"
        text = re.sub(r'\b([а-яА-ЯёЁ]*[еоЕО][гГ])\+([оО])\b', r'\1\2', text)

        # 3. Ставим юникод-ударение всем остальным гласным, кроме ё
        text = re.sub(r'\+([аеиоуыэюяАЕИОУЫЭЮЯ])', '\\1\u0301', text)

        # 4. Если остались плюсы перед ё/Ё, просто удаляем плюс
        text = re.sub(r'\+([ёЁ])', r'\1', text)

        return text

    def _normalize_plus_before_number(self, text: str) -> str:
        return re.sub(r'\+(?=\d)', ' плюс ', text)

    def _cleanup_final_text(self, text: str) -> str:
        return self._FINAL_CLEANUP_PATTERN.sub(' ', text)

    def _choose_percent_form(self, number_str: str) -> str:
        if '.' in number_str or ',' in number_str:
            return "процента"
        try:
            number = int(number_str)
            if 10 < number % 100 < 20:
                return "процентов"
            
            last_digit = number % 10
            if last_digit == 1:
                return "процент"
            if last_digit in [2, 3, 4]:
                return "процента"
                
            return "процентов"
        except (ValueError, OverflowError):
            return "процентов"

    def _normalize_percentages(self, text: str) -> str:
        def replace_match(match):
            number_str_clean = match.group(1).replace(',', '.')
            percent_word = self._choose_percent_form(number_str_clean)
            return f" {number_str_clean} {percent_word} "
            
        return re.sub(r'(\d+([.,]\d+)?)\s*%', replace_match, text)

    def _normalize_special_chars(self, text: str) -> str:
        text = self._emoji_pattern.sub('', text)
        text = text.translate(self._translation_table)
        text = text.replace('…', '.')
        text = re.sub(r':(?!\d)', ',', text)
        text = re.sub(r'([a-zA-Zа-яА-ЯёЁ])(\d)', r'\1 \2', text)
        text = re.sub(r'(\d)([a-zA-Zа-яА-ЯёЁ])', r'\1 \2', text)
        text = text.replace('\n', ' ').replace('\t', ' ')
        return text

    def _normalize_numbers(self, text: str) -> str:
        def replace_number(match):
            num_str = match.group(0).replace(',', '.')
            try:
                if '.' in num_str:
                    parts = num_str.split('.')
                    integer_part_str, fractional_part_str = parts[0], parts[1]
                    
                    if not integer_part_str or not fractional_part_str:
                        return num2words(
                            int(num_str.replace('.', '')), lang='ru'
                        )
                        
                    integer_words = num2words(int(integer_part_str), lang='ru')
                    fractional_words = num2words(int(fractional_part_str), lang='ru')
                    return f"{integer_words} и {fractional_words}"
                
                return num2words(int(num_str), lang='ru')
            except Exception:
                return num_str
                
        return re.sub(r'\b\d+([.,]\d+)?\b', replace_number, text)