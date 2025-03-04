# bot.py

import logging
import json
import os
import asyncio

from aiogram import Bot, Dispatcher, types
from aiogram.types import ReplyKeyboardMarkup, KeyboardButton
from aiogram.filters import Command

# --- Библиотеки для анализа текста ---
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import gensim
import numpy as np
from gensim.models import Word2Vec

import json

with open('faq.json', encoding='utf-8') as f:
  data = json.load(f)

# ------------------------------------------------------------------------------------
# ЗДЕСЬ СКРЫТЫЙ API ТОКЕН:
API_TOKEN = "NO"
# ------------------------------------------------------------------------------------

# Включаем логирование для удобства отладки
logging.basicConfig(level=logging.INFO)

# Инициализируем бота и диспетчер
bot = Bot(token=API_TOKEN)
dp = Dispatcher()

# ---------------------------
# 1. Загрузка FAQ из faq.json
# ---------------------------
with open("faq.json", "r", encoding="utf-8") as f:
    data = json.load(f)


faq_data = data["faq"]  # Извлекаем список объектов

faq_questions = [item["question"] for item in faq_data]
faq_answers   = [item["answer"]   for item in faq_data]

# ------------------------------------------
# 2. Настраиваем TF-IDF для списка вопросов
# ------------------------------------------
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(faq_questions)

# ------------------------------------------
# 3. Настраиваем Word2Vec для списка вопросов
# ------------------------------------------
# Разбиваем вопросы на списки токенов
sentences = [q.split() for q in faq_questions]
w2v_model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)

def sentence_vector(sentence: str, model: Word2Vec) -> np.ndarray:
    words = sentence.split()
    vectors = [model.wv[word] for word in words if word in model.wv]
    if not vectors:
        # Если в предложении нет слов из модели, вернём вектор из нулей
        return np.zeros(model.vector_size)
    return np.mean(vectors, axis=0)

faq_vectors = np.array([sentence_vector(q, w2v_model) for q in faq_questions])

# ------------------------------------------------------------------------
# Функции для поиска наиболее подходящего ответа (TF-IDF и Word2Vec варианты)
# ------------------------------------------------------------------------
def get_best_answer_tfidf(query: str) -> str:
    """Поиск наиболее близкого вопроса и ответа с помощью TF-IDF."""
    query_vector = tfidf_vectorizer.transform([query])
    similarities = cosine_similarity(query_vector, tfidf_matrix)
    best_match_idx = similarities.argmax()
    return faq_answers[best_match_idx]

def get_best_answer_w2v(query: str) -> str:
    """Поиск наиболее близкого вопроса и ответа с помощью Word2Vec."""
    query_vec = sentence_vector(query, w2v_model).reshape(1, -1)
    similarities = cosine_similarity(query_vec, faq_vectors)
    best_match_idx = similarities.argmax()
    return faq_answers[best_match_idx]

# -------------------------------------------------
# 4. Создаём клавиатуру с двумя кнопками
# -------------------------------------------------
keyboard = ReplyKeyboardMarkup(
        keyboard=[ [KeyboardButton(text='О компании')], [KeyboardButton(text='Пожаловаться')] ],
        resize_keyboard=True
    )

# -------------------------------------------------
# 5. Реакция на команду /start
# -------------------------------------------------
@dp.message(Command("start"))
async def cmd_start(message: types.Message):
    welcome_text = (
        "Привет! Я бот, который поможет ответить на вопросы. "
        "Просто задайте свой вопрос или воспользуйтесь кнопками ниже."
    )
    await message.answer(welcome_text, reply_markup=keyboard)

# -------------------------------------------------
# 6. Обработчик нажатия кнопок «О компании» и «Пожаловаться»
# -------------------------------------------------
@dp.message(lambda message: message.text == "О компании")
async def about_company(message: types.Message):
    text = "Наша компания занимается доставкой товаров по всей стране."
    await message.answer(text)

@dp.message(lambda message: message.text == "Пожаловаться")
async def complain(message: types.Message):
    text = (
        "Пожалуйста, пришлите изображение (скриншот или фото), "
        "опишите кратко проблему – и мы постараемся помочь."
    )
    await message.answer(text)

# -------------------------------------------------
# 7. Обработчик входящих фото (для жалобы)
# -------------------------------------------------
@dp.message(lambda message: message.content_type == "photo")
async def handle_photo(message: types.Message):
    """
    При получении фото бот сообщает его имя, размер, и подтверждает передачу запроса.
    """
    # Самый крупный вариант фото – в конце списка
    photo = message.photo[-1]
    photo_file = await bot.get_file(photo.file_id)

    # Название файла может быть пустым, если Telegram его не передаёт
    # Однако размер (file_size) доступен
    file_name = photo.file_id + ".jpg"  # Условное название (т.к. реального названия может не быть)
    file_size = photo_file.file_size

    response_text = (
        f"Фото получено!\n"
        f"Название (условное): {file_name}\n"
        f"Размер файла (байт): {file_size}\n\n"
        "Ваш запрос передан специалисту."
    )
    await message.answer(response_text)

# -------------------------------------------------
# 8. Ответ на любой текстовый вопрос (FAQ)
# -------------------------------------------------
@dp.message()
async def handle_question(message: types.Message):
    user_query = message.text.strip()

    # Выбираем метод: здесь для примера TF-IDF
    # Если использовать Word2Vec, заменить строку на:
    # best_answer = get_best_answer_w2v(user_query)
    best_answer = get_best_answer_tfidf(user_query)

    await message.answer(best_answer)

# -------------------------------------------------
# Запуск бота
# -------------------------------------------------
# Основной цикл
async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
