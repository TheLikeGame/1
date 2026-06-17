"""
Практическая работа: Машинное обучение для классификации текста
Алгоритм: TF-IDF + Naive Bayes (MultinomialNB)
Задача: Определить является ли введённый текст спамом или обычным сообщением
"""

import os
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import requests

N8N_WEBHOOK_URL = "https://thelikegame.app.n8n.cloud/webhook-test/spam-moderation"
UNCERTAINTY_THRESHOLD = 70  # % ниже которого отправляем на модерацию

def classify_and_route(text: str, pipeline):
    pred = int(pipeline.predict([text])[0])
    proba = pipeline.predict_proba([text])[0]
    confidence = float(proba[pred]) * 100

    label = "spam" if pred == 1 else "ham"

    if confidence < UNCERTAINTY_THRESHOLD:
        # Отправляем на ручную модерацию в n8n
        try:
            requests.post(N8N_WEBHOOK_URL, json={
                "text": text,
                "predicted": label,
                "confidence": round(confidence, 1)
            }, timeout=5)
            print(f"⏳ Отправлено на модерацию ({confidence:.1f}%): {text}")
        except requests.exceptions.RequestException as e:
            print(f"Ошибка отправки в n8n: {e}")
    else:
        print(f"{'🚨 СПАМ' if pred == 1 else '✅ Норма'} ({confidence:.1f}%): {text}")

# -------------------------------------------------------
# 1. Загрузка данных
# -------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
df = pd.read_csv(BASE_DIR / "data.csv")

print("=" * 50)
print("  КЛАССИФИКАТОР СПАМА — МАШИННОЕ ОБУЧЕНИЕ")
print("=" * 50)
print(f"\n[1] Загружено обучающих примеров: {len(df)}")
print(f"    Из них спам:     {df['label'].sum()}")
print(f"    Из них не спам:  {(df['label'] == 0).sum()}")

# -------------------------------------------------------
# 2. Разделение на обучающую и тестовую выборку
# -------------------------------------------------------
X = df['text']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

print(f"\n[2] Обучающая выборка: {len(X_train)} примеров")
print(f"    Тестовая выборка:  {len(X_test)} примеров")

# -------------------------------------------------------
# 3. Создание и обучение Pipeline
#    TfidfVectorizer — преобразует текст в числовые признаки
#    MultinomialNB   — вероятностный классификатор
# -------------------------------------------------------
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf',   MultinomialNB()),
])

pipeline.fit(X_train, y_train)
print("\n[3] Модель обучена успешно")

# -------------------------------------------------------
# 4. Оценка качества модели на тестовой выборке
# -------------------------------------------------------
preds = pipeline.predict(X_test)
acc = accuracy_score(y_test, preds)

print(f"\n[4] Точность модели на тестовой выборке: {acc * 100:.1f}%")
print()
print(classification_report(y_test, preds, target_names=['Не спам', 'Спам']))

# -------------------------------------------------------
# 5. Интерактивный ввод текста пользователем
# -------------------------------------------------------
print("=" * 50)
print("  ВВЕДИТЕ ТЕКСТ ДЛЯ ПРОВЕРКИ")
print("  (введите 'выход' для завершения)")
print("=" * 50)

while True:
    user_input = input("\nВведите сообщение: ").strip()

    if user_input.lower() in ('выход', 'exit', 'quit', 'q'):
        print("Завершение работы.")
        break

    if not user_input:
        print("Пустой ввод, попробуйте ещё раз.")
        continue

    result = pipeline.predict([user_input])[0]
    proba = pipeline.predict_proba([user_input])[0]

    if result == 1:
        print(f"  Результат: 🚨 СПАМ  (уверенность: {proba[1]*100:.1f}%)")
    else:
        print(f"  Результат: ✅ НЕ СПАМ  (уверенность: {proba[0]*100:.1f}%)")