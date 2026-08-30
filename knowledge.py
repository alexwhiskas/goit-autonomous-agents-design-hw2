"""База знань (ChromaDB) для agentic RAG та tool search_knowledge.

Документи описують DevOps-політики та процедури (runbook-стиль), які
не змінюються в реальному часі — на відміну від tools.py (статус,
логи, uptime), що повертають "живі" дані. Агент сам вирішує, який
тип інформації потрібен: довідковий (RAG) чи оперативний (tools).
"""

import os

import chromadb
from langchain_core.tools import tool

CHROMA_PATH = os.path.join(os.path.dirname(__file__), "chroma_db")
COLLECTION_NAME = "devops_knowledge"

DOCUMENTS = [
    "Політика вікон деплою: планові деплої в production дозволені з "
    "вівторка по четвер, з 10:00 до 16:00. Деплої заборонені в п'ятницю "
    "після обіду, у вихідні та за день до державних свят.",

    "Рівні серйозності інцидентів (severity): SL1 — повний відказ "
    "критичного сервісу, реакція негайно; SL2 — деградація сервісу для "
    "частини користувачів, реакція протягом 30 хв; SL3 — незначний "
    "дефект без впливу на користувачів, реакція протягом робочого дня.",

    "Процедура ескалації: якщо черговий інженер не відповів на алерт "
    "протягом 15 хвилин, система автоматично ескалує на тімліда. Якщо "
    "тімлід не відповів протягом наступних 15 хвилин — на chan #incidents "
    "виходить на зв'язок весь DevOps-канал.",

    "SLA сервісів: цільовий uptime для production-сервісів становить "
    "99.9% на місяць (не більше 43 хвилин простою). Порушення SLA "
    "фіксується у щомісячному звіті та обговорюється на post-mortem.",

    "Процедура rollback: якщо новий деплой спричинив зростання error "
    "rate більш ніж на 5% протягом 10 хвилин після релізу, необхідно "
    "негайно відкотити (rollback) до попередньої стабільної версії "
    "через CI/CD pipeline, а не вручну.",

    "Post-mortem процес: після кожного інциденту рівня SL1 або SL2 "
    "команда протягом 48 годин готує post-mortem документ без "
    "звинувачень (blameless), що описує причину, хронологію та action "
    "items для запобігання повторенню.",

    "Права доступу на деплой у production: деплоїти в production можуть "
    "тільки інженери з роллю 'release manager' або вище. Звичайні "
    "розробники можуть деплоїти лише у dev та staging середовища.",

    "Політика резервного копіювання: бази даних production бекапляться "
    "щодня о 03:00 UTC з ретенцією 30 днів. Перевірка відновлюваності "
    "бекапу (restore test) виконується щомісяця.",

    "Правила іменування сервісів: назви сервісів пишуться у kebab-case "
    "та повинні відображати домен (наприклад, payment-service, "
    "auth-service). Версії дотримуються семантичного версіювання "
    "(MAJOR.MINOR.PATCH) з префіксом 'v'.",

    "Політика алертів: алерт вважається actionable, якщо він вимагає "
    "дії людини протягом години. Неактуальні або 'шумні' алерти "
    "повинні бути відключені або переналаштовані протягом тижня після "
    "виявлення, щоб уникнути alert fatigue.",
]


def build_knowledge_base(client: "chromadb.ClientAPI" = None) -> "chromadb.Collection":
    """Створити (або відкрити існуючу) ChromaDB collection із документами.

    Args:
        client: Опційний ChromaDB client. За замовчуванням — PersistentClient
            (кеш embeddings між запусками у CHROMA_PATH).

    Returns:
        ChromaDB collection з завантаженими документами.
    """
    if client is None:
        client = chromadb.PersistentClient(path=CHROMA_PATH)

    collection = client.get_or_create_collection(COLLECTION_NAME)

    if collection.count() == 0:
        doc_ids = [f"doc_{i}" for i in range(len(DOCUMENTS))]
        collection.add(documents=DOCUMENTS, ids=doc_ids)

    return collection


_collection = None


def _get_collection():
    global _collection
    if _collection is None:
        _collection = build_knowledge_base()
    return _collection


@tool
def search_knowledge(query: str) -> str:
    """Пошук довідкової інформації у базі знань (політики, процедури, правила).

    Використовуйте цей інструмент для запитів про правила, процедури,
    політики та довідкову інформацію (наприклад, "коли можна деплоїти?",
    "які рівні серйозності інцидентів?", "як працює rollback?").
    НЕ використовуйте для отримання "живих" даних (поточний статус
    сервісу, логи, uptime) — для цього є інші tools.

    При посиланні на конкретне правило чи політику з результатів
    ЦИТУЙТЕ джерело дослівно (можна перевірити через verify_citation),
    а не переказуйте своїми словами — так знижується ризик, що агент
    "згадає" політику, якої насправді немає в базі знань.

    Args:
        query: Пошуковий запит.

    Returns:
        Топ-3 релевантних документи з бази знань.
    """
    collection = _get_collection()
    results = collection.query(query_texts=[query], n_results=3)
    docs = results["documents"][0]
    return "\n---\n".join(docs)


def verify_citation(quoted_text: str, source_docs: list[str] = None) -> bool:
    """Перевірити, що QUOTED_TEXT дійсно міститься в базі знань.

    Захист проти hallucinated citations: LLM може стверджувати "згідно
    з політикою X..." навіть якщо retrieved документи цього насправді
    не містять. Порівняння — після нормалізації пробілів і регістру,
    щоб не відхиляти цитату лише через незначну різницю форматування.

    Args:
        quoted_text: Текст, який агент стверджує, що процитував.
        source_docs: Документи для звірки (за замовчуванням — уся DOCUMENTS,
            або можна передати саме ті, що повернув конкретний search_knowledge).

    Returns:
        True, якщо quoted_text (дослівно, з точністю до пробілів/регістру)
        є підрядком одного з source_docs.
    """
    if source_docs is None:
        source_docs = DOCUMENTS
    normalized_quote = " ".join(quoted_text.split()).lower()
    if not normalized_quote:
        return False
    return any(normalized_quote in " ".join(doc.split()).lower() for doc in source_docs)


if __name__ == "__main__":
    kb = build_knowledge_base()
    print(f"Knowledge base: {kb.count()} documents loaded")

    print("\n=== Тест RAG-пошуку ===")
    for q in ["Коли можна деплоїти в продакшн?", "Що робити при падінні error rate після релізу?"]:
        print(f"\nЗапит: {q}")
        print(search_knowledge.invoke({"query": q}))

    print("\n=== Перевірка цитувань (захист від hallucinated policy) ===")
    real_quote = "планові деплої в production дозволені з вівторка по четвер"
    fake_quote = "деплої дозволені цілодобово без обмежень"
    print(f'Реальна цитата з бази знань -> verify_citation = {verify_citation(real_quote)}')
    print(f'Вигадана "цитата" -> verify_citation = {verify_citation(fake_quote)}')
