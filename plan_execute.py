"""Plan-and-Execute агент на LangGraph: planner -> executor -> replanner.

На відміну від ReAct (наступний крок вирішується "на льоту"), тут агент
СПОЧАТКУ складає повний план (список кроків), а ПОТІМ виконує його
крок за кроком, з можливістю переплановування (replanning) на основі
проміжних результатів.
"""

import operator
import os
import sqlite3
import time
from typing import Annotated, Literal, Optional, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from knowledge import search_knowledge
from tools import ALL_TOOLS

load_dotenv()

MODEL_NAME = os.environ.get("OPENAI_MODEL", "gpt-4.1")
DB_PATH = os.path.join(os.path.dirname(__file__), "agent_state.db")

EXECUTOR_TOOLS = ALL_TOOLS + [search_knowledge]


# === Структуровані моделі планування ======================================
class Plan(BaseModel):
    """План виконання задачі."""

    goal: str = Field(description="Головна ціль задачі")
    steps: list[str] = Field(description="Список конкретних, виконуваних кроків для досягнення цілі")


class ReplanDecision(BaseModel):
    """Рішення replanner: продовжити, перепланувати або завершити."""

    action: Literal["continue", "replan", "finish"] = Field(
        description="continue=виконати наступний крок, replan=змінити план, finish=завершити"
    )
    updated_steps: Optional[list[str]] = Field(
        default=None,
        description="Оновлені кроки плану (тільки якщо action='replan')",
    )
    reasoning: str = Field(description="Коротке пояснення рішення")


# === Стан графа ============================================================
class PlanExecuteState(TypedDict):
    messages: Annotated[list, add_messages]
    goal: str
    plan: list[str]
    current_step: int
    results: Annotated[list[str], operator.add]
    completed: bool


def _user_query(messages: list) -> str:
    for msg in messages:
        if isinstance(msg, HumanMessage):
            return str(msg.content)
    return ""


def planner_node(state: PlanExecuteState, llm) -> dict:
    """Генерує план (goal + steps) для запиту користувача."""
    planner_llm = llm.with_structured_output(Plan)
    query = _user_query(state["messages"])

    plan = planner_llm.invoke(
        f"Створи план для виконання задачі: {query}\n"
        "Розбий на 2-5 конкретних, виконуваних кроків. Кожен крок повинен "
        "бути чіткою дією, яку можна виконати одним викликом інструменту "
        "(перевірка статусу, отримання даних, пошук у базі знань, деплой)."
    )

    return {
        "goal": plan.goal,
        "plan": plan.steps,
        "current_step": 0,
        "results": [],
        "completed": False,
        "messages": [AIMessage(content=f"План ({plan.goal}): {plan.steps}")],
    }


def executor_node(state: PlanExecuteState, llm) -> dict:
    """Виконує поточний крок плану через tools (включно з search_knowledge)."""
    step_idx = state["current_step"]
    plan = state["plan"]

    if step_idx >= len(plan):
        return {"completed": True}

    current_step = plan[step_idx]
    llm_with_tools = llm.bind_tools(EXECUTOR_TOOLS)
    tools_by_name = {t.name: t for t in EXECUTOR_TOOLS}

    response = llm_with_tools.invoke(
        f"Виконай цей крок плану: {current_step}\n"
        f"Ціль: {state['goal']}\n"
        f"Попередні результати: {state['results']}"
    )

    result = response.content
    if getattr(response, "tool_calls", None):
        outputs = []
        for tc in response.tool_calls:
            tool_fn = tools_by_name.get(tc["name"])
            if tool_fn:
                tool_result = tool_fn.invoke(tc["args"])
                outputs.append(f'{tc["name"]}: {tool_result}')
        result = " | ".join(outputs) if outputs else result

    return {
        "current_step": step_idx + 1,
        "results": [f"Крок {step_idx + 1} ({current_step}): {result}"],
        "messages": [AIMessage(content=f"Виконано крок {step_idx + 1}: {result}")],
    }


def replanner_node(state: PlanExecuteState, llm) -> dict:
    """Аналізує прогрес і вирішує: продовжити, перепланувати чи завершити."""
    replanner_llm = llm.with_structured_output(ReplanDecision)
    plan = state["plan"]
    step_idx = state["current_step"]

    if step_idx >= len(plan):
        return {"completed": True}

    remaining = plan[step_idx:]
    decision = replanner_llm.invoke(
        "Оціни прогрес виконання плану:\n"
        f"Ціль: {state['goal']}\n"
        f"Повний план: {plan}\n"
        f"Виконано кроків: {step_idx}/{len(plan)}\n"
        f"Результати: {state['results']}\n"
        f"Залишилось: {remaining}\n"
        "Рішення: continue (виконати наступний крок), "
        "replan (змінити залишкові кроки, якщо план більше не підходить), "
        "finish (завершити, якщо цілі вже досягнуто)?"
    )

    if decision.action == "finish":
        return {
            "completed": True,
            "messages": [AIMessage(content=f"Завершено: {decision.reasoning}")],
        }
    if decision.action == "replan" and decision.updated_steps:
        return {
            "plan": plan[:step_idx] + decision.updated_steps,
            "messages": [AIMessage(content=f"План оновлено: {decision.reasoning}")],
        }
    return {}


def should_continue(state: PlanExecuteState) -> Literal["executor", "__end__"]:
    """Якщо задача завершена -> END, інакше -> executor."""
    if state.get("completed"):
        return "__end__"
    return "executor"


def build_graph(checkpointer=None, llm=None):
    """Побудувати та скомпілювати Plan-and-Execute граф.

    Args:
        checkpointer: Опційний LangGraph checkpointer (наприклад, SqliteSaver)
            для persistence стану між запусками.
        llm: Опційний готовий ChatOpenAI instance (для повторного використання).

    Returns:
        Скомпільований LangGraph app.
    """
    llm = llm or ChatOpenAI(model=MODEL_NAME, temperature=0.1)

    graph = StateGraph(PlanExecuteState)
    graph.add_node("planner", lambda state: planner_node(state, llm))
    graph.add_node("executor", lambda state: executor_node(state, llm))
    graph.add_node("replanner", lambda state: replanner_node(state, llm))

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "executor")
    graph.add_edge("executor", "replanner")
    graph.add_conditional_edges("replanner", should_continue, {"executor": "executor", "__end__": END})

    return graph.compile(checkpointer=checkpointer)


def run(app, query: str, config: Optional[dict] = None) -> dict:
    """Запустити Plan-and-Execute агента з одним запитом користувача."""
    initial_state = {
        "messages": [HumanMessage(content=query)],
        "goal": "",
        "plan": [],
        "current_step": 0,
        "results": [],
        "completed": False,
    }
    if config:
        return app.invoke(initial_state, config=config)
    return app.invoke(initial_state)


if __name__ == "__main__":
    print("=== Завдання 1: Plan-and-Execute (без checkpointer) ===")
    app = build_graph()
    result = run(app, "Перевір статус payment-service і проаналізуй помилки в логах за останню добу")
    print("Goal:", result["goal"])
    print("Plan:", result["plan"])
    print("Results:")
    for r in result["results"]:
        print(" -", r)

    print("\n=== Завдання 2: Checkpointer та persistence ===")
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    saver = SqliteSaver(conn)
    app_with_memory = build_graph(checkpointer=saver)

    config = {"configurable": {"thread_id": "session-001"}}
    result = run(app_with_memory, "Спланируй перевірку здоров'я auth-service", config=config)
    print(f"session-001: виконано кроків {result['current_step']}/{len(result['plan'])}")

    print("\n--- Симуляція перезапуску процесу (нове з'єднання до того ж файлу) ---")
    conn.close()
    conn2 = sqlite3.connect(DB_PATH, check_same_thread=False)
    saver2 = SqliteSaver(conn2)
    app_restored = build_graph(checkpointer=saver2)
    restored_state = app_restored.get_state(config)
    print(f"Відновлено stan: current_step={restored_state.values.get('current_step')}")
    print(f"Plan: {restored_state.values.get('plan')}")
    print(f"Results so far: {restored_state.values.get('results')}")

    print("\n--- Незалежність thread_id (нова сесія) ---")
    config2 = {"configurable": {"thread_id": "session-002"}}
    state2 = app_restored.get_state(config2)
    print(f"session-002 (ще не запускалась): values = {state2.values}")

    conn2.close()

    print("\n=== Демонстрація: відновлення після КРАХУ ПОСЕРЕД виконання ===")
    print("(на відміну від попередньої демонстрації, де 'перезапуск' стався")
    print("ПІСЛЯ успішного завершення — тут виключення штучно виникає ПІД ЧАС")
    print("виконання, після того як planner вже відпрацював і закешувався)")

    llm_for_crash_demo = ChatOpenAI(model=MODEL_NAME, temperature=0.1)
    _crash_flag = {"should_crash": True}

    def flaky_executor_node(state):
        if _crash_flag["should_crash"]:
            _crash_flag["should_crash"] = False
            raise RuntimeError("Симуляція краху процесу всередині executor_node")
        return executor_node(state, llm_for_crash_demo)

    crash_graph = StateGraph(PlanExecuteState)
    crash_graph.add_node("planner", lambda state: planner_node(state, llm_for_crash_demo))
    crash_graph.add_node("executor", flaky_executor_node)
    crash_graph.add_node("replanner", lambda state: replanner_node(state, llm_for_crash_demo))
    crash_graph.add_edge(START, "planner")
    crash_graph.add_edge("planner", "executor")
    crash_graph.add_edge("executor", "replanner")
    crash_graph.add_conditional_edges("replanner", should_continue, {"executor": "executor", "__end__": END})

    conn3 = sqlite3.connect(":memory:", check_same_thread=False)
    saver3 = SqliteSaver(conn3)
    crash_app = crash_graph.compile(checkpointer=saver3)
    crash_config = {"configurable": {"thread_id": "crash-demo"}}

    try:
        run(crash_app, "Перевір статус auth-service", config=crash_config)
    except RuntimeError as e:
        print(f"Очікуваний крах: {e}")

    checkpointed = crash_app.get_state(crash_config)
    print(f"Після краху: planner вже закешувався? goal={checkpointed.values.get('goal')!r}, "
          f"plan={checkpointed.values.get('plan')}")
    assert checkpointed.values.get("plan"), "planner мав закешуватися ДО того, як executor впав"

    print("-> Відновлення: invoke(None, config) — БЕЗ повторного виклику planner")
    recovered = crash_app.invoke(None, config=crash_config)
    print(f"Відновлено успішно: current_step={recovered.get('current_step')}, "
          f"results={recovered.get('results')}")
    assert recovered.get("goal") == checkpointed.values.get("goal"), (
        "goal не повинен був змінитися — planner НЕ мав перевиконуватися"
    )
    conn3.close()
