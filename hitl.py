"""Human-in-the-Loop: interrupt() перед виконанням ризикового tool.

Розширює executor з plan_execute.py: якщо LLM вирішує викликати
РИЗИКОВИЙ tool (deploy_service — незворотна дія в production), граф
зупиняється через interrupt(), показує деталі дії та чекає рішення
людини (approve / reject / edit) перед продовженням.

Для HITL ОБОВ'ЯЗКОВО потрібен checkpointer — без нього стан між
interrupt() та resume не зберігається.
"""

import os
import sqlite3

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from knowledge import search_knowledge
from plan_execute import (
    DB_PATH,
    MODEL_NAME,
    PlanExecuteState,
    planner_node,
    replanner_node,
    should_continue,
)
from tools import ALL_TOOLS, RISKY_TOOLS

HITL_TOOLS = ALL_TOOLS + [search_knowledge] + RISKY_TOOLS
RISKY_TOOL_NAMES = {t.name for t in RISKY_TOOLS}

# Дозволені поля для редагування під час approve/edit, per risky tool.
# Без allow-list "редагування" саме по собі стає каналом для
# неконтрольованих змін — наприклад, оператор міг би непомітно
# підмінити service_name замість того, щоб лише скоригувати версію.
EDITABLE_ARGS = {
    "deploy_service": {"version"},
}


def _apply_edit(tool_name: str, args: dict, edited_args: dict) -> dict:
    """Застосувати лише ДОЗВОЛЕНІ правки з edited_args до args."""
    allowed = EDITABLE_ARGS.get(tool_name, set())
    filtered = {k: v for k, v in edited_args.items() if k in allowed}
    return {**args, **filtered}


def executor_with_hitl(state: PlanExecuteState, llm) -> dict:
    """Executor, що зупиняється через interrupt() перед ризиковим tool.

    ВАЖЛИВО: усі interrupt() для ризикових викликів обробляються в
    ОКРЕМОМУ першому проході, ДО того, як виконується БУДЬ-ЯКИЙ tool
    (включно з безпечними) з цієї ж відповіді LLM. Причина: LangGraph
    переграє весь вузол з початку при resume після interrupt(). Якби
    безпечний tool ішов у списку ПЕРЕД ризиковим і виконувався одразу,
    він виконався б ДВІЧІ — один раз до паузи, і ще раз після resume.
    Розділення на "спершу всі підтвердження, потім усі виконання"
    усуває цей ризик незалежно від порядку tool calls.
    """
    step_idx = state["current_step"]
    plan = state["plan"]

    if step_idx >= len(plan):
        return {"completed": True}

    current_step = plan[step_idx]
    llm_with_tools = llm.bind_tools(HITL_TOOLS)
    tools_by_name = {t.name: t for t in HITL_TOOLS}

    response = llm_with_tools.invoke(
        f"Виконай цей крок плану: {current_step}\n"
        f"Ціль: {state['goal']}\n"
        f"Попередні результати: {state['results']}"
    )

    result = response.content
    if getattr(response, "tool_calls", None):
        # Прохід 1: зібрати рішення людини для КОЖНОГО ризикового виклику,
        # нічого ще не виконуючи.
        approvals = {}
        for tc in response.tool_calls:
            if tc["name"] in RISKY_TOOL_NAMES:
                approvals[tc["id"]] = interrupt({
                    "action": tc["name"],
                    "args": dict(tc["args"]),
                    "message": (
                        f'Підтвердіть ризикову дію:\n'
                        f'Tool: {tc["name"]}\n'
                        f'Параметри: {tc["args"]}'
                    ),
                })

        # Прохід 2: усі рішення вже відомі — тепер безпечно виконувати.
        outputs = []
        for tc in response.tool_calls:
            args = dict(tc["args"])

            if tc["name"] in RISKY_TOOL_NAMES:
                approval = approvals[tc["id"]]
                if isinstance(approval, dict) and approval.get("approved"):
                    if approval.get("edited_args"):
                        args = _apply_edit(tc["name"], args, approval["edited_args"])
                    tool_fn = tools_by_name[tc["name"]]
                    tool_result = tool_fn.invoke(args)
                    outputs.append(f'{tc["name"]}: {tool_result}')
                else:
                    reason = approval.get("reason", "не вказано") if isinstance(approval, dict) else "не вказано"
                    outputs.append(f'{tc["name"]}: ВІДХИЛЕНО оператором (причина: {reason})')
            else:
                tool_fn = tools_by_name.get(tc["name"])
                if tool_fn:
                    tool_result = tool_fn.invoke(args)
                    outputs.append(f'{tc["name"]}: {tool_result}')

        result = " | ".join(outputs) if outputs else result

    return {
        "current_step": step_idx + 1,
        "results": [f"Крок {step_idx + 1} ({current_step}): {result}"],
        "messages": [AIMessage(content=f"Виконано крок {step_idx + 1}: {result}")],
    }


def build_graph_with_hitl(checkpointer, llm=None):
    """Побудувати Plan-and-Execute граф з HITL-executor.

    Args:
        checkpointer: LangGraph checkpointer (ОБОВ'ЯЗКОВИЙ для HITL).
        llm: Опційний готовий ChatOpenAI instance.

    Returns:
        Скомпільований LangGraph app.
    """
    llm = llm or ChatOpenAI(model=MODEL_NAME, temperature=0.1)

    graph = StateGraph(PlanExecuteState)
    graph.add_node("planner", lambda state: planner_node(state, llm))
    graph.add_node("executor", lambda state: executor_with_hitl(state, llm))
    graph.add_node("replanner", lambda state: replanner_node(state, llm))

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "executor")
    graph.add_edge("executor", "replanner")
    graph.add_conditional_edges("replanner", should_continue, {"executor": "executor", "__end__": END})

    return graph.compile(checkpointer=checkpointer)


def _initial_state(query: str) -> dict:
    return {
        "messages": [HumanMessage(content=query)],
        "goal": "",
        "plan": [],
        "current_step": 0,
        "results": [],
        "completed": False,
    }


if __name__ == "__main__":
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    saver = SqliteSaver(conn)
    app = build_graph_with_hitl(checkpointer=saver)

    print("=== Сценарій APPROVE ===")
    config = {"configurable": {"thread_id": "hitl-approve"}}
    result = app.invoke(
        _initial_state("Задеплой нову версію v2.15.0 сервісу api-gateway в продакшн"),
        config=config,
    )
    if "__interrupt__" in result:
        interrupt_data = result["__interrupt__"][0].value
        print("Граф зупинено на interrupt():")
        print(" ", interrupt_data["message"])
        print("-> Оператор підтверджує дію (approve)")
        result = app.invoke(Command(resume={"approved": True}), config=config)
    print("Результати:")
    for r in result["results"]:
        print(" -", r)

    print("\n=== Сценарій REJECT ===")
    config = {"configurable": {"thread_id": "hitl-reject"}}
    result = app.invoke(
        _initial_state("Задеплой нову версію v9.9.9 сервісу auth-service в продакшн"),
        config=config,
    )
    if "__interrupt__" in result:
        interrupt_data = result["__interrupt__"][0].value
        print("Граф зупинено на interrupt():")
        print(" ", interrupt_data["message"])
        print("-> Оператор відхиляє дію (reject)")
        result = app.invoke(
            Command(resume={"approved": False, "reason": "Поза вікном деплою"}),
            config=config,
        )
    print("Результати:")
    for r in result["results"]:
        print(" -", r)

    print("\n=== Сценарій EDIT (підтвердження зі зміненими параметрами) ===")
    config = {"configurable": {"thread_id": "hitl-edit"}}
    result = app.invoke(
        _initial_state("Задеплой версію v1.0.0 сервісу notification-service в продакшн"),
        config=config,
    )
    if "__interrupt__" in result:
        interrupt_data = result["__interrupt__"][0].value
        print("Граф зупинено на interrupt():")
        print(" ", interrupt_data["message"])
        # Навмисно намагаємось відредагувати і дозволене поле (version), і
        # НЕдозволене (service_name) — друге має бути проігноровано
        # allow-list'ом, інакше "редагування" саме по собі стало б каналом
        # для непомітної підміни цілі дії.
        edited = {"version": "v1.0.1", "service_name": "payment-service"}
        print(f"-> Оператор підтверджує, редагуючи version -> v1.0.1 ТА намагаючись підмінити service_name")
        result = app.invoke(
            Command(resume={"approved": True, "edited_args": edited}),
            config=config,
        )
    print("Результати:")
    for r in result["results"]:
        print(" -", r)
    print("(service_name мав ЗАЛИШИТИСЬ notification-service — підміна заблокована EDITABLE_ARGS allow-list)")

    conn.close()
