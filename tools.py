"""Інструменти (tools) для DevOps-асистента.

Кожен tool має Pydantic v2 схему параметрів з Field(description=...)
для tool selection LLM та field_validator для захисту від некоректних
вхідних даних (галюцинованих параметрів).

`deploy_service` — навмисно РИЗИКОВИЙ tool (незворотна дія в продакшені),
використовується для демонстрації Human-in-the-Loop у hitl.py.
"""

import re

import hashlib

from langchain_core.tools import tool
from pydantic import BaseModel, Field, field_validator

VALID_ENVIRONMENTS = {"dev", "staging", "production"}

# --- Мок-дані реєстру сервісів ------------------------------------------
SERVICE_STATUS = {
    "api-gateway": {"dev": "healthy", "staging": "healthy", "production": "healthy"},
    "auth-service": {"dev": "healthy", "staging": "degraded", "production": "healthy"},
    "payment-service": {"dev": "healthy", "staging": "healthy", "production": "down"},
    "notification-service": {"dev": "unknown", "staging": "healthy", "production": "healthy"},
}

DEPLOYMENTS = {
    ("api-gateway", "production"): {
        "version": "v2.14.1", "deployed_at": "2026-08-26T09:12:00Z",
        "deployed_by": "ci-bot", "commit": "a1b2c3d",
    },
    ("auth-service", "staging"): {
        "version": "v1.9.0-rc2", "deployed_at": "2026-08-27T15:40:00Z",
        "deployed_by": "o.hal", "commit": "9f8e7d6",
    },
    ("payment-service", "production"): {
        "version": "v3.2.0", "deployed_at": "2026-08-20T11:05:00Z",
        "deployed_by": "ci-bot", "commit": "5c4b3a2",
    },
    ("notification-service", "production"): {
        "version": "v1.0.4", "deployed_at": "2026-08-25T18:30:00Z",
        "deployed_by": "ci-bot", "commit": "1122334",
    },
}


def _normalize(value: str) -> str:
    return value.strip().lower()


def _seeded_int(*parts: str, low: int, high: int) -> int:
    """Детермінований псевдо-випадковий int для стабільних мок-відповідей."""
    digest = hashlib.md5(":".join(parts).encode()).hexdigest()
    return low + (int(digest, 16) % (high - low + 1))


# === Tool 1: перевірка статусу сервісу ===================================
class CheckServiceStatusInput(BaseModel):
    """Параметри для перевірки статусу сервісу."""

    service_name: str = Field(
        description='Назва сервісу (наприклад, "api-gateway", "auth-service", '
        '"payment-service", "notification-service")'
    )
    environment: str = Field(
        default="production",
        description='Середовище: "dev", "staging" або "production" (за замовчуванням "production")',
    )

    @field_validator("service_name")
    @classmethod
    def service_name_not_empty(cls, v: str) -> str:
        v = _normalize(v)
        if len(v) < 3:
            raise ValueError("Назва сервісу повинна містити мінімум 3 символи")
        return v

    @field_validator("environment")
    @classmethod
    def environment_valid(cls, v: str) -> str:
        v = _normalize(v)
        if v not in VALID_ENVIRONMENTS:
            raise ValueError(
                f'Невідоме середовище "{v}". Дозволені значення: {sorted(VALID_ENVIRONMENTS)}'
            )
        return v


@tool(args_schema=CheckServiceStatusInput)
def check_service_status(service_name: str, environment: str = "production") -> str:
    """Отримати поточний статус сервісу (healthy / degraded / down / unknown).

    Використовуйте цей інструмент, коли потрібно дізнатися, чи працює
    конкретний сервіс у конкретному середовищі.

    Args:
        service_name: Назва сервісу.
        environment: Середовище розгортання.

    Returns:
        Рядок зі статусом сервісу.
    """
    statuses = SERVICE_STATUS.get(service_name)
    if statuses is None:
        return f'Сервіс "{service_name}" не знайдено в реєстрі.'
    status = statuses.get(environment, "unknown")
    return f'{service_name} ({environment}): {status}'


# === Tool 2: інформація про останній деплой ==============================
class GetDeploymentInfoInput(BaseModel):
    """Параметри для запиту інформації про деплой."""

    service_name: str = Field(description="Назва сервісу")
    environment: str = Field(
        default="production",
        description='Середовище: "dev", "staging" або "production"',
    )

    @field_validator("service_name")
    @classmethod
    def service_name_not_empty(cls, v: str) -> str:
        v = _normalize(v)
        if len(v) < 3:
            raise ValueError("Назва сервісу повинна містити мінімум 3 символи")
        return v

    @field_validator("environment")
    @classmethod
    def environment_valid(cls, v: str) -> str:
        v = _normalize(v)
        if v not in VALID_ENVIRONMENTS:
            raise ValueError(
                f'Невідоме середовище "{v}". Дозволені значення: {sorted(VALID_ENVIRONMENTS)}'
            )
        return v


@tool(args_schema=GetDeploymentInfoInput)
def get_deployment_info(service_name: str, environment: str = "production") -> str:
    """Отримати інформацію про останній деплой сервісу.

    Використовуйте цей інструмент, коли потрібно дізнатися версію,
    час деплою, автора або commit hash останнього релізу.

    Args:
        service_name: Назва сервісу.
        environment: Середовище розгортання.

    Returns:
        Рядок з інформацією про деплой.
    """
    info = DEPLOYMENTS.get((service_name, environment))
    if info is None:
        return f'Немає записів про деплой "{service_name}" у середовищі "{environment}".'
    return (
        f'{service_name} ({environment}): версія {info["version"]}, '
        f'задеплоєно {info["deployed_at"]} користувачем {info["deployed_by"]}, '
        f'commit {info["commit"]}'
    )


# === Tool 3: аналіз логів =================================================
class AnalyzeLogsInput(BaseModel):
    """Параметри для аналізу логів сервісу."""

    service_name: str = Field(description="Назва сервісу")
    environment: str = Field(
        default="production",
        description='Середовище: "dev", "staging" або "production"',
    )
    level: str = Field(
        default="error",
        description='Рівень логів для аналізу: "error", "warning" або "info"',
    )
    hours: int = Field(
        default=24,
        description="За скільки останніх годин аналізувати логи (1-168)",
    )

    @field_validator("service_name")
    @classmethod
    def service_name_not_empty(cls, v: str) -> str:
        v = _normalize(v)
        if len(v) < 3:
            raise ValueError("Назва сервісу повинна містити мінімум 3 символи")
        return v

    @field_validator("environment")
    @classmethod
    def environment_valid(cls, v: str) -> str:
        v = _normalize(v)
        if v not in VALID_ENVIRONMENTS:
            raise ValueError(
                f'Невідоме середовище "{v}". Дозволені значення: {sorted(VALID_ENVIRONMENTS)}'
            )
        return v

    @field_validator("level")
    @classmethod
    def level_valid(cls, v: str) -> str:
        v = _normalize(v)
        if v not in {"error", "warning", "info"}:
            raise ValueError('Рівень логів повинен бути "error", "warning" або "info"')
        return v

    @field_validator("hours")
    @classmethod
    def hours_in_range(cls, v: int) -> int:
        if not 1 <= v <= 168:
            raise ValueError("Кількість годин повинна бути від 1 до 168")
        return v


@tool(args_schema=AnalyzeLogsInput)
def analyze_logs(
    service_name: str,
    environment: str = "production",
    level: str = "error",
    hours: int = 24,
) -> str:
    """Проаналізувати логи сервісу за вказаний період та рівень.

    Використовуйте цей інструмент, коли потрібно дізнатися кількість
    помилок/попереджень сервісу за останній час.

    Args:
        service_name: Назва сервісу.
        environment: Середовище розгортання.
        level: Рівень логів (error / warning / info).
        hours: Кількість останніх годин для аналізу.

    Returns:
        Рядок з підсумком аналізу логів.
    """
    count = _seeded_int(service_name, environment, level, str(hours), low=0, high=hours * 2)
    sample_messages = {
        "error": "Connection timeout to database replica",
        "warning": "Response latency above 500ms threshold",
        "info": "Health check passed",
    }
    return (
        f'{service_name} ({environment}), останні {hours} год.: '
        f'{count} записів рівня "{level}". Приклад: "{sample_messages[level]}"'
    )


# === Tool 4: розрахунок uptime ============================================
class CalculateUptimeInput(BaseModel):
    """Параметри для розрахунку uptime сервісу."""

    service_name: str = Field(description="Назва сервісу")
    days: int = Field(
        default=7,
        description="За скільки останніх днів розрахувати uptime (1-90)",
    )

    @field_validator("service_name")
    @classmethod
    def service_name_not_empty(cls, v: str) -> str:
        v = _normalize(v)
        if len(v) < 3:
            raise ValueError("Назва сервісу повинна містити мінімум 3 символи")
        return v

    @field_validator("days")
    @classmethod
    def days_in_range(cls, v: int) -> int:
        if not 1 <= v <= 90:
            raise ValueError("Кількість днів повинна бути від 1 до 90")
        return v


@tool(args_schema=CalculateUptimeInput)
def calculate_uptime(service_name: str, days: int = 7) -> str:
    """Розрахувати відсоток uptime сервісу за останні N днів.

    Використовуйте цей інструмент, коли потрібно оцінити надійність
    сервісу за певний період.

    Args:
        service_name: Назва сервісу.
        days: Кількість останніх днів для розрахунку.

    Returns:
        Рядок з відсотком uptime.
    """
    incident_minutes = _seeded_int(service_name, str(days), low=0, high=days * 8)
    total_minutes = days * 24 * 60
    uptime_pct = round(100 * (1 - incident_minutes / total_minutes), 3)
    return (
        f'{service_name}: uptime за останні {days} дн. = {uptime_pct}% '
        f'(простій: {incident_minutes} хв.)'
    )


# === Ризиковий tool: деплой сервісу =======================================
class DeployServiceInput(BaseModel):
    """Параметри для деплою нової версії сервісу."""

    service_name: str = Field(description="Назва сервісу для деплою")
    environment: str = Field(
        default="production",
        description='Середовище: "dev", "staging" або "production"',
    )
    version: str = Field(description='Версія для деплою у форматі "vX.Y.Z" (наприклад, "v2.15.0")')

    @field_validator("service_name")
    @classmethod
    def service_name_not_empty(cls, v: str) -> str:
        v = _normalize(v)
        if len(v) < 3:
            raise ValueError("Назва сервісу повинна містити мінімум 3 символи")
        return v

    @field_validator("environment")
    @classmethod
    def environment_valid(cls, v: str) -> str:
        v = _normalize(v)
        if v not in VALID_ENVIRONMENTS:
            raise ValueError(
                f'Невідоме середовище "{v}". Дозволені значення: {sorted(VALID_ENVIRONMENTS)}'
            )
        return v

    @field_validator("version")
    @classmethod
    def version_format_valid(cls, v: str) -> str:
        v = v.strip()
        if not re.match(r"^v\d+\.\d+\.\d+", v):
            raise ValueError('Версія повинна бути у форматі "vX.Y.Z" (наприклад, "v2.15.0")')
        return v


@tool(args_schema=DeployServiceInput)
def deploy_service(service_name: str, environment: str = "production", version: str = "") -> str:
    """Задеплоїти нову версію сервісу (РИЗИКОВА, НЕЗВОРОТНА ДІЯ).

    Використовуйте цей інструмент тільки коли користувач явно попросив
    задеплоїти/розгорнути нову версію сервісу. Ця дія впливає на реальний
    сервіс і потребує підтвердження людини перед виконанням, особливо
    для середовища "production".

    Args:
        service_name: Назва сервісу.
        environment: Середовище розгортання.
        version: Версія для деплою.

    Returns:
        Рядок з підтвердженням деплою.
    """
    DEPLOYMENTS[(service_name, environment)] = {
        "version": version,
        "deployed_at": "щойно",
        "deployed_by": "agent",
        "commit": "n/a",
    }
    return f'Задеплоєно {service_name} ({environment}) версія {version}.'


ALL_TOOLS = [check_service_status, get_deployment_info, analyze_logs, calculate_uptime]
RISKY_TOOLS = [deploy_service]


# === Демонстрація / unit-тести =============================================
if __name__ == "__main__":
    print("=== Валідація Pydantic (мають впасти) ===")

    try:
        CheckServiceStatusInput(service_name="ab", environment="production")
    except Exception as e:
        print(f"OK, очікувана помилка (короткий service_name): {e}\n")

    try:
        CheckServiceStatusInput(service_name="api-gateway", environment="prod")
    except Exception as e:
        print(f"OK, очікувана помилка (невірне environment): {e}\n")

    try:
        AnalyzeLogsInput(service_name="auth-service", hours=500)
    except Exception as e:
        print(f"OK, очікувана помилка (hours поза діапазоном): {e}\n")

    try:
        CalculateUptimeInput(service_name="payment-service", days=0)
    except Exception as e:
        print(f"OK, очікувана помилка (days поза діапазоном): {e}\n")

    try:
        DeployServiceInput(service_name="api-gateway", version="2.15.0")
    except Exception as e:
        print(f"OK, очікувана помилка (невірний формат версії): {e}\n")

    print("=== Успішні виклики tools ===")
    print(check_service_status.invoke({"service_name": "payment-service", "environment": "production"}))
    print(get_deployment_info.invoke({"service_name": "api-gateway", "environment": "production"}))
    print(analyze_logs.invoke({"service_name": "auth-service", "environment": "staging", "level": "warning", "hours": 12}))
    print(calculate_uptime.invoke({"service_name": "notification-service", "days": 30}))

    print("\n=== Нормалізація вхідних даних (регістр/пробіли) ===")
    print(check_service_status.invoke({"service_name": "  API-Gateway ", "environment": "PRODUCTION"}))

    print("\n=== Ризиковий tool (deploy_service) ===")
    print(deploy_service.invoke({"service_name": "api-gateway", "environment": "production", "version": "v2.15.0"}))
