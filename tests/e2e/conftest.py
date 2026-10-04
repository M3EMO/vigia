"""Fixtures de la suite e2e: una plataforma real por sesión y un Chromium headless compartido."""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("playwright.sync_api")

from _e2e import Platform, ensure_accounts, free_port, run_calibration_flow, start_platform, stop_platform
from playwright.sync_api import Browser, BrowserContext, sync_playwright

# Módulos que pueden dejar la plataforma inutilizable si encuentran un defecto: van al final.
LAST_MODULES = ("test_e2e_streaming",)


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    items.sort(key=lambda item: getattr(item.module, "__name__", "") in LAST_MODULES)


@pytest.fixture(scope="session")
def platform(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Platform]:
    data_dir = Path(os.environ["VIGIA_E2E_DIR"]) if os.environ.get("VIGIA_E2E_DIR") else None
    if data_dir is None:
        data_dir = tmp_path_factory.mktemp("vigia_e2e")
    elif (data_dir / "vigia.db").exists():
        raise RuntimeError(f"VIGIA_E2E_DIR ya tiene una base: {data_dir}. Usá una carpeta vacía.")
    port = int(os.environ.get("VIGIA_E2E_PORT") or free_port())
    plat, proc, log = start_platform(data_dir, port)
    try:
        yield plat
    finally:
        stop_platform(proc, log)


@pytest.fixture(scope="session")
def browser() -> Iterator[Browser]:
    with sync_playwright() as pw:
        b = pw.chromium.launch(headless=True)
        try:
            yield b
        finally:
            b.close()


@pytest.fixture
def new_context(browser: Browser) -> Iterator[Callable[..., BrowserContext]]:
    """Fábrica de contextos aislados (un "dispositivo" cada uno); se cierran al terminar el test."""
    made: list[BrowserContext] = []

    def make(**kwargs: Any) -> BrowserContext:
        kwargs.setdefault("locale", "es-AR")
        ctx = browser.new_context(**kwargs)
        ctx.set_default_timeout(60_000)
        made.append(ctx)
        return ctx

    yield make
    for ctx in made:
        ctx.close()


@pytest.fixture
def accounts(platform: Platform, new_context: Callable[..., BrowserContext]) -> dict[str, str]:
    return ensure_accounts(platform, new_context())


@pytest.fixture
def calibrated(platform: Platform, new_context: Callable[..., BrowserContext]) -> dict[str, Any]:
    """Detector calibrado y activo para la org A (se calibra una sola vez por sesión, por el panel)."""
    return run_calibration_flow(platform, new_context())
