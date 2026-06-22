from __future__ import annotations

from pathlib import Path


APP_CN_NAME = "超透镜开源设计工作台"
APP_EN_NAME = "MetaLens Open Workbench"
APP_VERSION = "6.1.1"
APP_EDITION = "Open Source"
APP_DISPLAY_NAME = f"{APP_CN_NAME} / {APP_EN_NAME}"
APP_PUBLISHER = "MetaLens Open Source Contributors"
APP_PUBLISHER_EN = "MetaLens Open Source Contributors"
APP_COPYRIGHT = "Copyright (c) 2026 MetaLens Open Source Contributors. Released under the MIT License."
PRIMARY_RGB = (2, 82, 159)
PRIMARY_HEX = "#02529F"
ACCENT_HEX = "#0A6ED1"
APP_ID = "Open.MetaLensWorkbench"


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def asset_path(name: str) -> Path:
    return project_root() / "assets" / name
