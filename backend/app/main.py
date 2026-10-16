from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# CrownFlow 1.0 engineering-test geometry policy.  Import this before the API /
# service graph so patient-space sizing and safe anatomy refinement are the
# defaults used by all subsequently imported generation code.
from . import v1_test_geometry_overrides as _v1_test_geometry_overrides  # noqa: F401

from .api.crown import router as crown_router
from .api.importers import router as importers_router
from .api.margin import router as margin_router
from .api.routes import router
from .api.relocation import router as relocation_router
from .api.designer import router as designer_router
from .api.design_inputs import router as inputs_router
from .api.complete_generation import router as complete_generation_router
from .api.auto_fit import router as auto_fit_router
from .api.abutment_fast import router as abutment_fast_router
from .db import init_db


@asynccontextmanager
async def lifespan(_app: FastAPI):
    init_db()
    yield


# 建表必须在 import 时完成，不能只依赖 lifespan：
# 任何不走 lifespan 的调用方（TestClient 直接构造、脚本化导入）都会面对"没有表"的数据库。
init_db()

app = FastAPI(title="WelooAI牙冠自动设计 API", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Complete Crown 1.0 owns the accept + rebuild-cervical compatibility URLs.
# It must be registered before the legacy routes with the same paths so an
# already-generated intaglio/final is never discarded or rebuilt a second time.
app.include_router(complete_generation_router)
app.include_router(router)
# CAD-1 工程测试允许同病例/同扫描重复定位。该 router 必须先于 designer_router，
# 以保持原 URL 不变同时去掉旧的"已有 Margin/后续设计就禁止重定位"门禁。
app.include_router(relocation_router)
app.include_router(designer_router)
app.include_router(inputs_router)
app.include_router(auto_fit_router)
app.include_router(abutment_fast_router)
app.include_router(margin_router)
app.include_router(importers_router)
app.include_router(crown_router)
from .api.sculpt import router as sculpt_router
app.include_router(sculpt_router)
frontend_dist = Path(__file__).resolve().parents[2] / "frontend" / "dist"
if frontend_dist.is_dir():
    app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="frontend")
