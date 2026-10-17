from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# CrownFlow 1.0 engineering-test geometry policy.  Import this before the API /
# service graph so patient-space sizing and safe anatomy refinement are the
# defaults used by all subsequently imported generation code.
from . import v1_test_geometry_overrides as _v1_test_geometry_overrides  # noqa: F401
# Last-mile tuning is deliberately layered on top of the established policy:
# stronger patient B/L fullness, one-side proximal closure and antagonist-owned
# occlusal relief. Import before API/services capture the patched functions.
from . import v1_final_geometry_guards as _v1_final_geometry_guards  # noqa: F401
# Some real abutment scans create a second tiny boundary loop *on the confirmed
# Margin itself* after the non-planar clip.  Recover only that proven derived
# sliver before services bind extract_abutment_support; genuine holes and
# unconfirmed screw channels remain blocking review conditions.
from . import v1_intaglio_support_guard as _v1_intaglio_support_guard  # noqa: F401
# FDI46 showed two remaining 1.0 defects: whole-crown B/L measurement could hide
# a narrow axial body, and a Margin-clipped intaglio could still end ~0.15 mm from
# the confirmed curve.  Patch only those final owners; keep proximal/occlusal and
# template anatomy unchanged.
from . import v1_margin_bl_completion as _v1_margin_bl_completion  # noqa: F401
# The intaglio geometry semantics changed above, so force a fresh preview instead
# of reusing v3 cache meshes that predate exact Margin-boundary ownership.
from . import v1_intaglio_cache_v4 as _v1_intaglio_cache_v4  # noqa: F401

from .api.crown import router as crown_router
from .api.importers import router as importers_router
from .api.margin import router as margin_router
from .api.routes import router
from .api.relocation import router as relocation_router
from .api.designer import router as designer_router
from .api.design_inputs import router as inputs_router
from .api.v1_inner_complete import router as v1_inner_complete_router
from .api.complete_generation import router as complete_generation_router
from .api.auto_fit import router as auto_fit_router
from .api.abutment_fast import router as abutment_fast_router
from .api.inner_preview import router as inner_preview_router
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
# CrownFlow 1.0 inner lifecycle owns these compatibility URLs first: the intaglio
# generated during AUTO stays attached after accept/sculpt and rebuild-cervical
# cannot turn a valid design inner into a 400. The legacy routers remain below as
# fallbacks for cases without a generated inner or with a true finalized crown.
app.include_router(v1_inner_complete_router)
app.include_router(complete_generation_router)
app.include_router(inner_preview_router)
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
