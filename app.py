from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from routes import lease_routes, billing_routes

app = FastAPI(
    title="QiYoga Lease OCR API",
    description="API for analyzing lease agreements using OCR",
    version="1.0.0",
)

# CORS配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 修复：路由注册时指定前缀
app.include_router(lease_routes.router, prefix="/api/lease", tags=["lease"])
app.include_router(billing_routes.router, prefix="/api/billing", tags=["billing"])


@app.get("/")
async def root():
    return {
        "message": "QiYoga Lease OCR API is running",
        "version": "1.0.0",
        "endpoints": {
            "analyze": "/api/lease/analyze",
            "health": "/api/lease/health",
            "docs": "/docs",
        },
    }


@app.get("/health")
async def health_check():
    return {"status": "ok"}


@app.on_event("startup")
async def startup_event():
    print("Lease OCR API started successfully")


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
