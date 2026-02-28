from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from routes import lease_router, billing_router

app = FastAPI(
    title="TutorBox Lease OCR API",
    description="API for analyzing lease agreements using OCR",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(lease_router, prefix="/api/lease")
app.include_router(billing_router, prefix="/api/billing")


@app.get("/")
async def root():
    return {
        "message": "TutorBox Lease OCR API is running",
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
