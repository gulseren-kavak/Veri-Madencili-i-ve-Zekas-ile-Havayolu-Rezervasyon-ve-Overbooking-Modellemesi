"""FastAPI entry point for the Overbooking Risk Management System."""

import uvicorn
from fastapi import FastAPI, Depends
from fastapi.responses import JSONResponse

from .config import RISK_THRESHOLD
from .router import router

app = FastAPI(title="Overbooking Risk API", version="0.1.0")
app.include_router(router)

@app.get("/health")
def health_check():
    return {"status": "ok", "threshold": RISK_THRESHOLD}

if __name__ == "__main__":
    uvicorn.run("backend.app.main:app", host="0.0.0.0", port=8000, reload=True)

