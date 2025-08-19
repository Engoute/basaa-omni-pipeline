from fastapi import FastAPI

app = FastAPI(title="Basaa Omni Pipeline")

@app.get("/healthz")
def healthz():
    return {"ok": True, "service": "basaa-omni", "version": "0.0.1"}
