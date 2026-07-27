from fastapi import FastAPI


app = FastAPI(title="Toox 3D")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}

