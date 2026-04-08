from __future__ import annotations

import os

import uvicorn

from pharmatrials_env.api.server import app


def main() -> None:
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "7860"))
    uvicorn.run(app, host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
