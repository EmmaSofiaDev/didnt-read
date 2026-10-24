import os
import uvicorn

if __name__ == "__main__":
    port_str = os.getenv("PORT", "8000")
    try:
        port = int(port_str)
    except ValueError:
        port = 8000
    print(f"Starting Didn't Read on 0.0.0.0:{port}...")
    uvicorn.run("app:app", host="0.0.0.0", port=port)
