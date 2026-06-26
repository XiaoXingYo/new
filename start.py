import uvicorn
import webbrowser
import threading
import time
import os
def open_browser():
    time.sleep(1.5)
    print("\n🚀 底层计算引擎已就绪！正在自动切出画板...")
    webbrowser.open("http://127.0.0.1:8000")
if __name__ == "__main__":
    threading.Thread(target=open_browser, daemon=True).start()
    uvicorn.run("web.app:app", host="0.0.0.0", port=8000, log_level="warning")