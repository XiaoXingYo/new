# 文件：start.py
import uvicorn
import webbrowser
import threading
import time
import os


def open_browser():
    """旁路线程：等待服务挂载后，通过底层 Interop 唤起 Windows 浏览器"""
    time.sleep(1.5)  # 留出 1.5 秒给 uvicorn 绑定端口
    print("\n🚀 底层计算引擎已就绪！正在自动切出画板...")
    # 0.0.0.0 允许宿主机通过 localhost 访问 WSL 内部服务
    webbrowser.open("http://127.0.0.1:8000")


if __name__ == "__main__":
    # 1. 启动看门狗线程
    threading.Thread(target=open_browser, daemon=True).start()

    # 2. 启动核心服务 (关闭多余日志，保持终端整洁)
    print("🧠 正在唤醒 RTX 5060 与 OCR 双轨网络...")
    uvicorn.run("web.app:app", host="0.0.0.0", port=8000, log_level="warning")