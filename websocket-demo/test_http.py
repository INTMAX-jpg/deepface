from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import uvicorn

app = FastAPI()

@app.get("/")
async def root():
    return HTMLResponse("""
    <!DOCTYPE html>
    <html>
    <head><title>测试</title></head>
    <body>
        <h1>✅ 服务器运行正常</h1>
        <p>当前时间: <script>document.write(new Date().toLocaleString())</script></p>
    </body>
    </html>
    """)

if __name__ == "__main__":
    print("=" * 50)
    print("测试服务器启动")
    print("请访问: http://localhost:8000")
    print("=" * 50)
    uvicorn.run(app, host="0.0.0.0", port=8000)
