from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from io import BytesIO
import pathlib
import shutil
from mosai_art_execute import main, make_image_square

# FastAPIインスタンスを作成
app = FastAPI()

# CORS設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return {"message": "Hello from FastAPI!"}


@app.post("/generate")
async def generate(
    title: str = Form(...),  # フォームデータで送られるタイトル
    image: UploadFile = File(...)  # アップロードされる画像ファイル
):
    try:
        upload_dir = pathlib.Path("images/temp")
        upload_dir.mkdir(parents=True, exist_ok=True)
        save_path = upload_dir / image.filename
        
        with save_path.open("wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
        make_image_square(save_path)
        try:
            path = main(False, save_path)
        except:
            print("failed to create mosaic art")
            return FileResponse(save_path, media_type="image/png")
    finally:
        image.file.close()

    # バイトデータをレスポンスとして返す
    return FileResponse(path, media_type="image/png")

# サーバーの起動
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)