from fastapi import FastAPI
from routers import process_image

app = FastAPI()
app.include_router(process_image.route)