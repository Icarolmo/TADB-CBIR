from fastapi import APIRouter
from pydantic import BaseModel
from database import chroma
from engine import processing_engine as service

route = APIRouter()

class ImageModel(BaseModel):
    img_name: str
    path: str    
    description: str

@route.post("/process",)
async def process(imageModel : ImageModel):    
    return service.process_image(imageModel.path)