from fastapi import APIRouter, UploadFile, File
from pydantic import BaseModel
from typing import List, Optional
from database import chroma
from engine import processing_engine as service
import shutil
import os
from datetime import datetime

route = APIRouter()

class ImageModel(BaseModel):
    img_name: str
    path: str    
    description: str

class ImageResponse(BaseModel):
    id: str
    similarity: float
    metadata: dict

@route.post("/process",)
async def process(imageModel : ImageModel):    
    return service.process_image(imageModel.path)

@route.post("/upload-leaf")
async def upload_leaf(file: UploadFile = File(...), description: Optional[str] = None):
    """Upload e processamento de uma nova imagem de folha"""
    # Criar diretório de upload se não existir
    os.makedirs("image/uploads", exist_ok=True)
    
    # Salvar arquivo
    file_path = f"image/uploads/{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Processar imagem e salvar no banco
    result = service.process_image(file_path, save_to_db=True)
    
    if "error" in result:
        return {"error": result["error"]}
    
    return {
        "message": "Imagem processada com sucesso",
        "file_path": file_path,
        "features": result["features"][:5]  # Retorna apenas primeiros 5 valores para visualização
    }

@route.post("/search-similar")
async def search_similar(file: UploadFile = File(...), n_results: int = 5):
    """Busca imagens de folhas similares no banco"""
    # Salvar arquivo temporariamente
    temp_path = f"image/temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Processar imagem sem salvar no banco
    result = service.process_image(temp_path, save_to_db=False)
    
    if "error" in result:
        return {"error": result["error"]}
    
    # Buscar imagens similares
    similar = chroma.query_embedding(result["features"], n_results=n_results)
    
    # Limpar arquivo temporário
    os.remove(temp_path)
    
    # Formatar resultados
    results = []
    for i in range(len(similar["ids"][0])):
        results.append({
            "id": similar["ids"][0][i],
            "similarity": similar["distances"][0][i],
            "metadata": similar["metadatas"][0][i]
        })
    
    return results