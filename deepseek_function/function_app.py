import azure.functions as func
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential
import PyPDF2
import docx
import io
import re

# Cargar variables de entorno desde Azure Function App
endpoint = os.getenv("AZURE_INFERENCE_SDK_ENDPOINT")
model_name = os.getenv("DEPLOYMENT_NAME")
key = os.getenv("AZURE_INFERENCE_SDK_KEY")

# Verificar credenciales
if not endpoint or not key or not model_name:
    raise ValueError("❌ Faltan credenciales de Azure AI Inference en las variables de entorno.")

# Crear cliente con la clave de Azure
client = ChatCompletionsClient(endpoint=endpoint, credential=AzureKeyCredential(key))

# Crear la instancia de FastAPI
app = FastAPI()

# Configurar CORS para la Static Web App
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, reemplaza con dominios específicos
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "✅ API de DeepSeek en Azure Functions está funcionando correctamente."}

def extract_text_from_file(file: UploadFile) -> str:
    """ Extrae el texto de un archivo PDF, DOCX o TXT. """
    try:
        ext = file.filename.lower().split(".")[-1]
        content = io.BytesIO(file.file.read())
        file.file.close()  # Cerrar el archivo después de leerlo

        if ext == "pdf":
            pdf_reader = PyPDF2.PdfReader(content)
            return " ".join([page.extract_text() or "" for page in pdf_reader.pages])

        elif ext == "docx":
            doc = docx.Document(content)
            return " ".join([para.text for para in doc.paragraphs])

        elif ext == "txt":
            return content.getvalue().decode("utf-8")

        else:
            raise HTTPException(status_code=400, detail="❌ Formato de archivo no soportado.")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"⚠️ Error al extraer texto: {str(e)}")

@app.post("/process-file/")
async def process_file(file: UploadFile = File(...)):
    """ Procesa un archivo, extrae su texto y lo resume usando Azure AI. """
    try:
        file_text = extract_text_from_file(file)

        system_message = SystemMessage(content="You are a helpful assistant. Summarize the text provided.")
        user_message = UserMessage(content=f"Resumen del texto: {file_text}")

        response = client.complete(
            messages=[system_message, user_message],
            model=model_name,
            max_tokens=500
        )

        summary = response.choices[0].message.content.strip()
        summary = re.sub(r"<think>.*?</think>", "", summary, flags=re.DOTALL).strip()

        if not summary:
            raise HTTPException(status_code=500, detail="❌ No se pudo generar un resumen válido.")

        return {"summary": summary}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"⚠️ Error al procesar el archivo: {str(e)}")

@app.post("/chat")
async def chat_with_ai(user_input: dict):
    """ Recibe un mensaje del usuario y devuelve la respuesta del modelo IA. """
    try:
        user_message = user_input.get("message", "").strip()
        if not user_message:
            raise HTTPException(status_code=400, detail="❌ El mensaje no puede estar vacío.")

        system_message = SystemMessage(content="You are a helpful AI assistant.")
        user_chat_message = UserMessage(content=user_message)

        response = client.complete(
            messages=[system_message, user_chat_message],
            model=model_name,
            max_tokens=500
        )

        return {"response": response.choices[0].message.content.strip()}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"⚠️ Error en el chat: {str(e)}")

# Adaptar FastAPI a Azure Functions (Debe llamarse 'main' para que Azure lo detecte)
main = func.AsgiFunctionApp(app)