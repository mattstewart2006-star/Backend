import os
import uvicorn
import uuid
import librosa
import numpy as np
import joblib
import io
import soundfile as sf
from fastapi import FastAPI, APIRouter, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from gtts import gTTS
from groq import Groq

# --- 1. CONFIGURACIÓN E INFORMACIÓN DE USUARIO ---
os.environ["GROQ_API_KEY"] = "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"
client = Groq()

USER_DATA = {
    "nombre": "Matthew",
    "balance": 15000.0,
    "password": "banorte seguro"  # Contraseña simulada
}

try:
    MODELO_LIVENESS = joblib.load("modelo_liveness.joblib")
    print("✅ Modelo antispoofing cargado.")
except:
    print("⚠️ ERROR: No se encontró 'modelo_liveness.joblib'.")

if not os.path.exists("static"):
    os.makedirs("static")


# --- 2. LÓGICA DE ANTISPOOFING ---
def verificar_liveness(file_bytes):
    try:
        audio_data = io.BytesIO(file_bytes)
        data, sr = sf.read(audio_data)
        if len(data.shape) > 1:
            data = np.mean(data, axis=1)

        mfccs = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=40)
        mfccs_scaled = np.mean(mfccs.T, axis=0)
        rolloff = librosa.feature.spectral_rolloff(y=data, sr=sr)
        rolloff_scaled = np.mean(rolloff)
        features = np.hstack([mfccs_scaled, rolloff_scaled]).reshape(1, -1)

        prediccion = MODELO_LIVENESS.predict(features)[0]
        probabilidades = MODELO_LIVENESS.predict_proba(features)[0]
        return prediccion, probabilidades[prediccion]
    except Exception:
        return None, 0


# --- 3. TOOLS (HERRAMIENTAS) ---
from langchain_groq import ChatGroq
from langchain.agents import create_tool_calling_agent
from langchain.agents.agent import AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import SQLChatMessageHistory



@tool
def get_user_info(query: str = None) -> str:
    """Consulta el nombre del usuario y su balance actual de cuenta."""
    return f"El usuario se llama {USER_DATA['nombre']} y su balance actual es de ${USER_DATA['balance']}."


@tool
def realizar_retiro(amount: float) -> str:
    """Realiza un retiro y actualiza el balance del usuario."""
    if amount <= 0:
        return "Monto inválido."
    if amount > USER_DATA["balance"]:
        return "Fondos insuficientes."
    USER_DATA["balance"] -= amount
    return f"Retiro exitoso de ${amount}. Nuevo balance: ${USER_DATA['balance']}."


@tool
def bank_fraud_check(amount: float, password: str = None) -> str:
    """Verifica si una transferencia es riesgosa y pide contraseña si es necesario."""
    if amount > 10000:
        if password is None:
            return "Riesgo ALTO: Se requiere contraseña para continuar."
        elif password == USER_DATA["password"]:
            return "Transferencia validada con contraseña."
        else:
            return "Contraseña incorrecta. Operación bloqueada."
    return "Riesgo BAJO: Operación segura."


# --- 4. AGENTE BANCARIO ---
def get_session_history(session_id: str):
    return SQLChatMessageHistory(session_id=session_id, connection_string="sqlite:///banco.db")


def limpiar_transcripcion(texto: str) -> str:
    """Normaliza texto para mejor interpretación."""
    texto = texto.strip().lower()
    texto = texto.replace("retirar", "retiro")
    texto = texto.replace("transferir", "transferencia")
    return texto


class BankingAgent:
    def __init__(self):
        self.router = APIRouter(prefix="/agent", tags=["AI Agent"])
        self.llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", f"Eres un asistente bancario para {USER_DATA['nombre']}. "
                       "Puedes dar información sobre el balance de cuenta, realizar retiros y validar fraudes. "
                       "Sé amable y conciso."),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ])

        self.tools = [get_user_info, bank_fraud_check, realizar_retiro]
        agent = create_tool_calling_agent(self.llm, self.tools, self.prompt)
        self.executor = AgentExecutor(agent=agent, tools=self.tools, verbose=True)

        self.agent_with_memory = RunnableWithMessageHistory(
            self.executor,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
        )

        @self.router.post("/chat-voice-to-voice")
        async def chat_voice(session_id: str = Form(...), file: UploadFile = File(...)):
            try:
                file_bytes = await file.read()
                es_real, confianza = verificar_liveness(file_bytes)

                if es_real == 0:
                    return {"error": "ACCESO DENEGADO", "agente_dijo": "Detección de spoofing activada."}

                # STT
                transcription = client.audio.transcriptions.create(
                    file=(file.filename, file_bytes),
                    model="whisper-large-v3",
                    response_format="text",
                )
                transcription = limpiar_transcripcion(transcription)

                # LLM Processing
                config = {"configurable": {"session_id": session_id}}
                response = await self.agent_with_memory.ainvoke({"input": transcription}, config=config)
                text_res = response.get("output") if response else None 
                if not text_res:
                    text_res = "Lo siento, no pude procesar tu solicitud."

                # TTS
                audio_filename = f"{uuid.uuid4()}.mp3"
                audio_path = os.path.join("static", audio_filename)
                tts = gTTS(text=text_res, lang='es')
                tts.save(audio_path)

                return {
                    "usuario_dijo": transcription,
                    "agente_dijo": text_res,
                    "url_audio": f"https://backend-1k3i.onrender.com/static/{audio_filename}"
                }
            except Exception as e:
                return {"error": "Error interno", "detalle": str(e)}


# --- 5. INICIALIZACIÓN ---
app = FastAPI(title="Agente Bancario Pro")
origins = ["https://mattstewart2006-star.github.io"]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["*"], allow_headers=["*"], allow_credentials=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

ai_agent = BankingAgent()
app.include_router(ai_agent.router)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
