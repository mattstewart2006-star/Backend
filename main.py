import os
import uvicorn
import uuid
import librosa
import numpy as np
import joblib
import io
import unicodedata
import soundfile as sf
from fastapi import FastAPI, APIRouter, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from gtts import gTTS
from groq import Groq

# --- 1. CONFIGURACIÓN E INFORMACIÓN DE USUARIO ---
client = Groq(api_key=os.environ["GROQ_API_KEY"])

USER_DATA = {
    "nombre": "Matthew",
    "balance": 15000.0,
    "password": "banorte seguro"
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


# --- 3. TOOLS (HERRAMIENTAS OPTIMIZADAS) ---
from langchain_groq import ChatGroq
from langchain.agents import create_tool_calling_agent
from langchain.agents.agent import AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import SQLChatMessageHistory

@tool
def get_user_info() -> str:
    """Útil para cuando el usuario pregunta por su saldo, balance o quién es él. 
    No requiere argumentos."""
    return f"El usuario se llama {USER_DATA['nombre']} y su balance actual es de ${USER_DATA['balance']}."

@tool
def realizar_retiro(amount: float, destinatario: str) -> str:
    """Ejecuta un retiro o transferencia. 
    IMPORTANTE: Solo llamar si tienes AMBOS: el monto (amount) y el nombre de la persona (destinatario)."""
    if amount <= 0:
        return "Error: El monto debe ser mayor a cero."
    if amount > USER_DATA["balance"]:
        return f"Fondos insuficientes. Tu saldo actual es {USER_DATA['balance']}."
    
    USER_DATA["balance"] -= amount
    return f"OK. Transferencia de ${amount} a {destinatario} realizada. Saldo restante: ${USER_DATA['balance']}."
@tool
def bank_fraud_check(amount: float, password: str = None) -> str:
    """Verifica si una transferencia es riesgosa."""
    if amount > 1000:
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
    # Quitar acentos y normalizar a ASCII para evitar errores de Unicode en las Tools
    texto = unicodedata.normalize('NFKD', texto).encode('ascii', 'ignore').decode('ascii')
    texto = texto.strip().lower()
    texto = texto.replace("retirar", "retiro").replace("transferir", "transferencia")
    return texto

class BankingAgent:
    def __init__(self):
        self.router = APIRouter(prefix="/agent", tags=["AI Agent"])
        self.llm = ChatGroq(
            model="llama-3.1-8b-instant", 
            temperature=0.1,
            model_kwargs={"tool_choice": "auto"}
        )

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", f"""Eres el asistente virtual de Banorte para el usuario {USER_DATA['nombre']}.
            Tu objetivo es ayudar con saldos y transferencias.    
            REGLAS CRÍTICAS:
            1. Si el usuario quiere transferir pero NO ha dicho a quién o cuánto, NO uses la herramienta 'realizar_retiro'. Pregúntale educadamente el dato que falta.
            2. Una vez que tengas el MONTO y el DESTINATARIO, usa la herramienta 'realizar_retiro' inmediatamente.
            3. No inventes nombres ni montos.
            4. Responde siempre en español de forma concisa."""),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ])

        self.tools = [get_user_info, bank_fraud_check, realizar_retiro]
        agent = create_tool_calling_agent(self.llm, self.tools, self.prompt)
        
        # handle_tool_errors=True es vital para que el Error 400 no rompa el flujo
        self.executor = AgentExecutor(
            agent=agent, 
            tools=self.tools, 
            verbose=True,
            handle_parsing_errors=True,
            handle_tool_errors=True
        )

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

                transcription = client.audio.transcriptions.create(
                    file=(file.filename, file_bytes),
                    model="whisper-large-v3",
                    response_format="text",
                )
                transcription = limpiar_transcripcion(transcription)

                config = {"configurable": {"session_id": session_id}}
                try:
                    full_response = await self.agent_with_memory.ainvoke({"input": transcription}, config=config)
                    text_res = full_response.get("output")
                    
                    # Si el agente hizo la tarea en terminal pero no generó texto final
                    if not text_res:
                        text_res = "Operación realizada con éxito. ¿Deseas algo más?"
                except Exception as e:
                    print(f"Error en Agente: {e}")
                    text_res = "Lo siento, tuve un problema técnico. ¿Podrías repetir?"

                audio_filename = f"{uuid.uuid4()}.mp3"
                audio_path = os.path.join("static", audio_filename)
                
                tts = gTTS(text=str(text_res), lang='es')
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
