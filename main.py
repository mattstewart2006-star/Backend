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
        # Convertir a WAV PCM 16 bits, mono, 44.1 kHz
        y, sr = librosa.load(io.BytesIO(file_bytes), sr=44100, mono=True)
        temp_path = "temp_liveness.wav"
        sf.write(temp_path, y, sr, subtype="PCM_16")

        # Extraer características
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfccs_scaled = np.mean(mfccs.T, axis=0)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        rolloff_scaled = np.mean(rolloff)
        features = np.hstack([mfccs_scaled, rolloff_scaled]).reshape(1, -1)

        # Predicción con el modelo
        prediccion = MODELO_LIVENESS.predict(features)[0]
        probabilidades = MODELO_LIVENESS.predict_proba(features)[0]
        return prediccion, probabilidades[prediccion]

    except Exception as e:
        print(f"⚠️ Error en liveness: {e}")
        return None, 0
    finally:
        if os.path.exists("temp_liveness.wav"):
            os.remove("temp_liveness.wav")


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
    return f"El usuario se llama {USER_DATA['nombre']} y su balance actual es de ${USER_DATA['balance']}."

@tool
def realizar_retiro(amount: float, destinatario: str) -> str:
    if amount <= 0:
        return "Error: El monto debe ser mayor a cero."
    if amount > USER_DATA["balance"]:
        return f"Fondos insuficientes. Tu saldo actual es {USER_DATA['balance']}."
    USER_DATA["balance"] -= amount
    return f"Transferencia de ${amount} a {destinatario} exitosa. Tu nuevo saldo es ${USER_DATA['balance']}."

@tool
def bank_fraud_check(amount: float, password: str = None) -> str:
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
            ("system", f"""Eres el asistente virtual de Banorte para {USER_DATA['nombre']}.

REGLAS DE RESPUESTA:
1. Si el usuario pide su saldo, usa `get_user_info`.
2. Si el usuario pide retirar o transferir dinero, identifica el monto y el destinatario y llama `realizar_retiro(amount, destinatario)`.
3. Si el monto es mayor a 1000, PRIMERO llama `bank_fraud_check(amount, password)`:
   - Si la respuesta es "Transferencia validada con contraseña", procede a `realizar_retiro`.
   - Si la respuesta es "Riesgo ALTO: Se requiere contraseña para continuar", NO ejecutes `realizar_retiro` y pide la contraseña.
   - Si la respuesta es "Contraseña incorrecta. Operación bloqueada", NO ejecutes `realizar_retiro`.
4. Nunca ejecutes `realizar_retiro` para montos mayores a 1000 sin validación exitosa de `bank_fraud_check`.
5. Siempre responde con confirmación explícita: "Transferencia de [monto] a [destinatario] exitosa. Tu nuevo saldo es [saldo]".
6. Si ejecutas una herramienta, tu respuesta final DEBE incluir los datos obtenidos de dicha herramienta.
7. Nunca inventes datos. Usa siempre las herramientas.
8. No respondas con una pregunta inmediatamente después de una acción financiera; primero da el informe de éxito.
RESTRICCIÓN TÉCNICA: No intentes realizar múltiples llamadas a funciones en un solo mensaje si una depende del éxito de la anterior para evitar errores de validación (Error 400).
"""),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ])

        self.tools = [get_user_info, bank_fraud_check, realizar_retiro]
        agent = create_tool_calling_agent(self.llm, self.tools, self.prompt)

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
                    text_res = None

                    if isinstance(full_response, dict):
                        text_res = full_response.get("output")
                        if not text_res:
                            steps = full_response.get("intermediate_steps")
                            if steps and isinstance(steps, list) and len(steps) > 0:
                                text_res = steps[-1][1]

                    if not text_res:
                        text_res = "Operación finalizada. ¿Deseas algo más?"

                except Exception as e:
                    print(f"🔴 Error crítico en Agente: {e}")
                    text_res = "Tuve un problema técnico al consultar tu información."

                tts = gTTS(text=str(text_res), lang='es')
                audio_filename = f"{uuid.uuid4()}.mp3"
                audio_path = os.path.join("static", audio_filename)
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
origins = ["https://mattstewart2006-star.github.io"
