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
        y, sr = librosa.load(io.BytesIO(file_bytes), sr=44100, mono=True)
        temp_path = "temp_liveness.wav"
        sf.write(temp_path, y, sr, subtype="PCM_16")

        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfccs_scaled = np.mean(mfccs.T, axis=0)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        rolloff_scaled = np.mean(rolloff)
        features = np.hstack([mfccs_scaled, rolloff_scaled]).reshape(1, -1)

        prediccion = MODELO_LIVENESS.predict(features)[0]
        probabilidades = MODELO_LIVENESS.predict_proba(features)[0]
        return prediccion, probabilidades[prediccion]

    except Exception as e:
        print(f"⚠️ Error en liveness: {e}")
        return None, 0
    finally:
        if os.path.exists("temp_liveness.wav"):
            os.remove("temp_liveness.wav")

    if prediccion == 0:
        return {
            "status": "denegado",
            "mensaje": f"❌ Acceso denegado: intento de grabación detectado (confianza {confianza:.2f}%)"
        }
    else:
        return {
            "status": "permitido",
            "mensaje": f"✅ Voz real detectada (confianza {confianza:.2f}%)",
            "audio_wav": audio_data,  # audio completo para transcripción/agent
            "sr": sr
        }


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
    return f"Transferencia de ${amount} a {destinatario} exitosa. Tu nuevo saldo es ${USER_DATA['balance']}."

@tool
def bank_fraud_check(amount: float, password: str = None) -> str:
    """Verifica si una transferencia es riesgosa.
    Si el monto es mayor a 1000, requiere contraseña para continuar."""
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
    history = SQLChatMessageHistory(session_id=session_id, connection_string="sqlite:///banco.db")
    if len(history.messages) > 15:
        history.messages = history.messages[-15:]
    return history
    
def limpiar_transcripcion(texto: str) -> str:
    texto = unicodedata.normalize('NFKD', texto).encode('ascii', 'ignore').decode('ascii')
    texto = texto.strip().lower()
    texto = texto.replace("retirar", "retiro").replace("transferir", "transferencia")
    return texto
def contiene_palabra_clave(texto: str) -> bool:
    claves = ["saldo", "balance"]
    return any(palabra in texto.lower() for palabra in claves)

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
1. Si el usuario pide su saldo, balance o información de cuenta, SIEMPRE llama a `get_user_info` y usa exactamente el texto devuelto como respuesta final. 
   - Si la herramienta devuelve error, responde con ese error sin inventar datos.

2. Si el usuario pide retirar o transferir dinero:
   - Identifica claramente el monto y el destinatario.
   - Si alguno falta, responde: "Error: faltan parámetros para realizar la transferencia."
   - Solo entonces llama a `realizar_retiro(amount, destinatario)`.

3. Si el monto es mayor a 1000, PRIMERO llama a `bank_fraud_check(amount, password)`:
   - Si la respuesta es "Transferencia validada con contraseña", procede a `realizar_retiro`.
   - Si la respuesta es "Riesgo ALTO: Se requiere contraseña para continuar", NO ejecutes `realizar_retiro` y pide la contraseña.
   - Si la respuesta es "Contraseña incorrecta. Operación bloqueada", NO ejecutes `realizar_retiro`.

4. Nunca ejecutes `realizar_retiro` para montos mayores a 1000 sin validación exitosa de `bank_fraud_check`.
   - No combines llamadas en un mismo mensaje; espera siempre la validación antes de continuar.

5. Si ejecutas `realizar_retiro`, tu respuesta final DEBE ser exactamente el mensaje devuelto por la herramienta, sin modificarlo ni añadir frases adicionales.
   - Ejemplo correcto: "Transferencia de $500 a Juan exitosa. Tu nuevo saldo es $14500."

6. Si ejecutas cualquier herramienta, tu respuesta final DEBE incluir exactamente los datos obtenidos de esa herramienta.
   - Nunca inventes ni reformules resultados financieros.

7. Nunca inventes datos. Usa siempre las herramientas.
   - Si no hay salida (None), responde con: "No pude procesar tu solicitud. Intenta ser más específico."

8. No respondas con una pregunta inmediatamente después de una acción financiera; primero da el informe de éxito. 
   - Solo después puedes ofrecer ayuda adicional.

RESTRICCIÓN TÉCNICA:
- No intentes realizar múltiples llamadas a funciones en un solo mensaje si una depende del éxito de la anterior.
- Divide siempre en pasos: primero validación, luego ejecución.

EJEMPLOS DE COMPORTAMIENTO:
- Usuario: "Quiero transferir 500 pesos a Juan" → Llamar `realizar_retiro(amount=500, destinatario="Juan")`.
- Usuario: "Quiero transferir 2000 pesos a Ana" → Llamar `bank_fraud_check(amount=2000, password=None)` y pedir contraseña antes de continuar.
- Usuario: "¿Cuál es mi saldo?" → Llamar `get_user_info` y devolver exactamente el texto de la herramienta.
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
        # --- Transcripción con Groq ---
                transcription = client.audio.transcriptions.create(
                file=(file.filename, file_bytes),
                model="whisper-large-v3",
                response_format="text",)
                transcription = limpiar_transcripcion(transcription)
                if contiene_palabra_clave(transcription):
                    text_res = get_user_info.invoke({})
                else:
                    config = {"configurable": {"session_id": session_id}}
                    try:
                        full_response = await self.agent_with_memory.ainvoke({"input": transcription}, config=config)
                        text_res = None
                        if isinstance(full_response, dict):
                            text_res = full_response.get("output")
                            if not text_res:
                                steps = full_response.get("intermediate_steps")
                                if steps and isinstance(steps, list) and len(steps) > 0:
                                    # Log extra para depuración
                                    print(f"⚠️ Output vacío, usando intermediate_steps: {steps[-1]}")
                                    text_res = steps[-1][1]
                        if not text_res:
                            text_res = "Operación finalizada. ¿Deseas algo más?"
                    except Exception as e:
                        print(f"🔴 Error crítico en Agente: {e}")
                        text_res = "Tuve un problema técnico al consultar tu información."
                # --- Conversión a voz ---
                tts = gTTS(text=str(text_res), lang='es')
                audio_filename = f"{uuid.uuid4()}.mp3"
                audio_path = os.path.join("static", audio_filename)
                tts.save(audio_path)
                return {
                "usuario_dijo": transcription,
                "agente_dijo": text_res,
                "url_audio": f"https://backend-1k3i.onrender.com/static/{audio_filename}"}
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
