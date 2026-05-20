import os
import asyncio
import json
import unicodedata
import tempfile
import shutil
import pandas as pd
from datetime import datetime
from typing import Optional, Dict, Any, List

import gradio as gr
from PIL import Image
import pytesseract
from pypdf import PdfReader
import yaml

from notion_client import AsyncClient
from notion_client.errors import APIResponseError, RequestTimeoutError
import google.generativeai as genai
from google.generativeai import types
from google.generativeai.types import Content, Part

# Importaciones del framework Google ADK para gestión del agente autónomo
from google.adk.agents.llm_agent import Agent
from google.adk.tools import FunctionTool
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from dotenv import load_dotenv

# Carga de variables de entorno desde el archivo .env
load_dotenv()

# Configuración de credenciales de acceso para las API externas
NOTION_API_KEY = os.getenv("NOTION_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Identificadores de las bases de datos de Notion
NOTION_RULES_DB_ID = os.getenv("NOTION_RULES_DB_ID")
NOTION_CASES_DB_ID = os.getenv("NOTION_CASES_DB_ID")

# Validación de la presencia de las credenciales requeridas
if not NOTION_API_KEY or not GOOGLE_API_KEY:
    raise ValueError("Faltan credenciales en el entorno (Notion o Google).")

# Inicialización de clientes (Notion y Google Gemini)
notion = AsyncClient(auth=NOTION_API_KEY, notion_version="2025-09-03")
genai.configure(api_key=GOOGLE_API_KEY)

# Funciones auxiliares para la construcción de propiedades de Notion
def _title(text: Optional[str]) -> Dict[str, Any]: 
    return {"title": [{"type": "text", "text": {"content": text or ""}}]}

def _rich_text(text: Optional[str]) -> Dict[str, Any]:
    if not text: return {"rich_text": []}
    return {"rich_text": [{"type": "text", "text": {"content": text[:1900]}}]}

def _date(start: Optional[str], end: Optional[str] = None) -> Dict[str, Any]:
    if not start: return {"date": None}
    payload = {"start": start}
    if end: payload["end"] = end
    return {"date": payload}

def _status(name: Optional[str]) -> Dict[str, Any]: 
    return {"status": {"name": name}} if name else {"status": None}

def _files_external(urls: Optional[List[str]]) -> Dict[str, Any]:
    if not urls: return {"files": []}
    return {"files": [{"type": "external", "name": (u.split("/")[-1])[:100], "external": {"url": u}} for u in urls]}

# Normalización de texto para comparaciones semánticas consistentes
def _norm(s: str) -> str:
    if not s: return ""
    s = unicodedata.normalize("NFD", str(s))
    return "".join(c for c in s if unicodedata.category(c) != "Mn").lower().strip()

# Cálculo de la diferencia en días naturales entre dos fechas
def _dias_entre(inicio: Any, fin: Any) -> Optional[int]:
    try:
        d1 = pd.to_datetime(inicio).date()
        d2 = pd.to_datetime(fin).date()
        return (d2 - d1).days
    except Exception:
        return None

# Manejo de reintentos con espera exponencial para solicitudes a la API de Notion
async def notion_retry(coro_factory, max_attempts: int = 4, base_delay: float = 1.5):
    for attempt in range(1, max_attempts + 1):
        try: return await coro_factory()
        except (APIResponseError, RequestTimeoutError) as e:
            status = getattr(e, "status", None)
            if status in (429, 502, 503, 504) and attempt < max_attempts:
                await asyncio.sleep(base_delay * (2 ** (attempt - 1)))
                continue
            raise

# Caché de identificadores de orígenes de datos (Data Sources) para optimizar consultas
_DS_CACHE: Dict[str, str] = {}
async def _get_data_source_id(database_id: str) -> str:
    if database_id in _DS_CACHE: return _DS_CACHE[database_id]
    db = await notion_retry(lambda: notion.databases.retrieve(database_id=database_id))
    ds_id = db["data_sources"][0]["id"]
    _DS_CACHE[database_id] = ds_id
    return ds_id

# Módulos de extracción de datos soportados (PDF, OCR e Inteligencia Estructurada en Excel)
def extract_text_from_pdf(path: str) -> str:
    try:
        reader = PdfReader(path)
        return "".join((page.extract_text() or "") for page in reader.pages).strip()
    except Exception as e:
        print(f"Error procesando PDF {path}: {e}")
        return ""

def extract_text_from_image(path: str, lang: str = "spa") -> str:
    try:
        return pytesseract.image_to_string(Image.open(path), lang=lang).strip()
    except Exception as e:
        print(f"Error procesando OCR en {path}: {e}")
        return ""

# Selección automatizada del motor de extracción basado en la firma del documento
def extract_text_auto(path: Optional[str]) -> str:
    if not path or not os.path.exists(path): return ""
    low = path.lower()
    if low.endswith(".pdf"):
        return extract_text_from_pdf(path)
    if low.endswith((".png", ".jpg", ".jpeg", ".tiff", ".bmp")):
        return extract_text_from_image(path)
    return ""

# Extracción estructurada de diccionarios desde archivos corporativos Excel/CSV
def parsear_archivos_excel(ruta_informe: str, ruta_poliza: str) -> Dict[str, Any]:
    try:
        df_inf = pd.read_csv(ruta_informe) if ruta_informe.endswith('.csv') else pd.read_excel(ruta_informe)
        df_pol = pd.read_csv(ruta_poliza) if ruta_poliza.endswith('.csv') else pd.read_excel(ruta_poliza)
        
        inf_dict = df_inf.set_index("Celda / Campo")["Valor de Ejemplo"].to_dict()
        pol_dict = df_pol.set_index("Celda / Campo")["Valor de Ejemplo"].to_dict()
        return {"informe": inf_dict, "poliza": pol_dict, "error": None}
    except Exception as e:
        return {"error": str(e)}

# Operaciones transaccionales y de persistencia en la base de datos de Notion
DECISIONES_VALIDAS = {"Pendiente", "Faltan documentos", "Preaprobado", "No cubierto"}

# Sincronización de reglas maestras de cobertura desde Notion
async def fetch_notion_rules() -> List[Dict[str, Any]]:
    ds_id = await _get_data_source_id(NOTION_RULES_DB_ID)
    rules, cursor = [], None
    while True:
        resp = await notion_retry(lambda: notion.data_sources.query(data_source_id=ds_id, start_cursor=cursor))
        for page in resp["results"]:
            p = page["properties"]
            procedimiento = p.get("Procedimiento", {}).get("title", [{}])[0].get("plain_text") if p.get("Procedimiento", {}).get("title") else None
            cubierto = p.get("Cubierto", {}).get("select", {}).get("name") if p.get("Cubierto", {}).get("select") else None
            carencia = p.get("Carencia (días)", {}).get("number")
            exclusiones = "".join(t["plain_text"] for t in p.get("Exclusiones", {}).get("rich_text", [])) or None
            requisitos = [t["name"] for t in p.get("Requisitos documentales", {}).get("multi_select", [])]
            rules.append({"procedimiento": procedimiento, "cubierto": cubierto, "carencia_dias": carencia, "exclusiones": exclusiones, "requisitos": requisitos})
        if not resp["has_more"]: break
        cursor = resp["next_cursor"]
    return rules

# Creación de registro de siniestro en el dashboard administrativo
async def add_notion_case(caso: str, paciente: Optional[str] = None, procedimiento: Optional[str] = None, aseguradora_plan: Optional[str] = None, decision: str = "Pendiente", razones: Optional[str] = None, texto_extraido: Optional[str] = None) -> Optional[str]:
    properties = {
        "Caso": _title(caso), "Paciente": _rich_text(paciente), "Procedimiento solicitado": _rich_text(procedimiento),
        "Aseguradora / Plan": _rich_text(aseguradora_plan), "Decisión": _status(decision), 
        "Razones (resumen)": _rich_text(razones), "Texto extraído (fallback)": _rich_text(texto_extraido)
    }
    try:
        page = await notion_retry(lambda: notion.pages.create(parent={"database_id": NOTION_CASES_DB_ID}, properties=properties))
        return page["id"]
    except Exception as e:
        print(f"Error creando registro en Notion: {e}")
        return None

# Búsqueda de casos existentes por identificador de ticket
async def find_case_by_title(caso_title: str) -> Optional[str]:
    ds_id = await _get_data_source_id(NOTION_CASES_DB_ID)
    res = await notion_retry(lambda: notion.data_sources.query(data_source_id=ds_id, filter={"property": "Caso", "title": {"equals": caso_title}}, page_size=1))
    return res["results"][0]["id"] if res["results"] else None

# Motor de evaluación: Análisis de reglas contractuales directas (filtros duros)
def evaluar_reglas_duras(datos_inf: Dict, datos_pol: Dict) -> Optional[Dict[str, str]]:
    id_hosp = str(datos_inf.get("Paciente_Identificacion", "")).strip()
    id_pol = str(datos_pol.get("Titular_Identificacion", "")).strip()
    
    if id_hosp and id_pol and id_hosp != id_pol:
        return {"decision": "No cubierto", "razones": "Rechazo de Seguridad: La identidad del paciente no coincide con el titular."}
    
    try: edad = int(float(datos_inf.get("Paciente_Edad", 0)))
    except (ValueError, TypeError): edad = -1
    
    if 0 <= edad < 3 or edad > 65:
        return {"decision": "No cubierto", "razones": "Rechazo Contractual: Edad del paciente fuera del rango admitido (3-65 años)."}
    
    f_acc = datos_inf.get("Fecha_Accidente")
    f_adm = datos_inf.get("Fecha_Admision")
    if _dias_entre(f_acc, f_adm) and _dias_entre(f_acc, f_adm) > 30:
        return {"decision": "No cubierto", "razones": "Rechazo Contractual: Notificación extemporánea del evento (mayor a 30 días)."}
    
    estado_prima = _norm(str(datos_pol.get("Estado_Prima", "")))
    if estado_prima and "pagado" not in estado_prima and "al dia" not in estado_prima:
        return {"decision": "No cubierto", "razones": "Rechazo Administrativo: Contrato suspendido por falta de pago."}
    
    return None

# Motor de evaluación: Inferencia semántica y análisis causal utilizando Gemini
def auditoria_gemini(datos: Dict, reglas: List[Dict]) -> Dict[str, str]:
    client = genai.Client()
    contexto = "\n".join([f"Regla: {r['procedimiento']} | Cobertura: {r['cubierto']} | Exclusiones: {r['exclusiones']}" for r in reglas])
    prompt = f"Eres un auditor. Evalúa el caso. REGLAS:\n{contexto}\nDATOS: Diag: {datos.get('Diagnostico_Clinico')} | Proc: {datos.get('Procedimiento_Solicitado')}. Responde en JSON con 'decision' y 'razones'."
    
    try:
        resp = client.models.generate_content(
            model='gemini-2.5-flash', 
            contents=prompt, 
            config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0.1)
        )
        return json.loads(resp.text.strip())
    except Exception as e:
        return {"decision": "Pendiente", "razones": f"Error de evaluación semántica: {e}"}

# Orquestador del servicio unificado de procesamiento
async def process_new_case_request(caso: str, paciente: str, aseguradora_plan: str, procedimiento: str, informe_path: Optional[str] = None, poliza_path: Optional[str] = None) -> str:
    print(f"Iniciando evaluación de caso: {caso}")
    texto_extraido = ""
    datos_estructurados = None
    
    # Análisis dinámico del formato del archivo de entrada
    if informe_path and poliza_path and informe_path.lower().endswith(('.xls', '.xlsx', '.csv')):
        parseo = parsear_archivos_excel(informe_path, poliza_path)
        if parseo.get("error"): 
            return f"Error de lectura de estructura de datos: {parseo['error']}"
        datos_estructurados = parseo
        texto_extraido = f"Diagnóstico estructurado: {parseo['informe'].get('Diagnostico_Clinico')}"
    else:
        # Fallback a extracción OCR y PDF
        texto_extraido = extract_text_auto(informe_path) + "\n" + extract_text_auto(poliza_path)
        
    page_id = await find_case_by_title(caso)
    if not page_id:
        page_id = await add_notion_case(caso=caso, paciente=paciente, aseguradora_plan=aseguradora_plan, procedimiento=procedimiento)
    
    # Flujo secuencial de validaciones
    decision_final = None
    
    if datos_estructurados:
        decision_final = evaluar_reglas_duras(datos_estructurados["informe"], datos_estructurados["poliza"])
        
    if not decision_final:
        reglas = await fetch_notion_rules()
        if datos_estructurados:
            decision_final = auditoria_gemini(datos_estructurados["informe"], reglas)
        else:
            decision_final = {"decision": "Pendiente", "razones": "Revisión manual o por agente avanzado requerida. Documentos no estructurados."}

    # Registro de la decisión computada
    properties = {
        "Decisión": _status(decision_final["decision"]),
        "Razones (resumen)": _rich_text(decision_final["razones"]),
        "Texto extraído (fallback)": _rich_text(texto_extraido)
    }
    await notion_retry(lambda: notion.pages.update(page_id=page_id, properties=properties))
    
    return f"Proceso Finalizado.\nDecisión: {decision_final['decision']}\nMotivo: {decision_final['razones']}"

# Definición del entorno y orquestación del agente ADK
config_yaml = """
name: Agente_Pre_Autorizacion
model: gemini-2.5-flash
description: Agente para procesar pre-autorizaciones quirúrgicas.
instruction: |
  Eres un agente estricto y profesional encargado de auditoría médica.
  1. Solicita titulo del caso, paciente, aseguradora y procedimiento.
  2. Utiliza process_new_case_request para el registro en el sistema core.
  3. Proporciona respuestas concisas y directas. No utilices emojis ni lenguaje coloquial.
"""
config = yaml.safe_load(config_yaml)
pre_auth_tool = FunctionTool(func=process_new_case_request)

root_agent = Agent(
    model=config["model"], name=config["name"], description=config["description"],
    instruction=config["instruction"], tools=[pre_auth_tool]
)

session_service = InMemorySessionService()
runner = Runner(agent=root_agent, app_name="preauth", session_service=session_service)

# Configuración del frontend interactivo mediante Gradio
UPLOAD_DIR = tempfile.mkdtemp()

# Función segura para el procesamiento de archivos subidos al servidor
def _save_upload(file_obj) -> Optional[str]:
    if not file_obj: 
        return None
    file_path = file_obj if isinstance(file_obj, str) else getattr(file_obj, 'name', str(file_obj))
    if not os.path.exists(file_path):
        return None
    dest = os.path.join(UPLOAD_DIR, os.path.basename(file_path))
    shutil.copy(file_path, dest)
    return dest

# Función adaptadora para el flujo automatizado principal (Prioridad en carga de archivos)
async def submit_case_automatico(file_hosp, file_seg):
    if not file_hosp or not file_seg:
        return "Error crítico: Se requiere proporcionar ambos documentos (Hospital y Aseguradora) para proceder."
    
    path_hosp = _save_upload(file_hosp)
    path_seg = _save_upload(file_seg)
    
    # Inferencia de metadatos por defecto
    paciente_str = "Paciente Autogenerado"
    procedimiento_str = "Procedimiento por Evaluar"
    aseguradora_str = "Aseguradora del Sur"
    
    # Extracción de metadatos si los archivos poseen una estructura compatible
    if path_hosp and path_hosp.lower().endswith(('.xls', '.xlsx', '.csv')):
        parseo = parsear_archivos_excel(path_hosp, path_seg)
        if not parseo.get("error"):
            paciente_str = str(parseo["informe"].get("Paciente_Nombre", paciente_str))
            procedimiento_str = str(parseo["informe"].get("Procedimiento_Solicitado", procedimiento_str))
    
    titulo = f"C-{int(datetime.now().timestamp()) % 1000:03d} — {procedimiento_str[:15]} — {paciente_str.split()[-1]}"
    
    resultado = await process_new_case_request(
        caso=titulo,
        paciente=paciente_str,
        aseguradora_plan=aseguradora_str,
        procedimiento=procedimiento_str,
        informe_path=path_hosp,
        poliza_path=path_seg
    )
    return f"Expediente Ticket: {titulo}\n{resultado}"

# Función adaptadora para el flujo asistido con captura explícita
async def submit_case_ui(paciente, aseguradora, procedimiento, file_hosp, file_seg):
    if not paciente or not aseguradora or not procedimiento:
        return "Error: Faltan campos obligatorios en el formulario de registro."
    
    titulo = f"C-{int(datetime.now().timestamp()) % 1000:03d} — {procedimiento[:15]} — {paciente.split()[-1]}"
    resultado = await process_new_case_request(
        caso=titulo, paciente=paciente, aseguradora_plan=aseguradora, procedimiento=procedimiento,
        informe_path=_save_upload(file_hosp), poliza_path=_save_upload(file_seg)
    )
    return resultado

# Puente conversacional entre la interfaz gráfica y el agente ADK
async def chat_wrapper(message, history):
    try:
        msg = Content(role="user", parts=[Part(text=message)])
    except NameError:
        # Fallback en caso de incompatibilidad con versiones del SDK de genai
        msg = {"role": "user", "parts": [{"text": message}]}
        
    full = ""
    async for ev in runner.run_async(new_message=msg, user_id="user_admin", session_id="sesion_1"):
        if ev.is_final_response() and ev.content and hasattr(ev.content, 'parts') and ev.content.parts:
            full = ev.content.parts[0].text
    return full or "Evaluación en curso (Sin respuesta textual del agente)."

with gr.Blocks(title="Consola de Auditoría AI", theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate")) as demo:
    gr.Markdown("# Sistema Central de Auditoría Automatizada\n### Cruce de Identidad, Filtros de Tiempo y Análisis Documental Integral")
    
    with gr.Tabs():
        # Pestaña Principal (Enfoque en subida de archivos sin formularios)
        with gr.Tab("Auditoría Automática de Archivos", id=1):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Carga Directa de Expedientes Individuales")
                    file_hospital_auto = gr.File(label="1. Documento del Hospital (Excel, PDF, Imagen)", type="filepath")
                    file_aseguradora_auto = gr.File(label="2. Documento de Póliza (Excel, PDF, Imagen)", type="filepath")
                    btn_auto = gr.Button("PROCESAR EXPEDIENTE Y SINCRONIZAR", variant="primary")
                    
                with gr.Column(scale=1):
                    gr.Markdown("### Dictamen de Cobertura Quirúrgica")
                    out_auto = gr.Textbox(label="Resultados y Logs en Tiempo Real (Notion Connected)", lines=12, interactive=False)
            btn_auto.click(fn=submit_case_automatico, inputs=[file_hospital_auto, file_aseguradora_auto], outputs=out_auto)

        # Pestañas Secundarias (Formulario Asistido y Chat)
        with gr.Tab("Formulario de Carga Manual", id=2):
            gr.Markdown("### Registro Estructurado de Siniestros y Anexos")
            with gr.Row():
                with gr.Column():
                    paciente_in = gr.Textbox(label="Nombre del Paciente")
                    aseguradora_in = gr.Textbox(label="Aseguradora / Plan")
                    procedimiento_in = gr.Textbox(label="Procedimiento Solicitado")
                    file_hosp_manual = gr.File(label="Documentos de Soporte Médico (Opcional)", type="filepath")
                    file_seg_manual = gr.File(label="Documentos de Contrato (Opcional)", type="filepath")
                    btn_manual = gr.Button("Validar y Procesar Formulario")
                with gr.Column():
                    out_manual = gr.Textbox(label="Resultado de la Evaluación", lines=10, interactive=False)
            btn_manual.click(fn=submit_case_ui, inputs=[paciente_in, aseguradora_in, procedimiento_in, file_hosp_manual, file_seg_manual], outputs=out_manual)

        with gr.Tab("Asistente Conversacional ADK", id=3):
            gr.Markdown("### Interfaz de Consulta Dinámica con Agente Cognitivo")
            gr.ChatInterface(fn=chat_wrapper)

if __name__ == "__main__":
    puerto = int(os.environ.get("PORT", 7860))
    demo.launch(
        server_name="0.0.0.0", 
        server_port=puerto,
        quiet=True, 
        share=False
    )