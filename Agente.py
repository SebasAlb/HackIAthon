import os
import asyncio
import json
import unicodedata
import pandas as pd
from datetime import datetime
from typing import Optional, Dict, Any, List
import gradio as gr
from notion_client import AsyncClient
from notion_client.errors import APIResponseError, RequestTimeoutError
from google import genai
from google.genai import types
from dotenv import load_dotenv
import traceback

# ==============================================================================
# 1. CONFIGURACIÓN DE CREDENCIALES Y BASES DE DATOS DE NOTION
# ==============================================================================
# Configura tus tokens de acceso directamente o mediante variables de entorno
# Carga las variables de entorno desde el archivo .env
load_dotenv()

# Obtiene las credenciales y configuraciones de forma segura
NOTION_API_KEY = os.getenv("NOTION_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# IDs de tus bases de datos en Notion
NOTION_RULES_DB_ID = os.getenv("NOTION_RULES_DB_ID")
NOTION_CASES_DB_ID = os.getenv("NOTION_CASES_DB_ID")

# Opcional: Validación de seguridad para asegurarte de que se cargaron correctamente
if not NOTION_API_KEY or not GOOGLE_API_KEY:
    raise ValueError("Faltan claves de API en el archivo .env")

# ¡ATENCIÓN! Actualizado a la versión 2025-09-03 que soporta Data Sources
notion = AsyncClient(auth=os.environ["NOTION_API_KEY"], notion_version="2025-09-03")

# ==============================================================================
# 2. CONSTRUCTORES DE PROPIEDADES Y CACHÉ DE DATA SOURCES
# ==============================================================================
def _title(text: Optional[str]) -> Dict[str, Any]: 
    return {"title": [{"type": "text", "text": {"content": text or ""}}]}

def _rich_text(text: Optional[str]) -> Dict[str, Any]:
    if not text: return {"rich_text": []}
    chunks = [text[i:i + 1900] for i in range(0, len(text), 1900)]
    return {"rich_text": [{"type": "text", "text": {"content": c}} for c in chunks]}

def _date(start: Optional[str]) -> Dict[str, Any]: 
    return {"date": {"start": start}} if start else {"date": None}

def _status(name: Optional[str]) -> Dict[str, Any]: 
    return {"status": {"name": name}} if name else {"status": None}

def _number(num: Optional[int]) -> Dict[str, Any]: 
    return {"number": num}

def _norm(s: str) -> str:
    if not s: return ""
    s = unicodedata.normalize("NFD", s)
    return "".join(c for c in s if unicodedata.category(c) != "Mn").lower().strip()

def _dias_entre(inicio: Any, fin: Any) -> Optional[int]:
    try:
        d1 = pd.to_datetime(inicio).date()
        d2 = pd.to_datetime(fin).date()
        return (d2 - d1).days
    except Exception:
        return None

async def notion_retry(coro_factory, max_attempts: int = 4, base_delay: float = 1.5):
    for attempt in range(1, max_attempts + 1):
        try: return await coro_factory()
        except (APIResponseError, RequestTimeoutError) as e:
            status = getattr(e, "status", None)
            if status in (429, 502, 503, 504) and attempt < max_attempts:
                await asyncio.sleep(base_delay * (2 ** (attempt - 1)))
                continue
            raise

# --- SISTEMA DE CACHÉ DE DATA SOURCES (NUEVA ARQUITECTURA NOTION) ---
_DS_CACHE: Dict[str, str] = {}
async def _get_data_source_id(database_id: str) -> str:
    """Consulta a Notion el ID real del data source de una base de datos y lo guarda en caché."""
    if database_id in _DS_CACHE:
        return _DS_CACHE[database_id]
    
    db = await notion_retry(lambda: notion.databases.retrieve(database_id=database_id))
    ds_id = db["data_sources"][0]["id"]
    _DS_CACHE[database_id] = ds_id
    return ds_id

# ==============================================================================
# 3. EXTRACCIÓN DE REGLAS MAESTRAS DESDE NOTION
# ==============================================================================
async def fetch_notion_rules() -> List[Dict[str, Any]]:
    print("📋 Conectando con Notion para descargar el catálogo de reglas...")
    rules = []
    cursor = None
    
    # 1. Extraemos el ID del data source correcto para esta versión de la API
    ds_id = await _get_data_source_id(NOTION_RULES_DB_ID)
    
    while True:
        # 2. Consultamos directamente al endpoint data_sources
        resp = await notion_retry(lambda: notion.data_sources.query(
            data_source_id=ds_id, 
            start_cursor=cursor
        ))
        for page in resp["results"]:
            p = page["properties"]
            title_arr = p.get("Evento / Diagnóstico Traumático", {}).get("title", [])
            evento = title_arr[0]["plain_text"] if title_arr else None
            sel = p.get("Cubierto", {}).get("select")
            cubierto = sel["name"] if sel else None
            excl_arr = p.get("Exclusiones Semánticas Críticas", {}).get("rich_text", [])
            exclusiones = "".join(t["plain_text"] for t in excl_arr) or None
            req_arr = p.get("Requisitos Documentales Exigidos", {}).get("multi_select", [])
            requisitos = [t["name"] for t in req_arr]
            
            rules.append({
                "evento": evento, 
                "cubierto": cubierto, 
                "exclusiones": exclusiones,
                "requisitos": requisitos
            })
        if not resp["has_more"]: break
        cursor = resp["next_cursor"]
    print(f"✅ Sincronizadas {len(rules)} reglas de exclusión desde la nube.")
    return rules

# ==============================================================================
# 4. AGENTE DE GEMINI: AUDITORÍA SEMÁNTICA
# ==============================================================================
def ejecutar_auditoria_gemini(data_informe: Dict[str, Any], reglas_notion: List[Dict[str, Any]]) -> Dict[str, Any]:
    client = genai.Client()
    contexto_reglas = ""
    for r in reglas_notion:
        contexto_reglas += f"- Evento: {r['evento']} | Cubierto: {r['cubierto']} | Exclusiones: {r['exclusiones']} | Requisitos: {r['requisitos']}\n"

    prompt_solicitud = f"""
    Eres un auditor médico experto para "Aseguradora del Sur" encargado de evaluar pre-autorizaciones quirúrgicas bajo la póliza de Accidentes Personales.
    Tu misión es analizar el informe del hospital y contrastarlo contra las reglas de la compañía.

    REGLAS DE COBERTURA CONFIGURADAS (NOTION MAESTRO):
    {contexto_reglas}

    DATOS CLÍNICOS ACTUALES DEL PACIENTE (INFORME DEL HOSPITAL):
    - Diagnóstico Clínico: {data_informe.get('Diagnostico_Clinico')}
    - Procedimiento Solicitado: {data_informe.get('Procedimiento_Solicitado')}

    TAREA ANALÍTICA:
    1. Define el Nexo Causal: ¿La lesión descrita fue producida por una acción fortuita, repentina, violenta y de fuerza exterior (Accidente) o es una Enfermedad Común/Patología Crónica?
    2. Cruza el diagnóstico contra las exclusiones críticas (Las hernias, lumbalgias, várices y trasplantes están prohibidos por el Art. 3 del contrato).
    3. Evalúa si el diagnóstico médico del hospital se asocia semánticamente a alguna regla cubierta (ej. colisión vehicular mapea con Accidente de Tránsito).

    RESPUESTA REQUERIDA (FORMATO JSON ESTRICTO):
    {{
        "decision": "Preaprobado" o "No cubierto" o "Faltan documentos",
        "razones": "Explicación detallada (Máx 350 caracteres)."
    }}
    """

    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt_solicitud,
            config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0.1)
        )
        resultado_ia = json.loads(response.text.strip())
        return {
            "decision": resultado_ia.get("decision", "Pendiente"),
            "razones": resultado_ia.get("razones", "Evaluación completada.")
        }
    except Exception as e:
        return {"decision": "Faltan documentos", "razones": f"Error en la IA: {str(e)}"}

# ==============================================================================
# 5. PIPELINE INTEGRADO
# ==============================================================================
 # Añade esto arriba junto a tus otros imports si no lo tienes

# ==============================================================================
# 5. PIPELINE INTEGRADO (CORREGIDO Y OPTIMIZADO PARA NUBE)
# ==============================================================================
async def procesar_siniestro_pipeline(archivo_informe, archivo_poliza) -> str:
    if not archivo_informe or not archivo_poliza:
        return "❌ Error: Carga suspendida. Debes arrastrar Informe Médico y Póliza."
    
    try:
        # CORRECCIÓN 1: Manejo seguro de rutas de archivos para Gradio en la nube (Render/Railway)
        path_informe = archivo_informe if isinstance(archivo_informe, str) else archivo_informe.name
        path_poliza = archivo_poliza if isinstance(archivo_poliza, str) else archivo_poliza.name

        # LECTURA INTELIGENTE
        if path_informe.lower().endswith('.csv'):
            df_informe = pd.read_csv(path_informe).set_index("Celda / Campo")
        else:
            df_informe = pd.read_excel(path_informe).set_index("Celda / Campo")
            
        if path_poliza.lower().endswith('.csv'):
            df_poliza = pd.read_csv(path_poliza).set_index("Celda / Campo")
        else:
            df_poliza = pd.read_excel(path_poliza).set_index("Celda / Campo")
            
        data_informe = df_informe["Valor de Ejemplo"].to_dict()
        data_poliza = df_poliza["Valor de Ejemplo"].to_dict()
        
        # Extracción segura de IDs previniendo valores nulos
        id_hospital = str(data_informe.get("Paciente_Identificacion", "")).strip()
        id_poliza = str(data_poliza.get("Titular_Identificacion", "")).strip()
        
        print(f"🕵️ Cruzando identidades... ID Hospital: {id_hospital} | ID Póliza: {id_poliza}")
        
        if not id_hospital or not id_poliza or id_hospital != id_poliza:
            veredicto = {
                "decision": "No cubierto",
                "razones": "RECHAZO CRÍTICO DE SEGURIDAD: La cédula del paciente del hospital no coincide con el dueño registrado de la póliza (Posible fraude o falta de datos)."
            }
        else:
            # CORRECCIÓN 3: Conversión segura de la edad (evita error si el Excel tiene 28.0 o viene vacío)
            try:
                edad = int(float(data_informe.get("Paciente_Edad", 0)))
            except (ValueError, TypeError):
                edad = -1

            f_accidente = data_informe.get("Fecha_Accidente")
            f_notificacion = data_informe.get("Fecha_Admision")
            f_cirugia = data_informe.get("Fecha_Admision")
            estado_prima = _norm(str(data_poliza.get("Estado_Prima", "")))
            
            if edad < 0 or edad > 65:
                veredicto = {"decision": "No cubierto", "razones": f"Rechazado por Artículo 11: Fuera del rango estipulado (3 a 65 años)."}
            elif _dias_entre(f_accidente, f_notificacion) and _dias_entre(f_accidente, f_notificacion) > 30:
                veredicto = {"decision": "No cubierto", "razones": f"Rechazado por Artículo 14: Reporte extemporáneo (> 30 días)."}
            elif _dias_entre(f_accidente, f_cirugia) and _dias_entre(f_accidente, f_cirugia) > 180:
                veredicto = {"decision": "No cubierto", "razones": f"Rechazado por Parte IV: Fuera del límite de cobertura (180 días)."}
            elif "pagado" not in estado_prima and "al dia" not in estado_prima:
                veredicto = {"decision": "No cubierto", "razones": "Rechazado por Artículo 10: Póliza inactiva por mora de pago."}
            else:
                print("🧠 Pasa filtros duros de tiempo. Activando Agente de Gemini...")
                reglas_notion = await fetch_notion_rules()
                veredicto = ejecutar_auditoria_gemini(data_informe, reglas_notion)

        # --- GUARDADO EN LA BASE DE CASOS DE NOTION ---
        num_ticket = int(datetime.now().timestamp()) % 1000
        nombre_paciente = str(data_informe.get("Paciente_Nombre", "Paciente Anónimo"))
        partes_nombre = nombre_paciente.split()
        apellido_paciente = partes_nombre[-1] if len(partes_nombre) > 0 else "Paciente"
        
        proc_solicitado = str(data_informe.get('Procedimiento_Solicitado', 'Sin Diagnóstico'))
        titulo_siniestro = f"C-{num_ticket:03d} — {proc_solicitado[:12]} — {apellido_paciente}"
        
        # Formateo seguro de fechas para Notion (evita enviar 'nan')
        str_f_accidente = str(f_accidente)[:10] if pd.notna(f_accidente) else None
        str_f_notificacion = str(f_notificacion)[:10] if pd.notna(f_notificacion) else None
        str_f_cirugia = str(f_cirugia)[:10] if pd.notna(f_cirugia) else None
        
        properties = {
            "Caso": _title(titulo_siniestro),
            "Paciente": _rich_text(nombre_paciente[:100]),
            "Edad": _number(edad if edad >= 0 else 0),
            "Procedimiento solicitado": _rich_text(proc_solicitado[:1000]),
            "Decisión": _status(veredicto["decision"]),
            "Razones (resumen)": _rich_text(veredicto["razones"]),
            "Fecha de Accidente": _date(str_f_accidente),
            "Fecha de Notificación": _date(str_f_notificacion),
            "Fecha tentativa de cirugía": _date(str_f_cirugia),
            "Texto extraído (fallback)": _rich_text(f"Auditoría. Diagnóstico reportado: {data_informe.get('Diagnostico_Clinico', '')}")
        }
        
        # CORRECCIÓN 2: La API de páginas de Notion requiere que el parent sea explícitamente "database_id"
        await notion_retry(lambda: notion.pages.create(
            parent={"database_id": NOTION_CASES_DB_ID}, 
            properties=properties
        ))
        
        return f"🏁 PROCESO FINALIZADO CON ÉXITO\n\n🔹 Ticket de Siniestro: {titulo_siniestro}\n🔹 Dictamen de Auditoría: {veredicto['decision'].upper()}\n🔹 Justificación Técnica: {veredicto['razones']}"

    except Exception as e:
        # Añadimos un traceback en consola para diagnosticar instantáneamente si hay algún otro fallo de lectura
        print("🚨 OCURRIÓ UN ERROR EN EL PIPELINE:")
        traceback.print_exc()
        return f"❌ Error crítico procesando la estructura de los archivos: {str(e)}"

# ==============================================================================
# 6. INTERFAZ GRÁFICA DEL FRONTEND (GRADIO SERVER)
# ==============================================================================
with gr.Blocks(title="Consola de Auditoría AI") as front_app:
    gr.Markdown("# 🛡️ Consola Automatizada de Auditoría de Siniestros\n### Aseguradora del Sur — Cruce de Identidad, Filtros de Tiempo y Agente Evaluador Gemini")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📥 Carga de Archivos Individuales del Siniestro")
            # Ampliamos los tipos de archivo para que te acepte tus CSVs sin problema
            file_hospital = gr.File(label="1. Formato del Hospital", file_types=[".xlsx", ".xls", ".csv"], type="filepath")
            file_aseguradora = gr.File(label="2. Formato de Aseguradora", file_types=[".xlsx", ".xls", ".csv"], type="filepath")
            btn_run = gr.Button("⚡ AUDITAR EXPEDIENTE Y SINCRONIZAR", variant="primary")
            
        with gr.Column(scale=1):
            gr.Markdown("### 📋 Dictamen de Cobertura Quirúrgica")
            consola_logs = gr.Textbox(label="Resultados y Logs en Tiempo Real (Notion Connected)", lines=12, interactive=False)

    btn_run.click(fn=procesar_siniestro_pipeline, inputs=[file_hospital, file_aseguradora], outputs=consola_logs)

if __name__ == "__main__":
    # Extraer el puerto dinámico que asigna la nube (o usar 7860 de fallback en local)
    puerto = int(os.environ.get("PORT", 7860))
    
    front_app.launch(
        server_name="0.0.0.0",  # <- CRÍTICO PARA NUBE: Permite tráfico externo
        server_port=puerto,      # <- CRÍTICO PARA RAILWAY: Puerto dinámico
        quiet=True, 
        share=False, 
        theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate")
    )