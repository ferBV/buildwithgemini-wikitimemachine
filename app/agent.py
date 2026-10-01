# ruff: noqa
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import datetime
import json
import os
import urllib.parse
import urllib.request
import uuid
from zoneinfo import ZoneInfo
from dotenv import load_dotenv

load_dotenv()

from google import genai
from google.adk.agents import Agent
from google.adk.agents.callback_context import CallbackContext
from google.adk.apps import App
from google.adk.models import Gemini
from google.adk.tools.preload_memory_tool import PreloadMemoryTool
from google.adk.tools.tool_context import ToolContext
from google.cloud import storage
from google.genai import types

from .a2ui_utils import a2ui_callback


MODEL = "gemini-3.6-flash"
IMAGE_MODEL = "gemini-3.1-flash-lite-image"
GCS_BUCKET_NAME = "wikitimemachine-assets-qwiklabs-gcp-04-7328d6faf17f"


def search_wikipedia_event(topic: str, lang: str = "es") -> str:
    """Busca un evento, personaje o era histórica en Wikipedia y obtiene un resumen fáctico verificado.

    Args:
        topic: El nombre del acontecimiento histórico, personaje o tema a buscar (ej. 'Caída de Constantinopla', 'Revolución Francesa').
        lang: Código de idioma de Wikipedia (ej. 'es' para español, 'en' para inglés). Por defecto 'es'.

    Returns:
        Un resumen del hecho histórico o mensaje de error en caso de no encontrarse.
    """
    clean_topic = topic.strip().replace(" ", "_")
    encoded_title = urllib.parse.quote(clean_topic)
    url = f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/{encoded_title}"

    headers = {"User-Agent": "WikiTimeMachineAgent/1.0 (https://antigravity.google)"}
    req = urllib.request.Request(url, headers=headers)

    try:
        with urllib.request.urlopen(req, timeout=8) as response:
            data = json.loads(response.read().decode("utf-8"))
            title = data.get("title", topic)
            extract = data.get("extract", "")
            description = data.get("description", "")
            source_url = data.get("content_urls", {}).get("desktop", {}).get("page", "")

            if not extract:
                return f"No se encontró un extracto para '{topic}' en Wikipedia ({lang})."

            return (
                f"Título: {title}\n"
                f"Descripción: {description}\n"
                f"Resumen: {extract}\n"
                f"Fuente: {source_url}"
            )
    except urllib.error.HTTPError as e:
        if e.code == 404 and lang != "en":
            # Fallback a Wikipedia en inglés si no se encuentra en español
            return search_wikipedia_event(topic, lang="en")
        return f"Error al consultar Wikipedia para '{topic}': HTTP {e.code}"
    except Exception as e:
        return f"Error de conexión al buscar '{topic}' en Wikipedia: {str(e)}"


def geocode_address(address: str) -> dict:
    """Convierte una dirección o lugar en coordenadas geográficas (latitud y longitud).

    Args:
        address: Dirección, monumento o lugar a geocodificar (ej. 'Eiffel Tower, Paris', 'Zocalo, CDMX').

    Returns:
        Diccionario con 'name', 'address', 'location' ({'lat': ..., 'lng': ...}) o mensaje de error.
    """
    api_key = os.environ.get("GOOGLE_MAPS_API_KEY")
    if not api_key:
        return {"error": "GOOGLE_MAPS_API_KEY no está configurada en las variables de entorno."}

    encoded_address = urllib.parse.quote(address)
    url = f"https://maps.googleapis.com/maps/api/geocode/json?address={encoded_address}&key={api_key}"

    try:
        with urllib.request.urlopen(url, timeout=8) as response:
            data = json.loads(response.read().decode("utf-8"))
            if data.get("status") != "OK" or not data.get("results"):
                return {"error": f"No se encontraron coordenadas para: {address} (status: {data.get('status')})"}

            first_result = data["results"][0]
            return {
                "name": address,
                "address": first_result.get("formatted_address", ""),
                "location": first_result.get("geometry", {}).get("location", {}),
            }
    except Exception as e:
        return {"error": f"Fallo al geocodificar: {str(e)}"}


def find_nearby_places(latitude: float, longitude: float, place_type: str = "tourist_attraction", radius_meters: float = 1500.0, max_results: int = 5) -> dict:
    """Busca lugares cercanos de un tipo específico usando Google Places API (New).

    Args:
        latitude: Latitud del punto central.
        longitude: Longitud del punto central.
        place_type: Tipo de lugar (ej. 'tourist_attraction', 'museum', 'historical_landmark', 'restaurant', 'park').
        radius_meters: Radio de búsqueda en metros (máx 50000). Por defecto 1500m.
        max_results: Número máximo de resultados (1-20). Por defecto 5.

    Returns:
        Diccionario con la lista de lugares encontrados conteniendo 'name', 'address' y 'location'.
    """
    api_key = os.environ.get("GOOGLE_MAPS_API_KEY")
    if not api_key:
        return {"error": "GOOGLE_MAPS_API_KEY no está configurada en las variables de entorno."}

    url = "https://places.googleapis.com/v1/places:searchNearby"
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": api_key,
        "X-Goog-FieldMask": "places.displayName,places.formattedAddress,places.location",
    }
    payload = {
        "includedTypes": [place_type],
        "maxResultCount": min(max(max_results, 1), 20),
        "locationRestriction": {
            "circle": {
                "center": {
                    "latitude": latitude,
                    "longitude": longitude,
                },
                "radius": float(radius_meters),
            }
        },
    }

    try:
        req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers)
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode("utf-8"))
            places_list = []
            for place in data.get("places", []):
                name = place.get("displayName", {}).get("text", "")
                address = place.get("formattedAddress", "")
                loc = place.get("location", {})
                places_list.append({
                    "name": name,
                    "address": address,
                    "location": {"lat": loc.get("latitude"), "lng": loc.get("longitude")},
                })
            return {"places": places_list}
    except urllib.error.HTTPError as e:
        error_detail = e.read().decode("utf-8", errors="ignore")
        return {"error": f"HTTP {e.code} en Places API: {error_detail}"}
    except Exception as e:
        return {"error": f"Fallo al buscar lugares cercanos: {str(e)}"}


async def generate_historical_postcard(event_title: str, visual_description: str, era_style: str, tool_context: ToolContext) -> dict:
    """Genera una postal ilustrada del destino histórico, la guarda en los artefactos y la sube a Cloud Storage.

    Args:
        event_title: Título del evento histórico o lugar (ej. 'Caída de Constantinopla', 'Antigua Roma', 'París 1889').
        visual_description: Descripción detallada de la escena a plasmar en la ilustración.
        era_style: Estilo artístico acorde a la era (ej. 'óleo renacentista', 'fotografía vintage sepia', 'boceto antiguo', 'mosaico bizantino').

    Returns:
        Diccionario con 'status', 'event_title', 'artifact_filename' y 'public_url'.
    """
    project_id = os.environ.get("GOOGLE_CLOUD_PROJECT", "qwiklabs-gcp-04-7328d6faf17f")
    client = genai.Client(vertexai=True, project=project_id, location="global")

    prompt = (
        f"A vintage historical collectible postcard depicting {event_title}. "
        f"Scene: {visual_description}. "
        f"Art style: {era_style}. Highly detailed, museum archive quality, atmospheric lighting."
    )

    try:
        response = client.models.generate_content(
            model=IMAGE_MODEL,
            contents=[prompt],
            config=types.GenerateContentConfig(
                response_modalities=[types.Modality.IMAGE]
            ),
        )

        image_part = None
        for cand in response.candidates:
            for part in cand.content.parts:
                if part.inline_data:
                    image_part = part
                    break
            if image_part:
                break

        if not image_part or not image_part.inline_data:
            return {"error": "El modelo no generó una imagen en la respuesta."}

        image_bytes = image_part.inline_data.data
        mime_type = image_part.inline_data.mime_type or "image/png"
        extension = "jpg" if "jpeg" in mime_type else "png"

        # 1. Guardar como artefacto de sesión para el panel de artefactos de ADK / Playground
        clean_slug = "".join(c if c.isalnum() else "_" for c in event_title.lower())[:30]
        artifact_filename = f"postcard_{clean_slug}_{uuid.uuid4().hex[:6]}.{extension}"
        artifact_part = types.Part(inline_data=types.Blob(mime_type=mime_type, data=image_bytes))
        await tool_context.save_artifact(artifact_filename, artifact_part)

        # 2. Subir directamente los bytes a Cloud Storage y construir la URL pública
        storage_client = storage.Client(project=project_id)
        bucket = storage_client.bucket(GCS_BUCKET_NAME)
        blob = bucket.blob(artifact_filename)
        blob.upload_from_string(image_bytes, content_type=mime_type)

        public_url = f"https://storage.googleapis.com/{GCS_BUCKET_NAME}/{artifact_filename}"

        return {
            "status": "success",
            "event_title": event_title,
            "artifact_filename": artifact_filename,
            "public_url": public_url,
        }
    except Exception as e:
        return {"error": f"Fallo al generar o subir la postal histórica: {str(e)}"}


def get_weather(query: str) -> str:
    """Simulates a web search. Use it get information on weather.

    Args:
        query: A string containing the location to get weather information for.

    Returns:
        A string with the simulated weather information for the queried location.
    """
    if "sf" in query.lower() or "san francisco" in query.lower():
        return "It's 60 degrees and foggy."
    return "It's 90 degrees and sunny."


def get_current_time(query: str) -> str:
    """Simulates getting the current time for a city.

    Args:
        city: The name of the city to get the current time for.

    Returns:
        A string with the current time information.
    """
    if "sf" in query.lower() or "san francisco" in query.lower():
        tz_identifier = "America/Los_Angeles"
    else:
        return f"Sorry, I don't have timezone information for query: {query}."

    tz = ZoneInfo(tz_identifier)
    now = datetime.datetime.now(tz)
    return f"The current time for query {query} is {now.strftime('%Y-%m-%d %H:%M:%S %Z%z')}"


async def generate_memories_callback(callback_context: CallbackContext):
    """Guarda automáticamente la conversación en el Memory Bank para recordar preferencias entre sesiones."""
    await callback_context.add_session_to_memory()
    return None


_instruction_path = os.path.join(os.path.dirname(__file__), "instruction.txt")
if os.path.exists(_instruction_path):
    with open(_instruction_path, "r", encoding="utf-8") as _f:
        instruction = _f.read()
else:
    instruction = (
        "Eres WikiTimeMachine, un agente educativo e interactivo de viajes en el tiempo. "
        "Recuerdas las preferencias y hechos de conversaciones previas con el usuario (ej. su época favorita, "
        "su nombre o título, misiones completadas) para personalizar los viajes en el tiempo. "
        "Cuentas con herramientas para consultar eventos históricos en Wikipedia ('search_wikipedia_event'), "
        "geocodificar lugares o monumentos históricos para situarlos en el mapa actual ('geocode_address'), "
        "descubrir puntos de interés históricos o museos cercanos en el presente ('find_nearby_places'), "
        "y generar postales históricas ilustradas conmemorativas ('generate_historical_postcard') que se guardan "
        "como artefactos y se suben a almacenamiento público."
    )


root_agent = Agent(
    name="root_agent",
    model=Gemini(
        model=MODEL,
        retry_options=types.HttpRetryOptions(attempts=3),
    ),
    instruction=instruction,
    tools=[
        PreloadMemoryTool(),
        search_wikipedia_event,
        geocode_address,
        find_nearby_places,
        generate_historical_postcard,
        get_weather,
        get_current_time,
    ],
    after_model_callback=a2ui_callback,
    after_agent_callback=generate_memories_callback,
)

app = App(
    root_agent=root_agent,
    name="app",
)
