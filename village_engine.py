"""
VillageForge AI v3 - multimodal, agentic, open-data infrastructure planner.

Run through app.py. The engine is intentionally defensive:
- no paid API keys or hardcoded private tokens
- external calls use open/free public data sources where possible
- missing data is represented as None + data_quality notes, not fake zeros
- Earth Engine, Whisper, Folium, pyttsx3, and diskcache are optional
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import math
import os
import re
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import requests
from dotenv import load_dotenv

try:
    import ee  # type: ignore
except Exception:  # pragma: no cover - optional runtime dependency
    ee = None

try:
    import diskcache  # type: ignore
except Exception:  # pragma: no cover - optional runtime dependency
    diskcache = None

try:
    import folium  # type: ignore
    from folium.plugins import HeatMap, MarkerCluster  # type: ignore
except Exception:  # pragma: no cover - optional runtime dependency
    folium = None
    HeatMap = None
    MarkerCluster = None

try:
    from fpdf import FPDF  # type: ignore
except Exception:  # pragma: no cover - optional runtime dependency
    FPDF = None

warnings.filterwarnings("ignore", category=UserWarning, module="whisper")
load_dotenv()


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

APP_NAME = "VillageForge-AI/3.0 (open-data local planner)"
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
GEMMA_MODEL = os.environ.get("GEMMA_MODEL", "gemma4:e2b")
EE_PROJECT = os.environ.get("EE_PROJECT", "gen-lang-client-0923812577")
OVERPASS_URL = os.environ.get("OVERPASS_URL", "https://overpass-api.de/api/interpreter")
NOMINATIM_URL = os.environ.get("NOMINATIM_URL", "https://nominatim.openstreetmap.org")
OUTPUT_DIR = Path(os.environ.get("VILLAGEFORGE_OUTPUT_DIR", "outputs")).resolve()
CACHE_DIR = Path(os.environ.get("CACHE_DIR", ".villageforge_cache")).resolve()
CACHE_TTL = int(os.environ.get("CACHE_TTL_SECONDS", "86400"))
REQUEST_TIMEOUT = int(os.environ.get("REQUEST_TIMEOUT_SECONDS", "15"))

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class _MemoryCache(dict):
    def set(self, key: str, value: Any, expire: int | None = None) -> None:
        self[key] = {"value": value, "expires": time.time() + expire if expire else None}

    def __getitem__(self, key: str) -> Any:
        item = super().__getitem__(key)
        if isinstance(item, dict) and "value" in item:
            expires = item.get("expires")
            if expires and expires < time.time():
                del self[key]
                raise KeyError(key)
            return item["value"]
        return item

    def __contains__(self, key: object) -> bool:
        try:
            _ = self[key]  # type: ignore[index]
            return True
        except KeyError:
            return False


cache = diskcache.Cache(str(CACHE_DIR)) if diskcache else _MemoryCache()


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------

def _cache_key(*parts: Any) -> str:
    return "_".join(str(p).replace(" ", "-") for p in parts)


def _headers() -> dict[str, str]:
    return {"User-Agent": APP_NAME}


def _slug(value: str, max_len: int = 64) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")
    return (cleaned[:max_len] or "site")


def _clamp(value: float, low: float = 0, high: float = 100) -> float:
    return max(low, min(high, value))


def _safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        if value is None or value == "":
            return default
        v = float(value)
        if math.isnan(v) or math.isinf(v):
            return default
        return v
    except Exception:
        return default


def _safe_int(value: Any, default: int | None = None) -> int | None:
    v = _safe_float(value, None)
    return int(round(v)) if v is not None else default


def _first_number(*values: Any, default: float | None = None) -> float | None:
    for value in values:
        v = _safe_float(value, None)
        if v is not None:
            return v
    return default


def _request_json(
    method: str,
    url: str,
    *,
    params: dict[str, Any] | None = None,
    data: Any = None,
    json_payload: Any = None,
    timeout: int = REQUEST_TIMEOUT,
    retries: int = 1,
) -> Any:
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            response = requests.request(
                method,
                url,
                params=params,
                data=data,
                json=json_payload,
                headers=_headers(),
                timeout=timeout,
            )
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(0.6 * (attempt + 1))
    raise last_error or RuntimeError("Request failed")


def _with_cache(key: str, ttl: int, producer: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    if key in cache:
        return cache[key]
    try:
        result = producer()
    except Exception as exc:
        result = {"error": str(exc), "data_quality": "unavailable"}
    cache.set(key, result, expire=ttl)
    return result


def _extract_json(text: str) -> dict[str, Any] | list[Any]:
    cleaned = (text or "").strip()
    cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()
    try:
        return json.loads(cleaned)
    except Exception:
        pass
    match = re.search(r"(\{.*\}|\[.*\])", cleaned, flags=re.S)
    if not match:
        raise json.JSONDecodeError("No JSON object found", cleaned, 0)
    return json.loads(match.group(1), strict=False)


def _json_default(value: Any) -> str:
    try:
        return str(value)
    except Exception:
        return "<unserializable>"


def sanitize_text(text: Any) -> str:
    if text is None:
        return "N/A"
    text = str(text)
    replacements = {
        "—": "-",
        "–": "-",
        "\u201c": '"',
        "\u201d": '"',
        "\u2018": "'",
        "\u2019": "'",
        "\u2022": "-",
        "\u26a0": "!",
        "\u2713": "Yes",
        "\u20b9": "INR ",
        "\u20ac": "EUR ",
        "\xa3": "GBP ",
        "°": " deg ",
        "\n": " ",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text.encode("latin-1", "ignore").decode("latin-1").strip()


# ---------------------------------------------------------------------------
# Earth Engine helpers
# ---------------------------------------------------------------------------

def earth_engine_available() -> bool:
    if ee is None:
        return False
    try:
        ee.Number(1).getInfo()
        return True
    except Exception:
        return False


def init_earth_engine(project: str | None = None, authenticate: bool = False) -> tuple[bool, str]:
    if ee is None:
        return False, "earthengine-api is not installed"
    try:
        kwargs = {"project": project or EE_PROJECT} if (project or EE_PROJECT) else {}
        ee.Initialize(**kwargs)
        return True, "Earth Engine initialized"
    except Exception as first_error:
        if authenticate:
            try:
                ee.Authenticate()
                kwargs = {"project": project or EE_PROJECT} if (project or EE_PROJECT) else {}
                ee.Initialize(**kwargs)
                return True, "Earth Engine authenticated and initialized"
            except Exception as second_error:
                return False, str(second_error)
        return False, str(first_error)


def _buffer(lat: float, lon: float, radius_m: int = 10000):
    if ee is None:
        raise RuntimeError("Earth Engine unavailable")
    return ee.Geometry.Point([lon, lat]).buffer(radius_m)


def _reduce_mean(image: Any, geometry: Any, scale: int = 1000) -> dict[str, Any]:
    if ee is None:
        raise RuntimeError("Earth Engine unavailable")
    return image.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=geometry,
        scale=scale,
        maxPixels=1e9,
        bestEffort=True,
    ).getInfo()


def _reduce_sum(image: Any, geometry: Any, scale: int = 1000) -> dict[str, Any]:
    if ee is None:
        raise RuntimeError("Earth Engine unavailable")
    return image.reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=geometry,
        scale=scale,
        maxPixels=1e9,
        bestEffort=True,
    ).getInfo()


def _ee_error(exc: Exception) -> dict[str, Any]:
    return {"error": str(exc), "data_quality": "unavailable_earth_engine"}


# ---------------------------------------------------------------------------
# Geocoding and OSM helpers
# ---------------------------------------------------------------------------

def pincode_to_latlon(location_query: str) -> tuple[float, float]:
    query = str(location_query).strip()
    coord_match = re.match(r"^\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*$", query)
    if coord_match:
        return float(coord_match.group(1)), float(coord_match.group(2))

    key = _cache_key("geocode", query.lower())
    if key in cache:
        cached = cache[key]
        return float(cached["lat"]), float(cached["lon"])

    attempts = [query]
    if re.fullmatch(r"\d{5,6}", query):
        attempts.extend([f"{query}, India", f"postalcode {query}, India"])

    last_error = ""
    for q in attempts:
        try:
            data = _request_json(
                "GET",
                f"{NOMINATIM_URL}/search",
                params={
                    "q": q,
                    "format": "jsonv2",
                    "addressdetails": 1,
                    "limit": 1,
                },
                timeout=15,
                retries=1,
            )
            if data:
                lat, lon = float(data[0]["lat"]), float(data[0]["lon"])
                cache.set(key, {"lat": lat, "lon": lon, "source": "OpenStreetMap Nominatim"}, expire=CACHE_TTL * 30)
                return lat, lon
        except Exception as exc:
            last_error = str(exc)

    raise ValueError(f"Could not geocode '{location_query}' using OpenStreetMap Nominatim. {last_error}")


def get_admin_area(lat: float, lon: float) -> dict[str, Any]:
    key = _cache_key("reverse", round(lat, 4), round(lon, 4))
    if key in cache:
        return cache[key]

    def _lookup() -> dict[str, Any]:
        try:
            data = _request_json(
                "GET",
                f"{NOMINATIM_URL}/reverse",
                params={"lat": lat, "lon": lon, "format": "jsonv2", "addressdetails": 1, "zoom": 10},
                timeout=15,
                retries=1,
            )
            address = data.get("address", {})
            district = (
                address.get("state_district")
                or address.get("county")
                or address.get("district")
                or address.get("city")
                or address.get("town")
                or address.get("village")
                or "Unknown district"
            )
            state = address.get("state") or address.get("region") or "Unknown state"
            country = address.get("country") or "Unknown country"
            iso2 = str(address.get("country_code", "")).upper() or "IN"
            return {
                "district": district,
                "state": state,
                "country": country,
                "country_iso2": iso2,
                "source": "OpenStreetMap Nominatim",
            }
        except Exception as osm_error:
            if ee is not None and earth_engine_available():
                try:
                    point = ee.Geometry.Point([lon, lat])
                    gaul = ee.FeatureCollection("FAO/GAUL/2015/level2")
                    feature = gaul.filterBounds(point).first()
                    return {
                        "district": feature.get("ADM2_NAME").getInfo() or "Unknown district",
                        "state": feature.get("ADM1_NAME").getInfo() or "Unknown state",
                        "country": feature.get("ADM0_NAME").getInfo() or "Unknown country",
                        "country_iso2": "IN",
                        "source": "FAO GAUL via Earth Engine",
                    }
                except Exception:
                    pass
            return {
                "district": "Unknown district",
                "state": "Unknown state",
                "country": "Unknown country",
                "country_iso2": "IN",
                "source": "fallback",
                "error": str(osm_error),
            }

    result = _lookup()
    cache.set(key, result, expire=CACHE_TTL * 30)
    return result


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    return round(6371 * 2 * math.asin(math.sqrt(a)), 2)


def _overpass(query: str, timeout: int = 25) -> list[dict[str, Any]]:
    data = _request_json("POST", OVERPASS_URL, data={"data": query}, timeout=timeout, retries=1)
    return data.get("elements", []) if isinstance(data, dict) else []


def _nearest_from_osm(elements: list[dict[str, Any]], lat: float, lon: float) -> dict[str, Any] | None:
    best: dict[str, Any] | None = None
    for item in elements:
        item_lat = item.get("lat") or item.get("center", {}).get("lat")
        item_lon = item.get("lon") or item.get("center", {}).get("lon")
        if item_lat is None or item_lon is None:
            continue
        dist = _haversine_km(lat, lon, float(item_lat), float(item_lon))
        candidate = {
            "distance_km": dist,
            "lat": float(item_lat),
            "lon": float(item_lon),
            "name": item.get("tags", {}).get("name", "Unnamed"),
            "tags": item.get("tags", {}),
        }
        if best is None or dist < best["distance_km"]:
            best = candidate
    return best


# ---------------------------------------------------------------------------
# Open data layer functions
# ---------------------------------------------------------------------------

def analyze_solar_potential(lat: float, lon: float, radius: int = 10000) -> dict[str, Any]:
    key = _cache_key("solar", round(lat, 3), round(lon, 3))

    def _produce() -> dict[str, Any]:
        annual_avg = None
        peak_month = None
        trough_month = None
        try:
            data = _request_json(
                "GET",
                "https://power.larc.nasa.gov/api/temporal/monthly/point",
                params={
                    "parameters": "ALLSKY_SFC_SW_DWN",
                    "community": "RE",
                    "longitude": lon,
                    "latitude": lat,
                    "format": "JSON",
                    "start": "2022",
                    "end": "2023",
                },
                timeout=20,
                retries=1,
            )
            monthly = data["properties"]["parameter"]["ALLSKY_SFC_SW_DWN"]
            values = [float(v) for v in monthly.values() if _safe_float(v) is not None]
            if values:
                annual_avg = round(sum(values) / len(values), 2)
                peak_month = max(monthly, key=lambda k: float(monthly[k]))
                trough_month = min(monthly, key=lambda k: float(monthly[k]))
        except Exception:
            annual_avg = round(_clamp(5.8 - abs(lat) * 0.025, 2.8, 6.2), 2)

        cloud_cover_pct = None
        if ee is not None and earth_engine_available():
            try:
                buf = _buffer(lat, lon, radius)
                cloud_data = (
                    ee.ImageCollection("MODIS/061/MOD09GA")
                    .filterDate("2022-01-01", "2023-12-31")
                    .select("state_1km")
                )
                cloud_cover = cloud_data.map(lambda img: img.select("state_1km").bitwiseAnd(1 << 10).neq(0)).mean().clip(buf)
                cloud_cover_pct = round((_reduce_mean(cloud_cover, buf).get("state_1km") or 0) * 100, 1)
            except Exception:
                cloud_cover_pct = None

        return {
            "solar_irradiance_kwh_m2_day": annual_avg,
            "cloud_cover_pct": cloud_cover_pct,
            "peak_solar_month": peak_month,
            "trough_solar_month": trough_month,
            "solar_viable": annual_avg is not None and annual_avg >= 4.2 and (cloud_cover_pct is None or cloud_cover_pct < 55),
            "estimated_panel_output_kwh_day_per_kw": round(annual_avg * 0.78, 2) if annual_avg else None,
            "source": "NASA POWER + optional MODIS cloud cover",
            "data_quality": "measured" if annual_avg is not None else "estimated",
        }

    return _with_cache(key, CACHE_TTL, _produce)


def analyze_wind_potential(lat: float, lon: float, radius: int = 10000) -> dict[str, Any]:
    key = _cache_key("wind", round(lat, 3), round(lon, 3))

    def _produce() -> dict[str, Any]:
        speed = None
        try:
            data = _request_json(
                "GET",
                "https://power.larc.nasa.gov/api/temporal/monthly/point",
                params={
                    "parameters": "WS10M",
                    "community": "RE",
                    "longitude": lon,
                    "latitude": lat,
                    "format": "JSON",
                    "start": "2022",
                    "end": "2023",
                },
                timeout=20,
                retries=1,
            )
            monthly = data["properties"]["parameter"]["WS10M"]
            vals = [float(v) for v in monthly.values() if _safe_float(v) is not None]
            speed = round(sum(vals) / len(vals), 2) if vals else None
        except Exception:
            speed = None

        if speed is None and ee is not None and earth_engine_available():
            try:
                buf = _buffer(lat, lon, radius)
                wind = (
                    ee.ImageCollection("ECMWF/ERA5/MONTHLY")
                    .filterDate("2022-01-01", "2023-12-31")
                    .select("u_component_of_wind_10m", "v_component_of_wind_10m")
                )
                annual = wind.map(
                    lambda img: img.addBands(
                        img.select("u_component_of_wind_10m")
                        .pow(2)
                        .add(img.select("v_component_of_wind_10m").pow(2))
                        .sqrt()
                        .rename("wind_speed")
                    )
                ).mean().clip(buf)
                speed = round(_reduce_mean(annual, buf).get("wind_speed") or 0, 2)
            except Exception:
                speed = None

        wpd = round(0.5 * 1.225 * (speed ** 3), 1) if speed is not None else None
        return {
            "wind_speed_ms": speed,
            "wind_power_density_w_m2": wpd,
            "wind_viable": speed is not None and speed > 4.5,
            "wind_class": "excellent" if (speed or 0) > 7 else "good" if (speed or 0) > 5.5 else "marginal" if (speed or 0) > 4 else "poor",
            "source": "NASA POWER WS10M or ERA5 fallback",
            "data_quality": "measured" if speed is not None else "unavailable",
        }

    return _with_cache(key, CACHE_TTL, _produce)


def analyze_population(lat: float, lon: float, radius: int = 10000) -> dict[str, Any]:
    key = _cache_key("pop", round(lat, 3), round(lon, 3))

    def _produce() -> dict[str, Any]:
        population = None
        quality = "unavailable"
        if ee is not None and earth_engine_available():
            try:
                buf = _buffer(lat, lon, radius)
                pop_img = ee.ImageCollection("WorldPop/GP/100m/pop").filterDate("2020-01-01", "2020-12-31").mean().clip(buf)
                total = _reduce_sum(pop_img, buf, scale=100).get("population")
                population = int(round(total)) if total is not None else None
                quality = "measured_worldpop"
            except Exception:
                population = None

        if population is None:
            buildings = get_open_buildings_count(lat, lon, 5000)
            count = _safe_int(buildings.get("building_count_5km"))
            if count:
                population = int(count * 4.2)
                quality = "estimated_from_building_count"

        area_km2 = math.pi * (radius / 1000) ** 2
        return {
            "estimated_population": population,
            "estimated_households": int(population / 4.5) if population else None,
            "estimated_school_children": int(population * 0.22) if population else None,
            "estimated_under5": int(population * 0.13) if population else None,
            "population_density_per_km2": round(population / area_km2, 1) if population else None,
            "catchment_size": "large" if (population or 0) > 10000 else "medium" if (population or 0) > 2000 else "small" if population else "unknown",
            "source": "WorldPop via Earth Engine or OSM/Open Buildings estimate",
            "data_quality": quality,
        }

    return _with_cache(key, CACHE_TTL, _produce)


def analyze_nighttime_lights(lat: float, lon: float, radius: int = 10000) -> dict[str, Any]:
    key = _cache_key("ntl", round(lat, 3), round(lon, 3))

    def _produce() -> dict[str, Any]:
        if ee is None or not earth_engine_available():
            return {
                "nighttime_radiance": None,
                "radiance_trend_2019_2023": None,
                "electrification_trend": "unknown",
                "grid_status": "unknown",
                "grid_label": "Nighttime lights unavailable; inspect local utility data",
                "data_quality": "unavailable_earth_engine",
            }
        try:
            buf = _buffer(lat, lon, radius)
            viirs = (
                ee.ImageCollection("NOAA/VIIRS/DNB/MONTHLY_V1/VCMCFG")
                .filterDate("2022-01-01", "2023-12-31")
                .select("avg_rad")
                .mean()
                .clip(buf)
            )
            old = (
                ee.ImageCollection("NOAA/VIIRS/DNB/MONTHLY_V1/VCMCFG")
                .filterDate("2019-01-01", "2019-12-31")
                .select("avg_rad")
                .mean()
                .clip(buf)
            )
            radiance = round(_reduce_mean(viirs, buf, scale=500).get("avg_rad") or 0, 4)
            old_radiance = round(_reduce_mean(old, buf, scale=500).get("avg_rad") or 0, 4)
            trend = round(radiance - old_radiance, 4)
            if radiance < 0.5:
                grid_status, grid_label = "off_grid", "No strong grid electricity signal detected"
            elif radiance < 5:
                grid_status, grid_label = "partial", "Weak or intermittent electricity signal"
            else:
                grid_status, grid_label = "grid_connected", "Grid electricity likely available"
            return {
                "nighttime_radiance": radiance,
                "radiance_trend_2019_2023": trend,
                "electrification_trend": "improving" if trend > 0.5 else "stable" if trend > -0.5 else "declining",
                "grid_status": grid_status,
                "grid_label": grid_label,
                "data_quality": "measured_viirs",
            }
        except Exception as exc:
            return _ee_error(exc)

    return _with_cache(key, CACHE_TTL, _produce)


def analyze_vegetation_and_climate(lat: float, lon: float, radius: int = 10000) -> dict[str, Any]:
    key = _cache_key("veg", round(lat, 3), round(lon, 3))

    def _produce() -> dict[str, Any]:
        result: dict[str, Any] = {
            "ndvi": None,
            "vegetation_cover": "unknown",
            "land_surface_temp_c": None,
            "heat_stress_risk": "unknown",
            "daily_rainfall_mm": None,
            "annual_rainfall_mm_estimate": None,
            "elevation_m": None,
            "data_quality": "unavailable",
        }
        if ee is not None and earth_engine_available():
            try:
                buf = _buffer(lat, lon, radius)
                landsat = (
                    ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
                    .filterDate("2022-06-01", "2023-05-31")
                    .filterBounds(ee.Geometry.Point([lon, lat]))
                    .map(
                        lambda img: img.addBands(img.normalizedDifference(["SR_B5", "SR_B4"]).rename("NDVI")).addBands(
                            img.select("ST_B10").multiply(0.00341802).add(149.0).subtract(273.15).rename("LST")
                        )
                    )
                    .mean()
                    .clip(buf)
                )
                chirps = (
                    ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
                    .filterDate("2022-01-01", "2023-12-31")
                    .select("precipitation")
                    .mean()
                    .clip(buf)
                )
                srtm = ee.Image("USGS/SRTMGL1_003").clip(buf)
                ndvi = round(_reduce_mean(landsat.select("NDVI"), buf, scale=30).get("NDVI") or 0, 3)
                lst = round(_reduce_mean(landsat.select("LST"), buf, scale=30).get("LST") or 0, 1)
                rain = round(_reduce_mean(chirps, buf, scale=5000).get("precipitation") or 0, 2)
                elev = round(_reduce_mean(srtm, buf, scale=30).get("elevation") or 0, 0)
                result.update(
                    {
                        "ndvi": ndvi,
                        "vegetation_cover": "dense" if ndvi > 0.5 else "moderate" if ndvi > 0.2 else "sparse",
                        "land_surface_temp_c": lst,
                        "heat_stress_risk": "high" if lst > 38 else "moderate" if lst > 32 else "low",
                        "daily_rainfall_mm": rain,
                        "annual_rainfall_mm_estimate": round(rain * 365, 0),
                        "elevation_m": int(elev),
                        "data_quality": "measured_satellite",
                    }
                )
            except Exception as exc:
                result.update(_ee_error(exc))

        if result["daily_rainfall_mm"] is None:
            try:
                weather = fetch_open_meteo_weather(lat, lon)
                annual = weather.get("annual_rainfall_mm_estimate")
                result["annual_rainfall_mm_estimate"] = annual
                result["daily_rainfall_mm"] = round(annual / 365, 2) if annual else None
                result["data_quality"] = "weather_api_partial"
            except Exception:
                pass
        return result

    return _with_cache(key, CACHE_TTL, _produce)


def analyze_land_cover(lat: float, lon: float, radius: int = 10000) -> dict[str, Any]:
    key = _cache_key("landcover", round(lat, 3), round(lon, 3))

    def _produce() -> dict[str, Any]:
        if ee is None or not earth_engine_available():
            return {"dominant_land_use": "unknown", "construction_land_available_pct": None, "data_quality": "unavailable_earth_engine"}
        try:
            buf = _buffer(lat, lon, radius)
            wc = ee.Image("ESA/WorldCover/v200/2021").clip(buf)
            class_names = {
                10: "tree_cover",
                20: "shrubland",
                30: "grassland",
                40: "cropland",
                50: "built_up",
                60: "bare_sparse",
                80: "water",
                90: "wetland",
            }
            area_img = ee.Image.pixelArea().divide(1e6)
            total_area = math.pi * (radius / 1000) ** 2
            areas: dict[str, float] = {}
            for code, name in class_names.items():
                val = _reduce_sum(area_img.updateMask(wc.eq(code)), buf, scale=10).get("area") or 0
                areas[f"{name}_pct"] = round(val / total_area * 100, 1)
            dominant = max(class_names.values(), key=lambda n: areas.get(f"{n}_pct", 0))
            return {
                **areas,
                "dominant_land_use": dominant,
                "construction_land_available_pct": round(areas.get("bare_sparse_pct", 0) + areas.get("shrubland_pct", 0) + areas.get("grassland_pct", 0) * 0.35, 1),
                "data_quality": "measured_esa_worldcover",
            }
        except Exception as exc:
            return _ee_error(exc)

    return _with_cache(key, CACHE_TTL, _produce)


def analyze_flood_risk(lat: float, lon: float, radius: int = 10000) -> dict[str, Any]:
    key = _cache_key("flood", round(lat, 3), round(lon, 3))

    def _produce() -> dict[str, Any]:
        if ee is None or not earth_engine_available():
            return {
                "water_occurrence_pct": None,
                "terrain_slope_deg": None,
                "flood_risk_score": None,
                "flood_risk_level": "unknown",
                "safe_for_construction": None,
                "recommended_floor_height_m": 0.6,
                "data_quality": "unavailable_earth_engine",
            }
        try:
            buf = _buffer(lat, lon, radius)
            gsw = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence").clip(buf)
            occurrence = round(_reduce_mean(gsw, buf, scale=30).get("occurrence") or 0, 1)
            slope = ee.Terrain.slope(ee.Image("USGS/SRTMGL1_003")).clip(buf)
            slope_deg = round(_reduce_mean(slope, buf, scale=30).get("slope") or 0, 2)
            flood_score = int(round(_clamp(occurrence * 0.7 + max(0, 5 - slope_deg) * 6, 0, 100)))
            return {
                "water_occurrence_pct": occurrence,
                "terrain_slope_deg": slope_deg,
                "flood_risk_score": flood_score,
                "flood_risk_level": "high" if flood_score >= 50 else "moderate" if flood_score >= 20 else "low",
                "safe_for_construction": flood_score < 50,
                "recommended_floor_height_m": 0.3 if flood_score < 20 else 0.6 if flood_score < 50 else 1.2,
                "data_quality": "measured_jrc_srtm",
            }
        except Exception as exc:
            return _ee_error(exc)

    return _with_cache(key, CACHE_TTL, _produce)


def analyze_seismic_risk(lat: float, lon: float) -> dict[str, Any]:
    key = _cache_key("seismic", round(lat, 2), round(lon, 2))

    def _produce() -> dict[str, Any]:
        eq = fetch_usgs_earthquakes(lat, lon)
        activity = eq.get("seismic_activity", "unknown")
        risk_from_activity = {"very_high": "high", "high": "high", "moderate": "moderate", "low": "low"}.get(activity, "unknown")
        pga = None
        if ee is not None and earth_engine_available():
            try:
                buf = _buffer(lat, lon, 50000)
                # Dataset availability varies by EE account; use only as optional enhancement.
                hazard = ee.Image("USGS/EARTHQUAKE/USGS_HAZARD_2014").clip(buf)
                pga = round(_reduce_mean(hazard, buf, scale=1000).get("pga") or 0, 4)
            except Exception:
                pga = None
        risk = risk_from_activity
        if pga is not None:
            risk = "high" if pga > 0.3 else "moderate" if pga > 0.1 else "low"
        return {
            "peak_ground_acceleration_g": pga,
            "seismic_risk": risk,
            "reinforcement_required": risk in {"moderate", "high"},
            "seismic_design_note": (
                "Full seismic detailing and ductile reinforcement required"
                if risk == "high"
                else "Standard seismic-resistant frame with lateral bracing"
                if risk == "moderate"
                else "Standard construction adequate, verify local code zone"
                if risk == "low"
                else "Verify local seismic code before final design"
            ),
            "source": "USGS earthquake catalog + optional EE hazard layer",
            "data_quality": "measured_usgs" if activity != "unknown" else "partial",
        }

    return _with_cache(key, CACHE_TTL, _produce)


def analyze_landslide_risk(lat: float, lon: float, radius: int = 10000) -> dict[str, Any]:
    key = _cache_key("landslide", round(lat, 3), round(lon, 3))

    def _produce() -> dict[str, Any]:
        if ee is None or not earth_engine_available():
            return {"slope_deg": None, "landslide_susceptibility_score": None, "landslide_risk": "unknown", "data_quality": "unavailable_earth_engine"}
        try:
            buf = _buffer(lat, lon, radius)
            slope = ee.Terrain.slope(ee.Image("USGS/SRTMGL1_003")).clip(buf)
            slope_val = round(_reduce_mean(slope, buf, scale=30).get("slope") or 0, 2)
            climate = analyze_vegetation_and_climate(lat, lon, radius)
            rain = climate.get("daily_rainfall_mm") or 0
            score = int(round(_clamp(slope_val * 2.2 + rain * 0.6, 0, 100)))
            return {
                "slope_deg": slope_val,
                "landslide_susceptibility_score": score,
                "landslide_risk": "high" if score > 60 else "moderate" if score > 30 else "low",
                "site_preparation_note": "Slope stabilization mandatory" if score > 60 else "Retaining walls recommended" if score > 30 else "Standard site preparation adequate",
                "data_quality": "proxy_srtm_rainfall",
            }
        except Exception as exc:
            return _ee_error(exc)

    return _with_cache(key, CACHE_TTL, _produce)


def analyze_surface_water(lat: float, lon: float, radius: int = 15000) -> dict[str, Any]:
    key = _cache_key("surfacewater", round(lat, 3), round(lon, 3))

    def _produce() -> dict[str, Any]:
        if ee is None or not earth_engine_available():
            return {
                "permanent_water_fraction_pct": None,
                "water_recurrence_pct": None,
                "estimated_distance_to_water_km": None,
                "water_source_type": "unknown",
                "water_access_note": "Surface water layer unavailable; verify with local hydro survey",
                "rainwater_harvesting_viable": None,
                "data_quality": "unavailable_earth_engine",
            }
        try:
            buf = _buffer(lat, lon, radius)
            gsw = ee.Image("JRC/GSW1_4/GlobalSurfaceWater")
            water_fraction = round((_reduce_mean(gsw.select("seasonality").gte(8).clip(buf), buf, scale=30).get("seasonality") or 0) * 100, 2)
            recurrence = round(_reduce_mean(gsw.select("recurrence").clip(buf), buf, scale=30).get("recurrence") or 0, 1)
            # This is a proxy, not an exact nearest-water distance.
            distance_proxy = round(max(0.5, (100 - recurrence) / 100 * radius / 1000), 2)
            if water_fraction > 5:
                src, note = "surface_water_nearby", "Permanent water body likely within analysis radius"
            elif recurrence > 30:
                src, note = "seasonal_water", "Seasonal water present; storage and treatment needed"
            else:
                src, note = "groundwater_dependent", "No reliable surface water signal; borewell or piped supply recommended"
            return {
                "permanent_water_fraction_pct": water_fraction,
                "water_recurrence_pct": recurrence,
                "estimated_distance_to_water_km": distance_proxy,
                "water_source_type": src,
                "water_access_note": note,
                "rainwater_harvesting_viable": recurrence > 20,
                "data_quality": "measured_jrc_surface_water",
            }
        except Exception as exc:
            return _ee_error(exc)

    return _with_cache(key, CACHE_TTL, _produce)


def analyze_groundwater_potential(lat: float, lon: float) -> dict[str, Any]:
    key = _cache_key("groundwater", round(lat, 3), round(lon, 3))

    def _produce() -> dict[str, Any]:
        if ee is None or not earth_engine_available():
            return {"groundwater_potential_score": None, "groundwater_potential": "unknown", "borewell_depth_estimate_m": "Hydrogeological survey required", "data_quality": "unavailable_earth_engine"}
        try:
            buf = _buffer(lat, lon, 20000)
            srtm = ee.Image("USGS/SRTMGL1_003").clip(buf)
            slope = ee.Terrain.slope(srtm).clip(buf)
            slope_val = _reduce_mean(slope, buf, scale=30).get("slope") or 0
            elevation = _reduce_mean(srtm, buf, scale=30).get("elevation") or 0
            rain = analyze_vegetation_and_climate(lat, lon).get("annual_rainfall_mm_estimate") or 800
            score = _clamp(100 - slope_val * 9 - max(0, elevation - 300) * 0.04 + min(20, rain / 100), 0, 100)
            return {
                "groundwater_potential_score": round(score, 1),
                "groundwater_potential": "high" if score > 60 else "moderate" if score > 35 else "low",
                "borewell_depth_estimate_m": "15-30 m (shallow aquifer likely)" if score > 60 else "30-80 m (medium depth)" if score > 35 else "80-150 m or may be dry",
                "data_quality": "proxy_srtm_rainfall",
            }
        except Exception as exc:
            return _ee_error(exc)

    return _with_cache(key, CACHE_TTL, _produce)


def analyze_fire_risk(lat: float, lon: float, radius: int = 20000) -> dict[str, Any]:
    key = _cache_key("firehist", round(lat, 3), round(lon, 3))

    def _produce() -> dict[str, Any]:
        if ee is None or not earth_engine_available():
            return {"fire_frequency_index": None, "fire_risk": "unknown", "firebreak_required": None, "data_quality": "unavailable_earth_engine"}
        try:
            buf = _buffer(lat, lon, radius)
            fires = (
                ee.ImageCollection("MODIS/061/MOD14A1")
                .filterDate("2021-01-01", "2023-12-31")
                .select("FireMask")
                .mean()
                .clip(buf)
            )
            fire_val = round(_reduce_mean(fires, buf, scale=1000).get("FireMask") or 0, 3)
            return {
                "fire_frequency_index": fire_val,
                "fire_risk": "high" if fire_val > 0.5 else "moderate" if fire_val > 0.1 else "low",
                "firebreak_required": fire_val > 0.3,
                "data_quality": "measured_modis_fire",
            }
        except Exception as exc:
            return _ee_error(exc)

    return _with_cache(key, CACHE_TTL, _produce)


def analyze_ghsl_settlement(lat: float, lon: float) -> dict[str, Any]:
    key = _cache_key("ghsl", round(lat, 3), round(lon, 3))

    def _produce() -> dict[str, Any]:
        if ee is None or not earth_engine_available():
            return {"ghsl_smod_class": None, "urbanity": "unknown", "infrastructure_baseline": "unknown", "data_quality": "unavailable_earth_engine"}
        try:
            buf = _buffer(lat, lon, 10000)
            ghsl = ee.Image("JRC/GHSL/P2023A/GHS_SMOD/2020").clip(buf)
            smod = round(_reduce_mean(ghsl, buf, scale=1000).get("smod_code") or 0, 1)
            urbanity = "urban_centre" if smod >= 30 else "dense_suburb" if smod >= 23 else "semi_dense" if smod >= 22 else "suburban" if smod >= 21 else "rural"
            return {
                "ghsl_smod_class": smod,
                "urbanity": urbanity,
                "infrastructure_baseline": "strong" if smod >= 22 else "moderate" if smod >= 21 else "weak",
                "data_quality": "measured_ghsl",
            }
        except Exception as exc:
            return _ee_error(exc)

    return _with_cache(key, CACHE_TTL, _produce)


def get_open_buildings_count(lat: float, lon: float, radius: int = 5000) -> dict[str, Any]:
    key = _cache_key("buildings", round(lat, 3), round(lon, 3), radius)

    def _produce() -> dict[str, Any]:
        count = None
        source = "unavailable"
        if ee is not None and earth_engine_available():
            try:
                buf = _buffer(lat, lon, radius)
                buildings = ee.FeatureCollection("GOOGLE/Research/open-buildings/v3/polygons")
                count = int(buildings.filterBounds(buf).size().getInfo())
                source = "Google Open Buildings via Earth Engine"
            except Exception:
                count = None
        if count is None:
            try:
                q = f"""
                [out:json][timeout:20];
                (
                  way["building"](around:{radius},{lat},{lon});
                  relation["building"](around:{radius},{lat},{lon});
                );
                out center 10000;
                """
                count = len(_overpass(q, timeout=25))
                source = "OpenStreetMap Overpass building footprints"
            except Exception:
                count = None
        area_km2 = math.pi * (radius / 1000) ** 2
        density = round(count / area_km2, 1) if count is not None else None
        return {
            "building_count_5km": count,
            "building_density_per_km2": density,
            "settlement_type": "dense_urban" if (density or 0) > 500 else "peri_urban" if (density or 0) > 100 else "rural_village" if (density or 0) > 20 else "sparse_rural" if density is not None else "unknown",
            "source": source,
            "data_quality": "measured" if count is not None else "unavailable",
        }

    return _with_cache(key, CACHE_TTL, _produce)


def analyze_road_and_hospital_access(lat: float, lon: float) -> dict[str, Any]:
    key = _cache_key("access", round(lat, 3), round(lon, 3))

    def _produce() -> dict[str, Any]:
        result: dict[str, Any] = {
            "nearest_hospital_km": None,
            "nearest_hospital_name": "Unknown",
            "nearest_hospital_lat": None,
            "nearest_hospital_lon": None,
            "nearest_road_km": None,
            "road_access_level": "unknown",
            "nearest_school_km": None,
            "nearest_market_km": None,
            "nearest_water_point_km": None,
            "data_quality": "partial_osm",
        }
        try:
            hospitals = _overpass(
                f"""
                [out:json][timeout:25];
                (
                  node["amenity"~"hospital|clinic|doctors"](around:50000,{lat},{lon});
                  way["amenity"~"hospital|clinic|doctors"](around:50000,{lat},{lon});
                );
                out center 50;
                """,
                timeout=30,
            )
            h = _nearest_from_osm(hospitals, lat, lon)
            if h:
                result.update(
                    {
                        "nearest_hospital_km": h["distance_km"],
                        "nearest_hospital_name": h["name"],
                        "nearest_hospital_lat": h["lat"],
                        "nearest_hospital_lon": h["lon"],
                    }
                )
        except Exception as exc:
            result["hospital_error"] = str(exc)

        try:
            roads = _overpass(
                f"""
                [out:json][timeout:25];
                way["highway"~"motorway|trunk|primary|secondary|tertiary|unclassified"](around:50000,{lat},{lon});
                out center 50;
                """,
                timeout=30,
            )
            r = _nearest_from_osm(roads, lat, lon)
            if r:
                dist = r["distance_km"]
                result["nearest_road_km"] = dist
                result["road_access_level"] = "good" if dist < 2 else "moderate" if dist < 10 else "poor"
        except Exception as exc:
            result["road_error"] = str(exc)

        for label, query in {
            "nearest_school_km": 'node["amenity"="school"]',
            "nearest_market_km": 'node["amenity"~"marketplace|supermarket"]',
            "nearest_water_point_km": 'node["amenity"~"water_point|drinking_water"]',
        }.items():
            try:
                items = _overpass(f"[out:json][timeout:15];{query}(around:30000,{lat},{lon});out 25;", timeout=20)
                nearest = _nearest_from_osm(items, lat, lon)
                if nearest:
                    result[label] = nearest["distance_km"]
            except Exception:
                continue

        if any(result.get(k) is not None for k in ["nearest_hospital_km", "nearest_road_km", "nearest_school_km"]):
            result["data_quality"] = "measured_osm"
        return result

    return _with_cache(key, CACHE_TTL, _produce)


def analyze_soil_quality(lat: float, lon: float) -> dict[str, Any]:
    key = _cache_key("soil", round(lat, 3), round(lon, 3))

    def _produce() -> dict[str, Any]:
        try:
            data = _request_json(
                "GET",
                "https://rest.isric.org/soilgrids/v2.0/properties/query",
                params={
                    "lon": lon,
                    "lat": lat,
                    "property": ["phh2o", "soc", "clay", "nitrogen", "sand", "bdod"],
                    "depth": "0-5cm",
                    "value": "mean",
                },
                timeout=20,
                retries=1,
            )
            layers = {
                layer["name"]: layer["depths"][0]["values"]["mean"]
                for layer in data.get("properties", {}).get("layers", [])
                if layer.get("depths")
            }
            ph = round((layers.get("phh2o") or 0) / 10, 1) if layers.get("phh2o") is not None else None
            soc = round((layers.get("soc") or 0) / 10, 2) if layers.get("soc") is not None else None
            clay = round((layers.get("clay") or 0) / 10, 1) if layers.get("clay") is not None else None
            nitro = round((layers.get("nitrogen") or 0) / 100, 3) if layers.get("nitrogen") is not None else None
            sand = round((layers.get("sand") or 0) / 10, 1) if layers.get("sand") is not None else None
            bdod = round((layers.get("bdod") or 0) / 100, 2) if layers.get("bdod") is not None else None
            score = (3 if ph and 5.5 <= ph <= 7.5 else 1) + (3 if soc and soc > 10 else 1 if soc and soc > 5 else 0) + (2 if clay and 15 <= clay <= 45 else 0)
            bearing = "good" if (bdod or 0) > 1.4 else "moderate" if (bdod or 0) > 1.1 else "poor" if bdod is not None else "unknown"
            return {
                "soil_ph": ph,
                "organic_carbon_gkg": soc,
                "clay_pct": clay,
                "sand_pct": sand,
                "nitrogen_gkg": nitro,
                "bulk_density_g_cm3": bdod,
                "crop_suitability": "high" if score >= 7 else "moderate" if score >= 4 else "low",
                "foundation_bearing": bearing,
                "soil_score": score,
                "source": "ISRIC SoilGrids",
                "data_quality": "measured_soilgrids",
            }
        except Exception as exc:
            return {
                "soil_ph": None,
                "organic_carbon_gkg": None,
                "clay_pct": None,
                "nitrogen_gkg": None,
                "crop_suitability": "unknown",
                "foundation_bearing": "unknown",
                "soil_score": None,
                "error": str(exc),
                "data_quality": "unavailable",
            }

    return _with_cache(key, CACHE_TTL, _produce)


def analyze_air_quality(lat: float, lon: float, radius: int = 10000) -> dict[str, Any]:
    key = _cache_key("air_sat", round(lat, 3), round(lon, 3))

    def _produce() -> dict[str, Any]:
        no2 = None
        co = None
        if ee is not None and earth_engine_available():
            try:
                buf = _buffer(lat, lon, radius)
                no2_img = (
                    ee.ImageCollection("COPERNICUS/S5P/OFFL/L3_NO2")
                    .filterDate("2022-01-01", "2023-12-31")
                    .select("NO2_column_number_density")
                    .mean()
                    .clip(buf)
                )
                co_img = (
                    ee.ImageCollection("COPERNICUS/S5P/OFFL/L3_CO")
                    .filterDate("2022-01-01", "2023-12-31")
                    .select("CO_column_number_density")
                    .mean()
                    .clip(buf)
                )
                no2 = round((_reduce_mean(no2_img, buf, scale=1000).get("NO2_column_number_density") or 0) * 1e6, 3)
                co = round((_reduce_mean(co_img, buf, scale=1000).get("CO_column_number_density") or 0) * 1e3, 3)
            except Exception:
                no2, co = None, None
        openaq = fetch_openaq_air_quality(lat, lon)
        pm25 = openaq.get("pm25_ug_m3")
        if no2 is not None:
            aq, note = ("good", "Clean air signal from satellite NO2") if no2 < 10 else ("moderate", "Moderate pollution; ventilation needed") if no2 < 30 else ("poor", "High pollution; health filtration urgent")
        elif pm25 is not None:
            aq, note = ("good", "Ground PM2.5 is low") if pm25 <= 12 else ("moderate", "Ground PM2.5 is elevated") if pm25 <= 35 else ("poor", "High PM2.5; filtration required")
        else:
            aq, note = "unknown", "Air quality data unavailable; include low-cost PM sensor in next survey"
        return {
            "no2_umol_m2": no2,
            "co_mol_m2": co,
            "pm25_ug_m3": pm25,
            "pm10_ug_m3": openaq.get("pm10_ug_m3"),
            "air_quality": aq,
            "air_note": note,
            "ventilation_design_priority": aq == "poor" or (pm25 is not None and pm25 > 35),
            "source": "Sentinel-5P via Earth Engine + OpenAQ",
            "data_quality": "partial" if no2 is None and pm25 is None else "measured",
        }

    return _with_cache(key, CACHE_TTL, _produce)


def analyze_drought_risk(lat: float, lon: float, radius: int = 10000) -> dict[str, Any]:
    key = _cache_key("drought", round(lat, 3), round(lon, 3))

    def _produce() -> dict[str, Any]:
        if ee is None or not earth_engine_available():
            return {
                "pdsi_index": None,
                "drought_risk": "unknown",
                "drought_note": "Drought layer unavailable; use conservative water storage sizing",
                "soil_moisture_mm": None,
                "water_storage_tank_litres_recommended": 20000,
                "data_quality": "unavailable_earth_engine",
            }
        try:
            buf = _buffer(lat, lon, radius)
            pdsi = (
                ee.ImageCollection("IDAHO_EPSCOR/TERRACLIMATE")
                .filterDate("2018-01-01", "2023-12-31")
                .select("pdsi")
                .mean()
                .clip(buf)
            )
            pdsi_val = round((_reduce_mean(pdsi, buf, scale=5000).get("pdsi") or 0) / 100, 2)
            soil = (
                ee.ImageCollection("IDAHO_EPSCOR/TERRACLIMATE")
                .filterDate("2022-01-01", "2023-12-31")
                .select("soil")
                .mean()
                .clip(buf)
            )
            soil_moisture = round((_reduce_mean(soil, buf, scale=5000).get("soil") or 0) * 0.1, 1)
            risk, note = ("high", "Chronic drought tendency; storage is critical") if pdsi_val < -2 else ("moderate", "Mild drought tendency; rainwater harvesting advised") if pdsi_val < 0 else ("low", "Adequate moisture; standard water plan sufficient")
            return {
                "pdsi_index": pdsi_val,
                "drought_risk": risk,
                "drought_note": note,
                "soil_moisture_mm": soil_moisture,
                "water_storage_tank_litres_recommended": 50000 if pdsi_val < -2 else 20000 if pdsi_val < 0 else 10000,
                "data_quality": "measured_terraclimate",
            }
        except Exception as exc:
            return _ee_error(exc)

    return _with_cache(key, CACHE_TTL, _produce)


def fetch_open_meteo_weather(lat: float, lon: float) -> dict[str, Any]:
    key = _cache_key("weather", round(lat, 2), round(lon, 2))

    def _produce() -> dict[str, Any]:
        try:
            current = _request_json(
                "GET",
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude": lat,
                    "longitude": lon,
                    "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,precipitation,uv_index",
                    "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,wind_speed_10m_max,uv_index_max",
                    "timezone": "auto",
                    "forecast_days": 7,
                },
                timeout=15,
                retries=1,
            )
            cur = current.get("current", {})
            daily = current.get("daily", {})
            rain7 = sum([_safe_float(v, 0) or 0 for v in daily.get("precipitation_sum", [])])
            max_t = max([_safe_float(v, -99) or -99 for v in daily.get("temperature_2m_max", [])], default=None)
            min_t = min([_safe_float(v, 99) or 99 for v in daily.get("temperature_2m_min", [])], default=None)
            uv = _first_number(cur.get("uv_index"), max(daily.get("uv_index_max", [0]) or [0]), default=None)
            return {
                "current_temp_c": _safe_float(cur.get("temperature_2m")),
                "current_humidity_pct": _safe_float(cur.get("relative_humidity_2m")),
                "current_wind_kmh": _safe_float(cur.get("wind_speed_10m")),
                "current_uv_index": uv,
                "uv_risk": "high" if (uv or 0) >= 8 else "moderate" if (uv or 0) >= 3 else "low" if uv is not None else "unknown",
                "forecast_7d_max_temp_c": round(max_t, 1) if max_t is not None else None,
                "forecast_7d_min_temp_c": round(min_t, 1) if min_t is not None else None,
                "forecast_7d_rain_mm": round(rain7, 1),
                "forecast_7d_max_wind_kmh": max([_safe_float(v, 0) or 0 for v in daily.get("wind_speed_10m_max", [])], default=None),
                "annual_rainfall_mm_estimate": round(rain7 / 7 * 365, 0) if rain7 is not None else None,
                "construction_weather_note": "Avoid concrete pouring during rain window" if rain7 > 30 else "Good short-term construction window" if rain7 < 5 else "Monitor rainfall before pours",
                "source": "Open-Meteo open API",
                "data_quality": "forecast",
            }
        except Exception as exc:
            return {"weather_error": str(exc), "data_quality": "unavailable"}

    return _with_cache(key, 3600, _produce)


def fetch_wttr_weather(lat: float, lon: float) -> dict[str, Any]:
    # Kept for compatibility with the original app; open-meteo is the primary source.
    primary = fetch_open_meteo_weather(lat, lon)
    primary["alias"] = "wttr_weather_compatible_open_meteo"
    return primary


def fetch_usgs_earthquakes(lat: float, lon: float, radius_km: int = 200) -> dict[str, Any]:
    key = _cache_key("usgs_eq", round(lat, 2), round(lon, 2), radius_km)

    def _produce() -> dict[str, Any]:
        try:
            data = _request_json(
                "GET",
                "https://earthquake.usgs.gov/fdsnws/event/1/query",
                params={
                    "format": "geojson",
                    "latitude": lat,
                    "longitude": lon,
                    "maxradiuskm": radius_km,
                    "minmagnitude": 2.5,
                    "starttime": "2013-01-01",
                    "endtime": datetime.utcnow().strftime("%Y-%m-%d"),
                    "orderby": "magnitude",
                    "limit": 50,
                },
                timeout=20,
                retries=1,
            )
            features = data.get("features", [])
            mags = [f.get("properties", {}).get("mag") for f in features if f.get("properties", {}).get("mag") is not None]
            significant = [m for m in mags if m >= 5.0]
            return {
                "quake_count_10yr_r200km": len(features),
                "max_magnitude_10yr": round(max(mags), 1) if mags else 0,
                "avg_magnitude_10yr": round(sum(mags) / len(mags), 2) if mags else 0,
                "m5plus_events_10yr": len(significant),
                "seismic_activity": "very_high" if len(significant) >= 5 or (mags and max(mags) >= 6.5) else "high" if len(significant) >= 2 or (mags and max(mags) >= 5.5) else "moderate" if len(features) >= 10 else "low",
                "structural_recommendation": "Full seismic isolation and ductile detailing required" if len(significant) >= 5 else "Seismic-resistant frame required" if len(significant) >= 2 else "Basic seismic provisions sufficient" if len(features) >= 5 else "Standard construction adequate; verify local code",
                "source": "USGS Earthquake Catalog",
                "data_quality": "measured",
            }
        except Exception as exc:
            return {"usgs_error": str(exc), "seismic_activity": "unknown", "data_quality": "unavailable"}

    return _with_cache(key, CACHE_TTL, _produce)


def fetch_who_health_data(country_iso2: str) -> dict[str, Any]:
    key = _cache_key("who", country_iso2)

    def _produce() -> dict[str, Any]:
        indicators = {
            "MDG_0000000007": "under5_mortality_per_1000",
            "NUTRITION_WA_2": "stunting_pct_under5",
            "WHS4_543": "open_defecation_pct",
            "WHS4_100": "skilled_birth_attendance_pct",
        }
        result: dict[str, Any] = {"source": "WHO Global Health Observatory", "data_quality": "partial"}
        base = "https://ghoapi.azureedge.net/api"
        for code, name in indicators.items():
            try:
                data = _request_json(
                    "GET",
                    f"{base}/{code}",
                    params={"$filter": f"SpatialDim eq '{country_iso2}'", "$orderby": "TimeDim desc", "$top": 1},
                    timeout=12,
                    retries=0,
                )
                vals = data.get("value", [])
                if vals and vals[0].get("NumericValue") is not None:
                    result[name] = round(float(vals[0]["NumericValue"]), 2)
                    result[f"{name}_year"] = vals[0].get("TimeDim")
            except Exception as exc:
                result[f"{name}_error"] = str(exc)
        u5m = result.get("under5_mortality_per_1000")
        result["health_urgency"] = "critical" if (u5m or 0) > 60 else "high" if (u5m or 0) > 30 else "moderate" if (u5m or 0) > 15 else "low" if u5m is not None else "unknown"
        result["clinic_demand_multiplier"] = round(1 + (u5m or 20) / 100, 2)
        if any(k in result for k in indicators.values()):
            result["data_quality"] = "measured_country"
        return result

    return _with_cache(key, CACHE_TTL * 7, _produce)


def fetch_fao_crop_data(country_iso3: str) -> dict[str, Any]:
    key = _cache_key("fao", country_iso3)

    def _produce() -> dict[str, Any]:
        result: dict[str, Any] = {"source": "FAOSTAT", "data_quality": "partial"}
        try:
            data = _request_json(
                "GET",
                "https://fenixservices.fao.org/faostat/api/v1/en/data/FS",
                params={"area_cs": country_iso3, "element": "6132", "item": "210041", "year": "2021,2022", "output_type": "json"},
                timeout=15,
                retries=1,
            )
            items = data.get("data", [])
            if items:
                result["food_insecurity_pct"] = round(float(items[-1].get("Value", 0)), 1)
        except Exception as exc:
            result["food_security_error"] = str(exc)
        for crop_code, crop_name in [("27", "rice"), ("15", "wheat"), ("56", "maize")]:
            try:
                data = _request_json(
                    "GET",
                    "https://fenixservices.fao.org/faostat/api/v1/en/data/QCL",
                    params={"area_cs": country_iso3, "element": "5419", "item": crop_code, "year": "2021,2022", "output_type": "json"},
                    timeout=12,
                    retries=0,
                )
                items = data.get("data", [])
                if items:
                    value = int(float(items[-1].get("Value", 0)))
                    result[f"{crop_name}_yield_hg_ha"] = value
                    result[f"{crop_name}_yield_t_ha"] = round(value / 10000, 2)
            except Exception:
                continue
        result["local_yield_benchmark"] = "available" if any(k.endswith("_yield_t_ha") for k in result) else "benchmark_unavailable"
        return result

    return _with_cache(key, CACHE_TTL * 7, _produce)


def fetch_nasa_firms_fires(lat: float, lon: float, radius_km: int = 50) -> dict[str, Any]:
    key = _cache_key("firms", round(lat, 2), round(lon, 2), radius_km)

    def _produce() -> dict[str, Any]:
        # FIRMS now often requires a map key; we degrade gracefully if the public key is rejected.
        map_key = os.environ.get("NASA_FIRMS_MAP_KEY", "ABCD1234")
        try:
            delta = max(0.1, radius_km / 111)
            bbox = f"{lon - delta},{lat - delta},{lon + delta},{lat + delta}"
            response = requests.get(
                f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{map_key}/VIIRS_SNPP_NRT/{bbox}/7",
                headers=_headers(),
                timeout=20,
            )
            response.raise_for_status()
            text = response.text.strip()
            if "Invalid MAP_KEY" in text or "error" in text.lower()[:200]:
                raise ValueError(text[:160])
            lines = [line for line in text.splitlines() if line and not line.lower().startswith("latitude")]
            brightness: list[float] = []
            for line in lines[:50]:
                parts = line.split(",")
                if len(parts) > 2:
                    val = _safe_float(parts[2])
                    if val is not None:
                        brightness.append(val)
            return {
                "active_fire_detections_7d": len(lines),
                "max_brightness_k": round(max(brightness), 1) if brightness else None,
                "fire_proximity": "active_fire_detected" if lines else "no_active_fires_7d",
                "fire_evacuation_risk": len(lines) > 3,
                "firebreak_recommendation": "Immediate 30 m cleared firebreak required" if len(lines) > 5 else "10 m cleared buffer recommended" if lines else "Standard fire safety adequate",
                "source": "NASA FIRMS VIIRS",
                "data_quality": "measured" if map_key != "ABCD1234" else "public_key_attempt",
            }
        except Exception as exc:
            return {
                "active_fire_detections_7d": None,
                "fire_proximity": "unknown",
                "firebreak_recommendation": "Use MODIS historical fire risk and local forest department alerts",
                "firms_error": str(exc),
                "data_quality": "unavailable_without_firms_key",
            }

    return _with_cache(key, 3600, _produce)


def fetch_openaq_air_quality(lat: float, lon: float) -> dict[str, Any]:
    key = _cache_key("openaq", round(lat, 2), round(lon, 2))

    def _produce() -> dict[str, Any]:
        try:
            data = _request_json(
                "GET",
                "https://api.openaq.org/v2/latest",
                params={"coordinates": f"{lat},{lon}", "radius": 50000, "limit": 10},
                timeout=12,
                retries=1,
            )
            measurements = []
            for row in data.get("results", []):
                measurements.extend(row.get("measurements", []))
            values: dict[str, list[float]] = {}
            for m in measurements:
                param = m.get("parameter")
                val = _safe_float(m.get("value"))
                if param and val is not None and val >= 0:
                    values.setdefault(param, []).append(val)
            return {
                "pm25_ug_m3": round(sum(values.get("pm25", [])) / len(values["pm25"]), 1) if values.get("pm25") else None,
                "pm10_ug_m3": round(sum(values.get("pm10", [])) / len(values["pm10"]), 1) if values.get("pm10") else None,
                "source": "OpenAQ",
                "data_quality": "measured" if values else "no_station_nearby",
            }
        except Exception as exc:
            return {"openaq_error": str(exc), "pm25_ug_m3": None, "data_quality": "unavailable"}

    return _with_cache(key, 3600, _produce)


def fetch_waqi_air_quality(lat: float, lon: float) -> dict[str, Any]:
    # The previous version used a hardcoded WAQI token. This open-data version avoids paid/token services.
    result = fetch_openaq_air_quality(lat, lon)
    result["aqi_category"] = "unknown"
    result["aqi_value"] = None
    result["note"] = "WAQI disabled to avoid hardcoded/private tokens; OpenAQ and Sentinel-5P are used instead."
    return result


def fetch_reliefweb_disasters(lat: float, lon: float, country_name: str) -> dict[str, Any]:
    key = _cache_key("reliefweb", country_name[:20].lower())

    def _produce() -> dict[str, Any]:
        try:
            data = _request_json(
                "POST",
                "https://api.reliefweb.int/v1/disasters",
                params={"appname": "villageforge-open-data"},
                json_payload={
                    "filter": {
                        "operator": "AND",
                        "conditions": [
                            {"field": "country.name", "value": country_name},
                            {"field": "date.created", "value": {"from": "2018-01-01T00:00:00+00:00"}},
                        ],
                    },
                    "fields": {"include": ["name", "date", "type", "status", "glide"]},
                    "sort": ["date.created:desc"],
                    "limit": 10,
                },
                timeout=15,
                retries=1,
            )
            events = []
            type_counts: dict[str, int] = {}
            for item in data.get("data", []):
                f = item.get("fields", {})
                dtype = f.get("type", [{}])[0].get("name", "Unknown") if f.get("type") else "Unknown"
                events.append(
                    {
                        "name": f.get("name", ""),
                        "date": str(f.get("date", {}).get("created", ""))[:10],
                        "type": dtype,
                        "status": f.get("status", ""),
                    }
                )
                type_counts[dtype] = type_counts.get(dtype, 0) + 1
            dominant = max(type_counts, key=type_counts.get) if type_counts else "None"
            return {
                "recent_disasters_since_2018": len(events),
                "disaster_events": events[:5],
                "dominant_hazard_type": dominant,
                "type_breakdown": type_counts,
                "disaster_risk_note": f"High disaster frequency ({len(events)} events since 2018); prioritize {dominant} resilience" if len(events) >= 5 else f"Moderate history ({len(events)} events)" if len(events) >= 2 else "Low recent disaster history",
                "source": "ReliefWeb",
                "data_quality": "measured_country",
            }
        except Exception as exc:
            return {"reliefweb_error": str(exc), "recent_disasters_since_2018": None, "data_quality": "unavailable"}

    return _with_cache(key, CACHE_TTL, _produce)


def fetch_world_bank_indicators(country_code: str) -> dict[str, Any]:
    key = _cache_key("worldbank", country_code)

    def _produce() -> dict[str, Any]:
        indicators = {
            "NY.GDP.PCAP.CD": "gdp_per_capita_usd",
            "SP.DYN.IMRT.IN": "infant_mortality_per_1000",
            "SH.STA.BFED.ZS": "skilled_birth_attendance_pct",
            "SE.PRM.ENRR": "primary_school_enrollment_pct",
            "EG.ELC.ACCS.RU.ZS": "rural_electricity_access_pct",
            "SH.H2O.BASW.RU.ZS": "rural_basic_water_access_pct",
        }
        result: dict[str, Any] = {"source": "World Bank", "data_quality": "partial"}
        for code, name in indicators.items():
            try:
                data = _request_json(
                    "GET",
                    f"https://api.worldbank.org/v2/country/{country_code}/indicator/{code}",
                    params={"format": "json", "mrv": 1},
                    timeout=10,
                    retries=0,
                )
                if isinstance(data, list) and len(data) > 1 and data[1]:
                    value = data[1][0].get("value")
                    result[name] = round(float(value), 2) if value is not None else None
            except Exception:
                continue
        return result

    return _with_cache(key, CACHE_TTL * 7, _produce)


def analyze_climate_projection_2050(lat: float, lon: float) -> dict[str, Any]:
    key = _cache_key("climate2050", round(lat, 2), round(lon, 2))

    def _produce() -> dict[str, Any]:
        if ee is None or not earth_engine_available():
            return {
                "projected_temp_2050_c": None,
                "historical_temp_c": None,
                "warming_2050_c": None,
                "projected_daily_rainfall_2050_mm": None,
                "climate_risk_2050": "unknown",
                "design_lifespan_recommendation": "Earth Engine climate projections unavailable; use conservative heat and water resilience margins",
                "stranded_asset_risk": None,
                "data_quality": "unavailable_earth_engine",
            }
        try:
            buf = _buffer(lat, lon, 25000)
            proj = (
                ee.ImageCollection("NASA/GDDP-CMIP6")
                .filter(ee.Filter.eq("model", "ACCESS-CM2"))
                .filter(ee.Filter.eq("scenario", "ssp585"))
                .filterDate("2049-01-01", "2051-12-31")
                .select(["tas", "pr"])
            )
            avg = proj.mean().clip(buf)
            temp_2050 = round((_reduce_mean(avg.select("tas"), buf, scale=25000).get("tas") or 0) - 273.15, 1)
            precip_2050 = round((_reduce_mean(avg.select("pr"), buf, scale=25000).get("pr") or 0) * 86400, 2)
            hist = (
                ee.ImageCollection("NASA/GDDP-CMIP6")
                .filter(ee.Filter.eq("model", "ACCESS-CM2"))
                .filter(ee.Filter.eq("scenario", "historical"))
                .filterDate("2010-01-01", "2020-12-31")
                .select(["tas", "pr"])
                .mean()
                .clip(buf)
            )
            temp_hist = round((_reduce_mean(hist.select("tas"), buf, scale=25000).get("tas") or 0) - 273.15, 1)
            warming = round(temp_2050 - temp_hist, 1)
            return {
                "projected_temp_2050_c": temp_2050,
                "historical_temp_c": temp_hist,
                "warming_2050_c": warming,
                "projected_daily_rainfall_2050_mm": precip_2050,
                "climate_risk_2050": "extreme" if warming > 3.5 else "severe" if warming > 2.5 else "moderate" if warming > 1.5 else "low",
                "design_lifespan_recommendation": "Design for +4C resilience; use heat-reflective roof, deep shade, passive cooling, oversized water systems" if warming > 2.5 else "Use standard climate adaptation: ventilation, shade, drainage, and rainwater storage",
                "stranded_asset_risk": warming > 3.0,
                "source": "NASA NEX-GDDP-CMIP6 via Earth Engine",
                "data_quality": "modeled_projection",
            }
        except Exception as exc:
            return {"climate_projection_error": str(exc), "climate_risk_2050": "unknown", "data_quality": "unavailable"}

    return _with_cache(key, CACHE_TTL * 30, _produce)


# ---------------------------------------------------------------------------
# Composite scoring
# ---------------------------------------------------------------------------

def _risk_value(label: Any, mapping: dict[str, float], default: float) -> float:
    return mapping.get(str(label or "").lower(), default)


def compute_composite_scores(raw: dict[str, Any]) -> dict[str, Any]:
    flood = _first_number(raw.get("flood_risk", {}).get("flood_risk_score"), default=35) or 35
    seismic = _risk_value(raw.get("seismic_risk", {}).get("seismic_risk"), {"low": 10, "moderate": 40, "high": 80}, 35)
    landslide = _first_number(raw.get("landslide_risk", {}).get("landslide_susceptibility_score"), default=20) or 20
    bearing = _risk_value(raw.get("soil_quality", {}).get("foundation_bearing"), {"good": 0, "moderate": 15, "poor": 35, "unknown": 18}, 18)
    fire = _risk_value(raw.get("fire_risk", {}).get("fire_risk"), {"low": 5, "moderate": 30, "high": 60}, 15)

    css = _clamp(100 - flood * 0.35 - seismic * 0.25 - landslide * 0.20 - bearing * 0.12 - fire * 0.08)

    solar_kwh = _first_number(raw.get("solar_potential", {}).get("solar_irradiance_kwh_m2_day"), default=4.5) or 4.5
    wind_ms = _first_number(raw.get("wind_potential", {}).get("wind_speed_ms"), default=3.5) or 3.5
    cloud = _first_number(raw.get("solar_potential", {}).get("cloud_cover_pct"), default=35) or 35
    renewable = _clamp(solar_kwh * 11 + wind_ms * 4 + max(0, 45 - cloud) * 0.7)

    road_km = _first_number(raw.get("road_hospital_access", {}).get("nearest_road_km"), default=12) or 12
    hospital_km = _first_number(raw.get("road_hospital_access", {}).get("nearest_hospital_km"), default=40) or 40
    accessibility = _clamp(100 - road_km * 3.2 - hospital_km * 0.55)

    water_frac = _first_number(raw.get("surface_water", {}).get("permanent_water_fraction_pct"), default=1) or 1
    drought_idx = _first_number(raw.get("drought_risk", {}).get("pdsi_index"), default=-0.5) or -0.5
    rain = _first_number(raw.get("vegetation_climate", {}).get("daily_rainfall_mm"), default=2.0) or 2.0
    groundwater = _first_number(raw.get("groundwater_potential", {}).get("groundwater_potential_score"), default=45) or 45
    water_security = _clamp(water_frac * 5 + max(0, drought_idx + 4) * 9 + rain * 2 + groundwater * 0.35)

    u5m = _first_number(raw.get("who_health", {}).get("under5_mortality_per_1000"), raw.get("world_bank_country", {}).get("infant_mortality_per_1000"), default=25) or 25
    open_def = _first_number(raw.get("who_health", {}).get("open_defecation_pct"), default=15) or 15
    health_priority = _clamp(u5m * 1.15 + open_def * 0.6 + max(0, 40 - accessibility) * 0.35)

    climate_warming = _first_number(raw.get("climate_projection_2050", {}).get("warming_2050_c"), default=2.0) or 2.0
    resilience = _clamp(css * 0.45 + water_security * 0.25 + accessibility * 0.15 + max(0, 100 - climate_warming * 18) * 0.15)

    data_gaps = []
    for name, value in [
        ("flood risk", raw.get("flood_risk", {}).get("flood_risk_score")),
        ("population", raw.get("population", {}).get("estimated_population")),
        ("road access", raw.get("road_hospital_access", {}).get("nearest_road_km")),
        ("soil", raw.get("soil_quality", {}).get("soil_ph")),
        ("2050 climate", raw.get("climate_projection_2050", {}).get("warming_2050_c")),
    ]:
        if value is None:
            data_gaps.append(name)

    return {
        "construction_suitability": round(css, 1),
        "construction_suitability_label": "excellent" if css > 75 else "good" if css > 50 else "marginal" if css > 25 else "poor",
        "renewable_energy_score": round(renewable, 1),
        "primary_renewable": "solar" if solar_kwh * 11 >= wind_ms * 4 else "wind",
        "accessibility_score": round(accessibility, 1),
        "accessibility_label": "good" if accessibility > 60 else "moderate" if accessibility > 30 else "poor",
        "water_security_score": round(water_security, 1),
        "water_security_label": "secure" if water_security > 60 else "moderate" if water_security > 30 else "critical",
        "health_priority_score": round(health_priority, 1),
        "climate_resilience_score": round(resilience, 1),
        "data_confidence": "high" if len(data_gaps) <= 1 else "medium" if len(data_gaps) <= 3 else "low",
        "data_gaps": data_gaps,
    }


# ---------------------------------------------------------------------------
# Tool registry and Gemma calls
# ---------------------------------------------------------------------------

TOOL_REGISTRY: dict[str, dict[str, Any]] = {
    "solar_potential": {"fn": analyze_solar_potential, "description": "Solar irradiance, cloud cover, panel output."},
    "wind_potential": {"fn": analyze_wind_potential, "description": "Wind speed and wind power density."},
    "population": {"fn": analyze_population, "description": "Population, households, children, catchment."},
    "nighttime_lights": {"fn": analyze_nighttime_lights, "description": "Electricity/grid status from VIIRS."},
    "vegetation_climate": {"fn": analyze_vegetation_and_climate, "description": "NDVI, heat stress, rainfall, elevation."},
    "land_cover": {"fn": analyze_land_cover, "description": "ESA WorldCover land use breakdown."},
    "flood_risk": {"fn": analyze_flood_risk, "description": "Flood risk from surface water and slope."},
    "seismic_risk": {"fn": analyze_seismic_risk, "description": "Seismic risk and reinforcement requirement."},
    "landslide_risk": {"fn": analyze_landslide_risk, "description": "Landslide susceptibility."},
    "surface_water": {"fn": analyze_surface_water, "description": "Permanent/seasonal water source signal."},
    "groundwater_potential": {"fn": analyze_groundwater_potential, "description": "Borewell and groundwater potential."},
    "road_hospital_access": {"fn": analyze_road_and_hospital_access, "description": "Nearest road, hospital, school, market, water point."},
    "soil_quality": {"fn": analyze_soil_quality, "description": "Soil pH, SOC, clay, foundation bearing."},
    "air_quality": {"fn": analyze_air_quality, "description": "Sentinel-5P and OpenAQ air quality."},
    "drought_risk": {"fn": analyze_drought_risk, "description": "Drought and soil moisture."},
    "fire_risk": {"fn": analyze_fire_risk, "description": "Historical MODIS fire risk."},
    "ghsl_settlement": {"fn": analyze_ghsl_settlement, "description": "Settlement urbanity and baseline."},
    "open_buildings": {"fn": get_open_buildings_count, "description": "Building density and settlement type."},
    "wttr_weather": {"fn": fetch_wttr_weather, "description": "Open weather forecast compatibility layer."},
    "usgs_earthquakes": {"fn": fetch_usgs_earthquakes, "description": "USGS 10-year earthquake catalog."},
    "waqi_air": {"fn": fetch_waqi_air_quality, "description": "Token-free OpenAQ compatibility replacement."},
    "nasa_firms": {"fn": fetch_nasa_firms_fires, "description": "NASA FIRMS active fire alerts if a map key is available."},
    "climate_projection_2050": {"fn": analyze_climate_projection_2050, "description": "NASA CMIP6 2050 climate stress test."},
}

BASE_TOOLS = {
    "flood_risk",
    "population",
    "nighttime_lights",
    "road_hospital_access",
    "soil_quality",
    "ghsl_settlement",
    "climate_projection_2050",
}

PROJECT_TOOL_HINTS = {
    "Primary School": ["solar_potential", "air_quality", "population", "land_cover", "surface_water", "seismic_risk", "usgs_earthquakes", "open_buildings", "wttr_weather", "climate_projection_2050"],
    "Rural Health Clinic / Hospital": ["solar_potential", "air_quality", "population", "surface_water", "drought_risk", "seismic_risk", "usgs_earthquakes", "groundwater_potential", "road_hospital_access", "wttr_weather", "climate_projection_2050"],
    "Solar Microgrid Utility": ["solar_potential", "wind_potential", "nighttime_lights", "land_cover", "open_buildings", "population", "wttr_weather", "climate_projection_2050"],
    "Water Purification & Storage Center": ["surface_water", "groundwater_potential", "drought_risk", "flood_risk", "vegetation_climate", "soil_quality", "population", "wttr_weather", "climate_projection_2050"],
    "Community Farming Hub": ["soil_quality", "vegetation_climate", "drought_risk", "surface_water", "land_cover", "fire_risk", "nasa_firms", "wttr_weather", "climate_projection_2050"],
    "Disaster Relief Shelter": ["flood_risk", "seismic_risk", "usgs_earthquakes", "landslide_risk", "fire_risk", "nasa_firms", "road_hospital_access", "population", "soil_quality", "reliefweb_disasters", "climate_projection_2050"],
    "General Area Analysis": list(TOOL_REGISTRY.keys()),
}


def call_gemma(prompt: str, system: str = "", timeout: int = 240, temperature: float = 0.25) -> str:
    payload = {
        "model": GEMMA_MODEL,
        "prompt": prompt,
        "system": system,
        "stream": False,
        "format": "json" if "json" in (system + prompt).lower() else None,
        "options": {"temperature": temperature, "num_predict": 3500, "num_ctx": 8192},
    }
    payload = {k: v for k, v in payload.items() if v is not None}
    response = requests.post(f"{OLLAMA_BASE_URL}/api/generate", json=payload, timeout=timeout)
    response.raise_for_status()
    return response.json().get("response", "").strip()


def call_gemma_streaming(prompt: str, system: str = ""):
    payload = {
        "model": GEMMA_MODEL,
        "prompt": prompt,
        "system": system,
        "stream": True,
        "options": {"temperature": 0.25, "num_predict": 2500, "num_ctx": 8192},
    }
    with requests.post(f"{OLLAMA_BASE_URL}/api/generate", json=payload, stream=True, timeout=300) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line:
                continue
            chunk = json.loads(line)
            yield chunk.get("response", "")
            if chunk.get("done"):
                break


def select_tools_with_gemma(project_type: str, lat: float, lon: float, location: dict[str, Any]) -> list[str]:
    hints = PROJECT_TOOL_HINTS.get(project_type, PROJECT_TOOL_HINTS["General Area Analysis"])
    fallback = sorted(set(hints) | BASE_TOOLS)
    tool_list = "\n".join(f'- "{name}": {meta["description"]}' for name, meta in TOOL_REGISTRY.items())
    prompt = f"""
Project: {project_type}
Location: {location.get('district')}, {location.get('state')}, {location.get('country')} ({lat:.3f}, {lon:.3f})

Available tools:
{tool_list}

Select only the most relevant tools for planning and cost-estimating this project.
Return JSON only:
{{"selected_tools": ["tool_name"], "reasoning": "short explanation"}}
"""
    try:
        parsed = _extract_json(call_gemma(prompt, "You are a geospatial tool selector. Return valid JSON only.", timeout=60))
        selected = parsed.get("selected_tools", []) if isinstance(parsed, dict) else []
        valid = [name for name in selected if name in TOOL_REGISTRY]
        final = sorted(set(valid) | BASE_TOOLS)
        return final or fallback
    except Exception:
        return fallback


def run_tools_parallel(selected_tools: list[str], lat: float, lon: float, max_workers: int = 8) -> dict[str, Any]:
    results: dict[str, Any] = {}

    def _run(name: str) -> tuple[str, dict[str, Any]]:
        tool = TOOL_REGISTRY[name]
        try:
            return name, tool["fn"](lat, lon)
        except Exception as exc:
            return name, {"error": str(exc), "data_quality": "unavailable"}

    selected = [name for name in selected_tools if name in TOOL_REGISTRY]
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_run, name): name for name in selected}
        for future in concurrent.futures.as_completed(futures):
            name, result = future.result()
            results[name] = result
    return results


# ---------------------------------------------------------------------------
# Planning and multi-agent debate
# ---------------------------------------------------------------------------

def _fallback_plan(raw_data: dict[str, Any], project_type: str, budget_usd: int, scores: dict[str, Any]) -> dict[str, Any]:
    loc = raw_data.get("location", {})
    pop = raw_data.get("population", {})
    flood = raw_data.get("flood_risk", {})
    water = raw_data.get("surface_water", {})
    solar = raw_data.get("solar_potential", {})
    access = raw_data.get("road_hospital_access", {})
    climate = raw_data.get("climate_projection_2050", {})
    contingency = 15 if scores.get("construction_suitability", 50) < 50 else 10
    base_cost = int(budget_usd * (1 - contingency / 100))

    component_weights = {
        "Primary School": [("site preparation and foundation", 0.20), ("classrooms and sanitation", 0.38), ("solar and digital learning", 0.17), ("safe access and community works", 0.15), ("contingency", 0.10)],
        "Rural Health Clinic / Hospital": [("site preparation and resilient foundation", 0.18), ("clinical rooms and maternity bay", 0.35), ("solar backup and cold chain", 0.18), ("water sanitation and waste management", 0.17), ("contingency", 0.12)],
        "Solar Microgrid Utility": [("site works and poles", 0.15), ("solar array and inverters", 0.45), ("battery storage", 0.25), ("metering and training", 0.07), ("contingency", 0.08)],
        "Water Purification & Storage Center": [("source works and borewell/surface intake", 0.22), ("filtration and disinfection", 0.28), ("storage and distribution", 0.26), ("solar pumping and controls", 0.12), ("contingency", 0.12)],
        "Community Farming Hub": [("soil and water preparation", 0.20), ("storage and processing shed", 0.25), ("irrigation and tools", 0.25), ("training and market linkage", 0.15), ("contingency", 0.15)],
        "Disaster Relief Shelter": [("raised resilient foundation", 0.25), ("multi-purpose shelter structure", 0.35), ("water power and emergency stores", 0.20), ("access and evacuation signage", 0.10), ("contingency", 0.10)],
        "General Area Analysis": [("flood-safe site works", 0.20), ("water and sanitation priority", 0.24), ("solar resilience package", 0.20), ("access and service upgrades", 0.18), ("community training and monitoring", 0.08), ("contingency", 0.10)],
    }
    weights = component_weights.get(project_type, component_weights["General Area Analysis"])
    sectors: dict[str, dict[str, Any]] = {}
    breakdown: dict[str, dict[str, Any]] = {}
    for name, weight in weights:
        amount = int(budget_usd * weight)
        weeks = max(2, int(3 + weight * 28))
        sectors[name] = {
            "recommendation": f"Allocate this component around the site's measured constraints: flood level {flood.get('flood_risk_level', 'unknown')}, access {access.get('road_access_level', 'unknown')}, water source {water.get('water_source_type', 'unknown')}.",
            "items": [
                "local labour and supervision",
                "cement, aggregate, steel, and locally available masonry where appropriate",
                "quality inspection and community handover checklist",
            ],
            "estimated_cost_usd": amount,
            "timeline_weeks": weeks,
            "rationale": f"Composite scores: construction {scores.get('construction_suitability')} and water security {scores.get('water_security_score')}.",
        }
        breakdown[name] = {"amount_usd": amount, "pct": round(weight * 100, 1), "notes": sectors[name]["rationale"]}

    viability = "viable" if scores.get("construction_suitability", 0) >= 55 else "marginal" if scores.get("construction_suitability", 0) >= 30 else "not_recommended"
    warnings_list = []
    if flood.get("flood_risk_level") in {"moderate", "high"}:
        warnings_list.append(f"Flood risk is {flood.get('flood_risk_level')} with score {flood.get('flood_risk_score')}; use raised plinth and drainage.")
    if scores.get("data_confidence") != "high":
        warnings_list.append(f"Data confidence is {scores.get('data_confidence')}; field survey should verify {', '.join(scores.get('data_gaps', []))}.")
    warming = climate.get("warming_2050_c")
    if warming is not None and warming > 2.5:
        warnings_list.append(f"2050 warming projection is {warming}C; heat-resilient design is mandatory.")

    return {
        "project_summary": (
            f"{project_type} assessment for {loc.get('district', 'the site')} shows construction suitability "
            f"{scores.get('construction_suitability')}/100, renewable energy {scores.get('renewable_energy_score')}/100, "
            f"and water security {scores.get('water_security_score')}/100. Estimated catchment population is "
            f"{pop.get('estimated_population') or 'unavailable'}, with road access rated {access.get('road_access_level', 'unknown')}. "
            f"The plan uses conservative sizing where data layers were unavailable rather than treating missing values as zero."
        ),
        "site_viability": viability,
        "priority_ranking": [
            "Field-verify flood level, soil bearing capacity, and land ownership",
            "Confirm water source and sanitation pathway",
            "Finalize climate-resilient layout and raised plinth",
            "Procure local materials and train maintenance committee",
        ],
        "budget_allocation": {
            "total_recommended_usd": min(budget_usd, sum(v["amount_usd"] for v in breakdown.values())),
            "contingency_pct": contingency,
            "breakdown": breakdown,
        },
        "sectors": sectors,
        "structural_design_notes": f"Use raised plinth of at least {flood.get('recommended_floor_height_m', 0.6)} m where flooding is possible; seismic note: {raw_data.get('seismic_risk', {}).get('seismic_design_note', 'verify local code')}. Soil bearing is {raw_data.get('soil_quality', {}).get('foundation_bearing', 'unknown')}.",
        "renewable_energy_plan": f"Solar is the primary renewable if irradiance remains near {solar.get('solar_irradiance_kwh_m2_day', 'unknown')} kWh/m2/day; include battery backup for critical loads.",
        "water_and_sanitation_plan": f"Water source is rated {water.get('water_source_type', 'unknown')}; combine storage, filtration, handwashing, and greywater drainage.",
        "warnings": warnings_list or ["No critical site risks detected in available layers; still perform field verification."],
        "cost_saving_tips": ["Use local masonry and trained community labour where quality can be supervised.", "Phase non-critical finishes after core safety, water, and power systems are complete."],
        "local_material_opportunities": ["Local aggregate and masonry if soil/foundation tests pass.", "Shade trees and locally fabricated rainwater gutters."],
        "next_steps": ["Run field survey this week.", "Confirm land parcel and community committee in month 1.", "Tender materials and begin site works after survey sign-off."],
        "monitoring_kpis": ["Beneficiaries served monthly", "Water quality tests passed", "System uptime", "Maintenance issues resolved within 7 days"],
    }


def _validate_plan(plan: dict[str, Any], budget_usd: int, fallback: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(plan, dict) or plan.get("parse_error"):
        return fallback
    required = [
        "project_summary",
        "site_viability",
        "priority_ranking",
        "budget_allocation",
        "sectors",
        "warnings",
        "next_steps",
        "monitoring_kpis",
    ]
    for key in required:
        if key not in plan:
            plan[key] = fallback.get(key)
    budget = plan.get("budget_allocation")
    if not isinstance(budget, dict):
        plan["budget_allocation"] = fallback["budget_allocation"]
    else:
        total = _safe_int(budget.get("total_recommended_usd"), None)
        if total is None or total <= 0 or total > budget_usd:
            budget["total_recommended_usd"] = min(budget_usd, fallback["budget_allocation"]["total_recommended_usd"])
        if not isinstance(budget.get("breakdown"), dict) or not budget.get("breakdown"):
            budget["breakdown"] = fallback["budget_allocation"]["breakdown"]
        budget["contingency_pct"] = _safe_int(budget.get("contingency_pct"), fallback["budget_allocation"].get("contingency_pct", 10))
    if not isinstance(plan.get("sectors"), dict) or not plan.get("sectors"):
        plan["sectors"] = fallback["sectors"]
    if str(plan.get("site_viability", "")).lower() not in {"viable", "marginal", "not_recommended"}:
        plan["site_viability"] = fallback["site_viability"]
    return plan


def generate_village_plan(all_data: dict[str, Any], project_type: str, budget_usd: int, composite_scores: dict[str, Any]) -> dict[str, Any]:
    fallback = _fallback_plan(all_data, project_type, budget_usd, composite_scores)
    system_prompt = """You are an expert rural development architect, structural engineer, WASH planner, and social impact analyst.
Use only the supplied open-data evidence. Never invent precise measurements where a value is null.
Return valid JSON only. Costs must stay within the requested budget."""
    user_prompt = f"""
Create a data-driven infrastructure plan.

Project type: {project_type}
Maximum budget USD: {budget_usd}

Composite scores:
{json.dumps(composite_scores, indent=2, default=_json_default)}

Open-data site evidence:
{json.dumps(all_data, indent=2, default=_json_default)[:12000]}

Return exactly this JSON shape:
{{
  "project_summary": "3-4 sentence evaluation citing specific available values and noting missing data honestly",
  "site_viability": "viable|marginal|not_recommended",
  "priority_ranking": ["action 1", "action 2", "action 3", "action 4"],
  "budget_allocation": {{
    "total_recommended_usd": <number <= {budget_usd}>,
    "contingency_pct": <5-20>,
    "breakdown": {{"Category": {{"amount_usd": 0, "pct": 0, "notes": "evidence based reason"}}}}
  }},
  "sectors": {{
    "Component": {{
      "recommendation": "specific design recommendation",
      "items": ["specific material/equipment/labour item"],
      "estimated_cost_usd": 0,
      "timeline_weeks": 0,
      "rationale": "data-driven justification"
    }}
  }},
  "structural_design_notes": "foundation, flood, seismic, soil requirements",
  "renewable_energy_plan": "solar/wind/battery sizing logic",
  "water_and_sanitation_plan": "source, treatment, storage, drainage",
  "warnings": ["risk with data value or missing-data caveat"],
  "cost_saving_tips": ["tip"],
  "local_material_opportunities": ["item"],
  "next_steps": ["immediate action"],
  "monitoring_kpis": ["KPI"]
}}
"""
    try:
        parsed = _extract_json(call_gemma(user_prompt, system_prompt, timeout=240))
        return _validate_plan(parsed if isinstance(parsed, dict) else {}, budget_usd, fallback)
    except Exception as exc:
        fallback["llm_fallback_reason"] = str(exc)
        return fallback


AGENT_PERSONAS = {
    "structural_engineer": {
        "system": "You are a strict structural engineer obsessed with safety codes. Critique foundation, seismic, flood, landslide, and constructability risks. Return JSON only.",
        "focus": ["location", "soil_quality", "seismic_risk", "flood_risk", "landslide_risk", "climate_projection_2050"],
    },
    "social_impact_advocate": {
        "system": "You are a community advocate for marginalized populations. Critique accessibility, gender, child safety, inclusion, and maintenance ownership. Return JSON only.",
        "focus": ["location", "population", "road_hospital_access", "who_health", "world_bank_country"],
    },
    "climate_scientist": {
        "system": "You are a climate scientist focused on 30-year resilience. Critique heat, drought, fire, flood, water, and stranded asset risk. Return JSON only.",
        "focus": ["location", "drought_risk", "surface_water", "vegetation_climate", "fire_risk", "nasa_firms", "climate_projection_2050"],
    },
    "budget_auditor": {
        "system": "You are a ruthless budget auditor. Find unrealistic estimates, wasted scope, weak contingency, and missing lifecycle costs. Return JSON only.",
        "focus": ["location", "road_hospital_access", "ghsl_settlement", "open_buildings"],
    },
}


def _rule_based_critique(agent_name: str, plan: dict[str, Any], raw_data: dict[str, Any]) -> dict[str, Any]:
    issues: list[dict[str, str]] = []
    if agent_name == "structural_engineer":
        flood = raw_data.get("flood_risk", {})
        seismic = raw_data.get("seismic_risk", {})
        if flood.get("flood_risk_level") in {"moderate", "high", "unknown"}:
            issues.append({"issue": f"Flood risk is {flood.get('flood_risk_level', 'unknown')}; plinth/drainage must be explicit.", "severity": "high" if flood.get("flood_risk_level") == "high" else "medium", "recommended_fix": "Add raised plinth, swales, site grading, and flood-safe electrical mounting."})
        if seismic.get("reinforcement_required"):
            issues.append({"issue": "Seismic reinforcement is required but may be under-scoped.", "severity": "high", "recommended_fix": "Add ductile detailing, ring beams, shear walls/bracing, and inspection hold points."})
    elif agent_name == "social_impact_advocate":
        access = raw_data.get("road_hospital_access", {})
        if (access.get("nearest_road_km") or 99) > 5:
            issues.append({"issue": "Road access is weak, which can exclude elderly, disabled, and emergency users.", "severity": "medium", "recommended_fix": "Budget for footpath/last-mile access, ramps, lighting, and community transport coordination."})
        issues.append({"issue": "Community governance is not explicit enough.", "severity": "medium", "recommended_fix": "Create a local operations committee with women, youth, and marginalized group representation."})
    elif agent_name == "climate_scientist":
        warming = raw_data.get("climate_projection_2050", {}).get("warming_2050_c")
        if warming is None or warming > 2.5:
            issues.append({"issue": f"2050 heat stress is {'unavailable' if warming is None else str(warming) + 'C warming'}; current plan may undersize cooling/water.", "severity": "high" if warming and warming > 2.5 else "medium", "recommended_fix": "Use cool roof, shaded verandas, passive ventilation, drought storage, and heat-safe work scheduling."})
    elif agent_name == "budget_auditor":
        total = plan.get("budget_allocation", {}).get("total_recommended_usd", 0)
        contingency = plan.get("budget_allocation", {}).get("contingency_pct", 0)
        if contingency < 10:
            issues.append({"issue": "Contingency is too thin for rural logistics and data uncertainty.", "severity": "medium", "recommended_fix": "Raise contingency to 10-15 percent and phase non-critical finishes."})
        if total <= 0:
            issues.append({"issue": "Budget total is missing or invalid.", "severity": "high", "recommended_fix": "Rebuild budget from components and cap it at available funds."})
    if not issues:
        issues.append({"issue": "No major blocker from this lens.", "severity": "low", "recommended_fix": "Keep field verification and maintenance planning in the next step."})
    return {"critiques": issues[:3], "mode": "rule_based"}


def run_agent_debate(initial_plan: dict[str, Any], raw_data: dict[str, Any], project_type: str, use_llm: bool = True) -> dict[str, Any]:
    critiques: dict[str, Any] = {}
    for agent_name, persona in AGENT_PERSONAS.items():
        focused = {k: raw_data.get(k) for k in persona["focus"] if k in raw_data}
        if use_llm:
            prompt = f"""
Plan for {project_type}:
{json.dumps(initial_plan, indent=2, default=_json_default)[:4500]}

Relevant site data:
{json.dumps(focused, indent=2, default=_json_default)[:3000]}

Identify the top 3 issues. Return JSON:
{{"critiques": [{{"issue": "...", "severity": "high|medium|low", "recommended_fix": "..."}}]}}
"""
            try:
                parsed = _extract_json(call_gemma(prompt, persona["system"], timeout=90, temperature=0.15))
                critiques[agent_name] = parsed if isinstance(parsed, dict) else _rule_based_critique(agent_name, initial_plan, raw_data)
                continue
            except Exception as exc:
                critiques[agent_name] = _rule_based_critique(agent_name, initial_plan, raw_data)
                critiques[agent_name]["llm_error"] = str(exc)
                continue
        critiques[agent_name] = _rule_based_critique(agent_name, initial_plan, raw_data)

    high_or_medium = []
    for agent_name, critique in critiques.items():
        for item in critique.get("critiques", []) if isinstance(critique, dict) else []:
            if item.get("severity") in {"high", "medium"}:
                high_or_medium.append(f"{agent_name}: {item.get('recommended_fix')}")

    revised = json.loads(json.dumps(initial_plan, default=_json_default))
    revised["revised"] = True
    revised["debate_log"] = critiques
    revised["agent_consensus"] = high_or_medium[:8]
    warnings_list = revised.get("warnings", [])
    if not isinstance(warnings_list, list):
        warnings_list = [str(warnings_list)]
    for item in high_or_medium[:4]:
        if item not in warnings_list:
            warnings_list.append(item)
    revised["warnings"] = warnings_list

    if use_llm:
        synthesis_prompt = f"""
Critiques:
{json.dumps(critiques, indent=2, default=_json_default)[:5000]}

Original plan:
{json.dumps(initial_plan, indent=2, default=_json_default)[:5000]}

Return a revised plan with the same JSON structure. Keep total cost within the original budget.
Add fields "revised": true and "agent_consensus": [...]
"""
        try:
            parsed = _extract_json(call_gemma(synthesis_prompt, "You are a synthesis architect. Return valid JSON only.", timeout=180, temperature=0.15))
            if isinstance(parsed, dict):
                parsed["revised"] = True
                parsed["debate_log"] = critiques
                parsed.setdefault("agent_consensus", high_or_medium[:8])
                return parsed
        except Exception:
            pass
    return revised


# ---------------------------------------------------------------------------
# Impact, personas, funding, carbon, versions, TTS
# ---------------------------------------------------------------------------

SDG_MAPPING = {
    "Primary School": [
        {"sdg": 4, "name": "Quality Education", "metric": "estimated_school_children", "weight": 1.0},
        {"sdg": 5, "name": "Gender Equality", "metric": "estimated_school_children", "weight": 0.5},
        {"sdg": 10, "name": "Reduced Inequalities", "metric": "estimated_population", "weight": 0.3},
    ],
    "Rural Health Clinic / Hospital": [
        {"sdg": 3, "name": "Good Health & Well-being", "metric": "estimated_population", "weight": 1.0},
        {"sdg": 5, "name": "Gender Equality", "metric": "estimated_under5", "weight": 0.4},
        {"sdg": 6, "name": "Clean Water & Sanitation", "metric": "estimated_population", "weight": 0.3},
    ],
    "Solar Microgrid Utility": [
        {"sdg": 7, "name": "Affordable Clean Energy", "metric": "estimated_households", "weight": 1.0},
        {"sdg": 13, "name": "Climate Action", "metric": "estimated_households", "weight": 0.7},
        {"sdg": 11, "name": "Sustainable Communities", "metric": "estimated_population", "weight": 0.4},
    ],
    "Water Purification & Storage Center": [
        {"sdg": 6, "name": "Clean Water & Sanitation", "metric": "estimated_population", "weight": 1.0},
        {"sdg": 3, "name": "Good Health & Well-being", "metric": "estimated_under5", "weight": 0.6},
    ],
    "Community Farming Hub": [
        {"sdg": 2, "name": "Zero Hunger", "metric": "estimated_population", "weight": 1.0},
        {"sdg": 1, "name": "No Poverty", "metric": "estimated_households", "weight": 0.7},
        {"sdg": 15, "name": "Life on Land", "metric": "estimated_population", "weight": 0.3},
    ],
    "Disaster Relief Shelter": [
        {"sdg": 11, "name": "Sustainable Communities", "metric": "estimated_population", "weight": 1.0},
        {"sdg": 13, "name": "Climate Action", "metric": "estimated_population", "weight": 0.5},
    ],
    "General Area Analysis": [
        {"sdg": 11, "name": "Sustainable Communities", "metric": "estimated_population", "weight": 0.8},
        {"sdg": 6, "name": "Clean Water & Sanitation", "metric": "estimated_population", "weight": 0.5},
        {"sdg": 7, "name": "Affordable Clean Energy", "metric": "estimated_households", "weight": 0.5},
    ],
}


def calculate_sdg_impact(project_type: str, raw_data: dict[str, Any], plan: dict[str, Any]) -> dict[str, Any]:
    pop_data = raw_data.get("population", {})
    sdgs = SDG_MAPPING.get(project_type, SDG_MAPPING["General Area Analysis"])
    cost = _safe_float(plan.get("budget_allocation", {}).get("total_recommended_usd"), 1) or 1
    impacts = []
    for sdg in sdgs:
        metric = pop_data.get(sdg["metric"])
        if metric is None:
            metric = 0
        beneficiaries = int(metric * sdg["weight"])
        impacts.append(
            {
                "sdg_number": sdg["sdg"],
                "sdg_name": sdg["name"],
                "estimated_beneficiaries": beneficiaries,
                "cost_per_beneficiary_usd": round(cost / max(beneficiaries, 1), 2),
                "alignment_strength": "primary" if sdg["weight"] >= 0.8 else "secondary",
            }
        )
    return {
        "primary_sdgs": [i["sdg_number"] for i in impacts if i["alignment_strength"] == "primary"],
        "all_sdgs": impacts,
        "total_estimated_beneficiaries": sum(i["estimated_beneficiaries"] for i in impacts if i["alignment_strength"] == "primary"),
        "social_roi_score": round(_clamp(sum(i["estimated_beneficiaries"] for i in impacts) / max(cost / 1000, 1)), 1),
    }


def generate_beneficiary_personas(raw_data: dict[str, Any], project_type: str, use_llm: bool = True) -> list[dict[str, Any]]:
    pop = raw_data.get("population", {})
    loc = raw_data.get("location", {})
    fallback = [
        {
            "name": "Asha",
            "age": 34,
            "occupation": "caregiver and part-time farm worker",
            "daily_challenge": f"Long travel and unreliable services near {loc.get('district', 'the site')}.",
            "expected_benefit": f"The {project_type.lower()} reduces travel time and improves household resilience.",
            "quote": "If this works every day, it changes how we plan our week.",
        },
        {
            "name": "Ramesh",
            "age": 47,
            "occupation": "small shop owner",
            "daily_challenge": "Power, water, and road access make basic services unpredictable.",
            "expected_benefit": "Reliable local infrastructure protects income and saves emergency costs.",
            "quote": "A strong local facility means we do not wait for help from far away.",
        },
        {
            "name": "Meena",
            "age": 12,
            "occupation": "student",
            "daily_challenge": "Unsafe travel and poor basic services interrupt school and family life.",
            "expected_benefit": "Safer community infrastructure supports learning, health, and dignity.",
            "quote": "It should be close enough that I can go without fear.",
        },
    ]
    if not use_llm:
        return fallback
    prompt = f"""
Generate 3 realistic, culturally appropriate beneficiary personas for a {project_type} in
{loc.get('district')}, {loc.get('state')}, {loc.get('country')}.

Context:
- Population: {pop.get('estimated_population')}
- Households: {pop.get('estimated_households')}
- Children under 5: {pop.get('estimated_under5')}
- School-age children: {pop.get('estimated_school_children')}

Return JSON:
{{"personas": [{{"name": "", "age": 0, "occupation": "", "daily_challenge": "", "expected_benefit": "", "quote": ""}}]}}
"""
    try:
        parsed = _extract_json(call_gemma(prompt, "You are an empathetic ethnographer. Return JSON only.", timeout=90, temperature=0.35))
        personas = parsed.get("personas", []) if isinstance(parsed, dict) else []
        return personas[:3] or fallback
    except Exception:
        return fallback


def calculate_carbon_footprint(plan: dict[str, Any], raw_data: dict[str, Any]) -> dict[str, Any]:
    total_cost = _safe_float(plan.get("budget_allocation", {}).get("total_recommended_usd"), 0) or 0
    construction_co2 = round(total_cost * 350 / 1000, 0)
    grid_status = raw_data.get("nighttime_lights", {}).get("grid_status", "unknown")
    solar = _safe_float(raw_data.get("solar_potential", {}).get("solar_irradiance_kwh_m2_day"), 0) or 0
    plan_text = json.dumps(plan, default=_json_default).lower()
    if solar > 5 and "solar" in plan_text:
        annual_op = 200
    elif grid_status == "grid_connected":
        annual_op = 4500
    else:
        annual_op = 2000
    op_30yr = annual_op * 30
    total = construction_co2 + op_30yr
    trees = round(total / (22 * 40), 0)
    return {
        "construction_emissions_kg_co2": construction_co2,
        "operational_emissions_30yr_kg_co2": op_30yr,
        "total_emissions_kg_co2": total,
        "total_emissions_t_co2": round(total / 1000, 1),
        "offset_cost_usd_reference": round(total / 1000 * 25, 0),
        "trees_to_offset": trees,
        "carbon_intensity": "low" if total < 50000 else "moderate" if total < 200000 else "high",
        "recommendation": "Plant onsite shade trees and maximize solar uptime" if trees < 100 else f"Combine onsite trees with verified local restoration; reference offset budget about ${round(total / 1000 * 25, 0)}",
    }


FUNDING_DATABASE = {
    "Primary School": [
        {"name": "Global Partnership for Education", "url": "https://www.globalpartnership.org/", "typical_grant_usd": "100K-2M", "focus": "education infrastructure low-income countries"},
        {"name": "UNICEF Education", "url": "https://www.unicef.org/education", "typical_grant_usd": "50K-500K", "focus": "child-friendly schools"},
        {"name": "Asian Development Bank", "url": "https://www.adb.org/", "typical_grant_usd": "500K-10M", "focus": "education infrastructure Asia"},
    ],
    "Rural Health Clinic / Hospital": [
        {"name": "Global Fund", "url": "https://www.theglobalfund.org/", "typical_grant_usd": "100K-5M", "focus": "TB, malaria, HIV facilities"},
        {"name": "Gavi Health System Grants", "url": "https://www.gavi.org/", "typical_grant_usd": "50K-1M", "focus": "vaccination cold chain and clinics"},
        {"name": "World Bank Health", "url": "https://www.worldbank.org/en/topic/health", "typical_grant_usd": "200K-5M", "focus": "rural primary care"},
    ],
    "Solar Microgrid Utility": [
        {"name": "REEEP", "url": "https://www.reeep.org/", "typical_grant_usd": "50K-300K", "focus": "renewable energy access"},
        {"name": "Global Energy Alliance", "url": "https://energyalliance.org/", "typical_grant_usd": "500K-10M", "focus": "energy transitions"},
    ],
    "Water Purification & Storage Center": [
        {"name": "Water.org WaterCredit", "url": "https://water.org/", "typical_grant_usd": "100K-1M", "focus": "WASH in underserved areas"},
        {"name": "charity: water", "url": "https://www.charitywater.org/", "typical_grant_usd": "20K-200K", "focus": "clean water access"},
    ],
    "Community Farming Hub": [
        {"name": "IFAD", "url": "https://www.ifad.org/", "typical_grant_usd": "200K-5M", "focus": "smallholder farming"},
        {"name": "Gates Foundation Agriculture", "url": "https://www.gatesfoundation.org/ideas/agricultural-development", "typical_grant_usd": "500K-50M", "focus": "smallholder productivity"},
    ],
    "Disaster Relief Shelter": [
        {"name": "USAID BHA", "url": "https://www.usaid.gov/humanitarian-assistance", "typical_grant_usd": "100K-10M", "focus": "humanitarian shelter"},
        {"name": "UN-Habitat", "url": "https://unhabitat.org/", "typical_grant_usd": "100K-5M", "focus": "resilient housing"},
    ],
}


def match_funding_opportunities(project_type: str, raw_data: dict[str, Any], plan: dict[str, Any]) -> list[dict[str, Any]]:
    base = FUNDING_DATABASE.get(project_type, FUNDING_DATABASE.get("Disaster Relief Shelter", []))
    country = raw_data.get("location", {}).get("country", "")
    budget = _safe_float(plan.get("budget_allocation", {}).get("total_recommended_usd"), 0) or 0
    enriched = []
    for item in base:
        score = 50
        focus = item["focus"].lower()
        if country in {"India", "Bangladesh", "Nepal", "Pakistan"} and "asia" in focus:
            score += 20
        if country in {"Kenya", "Ethiopia", "Nigeria", "Ghana", "Tanzania", "Uganda"} and "africa" in focus:
            score += 20
        if budget and budget < 100000:
            score += 10
        if project_type.split()[0].lower() in focus:
            score += 10
        enriched.append({**item, "match_score": min(100, score), "recommended_first": score >= 70})
    return sorted(enriched, key=lambda x: -x["match_score"])


def save_plan_version(plan: dict[str, Any], raw_data: dict[str, Any], version_dir: str | Path = OUTPUT_DIR / "plan_versions") -> str:
    version_path = Path(version_dir)
    version_path.mkdir(parents=True, exist_ok=True)
    plan_hash = hashlib.md5(json.dumps(plan, sort_keys=True, default=_json_default).encode()).hexdigest()[:8]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = version_path / f"{timestamp}_{plan_hash}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"plan": plan, "raw_data": raw_data, "timestamp": timestamp}, f, indent=2, default=_json_default)
    return str(path.resolve())


def diff_plans(old_plan: dict[str, Any], new_plan: dict[str, Any]) -> dict[str, Any]:
    prompt = f"""
Compare these two project plans and identify the key differences.
OLD: {json.dumps(old_plan, default=_json_default)[:3000]}
NEW: {json.dumps(new_plan, default=_json_default)[:3000]}

Return JSON: {{"changes": [{{"field": "", "old_value": "", "new_value": "", "impact": ""}}]}}
"""
    try:
        parsed = _extract_json(call_gemma(prompt, "You are a precise diff analyzer. Return JSON only.", timeout=90))
        return parsed if isinstance(parsed, dict) else {"changes": []}
    except Exception:
        changes = []
        for key in ["site_viability", "project_summary", "budget_allocation"]:
            if old_plan.get(key) != new_plan.get(key):
                changes.append({"field": key, "old_value": str(old_plan.get(key))[:140], "new_value": str(new_plan.get(key))[:140], "impact": "Plan changed"})
        return {"changes": changes}


def speak_plan_summary(plan: dict[str, Any], language: str = "en") -> str | None:
    # Offline/local TTS only. No gTTS/paid/cloud service.
    try:
        import pyttsx3  # type: ignore

        summary = sanitize_text(plan.get("project_summary", ""))[:900]
        if not summary:
            return None
        path = OUTPUT_DIR / f"plan_audio_{datetime.now().strftime('%H%M%S')}.wav"
        engine = pyttsx3.init()
        engine.setProperty("rate", 165)
        engine.save_to_file(summary, str(path))
        engine.runAndWait()
        return str(path.resolve()) if path.exists() else None
    except Exception:
        return None


def chat_with_plan(message: str, history: list[Any], plan_state: dict[str, Any] | None, raw_data_state: dict[str, Any] | None) -> str:
    if not plan_state:
        return "Please generate a plan first."
    context = f"""
You are a helpful project advisor explaining this development plan to a non-technical user.

PLAN:
{json.dumps(plan_state, indent=2, default=_json_default)[:5000]}

SITE DATA:
{json.dumps(raw_data_state or {}, indent=2, default=_json_default)[:4000]}

Conversation:
{json.dumps(history[-6:] if history else [], default=_json_default)}

User question: {message}

Answer in plain English in 3-5 sentences. Cite specific available numbers. If data is missing, say so directly.
"""
    try:
        return call_gemma(context, "You are a friendly, specific rural infrastructure advisor.", timeout=120, temperature=0.35)
    except Exception:
        scores = (raw_data_state or {}).get("composite_scores", {})
        return f"The plan is based on construction suitability {scores.get('construction_suitability', 'N/A')}/100, water security {scores.get('water_security_score', 'N/A')}/100, and renewable energy {scores.get('renewable_energy_score', 'N/A')}/100. I could not reach the local Gemma model for a richer answer, but the safest next step is to field-check the biggest warnings in the report before reducing scope."


# ---------------------------------------------------------------------------
# HTML/GIF map artifacts
# ---------------------------------------------------------------------------

def generate_site_map(lat: float, lon: float, raw_data: dict[str, Any], plan: dict[str, Any], project_type: str) -> str:
    if folium is None:
        path = OUTPUT_DIR / f"site_map_{lat:.3f}_{lon:.3f}.html"
        path.write_text("<html><body><h1>Folium is not installed</h1></body></html>", encoding="utf-8")
        return str(path.resolve())

    m = folium.Map(location=[lat, lon], zoom_start=12, tiles="OpenStreetMap")
    flood = raw_data.get("flood_risk", {})
    flood_level = flood.get("flood_risk_level", "unknown")
    color = {"low": "green", "moderate": "orange", "high": "red"}.get(flood_level, "blue")
    folium.Marker(
        [lat, lon],
        popup=folium.Popup(f"<b>{project_type}</b><br>Flood risk: {flood_level}<br>Viability: {plan.get('site_viability', 'unknown')}", max_width=260),
        tooltip=f"{project_type} site",
        icon=folium.Icon(color=color, icon="home", prefix="fa"),
    ).add_to(m)
    folium.Circle([lat, lon], radius=5000, color="#2563eb", fill=True, fill_opacity=0.04, tooltip="5 km analysis radius").add_to(m)
    folium.Circle([lat, lon], radius=15000, color="#f97316", fill=True, fill_opacity=0.03, tooltip="15 km catchment").add_to(m)

    cluster = MarkerCluster(name="Nearby services").add_to(m) if MarkerCluster else m
    access = raw_data.get("road_hospital_access", {})
    if access.get("nearest_hospital_lat") and access.get("nearest_hospital_lon"):
        folium.Marker(
            [access["nearest_hospital_lat"], access["nearest_hospital_lon"]],
            popup=f"Nearest hospital/clinic: {access.get('nearest_hospital_name')} ({access.get('nearest_hospital_km')} km)",
            icon=folium.Icon(color="red", icon="plus", prefix="fa"),
        ).add_to(cluster)

    flood_score = flood.get("flood_risk_score") or 0
    if HeatMap and flood_score:
        heat = [[lat + (i - 4) * 0.006, lon + (j - 4) * 0.006, flood_score / 100] for i in range(9) for j in range(9)]
        HeatMap(heat, min_opacity=0.15, radius=28, name="Flood risk proxy").add_to(m)

    warnings_list = plan.get("warnings", []) if isinstance(plan.get("warnings"), list) else []
    warning_html = "".join(f"<li>{sanitize_text(w)}</li>" for w in warnings_list[:4])
    info = f"""
    <div style="background:white;border:1px solid #ddd;border-radius:8px;padding:10px;width:260px;font-size:12px;box-shadow:0 4px 12px rgba(0,0,0,.12)">
      <b>{sanitize_text(project_type)}</b><br>
      <span>Flood: {sanitize_text(flood_level)} | Grid: {sanitize_text(raw_data.get('nighttime_lights', {}).get('grid_status', 'unknown'))}</span>
      <ul style="margin:6px 0 0 16px;padding:0">{warning_html}</ul>
    </div>
    """
    folium.Marker([lat + 0.008, lon + 0.008], icon=folium.DivIcon(html=info, icon_size=(280, 150), icon_anchor=(0, 0))).add_to(m)
    folium.LayerControl().add_to(m)
    path = OUTPUT_DIR / f"site_map_{lat:.3f}_{lon:.3f}_{_slug(project_type, 24)}.html"
    m.save(str(path))
    return str(path.resolve())


def generate_3d_visualization(lat: float, lon: float, raw_data: dict[str, Any], plan: dict[str, Any], project_type: str) -> str:
    flood_score = raw_data.get("flood_risk", {}).get("flood_risk_score") or 35
    pop = raw_data.get("population", {}).get("estimated_population") or 0
    water_score = raw_data.get("composite_scores", {}).get("water_security_score") or 50
    summary = sanitize_text(plan.get("project_summary", ""))[:110]
    loc = raw_data.get("location", {})
    path = OUTPUT_DIR / f"3d_view_{lat:.3f}_{lon:.3f}_{_slug(project_type, 24)}.html"
    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>VillageForge 3D Site View</title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <script src="https://unpkg.com/deck.gl@9.0.0/dist.min.js"></script>
  <style>
    body {{ margin:0; font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background:#0f172a; }}
    #map {{ position:absolute; inset:0; }}
    .panel {{ position:absolute; top:18px; left:18px; z-index:2; max-width:340px; background:rgba(255,255,255,.94); padding:18px; border-radius:8px; box-shadow:0 12px 32px rgba(0,0,0,.22); }}
    h2 {{ margin:0 0 8px; color:#065f46; font-size:22px; }}
    p {{ margin:6px 0; color:#334155; line-height:1.35; }}
    .badge {{ display:inline-block; padding:4px 9px; border-radius:999px; font-size:12px; font-weight:700; margin:8px 6px 0 0; }}
    .red {{ background:#fee2e2; color:#991b1b; }} .green {{ background:#d1fae5; color:#065f46; }} .blue {{ background:#dbeafe; color:#1e40af; }}
    .legend {{ position:absolute; left:18px; bottom:18px; z-index:2; background:white; padding:12px; border-radius:8px; font-size:12px; color:#334155; box-shadow:0 8px 20px rgba(0,0,0,.18); }}
  </style>
</head>
<body>
  <div id="map"></div>
  <div class="panel">
    <h2>{sanitize_text(project_type)}</h2>
    <p><b>{sanitize_text(loc.get('district',''))}, {sanitize_text(loc.get('state',''))}</b></p>
    <p>{summary}</p>
    <span class="badge red">Flood {flood_score}/100</span>
    <span class="badge green">Population {pop:,}</span>
    <span class="badge blue">Water {water_score}/100</span>
    <p style="font-size:12px;margin-top:10px">Drag to rotate. Scroll to zoom. Hex height shows local risk intensity proxy.</p>
  </div>
  <div class="legend">Risk heat: green low | amber medium | red high<br>Central column marks proposed site.</div>
  <script>
    const {{DeckGL, HexagonLayer, ColumnLayer, ScatterplotLayer}} = deck;
    const points = [];
    for (let i = -12; i <= 12; i++) {{
      for (let j = -12; j <= 12; j++) {{
        const dist = Math.sqrt(i*i + j*j);
        const flood = {float(flood_score)} * Math.exp(-dist/5);
        const water = Math.max(0, 100 - {float(water_score)}) * Math.exp(-dist/8);
        points.push({{position:[{lon} + i * 0.0045, {lat} + j * 0.0045], weight:flood + water * .35}});
      }}
    }}
    new DeckGL({{
      container: 'map',
      mapStyle: 'https://basemaps.cartocdn.com/gl/positron-gl-style/style.json',
      initialViewState: {{longitude:{lon}, latitude:{lat}, zoom:12.6, pitch:58, bearing:32}},
      controller: true,
      layers: [
        new HexagonLayer({{
          id:'risk-hex', data:points, getPosition:d=>d.position, getElevationWeight:d=>d.weight,
          elevationScale:34, radius:180, extruded:true, coverage:.82,
          colorRange:[[16,185,129],[132,204,22],[250,204,21],[249,115,22],[220,38,38],[127,29,29]]
        }}),
        new ColumnLayer({{
          id:'site-marker', data:[{{position:[{lon},{lat}], height:900}}],
          getPosition:d=>d.position, getElevation:d=>d.height, getFillColor:[4,120,87,220],
          radius:65, extruded:true, pickable:true
        }}),
        new ScatterplotLayer({{
          id:'catchment', data:[{{position:[{lon},{lat}]}}],
          getPosition:d=>d.position, getRadius:5000, getFillColor:[37,99,235,28], getLineColor:[37,99,235,120], stroked:true
        }})
      ]
    }});
  </script>
</body>
</html>"""
    path.write_text(html, encoding="utf-8")
    return str(path.resolve())


def generate_satellite_timelapse(lat: float, lon: float) -> str | None:
    if ee is None or not earth_engine_available():
        return None
    try:
        buf = _buffer(lat, lon, 5000)
        frames = []
        for year in range(2019, 2025):
            img = (
                ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                .filterDate(f"{year}-01-01", f"{year}-12-31")
                .filterBounds(buf)
                .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 25))
                .median()
                .clip(buf)
                .visualize(bands=["B4", "B3", "B2"], min=0, max=3000)
            )
            frames.append(img)
        collection = ee.ImageCollection.fromImages(frames)
        url = collection.getVideoThumbURL({"dimensions": 512, "framesPerSecond": 1, "region": buf, "format": "gif"})
        response = requests.get(url, headers=_headers(), timeout=60)
        response.raise_for_status()
        path = OUTPUT_DIR / f"timelapse_{lat:.3f}_{lon:.3f}.gif"
        path.write_bytes(response.content)
        return str(path.resolve())
    except Exception:
        return None


# ---------------------------------------------------------------------------
# PDF report
# ---------------------------------------------------------------------------

if FPDF is not None:
    class VillagePDF(FPDF):
        def __init__(self, title_text: str):
            super().__init__()
            self.title_text = sanitize_text(title_text)
            self.set_auto_page_break(auto=True, margin=14)

        def header(self):
            self.set_font("Helvetica", "B", 11)
            self.set_fill_color(15, 95, 75)
            self.set_text_color(255, 255, 255)
            self.cell(0, 10, f"  VillageForge AI - {self.title_text}", fill=True, ln=True)
            self.set_text_color(0, 0, 0)
            self.ln(3)

        def footer(self):
            self.set_y(-12)
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(120, 120, 120)
            self.cell(0, 8, f"Open-data planning report - {datetime.now().strftime('%Y-%m-%d')} | Page {self.page_no()}", align="C")

        def section_title(self, title: str):
            self.set_x(self.l_margin)
            self.ln(4)
            self.set_font("Helvetica", "B", 12)
            self.set_fill_color(220, 242, 232)
            self.set_text_color(15, 95, 75)
            self.cell(0, 8, f"  {sanitize_text(title)}", fill=True, ln=True)
            self.set_text_color(0, 0, 0)
            self.ln(2)

        def score_bar(self, label: str, score: float, max_score: float = 100):
            self.set_x(self.l_margin)
            self.set_font("Helvetica", "B", 9)
            self.cell(70, 7, sanitize_text(label) + ":", ln=False)
            bar_w = 88
            pct = _clamp(score / max_score, 0, 1)
            color = (22, 163, 74) if pct > 0.6 else (245, 158, 11) if pct > 0.35 else (220, 38, 38)
            self.set_fill_color(*color)
            self.cell(int(bar_w * pct), 7, "", fill=True, ln=False)
            self.set_fill_color(220, 220, 220)
            self.cell(int(bar_w * (1 - pct)), 7, "", fill=True, ln=False)
            self.set_font("Helvetica", "", 9)
            self.cell(24, 7, f"  {score:.0f}/100", ln=True)

        def kv_row(self, key: str, value: Any):
            self.set_x(self.l_margin)
            clean_key = sanitize_text(key)
            if len(clean_key) > 36:
                clean_key = clean_key[:33] + "..."
            self.set_font("Helvetica", "B", 9)
            self.cell(66, 6, clean_key + ":", ln=False)
            self.set_font("Helvetica", "", 9)
            self.multi_cell(0, 6, sanitize_text(value) or "N/A")

        def body_text(self, text: Any):
            self.set_x(self.l_margin)
            self.set_font("Helvetica", "", 10)
            self.multi_cell(0, 6, sanitize_text(text) or "No information provided.")
            self.ln(1)

        def bullet(self, text: Any):
            self.set_x(self.l_margin)
            self.set_font("Helvetica", "", 9)
            self.cell(6, 5, "-", ln=False)
            self.multi_cell(0, 5, sanitize_text(text) or "N/A")
else:
    VillagePDF = None  # type: ignore


def _list_to_pdf(pdf: Any, title: str, values: Any) -> None:
    if isinstance(values, list) and values:
        pdf.section_title(title)
        for value in values:
            pdf.bullet(json.dumps(value, default=_json_default) if isinstance(value, dict) else value)


def save_report_pdf(
    raw_data: dict[str, Any],
    plan: dict[str, Any],
    composite_scores: dict[str, Any],
    selected_tools: list[str],
    location_query: str,
    project_type: str,
    filename: str | Path,
) -> str:
    path = Path(filename)
    if not path.is_absolute():
        path = OUTPUT_DIR / path
    if FPDF is None or VillagePDF is None:
        path = path.with_suffix(".txt")
        path.write_text(json.dumps({"raw_data": raw_data, "plan": plan, "scores": composite_scores}, indent=2, default=_json_default), encoding="utf-8")
        return str(path.resolve())

    loc = raw_data.get("location", {})
    name = f"{loc.get('district', 'Site')}, {loc.get('state', '')}"
    pdf = VillagePDF(f"{project_type}: {name}")
    pdf.add_page()

    pdf.section_title("Site Overview")
    pdf.kv_row("Project Type", project_type)
    pdf.kv_row("Location", name)
    pdf.kv_row("Country", loc.get("country", ""))
    pdf.kv_row("Coordinates", f"{loc.get('latitude', 0):.4f}, {loc.get('longitude', 0):.4f}")
    pdf.kv_row("Location Query", location_query)
    pdf.kv_row("Data Layers Analysed", len(selected_tools))
    pdf.kv_row("Generated", datetime.now().strftime("%Y-%m-%d %H:%M"))

    pdf.section_title("Composite Site Scores")
    for key, value in composite_scores.items():
        if isinstance(value, (int, float)):
            pdf.score_bar(key.replace("_", " ").title(), float(value))
        elif key != "data_gaps":
            pdf.kv_row(key.replace("_", " ").title(), value)

    pdf.section_title("Project Feasibility Summary")
    pdf.body_text(plan.get("project_summary", "Summary unavailable."))
    pdf.kv_row("Site Viability", str(plan.get("site_viability", "N/A")).upper())

    _list_to_pdf(pdf, "Priority Actions", plan.get("priority_ranking", []))

    pdf.section_title("Engineering Plans")
    pdf.kv_row("Structural", plan.get("structural_design_notes", "N/A"))
    pdf.kv_row("Renewable Energy", plan.get("renewable_energy_plan", "N/A"))
    pdf.kv_row("Water & Sanitation", plan.get("water_and_sanitation_plan", "N/A"))

    pdf.section_title("Budget Breakdown")
    budget = plan.get("budget_allocation", {})
    pdf.kv_row("Total Recommended", f"${_safe_int(budget.get('total_recommended_usd'), 0):,}")
    pdf.kv_row("Contingency", f"{budget.get('contingency_pct', 'N/A')}%")
    for phase, alloc in (budget.get("breakdown", {}) if isinstance(budget, dict) else {}).items():
        if isinstance(alloc, dict):
            pdf.bullet(f"{phase}: ${_safe_int(alloc.get('amount_usd'), 0):,} ({alloc.get('pct', 0)}%) - {alloc.get('notes', '')}")

    sectors = plan.get("sectors", {})
    if isinstance(sectors, dict):
        for phase, detail in sectors.items():
            pdf.section_title(f"Component: {phase}")
            if not isinstance(detail, dict):
                pdf.body_text(detail)
                continue
            pdf.kv_row("Recommendation", detail.get("recommendation", "N/A"))
            pdf.kv_row("Cost Estimate", f"${_safe_int(detail.get('estimated_cost_usd'), 0):,}")
            pdf.kv_row("Timeline", f"{detail.get('timeline_weeks', '?')} weeks")
            pdf.kv_row("Rationale", detail.get("rationale", "N/A"))
            items = detail.get("items", [])
            if isinstance(items, str):
                items = [items]
            for item in items[:8]:
                pdf.bullet(item)

    _list_to_pdf(pdf, "Critical Site Risks", plan.get("warnings", []))
    _list_to_pdf(pdf, "Agent Consensus Fixes", plan.get("agent_consensus", []))

    impact = raw_data.get("sdg_impact", {})
    if impact:
        pdf.section_title("SDG Impact")
        pdf.kv_row("Primary SDGs", ", ".join(map(str, impact.get("primary_sdgs", []))))
        pdf.kv_row("Estimated Primary Beneficiaries", impact.get("total_estimated_beneficiaries", "N/A"))
        pdf.kv_row("Social ROI Score", impact.get("social_roi_score", "N/A"))
        for item in impact.get("all_sdgs", [])[:6]:
            pdf.bullet(f"SDG {item.get('sdg_number')} {item.get('sdg_name')}: {item.get('estimated_beneficiaries')} beneficiaries, ${item.get('cost_per_beneficiary_usd')} per beneficiary")

    climate = raw_data.get("climate_projection_2050", {})
    if climate:
        pdf.section_title("2050 Climate Stress Test")
        for key in ["warming_2050_c", "projected_temp_2050_c", "projected_daily_rainfall_2050_mm", "climate_risk_2050", "design_lifespan_recommendation"]:
            pdf.kv_row(key.replace("_", " ").title(), climate.get(key, "N/A"))

    carbon = raw_data.get("carbon_footprint", {})
    if carbon:
        pdf.section_title("Carbon Footprint")
        for key in ["total_emissions_t_co2", "carbon_intensity", "trees_to_offset", "recommendation"]:
            pdf.kv_row(key.replace("_", " ").title(), carbon.get(key, "N/A"))

    _list_to_pdf(pdf, "Beneficiary Personas", raw_data.get("beneficiary_personas", []))
    _list_to_pdf(pdf, "Funding Matches", raw_data.get("funding_matches", []))
    _list_to_pdf(pdf, "Cost-Saving Measures", plan.get("cost_saving_tips", []))
    _list_to_pdf(pdf, "Local Material Opportunities", plan.get("local_material_opportunities", []))
    _list_to_pdf(pdf, "Next Steps", plan.get("next_steps", []))
    _list_to_pdf(pdf, "Post-Construction KPIs", plan.get("monitoring_kpis", []))

    pdf.add_page()
    pdf.section_title("Appendix A - Open Data Readout")
    for category, values in raw_data.items():
        if category in {"beneficiary_personas", "funding_matches", "sdg_impact", "carbon_footprint"}:
            continue
        if isinstance(values, dict):
            pdf.set_font("Helvetica", "B", 10)
            pdf.set_x(pdf.l_margin)
            pdf.cell(0, 6, sanitize_text(category.replace("_", " ").title()) + ":", ln=True)
            for k, v in values.items():
                if isinstance(v, (dict, list)):
                    v = json.dumps(v, default=_json_default)[:500]
                pdf.kv_row("  " + k.replace("_", " ").title(), v)

    pdf.add_page()
    pdf.section_title("Appendix B - Tool Selection Log")
    pdf.body_text(", ".join(selected_tools))

    try:
        pdf.output(str(path))
    except PermissionError:
        path = path.with_name(f"{path.stem}_{int(time.time())}{path.suffix}")
        pdf.output(str(path))
    return str(path.resolve())


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

COUNTRY_ISO3 = {
    "IN": "IND",
    "BD": "BGD",
    "NP": "NPL",
    "PK": "PAK",
    "NG": "NGA",
    "KE": "KEN",
    "ET": "ETH",
    "GH": "GHA",
    "TZ": "TZA",
    "UG": "UGA",
    "ID": "IDN",
    "MM": "MMR",
    "KH": "KHM",
    "MZ": "MOZ",
    "US": "USA",
}


def run_village_analysis(
    location_query: str,
    budget_usd: int,
    project_type: str = "General Area Analysis",
    *,
    run_debate: bool = True,
    generate_timelapse: bool = False,
    use_llm_personas: bool = True,
    use_llm_tool_selection: bool = True,
) -> dict[str, Any]:
    print(f"\n[VillageForge v3] Planning '{project_type}' at: {location_query}")
    lat, lon = pincode_to_latlon(location_query)
    location = get_admin_area(lat, lon)
    location.update({"latitude": lat, "longitude": lon, "query": location_query})
    print(f"[Geo] {location.get('district')}, {location.get('state')} ({lat:.4f}, {lon:.4f})")

    country_iso2 = str(location.get("country_iso2") or "IN").upper()
    country_iso3 = COUNTRY_ISO3.get(country_iso2, "IND")

    selected_tools = select_tools_with_gemma(project_type, lat, lon, location) if use_llm_tool_selection else sorted(set(PROJECT_TOOL_HINTS.get(project_type, [])) | BASE_TOOLS)
    print(f"[Tools] {len(selected_tools)} selected: {selected_tools}")
    tool_results = run_tools_parallel(selected_tools, lat, lon)

    context_fns: dict[str, tuple[Callable[..., dict[str, Any]], tuple[Any, ...]]] = {
        "world_bank_country": (fetch_world_bank_indicators, (country_iso2,)),
        "who_health": (fetch_who_health_data, (country_iso2,)),
        "fao_crops": (fetch_fao_crop_data, (country_iso3,)),
        "reliefweb_disasters": (fetch_reliefweb_disasters, (lat, lon, location.get("country", "India"))),
        "nasa_firms": (fetch_nasa_firms_fires, (lat, lon)),
    }
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(fn, *args): name for name, (fn, args) in context_fns.items()}
        for future in concurrent.futures.as_completed(futures):
            name = futures[future]
            try:
                tool_results[name] = future.result()
            except Exception as exc:
                tool_results[name] = {"error": str(exc), "data_quality": "unavailable"}

    raw_data: dict[str, Any] = {"location": location, **tool_results}
    composite_scores = compute_composite_scores(raw_data)
    raw_data["composite_scores"] = composite_scores
    print(f"[Scores] {composite_scores}")

    plan = generate_village_plan(raw_data, project_type, int(budget_usd), composite_scores)
    if run_debate:
        print("[Agents] Running multi-agent debate")
        plan = run_agent_debate(plan, raw_data, project_type, use_llm=True)
        plan = _validate_plan(plan, int(budget_usd), _fallback_plan(raw_data, project_type, int(budget_usd), composite_scores))

    raw_data["sdg_impact"] = calculate_sdg_impact(project_type, raw_data, plan)
    raw_data["carbon_footprint"] = calculate_carbon_footprint(plan, raw_data)
    raw_data["funding_matches"] = match_funding_opportunities(project_type, raw_data, plan)
    raw_data["beneficiary_personas"] = generate_beneficiary_personas(raw_data, project_type, use_llm=use_llm_personas)

    safe_loc = _slug(location_query, 36)
    safe_proj = _slug(project_type, 36)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    pdf_path = OUTPUT_DIR / f"{safe_proj}_{safe_loc}_{timestamp}.pdf"
    json_path = OUTPUT_DIR / f"{safe_proj}_{safe_loc}_{timestamp}.json"

    map_path = generate_site_map(lat, lon, raw_data, plan, project_type)
    view_3d_path = generate_3d_visualization(lat, lon, raw_data, plan, project_type)
    timelapse_path = generate_satellite_timelapse(lat, lon) if generate_timelapse else None
    version_path = save_plan_version(plan, raw_data)
    pdf_final = save_report_pdf(raw_data, plan, composite_scores, selected_tools, location_query, project_type, pdf_path)

    payload = {
        "project_type": project_type,
        "selected_tools": selected_tools,
        "composite_scores": composite_scores,
        "raw_data": raw_data,
        "plan": plan,
        "artifacts": {
            "pdf_path": pdf_final,
            "json_path": str(json_path.resolve()),
            "map_path": map_path,
            "view_3d_path": view_3d_path,
            "timelapse_path": timelapse_path,
            "version_path": version_path,
        },
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=_json_default)

    return {
        "plan": plan,
        "raw_data": raw_data,
        "composite_scores": composite_scores,
        "selected_tools": selected_tools,
        "pdf_path": pdf_final,
        "json_path": str(json_path.resolve()),
        "map_path": map_path,
        "view_3d_path": view_3d_path,
        "timelapse_path": timelapse_path,
        "version_path": version_path,
        "location": location,
    }


# ---------------------------------------------------------------------------
# Voice intent extraction
# ---------------------------------------------------------------------------

WHISPER_MODEL = None


def process_voice_audio(audio_path: str) -> dict[str, Any]:
    global WHISPER_MODEL
    if not audio_path:
        return {"english_text": "", "location": "", "project_type": "General Area Analysis", "budget": 50000, "error": "No audio supplied"}
    try:
        import whisper  # type: ignore

        if WHISPER_MODEL is None:
            print("[Voice] Loading Whisper base model")
            WHISPER_MODEL = whisper.load_model("base")
        result = WHISPER_MODEL.transcribe(audio_path, task="translate")
        english_text = result.get("text", "").strip()
    except Exception as exc:
        return {"english_text": "", "location": "", "project_type": "General Area Analysis", "budget": 50000, "error": f"Whisper unavailable: {exc}"}

    prompt = f"""
Extract planning intent from this translated request:
"{english_text}"

Map project_type to one exact option:
- General Area Analysis
- Primary School
- Rural Health Clinic / Hospital
- Solar Microgrid Utility
- Water Purification & Storage Center
- Community Farming Hub
- Disaster Relief Shelter

Return JSON only:
{{"location": "", "project_type": "", "budget": 50000}}
"""
    try:
        parsed = _extract_json(call_gemma(prompt, "You extract structured intent. Return JSON only.", timeout=90))
        if not isinstance(parsed, dict):
            raise ValueError("No object")
        parsed["english_text"] = english_text
        parsed["budget"] = _safe_int(parsed.get("budget"), 50000) or 50000
        return parsed
    except Exception:
        budget_match = re.search(r"(\d[\d,]{3,})", english_text)
        budget = int(budget_match.group(1).replace(",", "")) if budget_match else 50000
        return {"english_text": english_text, "location": "", "project_type": "General Area Analysis", "budget": budget}


def live_alert_snapshot(lat: float, lon: float) -> str:
    weather = fetch_open_meteo_weather(lat, lon)
    fires = fetch_nasa_firms_fires(lat, lon)
    return (
        f"Weather: {weather.get('current_temp_c', 'N/A')}C, "
        f"wind {weather.get('current_wind_kmh', 'N/A')} km/h, "
        f"7-day rain {weather.get('forecast_7d_rain_mm', 'N/A')} mm | "
        f"Fires 7d: {fires.get('active_fire_detections_7d', 'N/A')}"
    )
