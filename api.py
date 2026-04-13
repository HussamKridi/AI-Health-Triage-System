import json
import math
import os
from urllib import parse, request as urllib_request

from flask import Blueprint, jsonify, request

from models import get_db_connection
from triage_flow import continue_triage, load_json, run_initial_triage


api_bp = Blueprint("api", __name__)
OVERPASS_API_URL = os.getenv("OVERPASS_API_URL", "https://overpass-api.de/api/interpreter")
HOSPITAL_SEARCH_RADIUS_METERS = int(os.getenv("HOSPITAL_SEARCH_RADIUS_METERS", "12000"))
DEFAULT_HOSPITAL_LIMIT = int(os.getenv("HOSPITAL_LIMIT", "5"))
GENERIC_NEARBY_MAP_URL = "https://www.google.com/maps/search/hospitals+near+me"


def _parse_payload(data):
    return {
        "age": int(data["age"]),
        "gender": str(data["gender"]),
        "weight": float(data["weight"]),
        "height": float(data["height"]),
        "spo2": float(data["spo2"]),
        "temperature": float(data["temperature"]),
        "heart_rate": int(data["heart_rate"]),
    }


def _haversine_distance_meters(lat1, lng1, lat2, lng2):
    radius_meters = 6371000
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lng2 - lng1)
    a = (
        math.sin(delta_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    )
    return 2 * radius_meters * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _compose_address(tags):
    if not isinstance(tags, dict):
        return ""
    pieces = [
        tags.get("addr:street"),
        tags.get("addr:housenumber"),
        tags.get("addr:suburb"),
        tags.get("addr:district"),
        tags.get("addr:city") or tags.get("addr:town") or tags.get("addr:village"),
    ]
    return ", ".join(str(piece).strip() for piece in pieces if piece)


def _fetch_hospitals_from_overpass(lat, lng, radius_meters=HOSPITAL_SEARCH_RADIUS_METERS, limit=DEFAULT_HOSPITAL_LIMIT):
    overpass_query = f"""
[out:json][timeout:12];
(
  node["amenity"="hospital"](around:{int(radius_meters)},{lat},{lng});
  way["amenity"="hospital"](around:{int(radius_meters)},{lat},{lng});
  relation["amenity"="hospital"](around:{int(radius_meters)},{lat},{lng});
);
out center tags;
""".strip()
    encoded_body = parse.urlencode({"data": overpass_query}).encode("utf-8")
    http_request = urllib_request.Request(
        OVERPASS_API_URL,
        data=encoded_body,
        headers={
            "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
            "Accept": "application/json",
        },
        method="POST",
    )

    with urllib_request.urlopen(http_request, timeout=15) as response:
        payload = json.loads(response.read().decode("utf-8"))

    hospitals = []
    for element in payload.get("elements", []):
        tags = element.get("tags", {})
        hospital_lat = element.get("lat")
        hospital_lng = element.get("lon")
        center = element.get("center", {})
        if hospital_lat is None:
            hospital_lat = center.get("lat")
        if hospital_lng is None:
            hospital_lng = center.get("lon")
        if hospital_lat is None or hospital_lng is None:
            continue

        distance_meters = _haversine_distance_meters(lat, lng, float(hospital_lat), float(hospital_lng))
        hospitals.append(
            {
                "name": tags.get("name") or "Nearby hospital",
                "address": _compose_address(tags),
                "lat": float(hospital_lat),
                "lng": float(hospital_lng),
                "distance_meters": round(distance_meters),
            }
        )

    hospitals.sort(key=lambda item: item["distance_meters"])
    return hospitals[:limit]


def lookup_nearby_hospitals(lat, lng, limit=DEFAULT_HOSPITAL_LIMIT):
    hospitals = _fetch_hospitals_from_overpass(lat, lng, limit=limit)
    return {
        "mode": "resolved_location",
        "origin": {
            "label": "Resolved emergency location",
            "lat": float(lat),
            "lng": float(lng),
        },
        "hospitals": hospitals,
        "fallback_used": False,
    }


def build_hospital_lookup_fallback(lat, lng):
    return {
        "status": "success",
        "origin": {
            "label": "Resolved emergency location",
            "lat": float(lat),
            "lng": float(lng),
        },
        "hospitals": [],
        "fallback_used": True,
        "mode": "maps_fallback",
        "message": "Live hospital lookup is unavailable right now. You can still open nearby hospitals in Google Maps.",
        "fallback_map_url": GENERIC_NEARBY_MAP_URL,
    }


@api_bp.route("/vitals", methods=["POST"])
def receive_vitals():
    data = request.get_json() or {}
    required = {"patient_id", "age", "gender", "weight", "height", "spo2", "temperature", "heart_rate"}
    if not required.issubset(data):
        return jsonify({"error": "Missing required triage fields"}), 400

    patient_id = data["patient_id"]
    payload = _parse_payload(data)
    state = run_initial_triage(payload)

    conn = get_db_connection()
    conn.execute(
        """
        INSERT INTO vitals
        (patient_id, age, gender, weight, height, spo2, temperature, heart_rate, classification,
         triage_status, ai_recommendation, conversation_history, model_assessment, disagreement_logged, final_source)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            patient_id,
            payload["age"],
            payload["gender"],
            payload["weight"],
            payload["height"],
            payload["spo2"],
            payload["temperature"],
            payload["heart_rate"],
            state["classification"],
            state["triage_status"],
            state["ai_recommendation_json"],
            state["conversation_history_json"],
            state["model_assessment_json"],
            state["disagreement_logged"],
            state["final_source"],
        ),
    )
    vital_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    conn.commit()
    conn.close()

    return jsonify(
        {
            "status": "success",
            "vital_id": vital_id,
            "triage_status": state["triage_status"],
            **state["result"],
        }
    ), 201


@api_bp.route("/vitals/<int:vital_id>/triage", methods=["POST"])
def continue_triage_route(vital_id):
    data = request.get_json() or {}
    patient_id = data.get("patient_id")
    answer = str(data.get("answer", "")).strip()

    if not patient_id or not answer:
        return jsonify({"error": "patient_id and answer are required"}), 400

    conn = get_db_connection()
    vital = conn.execute(
        "SELECT * FROM vitals WHERE id = ? AND patient_id = ?",
        (vital_id, patient_id),
    ).fetchone()
    if vital is None:
        conn.close()
        return jsonify({"error": "Vitals record not found"}), 404

    current_triage = load_json(vital["ai_recommendation"], {})
    if current_triage.get("type") != "question":
        conn.close()
        return jsonify({"error": "Triage has already reached a final decision"}), 409

    state = continue_triage(vital, answer)
    conn.execute(
        """
        UPDATE vitals
        SET classification = ?, triage_status = ?, ai_recommendation = ?, conversation_history = ?,
            model_assessment = ?, disagreement_logged = ?, final_source = ?
        WHERE id = ?
        """,
        (
            state["classification"],
            state["triage_status"],
            state["ai_recommendation_json"],
            state["conversation_history_json"],
            state["model_assessment_json"],
            state["disagreement_logged"],
            state["final_source"],
            vital_id,
        ),
    )
    conn.commit()
    conn.close()

    return jsonify(
        {
            "status": "success",
            "vital_id": vital_id,
            "triage_status": state["triage_status"],
            "conversation_history": state["conversation_history"],
            **state["result"],
        }
    )


@api_bp.route("/nearby-hospitals", methods=["GET"])
def nearby_hospitals():
    lat_raw = request.args.get("lat")
    lng_raw = request.args.get("lng")
    if lat_raw is None or lng_raw is None:
        return jsonify({"error": "lat and lng are required"}), 400

    try:
        lat = float(lat_raw)
        lng = float(lng_raw)
    except (TypeError, ValueError):
        return jsonify({"error": "lat and lng must be valid numbers"}), 400

    if not (-90 <= lat <= 90 and -180 <= lng <= 180):
        return jsonify({"error": "lat or lng out of range"}), 400

    try:
        result = lookup_nearby_hospitals(lat, lng)
    except Exception:
        return jsonify(build_hospital_lookup_fallback(lat, lng))

    if not result.get("hospitals"):
        return jsonify(
            {
                "status": "success",
                "origin": result.get("origin", {}),
                "hospitals": [],
                "fallback_used": bool(result.get("fallback_used")),
                "mode": result.get("mode", "unknown"),
                "message": "No nearby hospitals were found for the selected search area.",
                "fallback_map_url": result.get("fallback_map_url"),
            }
        )

    return jsonify(
        {
            "status": "success",
            "origin": result["origin"],
            "hospitals": result["hospitals"],
            "fallback_used": bool(result.get("fallback_used")),
            "mode": result.get("mode", "live_location"),
            "fallback_map_url": result.get("fallback_map_url"),
        }
    )
