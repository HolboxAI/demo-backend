# calorie_content.py
import os
import json
import base64
import requests
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_VISION_MODEL = os.getenv("OPENAI_VISION_MODEL", "gpt-4o-mini")  # must be vision-capable
USDA_API_KEY = os.getenv("USDA_API_KEY")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY missing in environment")
if not USDA_API_KEY:
    raise RuntimeError("USDA_API_KEY missing in environment")


def identify_foods_with_llm(
    image_bytes: bytes,
    user_portion_hint: Optional[str] = None,
    mime: str = "image/jpeg",
) -> List[Dict[str, Any]]:
    """
    Use OpenAI vision model to identify foods and rough portions.
    Returns a list of {name, confidence, portion_text, estimated_grams}.
    """
    from openai import OpenAI
    import base64, json, re

    def _extract_json(text: str) -> dict:
        # Try direct parse first
        try:
            return json.loads(text)
        except Exception:
            pass
        # Fallback: grab the first JSON object in the text
        m = re.search(r"\{.*\}", text, flags=re.S)
        if not m:
            raise ValueError("No JSON object in model output")
        return json.loads(m.group(0))

    client = OpenAI(api_key=OPENAI_API_KEY)

    system = (
    
    "You are a nutrition analyst. Identify distinct edible items in the image, "
    "estimate a plausible portion per item, and RETURN ONLY a JSON object with this schema: "
    '{"items":[{"name":"","confidence":0.0,"portion_text":"","estimated_grams":0}]}. '
    "Rules: "
    "1) If the photo shows multiple of the SAME food, AGGREGATE them into ONE item and estimate the TOTAL grams. "
    "   For example, if there are multiple apples or bananas, count them as a group, not individual items. "
    "2) Do NOT split the same food into variants like 'apple' and 'apple slice' simultaneously—pick one aggregate entry. "
    "3) Make sure to count all visible portions of the food in the image, including food partially out of view or behind other items. "
    "4) Prefer generic, fresh/raw forms (e.g., 'apple (raw)') unless clear evidence shows otherwise (e.g., canned label). "
    "5) Use concise common names; avoid brands unless a logo is clearly visible."

    )

    user_prompt = (
        "Detect foods in the photo. For each item provide:\n"
        " - name (common food name)\n"
        " - confidence (0..1)\n"
        " - portion_text (e.g., '1 bowl', '2 slices')\n"
        " - estimated_grams (integer)"
    )
    if user_portion_hint:
        user_prompt += f"\nPortion hint from user: {user_portion_hint}"

    data_url = f"data:{mime};base64,{base64.b64encode(image_bytes).decode('utf-8')}"

    messages = [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_prompt},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        },
    ]

    content_text = None

    # Try Responses API first (newer SDKs)
    try:
        resp = client.responses.create(
            model=OPENAI_VISION_MODEL,
            temperature=0,
            # Some older SDKs don't accept response_format; keep it here, but handle TypeError
            response_format={"type": "json_object"},
            input=[
                {"role": "system", "content": [{"type": "text", "text": system}]},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {"type": "input_image", "image_url": data_url},
                    ],
                },
            ],
        )
        # Pull out text from the streaming-style output shape
        chunks = []
        for out in getattr(resp, "output", []) or []:
            if getattr(out, "type", None) == "message":
                for c in getattr(out, "content", []) or []:
                    if getattr(c, "type", None) == "output_text":
                        chunks.append(getattr(c, "text", ""))
        content_text = "".join(chunks).strip() or getattr(resp, "output_text", "")
    except TypeError:
        # Old SDK: fall back to chat.completions
        pass
    except AttributeError:
        # Very old SDK shape: fall back
        pass

    if not content_text:
        # Fallback: chat.completions (works widely)
        resp = client.chat.completions.create(
            model=OPENAI_VISION_MODEL,   # must be vision-capable (gpt-4o / gpt-4o-mini)
            temperature=0,
            messages=messages,
        )
        content_text = resp.choices[0].message.content

    data = _extract_json(content_text)

    items = data.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError("Model returned no items")

    cleaned: List[Dict[str, Any]] = []
    for it in items:
        name = str(it.get("name", "")).strip()
        if not name:
            continue
        cleaned.append(
            {
                "name": name.lower(),
                "confidence": float(it.get("confidence", 0.5)),
                "portion_text": (str(it.get("portion_text", "")).strip() or None),
                "estimated_grams": int(max(0, round(float(it.get("estimated_grams", 0))))),
            }
        )

    if not cleaned:
        raise ValueError("No valid items after cleaning")

    return cleaned




def usda_search_top_hit(query: str) -> Optional[Dict[str, Any]]:
    """
    Search FoodData Central for a food item and return the most relevant single result.
    Try strict first, then relaxed.
    """
    try:
        url = "https://api.nal.usda.gov/fdc/v1/foods/search"
        for require_all in (True, False):
            params = {
                "api_key": USDA_API_KEY,
                "query": query,
                "pageSize": 1,
                "dataType": "Survey (FNDDS),SR Legacy,Foundation,Branded",
                "requireAllWords": require_all,
            }
            r = requests.get(url, params=params, timeout=10)
            if r.status_code == 200:
                foods = (r.json().get("foods") or [])
                if foods:
                    return foods[0]
        return None
    except Exception:
        return None


def _find_label_kcal(label: Optional[Dict[str, Any]]) -> Optional[float]:
    if not label:
        return None
    cal = label.get("calories")
    if isinstance(cal, dict) and cal.get("value") is not None:
        return float(cal["value"])
    return None


def extract_calories_per_100g(usda_food: Dict[str, Any]) -> Optional[float]:
    """
    Extract kcal per 100g from USDA food data.
    """
    nutrients = usda_food.get("foodNutrients") or []
    kcal_per_100g = None

    for n in nutrients:
        unit = (n.get("unitName") or "").lower()
        name = (n.get("nutrientName") or "").lower()
        if unit == "kcal" and ("energy" in name or "calories" in name):
            val = n.get("value")
            if val is not None:
                kcal_per_100g = float(val)
                break

    # Try label-based calc (common for branded)
    if kcal_per_100g is None:
        serving_size = usda_food.get("servingSize")
        serving_unit = (usda_food.get("servingSizeUnit") or "").lower()
        label_kcal = _find_label_kcal(usda_food.get("labelNutrients"))
        if (
            label_kcal is not None
            and serving_size
            and serving_unit in ("g", "gram", "grams")
        ):
            try:
                kcal_per_100g = (float(label_kcal) / float(serving_size)) * 100.0
            except Exception:
                pass

    return kcal_per_100g



def estimate_item_calories(item: Dict[str, Any], usda_food: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    grams = int(item.get("estimated_grams") or 0)
    if grams <= 0:
        grams = 150  # fallback

    kcal_per_100g = extract_calories_per_100g(usda_food) if usda_food else None
    est_calories = (kcal_per_100g * grams / 100.0) if kcal_per_100g is not None else None

    return {
        "name": item["name"],
        "confidence": round(float(item["confidence"]), 3),
        "portion_text": item.get("portion_text"),
        "estimated_grams": grams,
        "usda_fdc_id": usda_food.get("fdcId") if usda_food else None,
        "kcal_per_100g": round(kcal_per_100g, 2) if kcal_per_100g is not None else None,
        "estimated_calories": round(est_calories, 1) if est_calories is not None else None,
        "usda_brand": usda_food.get("brandOwner") if usda_food else None,
        "description": usda_food.get("description") if usda_food else None,
    }

def analyze_calories_flow(
    image_bytes: bytes,
    portion_hint: Optional[str] = None,
    mime: str = "image/jpeg",
) -> Dict[str, Any]:
    """
    Full flow: detect foods (vision), look each up in USDA, estimate calories, sum.
    """
    detections = identify_foods_with_llm(image_bytes, portion_hint, mime=mime)

    detailed_items = []
    for det in detections:
        usda_hit = usda_search_top_hit(det["name"])
        detailed_items.append(estimate_item_calories(det, usda_hit))

    total_cal = sum(
        i["estimated_calories"] for i in detailed_items if i["estimated_calories"] is not None
    )
    return {
        "items": detailed_items,
        "total_calories_estimate": round(total_cal, 1),
    }
