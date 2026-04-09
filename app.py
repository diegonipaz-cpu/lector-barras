import base64
import io
import cv2
import numpy as np
import zxingcpp
from flask import Flask, render_template, request, jsonify
from PIL import Image

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # 32 MB


# ── Conversiones ──────────────────────────────────────────────────────────────

def pil_a_cv(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)

def cv_a_rgb(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


# ── Helpers de imagen ─────────────────────────────────────────────────────────

def escalar(bgr: np.ndarray, factor: float) -> np.ndarray:
    h, w = bgr.shape[:2]
    return cv2.resize(bgr, (int(w * factor), int(h * factor)), interpolation=cv2.INTER_LANCZOS4)

def escalar_max(bgr: np.ndarray, max_lado: int = 2000) -> np.ndarray:
    """Escala la imagen para que el lado mayor no supere max_lado px."""
    h, w = bgr.shape[:2]
    if max(h, w) <= max_lado:
        return bgr
    factor = max_lado / max(h, w)
    return cv2.resize(bgr, (int(w * factor), int(h * factor)), interpolation=cv2.INTER_LANCZOS4)

def clahe(bgr: np.ndarray, clip: float = 3.0, grid: int = 8) -> np.ndarray:
    cl = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid))
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(cl.apply(g), cv2.COLOR_GRAY2BGR)

def enfocar(bgr: np.ndarray, fuerza: float = 1.5) -> np.ndarray:
    blur = cv2.GaussianBlur(bgr, (0, 0), 3)
    return cv2.addWeighted(bgr, 1 + fuerza, blur, -fuerza, 0)

def gamma(bgr: np.ndarray, g: float) -> np.ndarray:
    tabla = np.array([(i / 255.0) ** g * 255 for i in range(256)], dtype=np.uint8)
    return cv2.LUT(bgr, tabla)

def brillo_contraste(bgr: np.ndarray, brillo: int, contraste: float) -> np.ndarray:
    out = bgr.astype(np.float32) * contraste + brillo
    return np.clip(out, 0, 255).astype(np.uint8)

def umbral_adaptativo(bgr: np.ndarray, tam: int = 21) -> np.ndarray:
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    th = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, tam, 2)
    return cv2.cvtColor(th, cv2.COLOR_GRAY2BGR)

def rotar(bgr: np.ndarray, angulo: float) -> np.ndarray:
    h, w = bgr.shape[:2]
    cx, cy = w // 2, h // 2
    M = cv2.getRotationMatrix2D((cx, cy), angulo, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nw = int(h * sin + w * cos)
    nh = int(h * cos + w * sin)
    M[0, 2] += (nw / 2) - cx
    M[1, 2] += (nh / 2) - cy
    return cv2.warpAffine(bgr, M, (nw, nh), flags=cv2.INTER_LANCZOS4,
                          borderMode=cv2.BORDER_REPLICATE)


# ── Detección de regiones con códigos de barras ───────────────────────────────

def detectar_regiones_barcode(bgr: np.ndarray, margen: float = 0.15) -> list[np.ndarray]:
    """
    Usa gradientes de Scharr para encontrar zonas de la imagen con
    alta densidad de bordes verticales — característica de los códigos de barras.
    Retorna recortes de esas zonas con margen extra.
    """
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # Gradiente horizontal fuerte = barras verticales del código
    grad_x = cv2.Scharr(g, cv2.CV_32F, 1, 0)
    grad_x = cv2.convertScaleAbs(grad_x)

    # Suavizar verticalmente para unir las barras del mismo código
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    cerrado = cv2.morphologyEx(grad_x, cv2.MORPH_CLOSE, kernel)

    # Umbralizar y limpiar ruido
    _, thresh = cv2.threshold(cerrado, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
    dilatado = cv2.dilate(thresh, kernel2, iterations=3)
    eroded   = cv2.erode(dilatado, kernel2, iterations=1)

    # Encontrar contornos de las regiones candidatas
    contornos, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h, w = bgr.shape[:2]
    regiones = []
    for cnt in contornos:
        x, y, cw, ch = cv2.boundingRect(cnt)
        area = cw * ch
        # Descartar regiones demasiado pequeñas o con forma incorrecta
        if area < 800 or cw < ch:
            continue
        # Agregar margen
        mx = int(cw * margen)
        my = int(ch * margen)
        x1 = max(0, x - mx)
        y1 = max(0, y - my)
        x2 = min(w, x + cw + mx)
        y2 = min(h, y + ch + my)
        regiones.append(bgr[y1:y2, x1:x2])

    return regiones


# ── Variantes de preprocesado para un fragmento ───────────────────────────────

def _variantes(bgr: np.ndarray, prefijo: str = ""):
    p = prefijo
    yield f"{p}orig",       cv_a_rgb(bgr)
    yield f"{p}clahe",      cv_a_rgb(clahe(bgr))
    yield f"{p}clahe+enf",  cv_a_rgb(enfocar(clahe(bgr)))
    yield f"{p}g0.5",       cv_a_rgb(gamma(bgr, 0.5))
    yield f"{p}g0.4",       cv_a_rgb(gamma(bgr, 0.4))
    yield f"{p}g0.4+enf",   cv_a_rgb(enfocar(gamma(bgr, 0.4)))
    yield f"{p}g0.4+clahe", cv_a_rgb(clahe(gamma(bgr, 0.4)))
    yield f"{p}b80c1.6",    cv_a_rgb(brillo_contraste(bgr, 80, 1.6))
    yield f"{p}b80+clahe",  cv_a_rgb(clahe(brillo_contraste(bgr, 80, 1.6)))
    yield f"{p}adapt21",    cv_a_rgb(umbral_adaptativo(bgr, 21))
    yield f"{p}adapt11",    cv_a_rgb(umbral_adaptativo(bgr, 11))


# ── Lector ───────────────────────────────────────────────────────────────────

def _acumular(arr_rgb: np.ndarray, nombre: str, votos: dict) -> int:
    """Escanea un array y acumula votos. Retorna cantidad de códigos nuevos."""
    nuevos = 0
    for c in zxingcpp.read_barcodes(arr_rgb):
        if not c.text:
            continue
        if c.text not in votos:
            votos[c.text] = {
                "datos": c.text,
                "tipo": c.format.name,
                "detecciones": [nombre],
                "posicion": {
                    "x":     c.position.top_left.x,
                    "y":     c.position.top_left.y,
                    "ancho": abs(c.position.top_right.x  - c.position.top_left.x),
                    "alto":  abs(c.position.bottom_left.y - c.position.top_left.y),
                },
            }
            nuevos += 1
        else:
            votos[c.text]["detecciones"].append(nombre)
    return nuevos


def leer_codigos_barras(imagen: Image.Image) -> list[dict]:
    bgr_orig = pil_a_cv(imagen)
    bgr = escalar_max(bgr_orig, max_lado=2000)
    votos: dict[str, dict] = {}

    def encontrados():
        return [v for v in votos.values() if len(v["detecciones"]) >= 2]

    # ── Fase 1: regiones detectadas automáticamente (rápido y efectivo)
    regiones = detectar_regiones_barcode(bgr)
    for i, region in enumerate(regiones):
        if region.size == 0:
            continue
        region_esc = escalar(region, 2.0)
        for nombre, arr in _variantes(region_esc, prefijo=f"reg{i}_"):
            _acumular(arr, nombre, votos)

    if encontrados():
        return _formatear(votos)

    # ── Fase 2: imagen completa con preprocesados base
    for nombre, arr in _variantes(bgr):
        _acumular(arr, nombre, votos)

    if encontrados():
        return _formatear(votos)

    # ── Fase 3: imagen escalada ×2 (para códigos pequeños)
    esc2 = escalar(bgr, 2.0)
    for nombre, arr in _variantes(esc2, prefijo="x2_"):
        _acumular(arr, nombre, votos)

    if encontrados():
        return _formatear(votos)

    # ── Fase 4: regiones con rotaciones (foto inclinada)
    for angulo in [-5, 5, -10, 10]:
        rot = rotar(bgr, angulo)
        for i, region in enumerate(detectar_regiones_barcode(rot)):
            if region.size == 0:
                continue
            region_esc = escalar(region, 2.0)
            for nombre, arr in _variantes(region_esc, prefijo=f"rot{angulo}r{i}_"):
                _acumular(arr, nombre, votos)
        if encontrados():
            return _formatear(votos)

    # ── Fase 5: tiles manuales 3×3
    h, w = bgr.shape[:2]
    sh, sw = h // 3, w // 3
    for r in range(3):
        for c in range(3):
            tile = bgr[r*sh:(r+1)*sh, c*sw:(c+1)*sw]
            esc  = escalar(tile, 2.0)
            _acumular(cv_a_rgb(esc),              f"t{r}{c}_orig",  votos)
            _acumular(cv_a_rgb(clahe(esc)),        f"t{r}{c}_clahe", votos)
            _acumular(cv_a_rgb(gamma(esc, 0.4)),   f"t{r}{c}_g0.4",  votos)

    if encontrados():
        return _formatear(votos)

    # ── Fase 6: último recurso — aceptar detecciones únicas
    return _formatear(votos, minimo=1)


def _formatear(votos: dict, minimo: int = 2) -> list[dict]:
    confirmados = [v for v in votos.values() if len(v["detecciones"]) >= minimo]
    if not confirmados and minimo > 1:
        confirmados = list(votos.values())
    return [
        {
            "datos":          v["datos"],
            "tipo":           v["tipo"],
            "detectado_con":  v["detecciones"][0],
            "confirmaciones": len(v["detecciones"]),
            "posicion":       v["posicion"],
        }
        for v in confirmados
    ]

    return [
        {
            "datos":          v["datos"],
            "tipo":           v["tipo"],
            "detectado_con":  v["detecciones"][0],
            "confirmaciones": len(v["detecciones"]),
            "posicion":       v["posicion"],
        }
        for v in confirmados
    ]


# ── Flask ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/escanear", methods=["POST"])
def escanear():
    if "imagen" not in request.files:
        return jsonify({"error": "No se recibió ninguna imagen."}), 400

    archivo = request.files["imagen"]
    if archivo.filename == "":
        return jsonify({"error": "Nombre de archivo vacío."}), 400

    try:
        imagen = Image.open(archivo.stream)
        imagen.load()
    except Exception:
        return jsonify({"error": "No se pudo abrir la imagen. Verificá el formato."}), 400

    resultados = leer_codigos_barras(imagen)

    buffer = io.BytesIO()
    fmt = imagen.format or "PNG"
    imagen.save(buffer, format=fmt)
    img_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    mime = f"image/{fmt.lower()}"

    return jsonify({
        "total": len(resultados),
        "nombre": archivo.filename,
        "imagen_b64": f"data:{mime};base64,{img_b64}",
        "codigos": resultados,
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
