import io
import re
import cv2
import numpy as np
import zxingcpp
from flask import Flask, render_template, request, jsonify
from PIL import Image, ExifTags

try:
    import easyocr
    _EASYOCR_DISPONIBLE = True
except Exception:
    _EASYOCR_DISPONIBLE = False

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # 32 MB

# Inicializar OCR una sola vez al arrancar (tarda la primera vez)
_ocr = None
def get_ocr():
    global _ocr
    if not _EASYOCR_DISPONIBLE:
        return None
    if _ocr is None:
        _ocr = easyocr.Reader(["en"], gpu=False, verbose=False)
    return _ocr


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

def rotar_90(bgr: np.ndarray, veces: int) -> np.ndarray:
    """Rota 90° * veces en sentido antihorario (1=90, 2=180, 3=270)."""
    return np.rot90(bgr, k=veces)

def corregir_exif(imagen: Image.Image) -> Image.Image:
    """Corrige la orientación usando los metadatos EXIF del celular."""
    try:
        exif = imagen._getexif()
        if exif is None:
            return imagen
        orientacion_tag = next(
            k for k, v in ExifTags.TAGS.items() if v == "Orientation"
        )
        orientacion = exif.get(orientacion_tag)
        rotaciones = {3: 180, 6: 270, 8: 90}
        if orientacion in rotaciones:
            return imagen.rotate(rotaciones[orientacion], expand=True)
    except Exception:
        pass
    return imagen


# ── Corrección de perspectiva ─────────────────────────────────────────────────

def corregir_perspectiva(bgr: np.ndarray) -> list[np.ndarray]:
    """
    Detecta rectángulos (etiquetas) en la imagen y aplica transformación
    de perspectiva para aplanarlos. Muy útil para fotos tomadas de costado.
    """
    resultados = []
    gris = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gris, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges = cv2.dilate(edges, kernel, iterations=2)

    contornos, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = bgr.shape[:2]
    area_img = h * w

    for cnt in sorted(contornos, key=cv2.contourArea, reverse=True)[:5]:
        area = cv2.contourArea(cnt)
        # Solo rectángulos que ocupen entre 5% y 80% de la imagen
        if area < area_img * 0.05 or area > area_img * 0.80:
            continue

        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        if len(approx) == 4:
            pts = approx.reshape(4, 2).astype(np.float32)
            # Ordenar: top-left, top-right, bottom-right, bottom-left
            s = pts.sum(axis=1)
            d = np.diff(pts, axis=1)
            ordered = np.array([
                pts[np.argmin(s)],
                pts[np.argmin(d)],
                pts[np.argmax(s)],
                pts[np.argmax(d)],
            ], dtype=np.float32)

            wa = np.linalg.norm(ordered[1] - ordered[0])
            wb = np.linalg.norm(ordered[2] - ordered[3])
            ha = np.linalg.norm(ordered[3] - ordered[0])
            hb = np.linalg.norm(ordered[2] - ordered[1])
            nw, nh = int(max(wa, wb)), int(max(ha, hb))

            if nw < 50 or nh < 50:
                continue

            dst = np.array([[0,0],[nw,0],[nw,nh],[0,nh]], dtype=np.float32)
            M = cv2.getPerspectiveTransform(ordered, dst)
            warped = cv2.warpPerspective(bgr, M, (nw, nh))
            resultados.append(warped)

    return resultados


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
    yield f"{p}gris",       cv_a_rgb(cv2.cvtColor(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR))
    yield f"{p}clahe",      cv_a_rgb(clahe(bgr))
    yield f"{p}clahe+enf",  cv_a_rgb(enfocar(clahe(bgr)))
    yield f"{p}g0.5+enf",   cv_a_rgb(enfocar(gamma(bgr, 0.5)))
    yield f"{p}g0.4+enf",   cv_a_rgb(enfocar(gamma(bgr, 0.4)))
    yield f"{p}g0.4+clahe", cv_a_rgb(clahe(gamma(bgr, 0.4)))
    yield f"{p}b80+clahe",  cv_a_rgb(clahe(brillo_contraste(bgr, 80, 1.6)))
    yield f"{p}b80+enf",    cv_a_rgb(enfocar(brillo_contraste(bgr, 80, 1.6)))
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


def _escanear_franja(franja: np.ndarray, prefijo: str, votos: dict):
    f = escalar(franja, 2.0)
    _acumular(cv_a_rgb(f),                            f"{prefijo}orig",  votos)
    _acumular(cv_a_rgb(clahe(f)),                     f"{prefijo}clahe", votos)
    _acumular(cv_a_rgb(enfocar(clahe(f))),             f"{prefijo}c+enf", votos)
    _acumular(cv_a_rgb(enfocar(gamma(f, 0.4))),       f"{prefijo}g+enf", votos)
    _acumular(cv_a_rgb(umbral_adaptativo(f, 21)),     f"{prefijo}adp21", votos)
    _acumular(cv_a_rgb(brillo_contraste(f, 80, 1.6)), f"{prefijo}b80",   votos)


def _escanear_region_completa(region: np.ndarray, prefijo: str, votos: dict,
                               hay_resultados_fn=None):
    """
    Escanea región completa. Si ya hay resultados tras el escaneo base,
    no continúa con las franjas.
    """
    h = region.shape[0]

    # Primero la región completa con todas las variantes
    for nombre, arr in _variantes(region, prefijo=prefijo):
        _acumular(arr, nombre, votos)

    # Si ya encontró algo, no hace falta escanear por franjas
    if hay_resultados_fn and hay_resultados_fn():
        return

    # Franjas solo si la región completa no fue suficiente
    franja_h = h // 3
    if franja_h < 15:
        return
    for fi in range(4):
        y1 = max(0, fi * franja_h - franja_h // 4)
        y2 = min(h, y1 + franja_h + franja_h // 4)
        franja = region[y1:y2, :]
        if franja.shape[0] < 15:
            continue
        _escanear_franja(franja, f"{prefijo}f{fi}_", votos)


def leer_codigos_barras(imagen: Image.Image) -> list[dict]:
    imagen = corregir_exif(imagen)
    bgr_orig = pil_a_cv(imagen)
    bgr = escalar_max(bgr_orig, max_lado=2000)
    votos: dict[str, dict] = {}

    def hay_resultados():
        return any(len(v["detecciones"]) >= 2 for v in votos.values())

    # ── Fase 1: imagen completa
    for nombre, arr in _variantes(bgr):
        _acumular(arr, nombre, votos)

    # ── Fase 2: regiones detectadas (escanea TODAS para no perder barcodes)
    regiones = detectar_regiones_barcode(bgr)
    for i, region in enumerate(regiones):
        if region.size == 0:
            continue
        region_esc = escalar(region, 2.0)
        # hay_resultados se pasa para saltear franjas dentro de una región
        # si ya encontró algo, pero NO se hace break — escanea todas las regiones
        _escanear_region_completa(region_esc, f"reg{i}_", votos, hay_resultados)

    if hay_resultados():
        return _formatear(votos)

    # ── Fase 3: perspectiva + regiones
    for i, plano in enumerate(corregir_perspectiva(bgr)):
        plano_esc = escalar(plano, 2.0) if min(plano.shape[:2]) < 400 else plano
        _escanear_region_completa(plano_esc, f"persp{i}_", votos, hay_resultados)
        if hay_resultados():
            break
        for j, reg in enumerate(detectar_regiones_barcode(plano_esc)):
            reg_esc = escalar(reg, 2.0)
            _escanear_region_completa(reg_esc, f"persp{i}r{j}_", votos, hay_resultados)
            if hay_resultados():
                break

    if hay_resultados():
        return _formatear(votos)

    # ── Fase 3: rotaciones 90/180/270 + regiones
    for veces in [0, 1, 2, 3]:
        base = rotar_90(bgr, veces) if veces > 0 else bgr
        for i, region in enumerate(detectar_regiones_barcode(base)):
            if region.size == 0:
                continue
            region_esc = escalar(region, 2.0)
            _escanear_region_completa(region_esc, f"r90x{veces}r{i}_", votos, hay_resultados)
            if hay_resultados():
                break
        if hay_resultados():
            break

    if hay_resultados():
        return _formatear(votos)

    # ── Fase 4: imagen completa (todas las rotaciones)
    for veces in [0, 1, 2, 3]:
        base = rotar_90(bgr, veces) if veces > 0 else bgr
        for nombre, arr in _variantes(base, prefijo=f"full{veces*90}_"):
            _acumular(arr, nombre, votos)
        if hay_resultados():
            return _formatear(votos)

    # ── Fase 5: imagen escalada ×2
    esc2 = escalar(bgr, 2.0)
    for nombre, arr in _variantes(esc2, prefijo="x2_"):
        _acumular(arr, nombre, votos)

    if hay_resultados():
        return _formatear(votos)

    # ── Fase 6: rotaciones pequeñas + regiones
    for angulo in [-5, 5, -10, 10]:
        rot = rotar(bgr, angulo)
        for i, region in enumerate(detectar_regiones_barcode(rot)):
            if region.size == 0:
                continue
            _escanear_region_completa(escalar(region, 2.0), f"rot{angulo}r{i}_", votos)
        if hay_resultados():
            return _formatear(votos)

    # ── Fase 7: tiles manuales 3×3
    h, w = bgr.shape[:2]
    sh, sw = h // 3, w // 3
    for r in range(3):
        for c in range(3):
            tile = bgr[r*sh:(r+1)*sh, c*sw:(c+1)*sw]
            esc  = escalar(tile, 2.0)
            _acumular(cv_a_rgb(esc),            f"t{r}{c}_orig",  votos)
            _acumular(cv_a_rgb(clahe(esc)),      f"t{r}{c}_clahe", votos)
            _acumular(cv_a_rgb(gamma(esc, 0.4)), f"t{r}{c}_g0.4",  votos)

    if hay_resultados():
        return _formatear(votos)

    return _formatear(votos, minimo=1)


# ── OCR fallback ──────────────────────────────────────────────────────────────

# Patrones comunes en etiquetas de equipos de red
_PATRONES_OCR = [
    re.compile(r'\bSN[:\s]*([A-Z0-9]{8,})\b', re.IGNORECASE),
    re.compile(r'\bMAC[:\s]*([A-F0-9]{12})\b', re.IGNORECASE),
    re.compile(r'\bMAC[:\s]*((?:[A-F0-9]{2}[:\-]){5}[A-F0-9]{2})\b', re.IGNORECASE),
    re.compile(r'\bPROD\s*ID[:\s]*([A-Z0-9]{8,})\b', re.IGNORECASE),
    re.compile(r'\b([A-Z0-9]{12,})\b'),   # cadena alfanumérica larga (genérico)
]

def extraer_con_ocr(bgr: np.ndarray) -> list[dict]:
    """
    Usa OCR para leer texto de la imagen cuando el barcode no pudo ser leído.
    Extrae valores que parecen códigos (SN, MAC, PROD ID, etc.)
    """
    try:
        ocr = get_ocr()
        if ocr is None:
            return []
        img_esc = escalar(bgr, 2.0)
        resultados_ocr = ocr.readtext(img_esc, detail=0, paragraph=False)
        texto_completo = " ".join(resultados_ocr)

        encontrados = {}
        for linea in resultados_ocr:
            linea = linea.strip()
            for patron in _PATRONES_OCR:
                for m in patron.finditer(linea):
                    valor = m.group(1) if m.lastindex else m.group(0)
                    valor = valor.strip(":- ")
                    if len(valor) >= 8 and valor not in encontrados:
                        encontrados[valor] = {
                            "datos": valor,
                            "tipo": "OCR",
                            "detectado_con": "ocr",
                            "confirmaciones": 1,
                            "posicion": {"x": 0, "y": 0, "ancho": 0, "alto": 0},
                        }
        return list(encontrados.values())
    except Exception as e:
        return []


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

    # OCR solo si no se encontró ningún código de barras
    if not resultados:
        bgr = pil_a_cv(imagen)
        resultados = extraer_con_ocr(bgr)

    return jsonify({
        "total": len(resultados),
        "nombre": archivo.filename,
        "codigos": resultados,
    })


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
