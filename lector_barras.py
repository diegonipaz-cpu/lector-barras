import sys
import json
import webbrowser
from pathlib import Path
from datetime import datetime
from PIL import Image
from pyzbar.pyzbar import decode


def leer_codigos_barras(ruta_imagen: str) -> list[dict]:
    ruta = Path(ruta_imagen)
    if not ruta.exists():
        raise FileNotFoundError(f"No se encontró la imagen: {ruta_imagen}")

    imagen = Image.open(ruta)
    codigos = decode(imagen)

    resultados = []
    for codigo in codigos:
        resultados.append({
            "datos": codigo.data.decode("utf-8"),
            "tipo": codigo.type,
            "posicion": {
                "x": codigo.rect.left,
                "y": codigo.rect.top,
                "ancho": codigo.rect.width,
                "alto": codigo.rect.height,
            },
        })

    return resultados


def generar_reporte_html(ruta_imagen: str, resultados: list[dict]) -> Path:
    template_path = Path(__file__).parent / "template.html"
    template = template_path.read_text(encoding="utf-8")

    datos_json = json.dumps(resultados, ensure_ascii=False)
    fecha = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

    html = (
        template
        .replace("{{IMAGEN}}", ruta_imagen.replace("\\", "/"))
        .replace("{{FECHA}}", fecha)
        .replace("{{TOTAL}}", str(len(resultados)))
        .replace("{{DATOS_JSON}}", datos_json)
    )

    salida = Path(__file__).parent / "reporte.html"
    salida.write_text(html, encoding="utf-8")
    return salida


def main():
    if len(sys.argv) < 2:
        print("Uso: python lector_barras.py <ruta_imagen>")
        sys.exit(1)

    ruta_imagen = sys.argv[1]

    try:
        resultados = leer_codigos_barras(ruta_imagen)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    salida = generar_reporte_html(ruta_imagen, resultados)
    print(f"Reporte generado: {salida}")
    webbrowser.open(salida.as_uri())


if __name__ == "__main__":
    main()
