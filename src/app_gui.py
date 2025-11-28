import gradio as gr
from ultralytics import YOLO
import cv2
import easyocr
import re
import numpy as np
from PIL import Image

model_path = 'runs/detect/detector_placas3/weights/best.pt'
ocr_languages = ['es', 'en']

print("Cargando modelos... Espere un momento.")
try:
    model = YOLO(model_path)
    reader = easyocr.Reader(ocr_languages, gpu=True)
    print("¡Modelos cargados correctamente!")
except Exception as e:
    print(f"Error cargando modelos: {e}")
    print("Verifica que el archivo best.pt exista en la ruta correcta.")

def preprocess_plate_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    denoised = cv2.bilateralFilter(gray, 11, 17, 17)
    thresh = cv2.adaptiveThreshold(
        denoised, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )
    return thresh

def clean_plate_text(text):
    return re.sub(r'[^A-Z0-9-]', '', text.upper())

def detectar_placa(input_image):
    if input_image is None:
        return None, "No se subió ninguna imagen."

    # Convertir a formato que usa OpenCV (BGR)
    image = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)
    
    # Detectar
    results = model.predict(source=image, conf=0.45, verbose=False)
    result = results[0]
    boxes = result.boxes
    
    texto_resultado = ""
    
    if len(boxes) == 0:
        return input_image, "No se detectaron placas."

    # Procesar cada placa
    for idx, box in enumerate(boxes, 1):
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf_detection = box.conf[0].item()
        
        # Recorte inteligente
        placa_crop = image[y1:y2, x1:x2]
        h, w = placa_crop.shape[:2]
        
        # Margen interno (5%)
        margin_y = int(h * 0.05)
        margin_x = int(w * 0.05)
        if margin_y > 0 and margin_x > 0:
            placa_crop = placa_crop[margin_y:h-margin_y, margin_x:w-margin_x]
            
        # Escalado
        if w < 300:
            scale = 300 / w
            placa_crop = cv2.resize(placa_crop, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
            
        # OCR
        placa_processed = preprocess_plate_image(placa_crop)
        candidates = []
        try:
            res_orig = reader.readtext(placa_crop)
            res_proc = reader.readtext(placa_processed)
            if res_orig: candidates.extend(res_orig)
            if res_proc: candidates.extend(res_proc)
        except Exception:
            pass

        texto_detectado = "NO LEÍDO"
        conf_ocr = 0.0
        
        if candidates:
            valid_candidates = [c for c in candidates if len(clean_plate_text(c[1])) >= 5]
            if valid_candidates:
                best = max(valid_candidates, key=lambda x: x[2])
            else:
                best = max(candidates, key=lambda x: x[2])
            
            raw_text = best[1]
            conf_ocr = best[2]
            texto_detectado = clean_plate_text(raw_text)

        # Dibujar (Verde fosforescente para que se vea bien)
        color = (0, 255, 0)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 4)
        
        # Fondo negro para el texto para que se lea mejor
        label = f"{texto_detectado}"
        (w_text, h_text), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        cv2.rectangle(image, (x1, y1 - 30), (x1 + w_text, y1), color, -1)
        cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
        
        texto_resultado += f"Placa #{idx}: {texto_detectado}\n"
        texto_resultado += f"   └ Confianza Detección: {conf_detection:.1%}\n"
        texto_resultado += f"   └ Confianza Lectura: {conf_ocr:.1%}\n\n"

    # Convertir BGR a RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Convertir a objeto Imagen de PIL
    final_pil_image = Image.fromarray(image_rgb)
    
    return final_pil_image, texto_resultado

interfaz = gr.Interface(
    fn=detectar_placa,
    inputs=gr.Image(label="Sube una imagen de un auto", type="numpy"),
    outputs=[
        gr.Image(label="Imagen Procesada", type="pil"),
        gr.Textbox(label="Resultados detallados", lines=5)
    ],
    title="Detector de Placas de Vehículos",
    description="Sube una foto para detectar la placa y leer su contenido automáticamente.",
)

if __name__ == "__main__":
    interfaz.launch()
