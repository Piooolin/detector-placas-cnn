from ultralytics import YOLO
import cv2
import os
import easyocr
import re
import numpy as np
from datetime import datetime


def preprocess_plate_image(image):
    # Convertir a escala de grises
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Reducción de ruido manteniendo bordes
    denoised = cv2.bilateralFilter(gray, 11, 17, 17)
    
    # Threshold adaptativo para resaltar caracteres
    thresh = cv2.adaptiveThreshold(
        denoised, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )
    
    return thresh


def clean_plate_text(text):
    # Dejar solo letras mayúsculas, números y guiones
    text = re.sub(r'[^A-Z0-9-]', '', text.upper())
    return text


if __name__ == '__main__':
    model_path = 'runs/detect/detector_placas3/weights/best.pt'
    # Cambia esto por la imagen que quieras probar
    image_path = 'images/test_image13.jpg'  
    output_folder = 'detecciones_con_ocr'
    conf_threshold = 0.5
    ocr_languages = ['es', 'en']
    
    print("\n" + "="*70)
    print("DETECTOR Y LECTOR AUTOMÁTICO (VERSIÓN FINAL)")
    print("="*70 + "\n")
    
    os.makedirs(output_folder, exist_ok=True)
    
    print("Cargando modelo de detección...")
    model = YOLO(model_path)
    
    print("Inicializando motor OCR...")
    reader = easyocr.Reader(ocr_languages, gpu=True)
    print("OCR listo\n")
    
    print(f"Cargando imagen: {image_path}")
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"ERROR: No se pudo cargar '{image_path}'\n")
        exit()
    
    print("Detectando placas...")
    results = model.predict(source=image, conf=conf_threshold, verbose=False)
    
    result = results[0]
    boxes = result.boxes
    
    print("="*70)
    print(f"DETECCIÓN COMPLETADA: {len(boxes)} placa(s) encontrada(s)")
    print("="*70 + "\n")
    
    if len(boxes) == 0:
        print("No se detectaron placas en la imagen\n")
    else:
        all_results = []
        
        for idx, box in enumerate(boxes, 1):
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf_detection = box.conf[0].item()
            
            # 1. Recorte inicial
            placa_crop = image[y1:y2, x1:x2]
            
            # 2. Recorte interno (5% por lado) para quitar marcos
            h, w = placa_crop.shape[:2]
            margin_y = int(h * 0.05)
            margin_x = int(w * 0.05)
            if margin_y > 0 and margin_x > 0:
                placa_crop = placa_crop[margin_y:h-margin_y, margin_x:w-margin_x]
            
            # 3. Escalado si es pequeña
            h, w = placa_crop.shape[:2]
            if w < 300:
                scale = 300 / w
                placa_crop = cv2.resize(placa_crop, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)

            print(f"{'─'*70}")
            print(f"PLACA #{idx}")
            print(f"{'─'*70}")
            
            # Preprocesamiento
            placa_processed = preprocess_plate_image(placa_crop)
            
            print(f"Leyendo texto...")
            
            try:
                # OCR en original y procesada
                candidates = []
                res_orig = reader.readtext(placa_crop)
                res_proc = reader.readtext(placa_processed)
                
                if res_orig: candidates.extend(res_orig)
                if res_proc: candidates.extend(res_proc)
                
                if candidates:
                    # Lógia de selección inteligente
                    valid_candidates = [c for c in candidates if len(clean_plate_text(c[1])) >= 5]
                    
                    if valid_candidates:
                        # Si hay candidatos válidos, tomamos el de mayor confianza entre ellos
                        best_candidate = max(valid_candidates, key=lambda x: x[2])
                        print("  -> Seleccionado por longitud válida y confianza")
                    else:
                        # Si no, nos quedamos con el de mayor confianza global (fallback)
                        best_candidate = max(candidates, key=lambda x: x[2])

                    raw_text = best_candidate[1]
                    ocr_confidence = best_candidate[2]
                    cleaned_text = clean_plate_text(raw_text)
                    
                    print(f"Texto detectado: {cleaned_text}")
                    print(f"Confianza OCR: {ocr_confidence:.2%}")
                    
                    result_data = {
                        'placa_num': idx,
                        'texto': cleaned_text,
                        'conf_deteccion': conf_detection,
                        'conf_ocr': ocr_confidence,
                        'coordenadas': (x1, y1, x2, y2)
                    }
                else:
                    print(f"No se pudo leer texto claro")
                    result_data = {
                        'placa_num': idx,
                        'texto': 'NO LEÍDO',
                        'conf_deteccion': conf_detection,
                        'conf_ocr': 0.0,
                        'coordenadas': (x1, y1, x2, y2)
                    }
                
                all_results.append(result_data)
                    
            except Exception as e:
                print(f"Error en OCR: {str(e)}")
            
            # Guardar recorte para verificar
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            crop_filename = f'placa_{idx:02d}_{timestamp}.jpg'
            cv2.imwrite(os.path.join(output_folder, crop_filename), placa_crop)
        
        # Imagen anotada
        annotated = result.plot()
        for res in all_results:
            x1, y1, x2, y2 = res['coordenadas']
            text = f"{res['texto']}"
            cv2.putText(annotated, text, (x1, y1-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        annotated_filename = f'resultado_completo_{datetime.now().strftime("%Y%m%d_%H%M%S")}.jpg'
        cv2.imwrite(os.path.join(output_folder, annotated_filename), annotated)
        
        print("="*70)
        print("RESUMEN FINAL")
        print("="*70)
        for res in all_results:
            print(f" Placa #{res['placa_num']}: {res['texto']} (OCR: {res['conf_ocr']:.1%})")
        print(f"\nArchivos en: {output_folder}/")
        print("="*70 + "\n")
    
    print("Proceso completado!\n")
