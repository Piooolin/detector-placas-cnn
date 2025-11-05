from ultralytics import YOLO
import cv2
import os
import easyocr
import re
from datetime import datetime

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
    text = re.sub(r'[^A-Z0-9-]', '', text.upper())
    
    text = ' '.join(text.split())
    
    return text

if __name__ == '__main__':
    model_path = 'runs/detect/detector_placas3/weights/best.pt'
    image_path = 'test_image.jpg'
    output_folder = 'detecciones_con_ocr'
    conf_threshold = 0.5
    
    ocr_languages = ['es', 'en']
    
    print("\n" + "="*70)
    print("DETECTOR Y LECTOR AUTOMÁTICO DE PLACAS")
    print("="*70 + "\n")
    
    os.makedirs(output_folder, exist_ok=True)
    
    print("Cargando modelo de detección...")
    model = YOLO(model_path)
    
    print("Inicializando motor OCR...")
    print("   (Esto puede tardar unos segundos la primera vez)")
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
            
            placa_crop = image[y1:y2, x1:x2]
            
            print(f"{'─'*70}")
            print(f"PLACA #{idx}")
            print(f"{'─'*70}")
            print(f"Confianza detección: {conf_detection:.2%}")
            print(f"Ubicación: ({x1}, {y1}) → ({x2}, {y2})")
            print(f"Tamaño: {x2-x1}x{y2-y1} píxeles")
            
            placa_processed = preprocess_plate_image(placa_crop)
            
            print(f"Leyendo texto...")
            
            try:
                ocr_result_original = reader.readtext(placa_crop)
                
                ocr_result_processed = reader.readtext(placa_processed)
                
                if ocr_result_original and ocr_result_processed:
                    conf_original = max([r[2] for r in ocr_result_original]) if ocr_result_original else 0
                    conf_processed = max([r[2] for r in ocr_result_processed]) if ocr_result_processed else 0
                    
                    if conf_processed > conf_original:
                        ocr_result = ocr_result_processed
                    else:
                        ocr_result = ocr_result_original
                else:
                    ocr_result = ocr_result_original or ocr_result_processed
                
                if ocr_result:
                    text_parts = [detection[1] for detection in ocr_result]
                    raw_text = ' '.join(text_parts)
                    cleaned_text = clean_plate_text(raw_text)
                    
                    ocr_confidence = sum([detection[2] for detection in ocr_result]) / len(ocr_result)
                    
                    print(f"Texto detectado: {cleaned_text}")
                    print(f"Confianza OCR: {ocr_confidence:.2%}")
                    
                    result_data = {
                        'placa_num': idx,
                        'texto': cleaned_text,
                        'conf_deteccion': conf_detection,
                        'conf_ocr': ocr_confidence,
                        'coordenadas': (x1, y1, x2, y2)
                    }
                    all_results.append(result_data)
                    
                else:
                    print(f"No se pudo leer texto en esta placa")
                    result_data = {
                        'placa_num': idx,
                        'texto': 'NO DETECTADO',
                        'conf_deteccion': conf_detection,
                        'conf_ocr': 0,
                        'coordenadas': (x1, y1, x2, y2)
                    }
                    all_results.append(result_data)
                    
            except Exception as e:
                print(f"Error en OCR: {str(e)}")
                result_data = {
                    'placa_num': idx,
                    'texto': 'ERROR',
                    'conf_deteccion': conf_detection,
                    'conf_ocr': 0,
                    'coordenadas': (x1, y1, x2, y2)
                }
                all_results.append(result_data)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            crop_filename = f'placa_{idx:02d}_{cleaned_text if "cleaned_text" in locals() else "unknown"}_{timestamp}.jpg'
            crop_path = os.path.join(output_folder, crop_filename)
            cv2.imwrite(crop_path, placa_crop)
            
            processed_filename = f'placa_{idx:02d}_processed_{timestamp}.jpg'
            processed_path = os.path.join(output_folder, processed_filename)
            cv2.imwrite(processed_path, placa_processed)
            
            print(f"Guardada: {crop_filename}\n")
        
        annotated = result.plot()
        
        for res in all_results:
            x1, y1, x2, y2 = res['coordenadas']
            text = res['texto']
            cv2.putText(annotated, text, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        annotated_filename = f'resultado_completo_{datetime.now().strftime("%Y%m%d_%H%M%S")}.jpg'
        annotated_path = os.path.join(output_folder, annotated_filename)
        cv2.imwrite(annotated_path, annotated)
        
        print("="*70)
        print("RESUMEN DE RESULTADOS")
        print("="*70)
        for res in all_results:
            print(f"  Placa #{res['placa_num']}: {res['texto']} "
                  f"(Det: {res['conf_deteccion']:.1%}, OCR: {res['conf_ocr']:.1%})")
        print(f"\nArchivos guardados en: {output_folder}/")
        print("="*70 + "\n")
    
    print("Proceso completado!\n")
