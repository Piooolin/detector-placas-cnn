from ultralytics import YOLO
import cv2
import os
from datetime import datetime

if __name__ == '__main__':
    model_path = 'runs/detect/detector_placas3/weights/best.pt'
    
    image_path = 'test_image.jpg'
    
    output_folder = 'detecciones_placas'
    
    conf_threshold = 0.5
    
    print("\n" + "="*60)
    print("DETECTOR AVANZADO DE PLACAS")
    print("="*60 + "\n")
    
    os.makedirs(output_folder, exist_ok=True)
    print(f"Carpeta de salida: {output_folder}")
    
    print(f"Cargando modelo desde: {model_path}")
    model = YOLO(model_path)
    
    print(f"Cargando imagen: {image_path}\n")
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"ERROR: No se pudo cargar la imagen '{image_path}'")
        print("Verifica que la imagen exista en la ruta especificada.\n")
        exit()
    
    height, width = image.shape[:2]
    print(f"Dimensiones de la imagen: {width}x{height} píxeles\n")
    
    print(f"Detectando placas con confianza mínima de {conf_threshold*100:.0f}%...")
    results = model.predict(
        source=image,
        conf=conf_threshold,
        verbose=False
    )
    
    result = results[0]
    boxes = result.boxes
    
    print("="*60)
    print(f"DETECCIÓN COMPLETADA")
    print("="*60)
    print(f"Total de placas detectadas: {len(boxes)}\n")
    
    if len(boxes) == 0:
        print("No se detectaron placas en la imagen.")
        print("   Intenta:")
        print("   - Reducir el umbral de confianza")
        print("   - Verificar que la imagen contenga placas")
        print("   - Revisar la calidad de la imagen\n")
    else:
        print("="*60)
        print("DETALLES DE CADA PLACA DETECTADA")
        print("="*60 + "\n")
        
        for idx, box in enumerate(boxes, 1):
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = box.conf[0].item()
            
            ancho = x2 - x1
            alto = y2 - y1
            
            placa_crop = image[y1:y2, x1:x2]
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            crop_filename = f'placa_{idx:02d}_conf_{conf:.3f}_{timestamp}.jpg'
            crop_path = os.path.join(output_folder, crop_filename)
            
            cv2.imwrite(crop_path, placa_crop)
            
            print(f"Placa #{idx}:")
            print(f"  Confianza: {conf:.2%}")
            print(f"  Coordenadas: ({x1}, {y1}) → ({x2}, {y2})")
            print(f"  Dimensiones: {ancho}x{alto} píxeles")
            print(f"  Guardada como: {crop_filename}")
            print()
        
        annotated = result.plot()
        annotated_filename = f'imagen_anotada_{datetime.now().strftime("%Y%m%d_%H%M%S")}.jpg'
        annotated_path = os.path.join(output_folder, annotated_filename)
        cv2.imwrite(annotated_path, annotated)
        
        print("="*60)
        print("RESUMEN DE ARCHIVOS GENERADOS")
        print("="*60)
        print(f"{len(boxes)} placas individuales guardadas")
        print(f"Imagen completa anotada: {annotated_filename}")
        print(f"Todos los archivos en: {output_folder}/")
        print("="*60 + "\n")
        
        confianzas = [box.conf[0].item() for box in boxes]
        print("ESTADÍSTICAS")
        print("="*60)
        print(f"Confianza promedio: {sum(confianzas)/len(confianzas):.2%}")
        print(f"Confianza máxima: {max(confianzas):.2%}")
        print(f"Confianza mínima: {min(confianzas):.2%}")
        print("="*60 + "\n")
    
    print("Proceso completado exitosamente!\n")
