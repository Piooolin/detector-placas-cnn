# Detector de Placas Vehiculares con YOLOv8

Sistema de detección y reconocimiento de placas usando Deep Learning.

## Autores
- Leonardo Chuquilin
- Henry Pillpe
- Jesus Reyes
- Diego Ojeda
- Rolando Roller

**Universidad San Ignacio de Loyola - 2025**

## Instalación

1. Clonar repositorio:

git clone https://github.com/Piooolin/detector-placas-cnn.git
cd detector-placas-cnn

2. Crear entorno virtual:

python -m venv venv-cnn
venv-cnn\Scripts\actívate

3. Instalar dependencias:

pip install -r requirements.txt

## Uso

python src/detect_and_read_plates.py

## Resultados

- Precisión: 98.3%
- mAP@0.5: 97.2%
- Velocidad: 1ms por imagen

## Tecnologías

- Python 3.11
- PyTorch 2.5.1
- YOLOv8
- EasyOCR
