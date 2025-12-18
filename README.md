# Reconhecimento de Placas (YOLOv11 + PaddleOCR)

Pipeline para detecção de placas em cenas de veículos e leitura via OCR.

## Estrutura
- `dataset/`: Roboflow export (YOLO) com `data.yaml`, splits `train/valid/test`.
- `runs/plates/train/weights/best.pt`: checkpoint treinado (base `yolo11n.pt`, 30 épocas).
- `main.py`: treino, inferência em imagem/vídeo/stream, OCR e salvamento.

## Ambiente
```bash
python3 -m venv env
env/bin/pip install --upgrade pip
env/bin/pip install torch==2.3.1+cu121 torchvision==0.18.1+cu121 --index-url https://download.pytorch.org/whl/cu121
env/bin/pip install numpy==1.26.4 paddlepaddle==2.6.2 paddleocr==2.7.3
env/bin/pip install "ultralytics>=8.3.0" opencv-python
```

## Treino
```bash
ULTRALYTICS_CONFIG_DIR=.config XDG_CONFIG_HOME=.config \
env/bin/python main.py --mode train \
  --weights yolo11n.pt --epochs 30 --imgsz 640 --batch 16
```
Saída: `runs/plates/train/weights/best.pt` e métricas em `runs/plates/train/results.csv`.

## Inferência em imagem + OCR
```bash
ULTRALYTICS_CONFIG_DIR=.config XDG_CONFIG_HOME=.config \
env/bin/python main.py --mode infer \
  --weights runs/plates/train/weights/best.pt \
  --source exemplo.jpg --conf 0.25
```
Resultado anotado: `runs/plates/predict/<nome>_ocr.jpg`.

## Vídeo ou stream (RTSP/RTMP/HTTP)
```bash
ULTRALYTICS_CONFIG_DIR=.config XDG_CONFIG_HOME=.config \
env/bin/python main.py --mode infer \
  --weights runs/plates/train/weights/best.pt \
  --source rtsp://localhost:8554/cam1 \
  --conf 0.25 --imgsz 640 \
  --show --save-video
```
Resultado anotado: `runs/plates/predict/<stream>_ocr.mp4`. Pressione `q` para sair da janela.

## Subir o MediaMTX (RTSP/RTMP/HLS) com Docker
`docker-compose.yml` expõe RTSP (8554), RTMP (1935) e HTTP/HLS/API (8888).
```bash
docker compose up -d mediamtx
# RTSP: rtsp://localhost:8554/cam1
# RTMP: rtmp://localhost:1935/cam1
# UI/API/HLS: http://localhost:8888
```
Publique um fluxo empurrando para `rtsp://localhost:8554/cam1` ou `rtmp://localhost:1935/cam1` (ex.: OBS/ffmpeg). Depois rode a inferência apontando para RTSP (mais estável):
```bash
ULTRALYTICS_CONFIG_DIR=.config XDG_CONFIG_HOME=.config \
env/bin/python main.py --mode infer \
  --weights runs/plates/train/weights/best.pt \
  --source rtsp://localhost:8554/cam1 \
  --conf 0.25 --imgsz 640 --show --save-video
```

## Observações
- Use `--show` para janela ao vivo; `--save-video` para gravar.
- `_expand_box` em `main.py` define margem do crop para OCR (padrão 5%).
- Para streams, RTSP é a opção mais estável (`rtsp://localhost:8554/cam1`).
