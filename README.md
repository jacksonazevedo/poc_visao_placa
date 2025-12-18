# Reconhecimento de Placas (YOLOv11 + PaddleOCR)

Pipeline para detecção de placas em cenas de veículos e leitura via OCR.

## Estrutura
- `dataset/`: Roboflow export (YOLO) com `data.yaml`, splits `train/valid/test`.
- `runs/plates/train/weights/best.pt`: checkpoint treinado (base `yolo11n.pt`, 30 épocas).
- `main.py`: treino, inferência em imagem/vídeo/stream, OCR e salvamento.
- `.config/`: diretório local para caches do Ultralytics (criado automaticamente).

## Ambiente
```bash
python3 -m venv env
env/bin/pip install --upgrade pip
env/bin/pip install torch==2.3.1+cu121 torchvision==0.18.1+cu121 --index-url https://download.pytorch.org/whl/cu121
env/bin/pip install numpy==1.26.4 paddlepaddle==2.6.2 paddleocr==2.7.3
env/bin/pip install "ultralytics>=8.3.0" opencv-python
```

## Uso rápido
- `--mode train|infer`: escolhe treino ou inferência.
- `--weights`: checkpoint para treinar ou inferir (`yolo11n.pt` ou `runs/plates/train/weights/best.pt`).
- `--source`: imagem/vídeo local ou stream (rtsp/rtmp/http).
- `--show`: abre janela ao vivo (vídeo/stream).
- `--save-video`: grava saída anotada ao inferir vídeo.
- `--track`: ativa ByteTrack para manter IDs e cachear OCR por objeto.
- `--ocr-model-dir`: caminho para modelo PaddleOCR de LPR (usa `det=False`).

## Treino
```bash
ULTRALYTICS_CONFIG_DIR=.config XDG_CONFIG_HOME=.config \
env/bin/python main.py --mode train \
  --weights yolo11n.pt --epochs 30 --imgsz 640 --batch 16
```
Saída: `runs/plates/train/weights/best.pt` e métricas em `runs/plates/train/results.csv`.

## Inferência (exemplos prontos)

### Imagem
```bash
ULTRALYTICS_CONFIG_DIR=.config XDG_CONFIG_HOME=.config \
env/bin/python main.py --mode infer \
  --weights runs/plates/train/weights/best.pt \
  --source input/exemplo1.jpg --conf 0.25
```
Saída: `runs/plates/predict/exemplo1_ocr.jpg`. Troque para `input/exemplo2.png` se preferir.

### Vídeo local (`input/video1.mp4`)
```bash
ULTRALYTICS_CONFIG_DIR=.config XDG_CONFIG_HOME=.config \
env/bin/python main.py --mode infer \
  --weights runs/plates/train/weights/best.pt \
  --source input/video1.mp4 \
  --conf 0.25 --imgsz 640 --save-video
```
Saída: `runs/plates/predict/video1_ocr.mp4`.

### Vídeo com rastreamento (cache OCR por ID)
```bash
ULTRALYTICS_CONFIG_DIR=.config XDG_CONFIG_HOME=.config \
env/bin/python main.py --mode infer \
  --weights runs/plates/train/weights/best.pt \
  --source input/video2.mp4 \
  --conf 0.25 --imgsz 640 --track --save-video --show
```
Saída: `runs/plates/predict/video2_ocr.mp4`. Pressione `q` para fechar a janela.

### Stream (RTSP/RTMP/HTTP)
```bash
ULTRALYTICS_CONFIG_DIR=.config XDG_CONFIG_HOME=.config \
env/bin/python main.py --mode infer \
  --weights runs/plates/train/weights/best.pt \
  --source rtsp://localhost:8554/cam1 \
  --conf 0.25 --imgsz 640 --show --save-video
```
Saída: `runs/plates/predict/<stream>_ocr.mp4`.

Arquivos disponíveis em `input/`: `exemplo1.jpg`, `exemplo2.png`, `video1.mp4`, `video2.mp4`, `video3.mp4`, `video4.mp4`.

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
