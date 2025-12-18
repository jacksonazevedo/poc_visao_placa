
"""
Pipeline simples para treinar YOLOv11 em placas (dataset Roboflow) e rodar OCR nas detecções.
Uso rápido:
  - Treino:  env/bin/python main.py --mode train
  - Inferência + OCR: env/bin/python main.py --mode infer --source exemplo.jpg --weights runs/plates/train/weights/best.pt
"""

from __future__ import annotations

import argparse
import os
from functools import lru_cache
import re
from pathlib import Path
from typing import List, Tuple
from urllib.parse import urlparse

import cv2
import numpy as np
import torch
from paddleocr import PaddleOCR


ROOT = Path(__file__).parent
DATA_YAML = ROOT / "dataset" / "data.yaml"
CONFIG_ROOT = ROOT / ".config"

os.environ.setdefault("XDG_CONFIG_HOME", str(CONFIG_ROOT))
os.environ.setdefault("ULTRALYTICS_CONFIG_DIR", str(CONFIG_ROOT / "ultralytics"))
CONFIG_ROOT.mkdir(parents=True, exist_ok=True)
(CONFIG_ROOT / "ultralytics").mkdir(parents=True, exist_ok=True)


def ensure_local_ultralytics_config() -> None:
    """Redireciona cache/config do Ultralytics para dentro do workspace (sandbox-friendly)."""
    config_root = ROOT / ".config"
    os.environ.setdefault("XDG_CONFIG_HOME", str(config_root))
    os.environ.setdefault("ULTRALYTICS_CONFIG_DIR", str(config_root / "ultralytics"))
    config_root.mkdir(parents=True, exist_ok=True)
    (config_root / "ultralytics").mkdir(parents=True, exist_ok=True)


def get_device() -> str:
    """Escolhe GPU se disponível, caso contrário CPU."""
    try:
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass  # Ambientes sem acesso à GPU podem disparar warnings; seguimos no CPU.
    return "cpu"


def train_detector(
    data_yaml: Path = DATA_YAML,
    base_weights: str = "yolo11n.pt",
    epochs: int = 30,
    imgsz: int = 640,
    batch: int = 16,
    project: Path | None = None,
    name: str = "train",
) -> Path:
    """Treina YOLOv11 usando o dataset local."""
    from ultralytics import YOLO

    device = get_device()
    model = YOLO(base_weights)
    project = project or (ROOT / "runs" / "plates")
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=str(project),
        name=name,
        exist_ok=True,
    )
    return Path(results.save_dir) / "weights" / "best.pt"


def _expand_box(xyxy: np.ndarray, img_shape: Tuple[int, int, int], margin: float = 0.05) -> Tuple[int, int, int, int]:
    """Expande bounding box em porcentagem e clampa nos limites da imagem."""
    x1, y1, x2, y2 = xyxy
    h, w = img_shape[:2]
    dx = (x2 - x1) * margin
    dy = (y2 - y1) * margin
    x1 = int(max(0, x1 - dx))
    y1 = int(max(0, y1 - dy))
    x2 = int(min(w - 1, x2 + dx))
    y2 = int(min(h - 1, y2 + dy))
    return x1, y1, x2, y2


def _extract_plate_text(texts: List[str]) -> str:
    """Filtra texto de placa: tenta padrões BR (Mercosul/antigo) e cai para o mais longo."""
    patterns = [
        r"[A-Z]{3}[0-9][A-Z][0-9]{2}",  # Mercosul: ABC1D23
        r"[A-Z]{3}[0-9]{4}",            # Antiga: ABC1234
    ]

    def normalize(txt: str) -> str:
        return "".join(ch for ch in txt.upper() if ch.isalnum())

    norms = [normalize(str(t)) for t in texts if str(t).strip()]
    # Prioridade: primeiro match de regex na ordem dos textos
    for pat in patterns:
        for n in norms:
            m = re.search(pat, n)
            if m:
                return m.group(0)
    if not norms:
        return ""
    # Caso sem match, retorna o mais longo
    norms.sort(key=len, reverse=True)
    return norms[0]


@lru_cache(maxsize=8)
def _get_cached_paddle_ocr(use_gpu: bool, rec_model_dir: str | None) -> PaddleOCR:
    return PaddleOCR(
        use_gpu=use_gpu,
        lang="en",
        det=False,          # já recebemos um recorte da placa, não precisa detector do OCR
        rec=True,
        use_angle_cls=False,
        rec_batch_num=8,    # pequenas placas cabem em batch maior
        rec_model_dir=rec_model_dir,
    )


def get_paddle_ocr(use_gpu: bool, rec_model_dir: str | Path | None = None) -> PaddleOCR:
    """Inicializa PaddleOCR com cache; aceita modelo LPR via rec_model_dir."""
    rec_dir = str(rec_model_dir) if rec_model_dir else None
    return _get_cached_paddle_ocr(use_gpu, rec_dir)


def run_plate_ocr(crop: np.ndarray, use_gpu: bool, rec_model_dir: str | Path | None = None) -> str:
    """Executa OCR no recorte de placa e retorna texto filtrado."""
    # PaddleOCR espera RGB; convertendo de BGR (OpenCV) para RGB.
    if crop is None or crop.size == 0:
        return ""
    if len(crop.shape) == 3 and crop.shape[2] == 3:
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

    ocr = get_paddle_ocr(use_gpu, rec_model_dir)
    result = ocr.ocr(crop, cls=False)
    texts: List[str] = []
    if result:
        lines = result[0] if isinstance(result[0], list) else result
        for line in lines:
            if isinstance(line, (list, tuple)) and len(line) >= 2:
                # formato típico: [ [ [x1,y1]... ], (text, score) ]
                rec = line[1]
                if isinstance(rec, (list, tuple)) and len(rec) >= 1:
                    texts.append(str(rec[0]))
            elif isinstance(line, str):
                texts.append(line)
    return _extract_plate_text(texts)


def _draw_box_label(img: np.ndarray, xyxy: Tuple[int, int, int, int], label: str, color=(0, 255, 0)) -> None:
    """Desenha bbox e rótulo acima da caixa para não cobrir a placa."""
    x1, y1, x2, y2 = map(int, xyxy)
    thickness = 2
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    if not label:
        return
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    (tw, th), baseline = cv2.getTextSize(label, font, font_scale, 2)
    pad = 4
    # posiciona acima da caixa, clampa no topo
    ty2 = max(0, y1 - baseline - pad)
    ty1 = max(0, ty2 - th - pad)
    tx1 = x1
    tx2 = x1 + tw + 2 * pad
    cv2.rectangle(img, (tx1, ty1), (tx2, ty2), color, -1)
    cv2.putText(img, label, (tx1 + pad, ty2 - pad), font, font_scale, (0, 0, 0), 2, cv2.LINE_AA)


def run_inference_with_ocr(
    weights: Path,
    source: str | Path,
    conf: float = 0.25,
    project: Path | None = None,
    name: str = "predict",
    rec_model_dir: str | Path | None = None,
) -> List[dict]:
    """Roda detecção e aplica OCR nas boxes; retorna metadados e salva imagem anotada."""
    from ultralytics import YOLO

    device = get_device()
    model = YOLO(str(weights))
    project = project or (ROOT / "runs" / "plates")

    results = model.predict(
        source=str(source),
        conf=conf,
        device=device,
        save=False,
        stream=False,
        imgsz=640,
    )

    annotations: List[dict] = []

    for res in results:
        img = res.orig_img
        boxes = res.boxes.xyxy.cpu().numpy()
        scores = res.boxes.conf.cpu().numpy()
        cls_ids = res.boxes.cls.cpu().numpy()

        for xyxy, score, cls_id in zip(boxes, scores, cls_ids):
            x1, y1, x2, y2 = _expand_box(xyxy, img.shape)
            crop = img[int(y1) : int(y2), int(x1) : int(x2)]
            text = run_plate_ocr(crop, device == "cuda", rec_model_dir=rec_model_dir)
            annotations.append(
                {
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "score": float(score),
                    "class_id": int(cls_id),
                    "text": text,
                }
            )
            label = f"{text or 'plate'} ({score:.2f})"
            _draw_box_label(img, (x1, y1, x2, y2), label)

        save_dir = project / name
        save_dir.mkdir(parents=True, exist_ok=True)
        out_path = save_dir / f"{Path(source).stem}_ocr.jpg"
        cv2.imwrite(str(out_path), img)
        print(f"Resultado salvo em: {out_path}")

    return annotations


def run_video_with_ocr(
    weights: Path,
    source: str | Path,
    conf: float = 0.25,
    imgsz: int = 640,
    show: bool = False,
    save: bool = True,
    use_track: bool = False,
    tracker: str = "bytetrack.yaml",
    project: Path | None = None,
    name: str = "predict",
    rec_model_dir: str | Path | None = None,
) -> None:
    """Processa vídeo frame a frame, exibindo/gravando com detecção + OCR."""
    from ultralytics import YOLO

    device = get_device()
    model = YOLO(str(weights))
    project = project or (ROOT / "runs" / "plates")
    save_dir = project / name
    save_dir.mkdir(parents=True, exist_ok=True)
    def _stem(src: str | Path) -> str:
        s = str(src)
        if "://" in s:
            parsed = urlparse(s)
            return Path(parsed.path).stem or (parsed.netloc.replace(":", "_") or "stream")
        return Path(s).stem

    out_path = save_dir / f"{_stem(source)}_ocr.mp4"
    writer = None
    frame_count = 0
    tracker_arg: str | None = tracker
    track_text_cache: dict[int, str] = {}  # evita repetir OCR para o mesmo track_id

    def init_writer_if_needed(frame: np.ndarray, fps: float) -> None:
        nonlocal writer
        if writer is not None or not save:
            return
        h, w = frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
        if not writer.isOpened():
            print(f"Aviso: VideoWriter não abriu para {out_path}, desabilitando save.")
            writer = None

    def source_fps(src: str | Path) -> float:
        cap = cv2.VideoCapture(str(src))
        if not cap.isOpened():
            return 30.0
        fps_val = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()
        return fps_val if fps_val > 1e-3 else 30.0

    try:
        if use_track:
            # valida tracker: se o caminho não existir, cai para padrão 'bytetrack.yaml'
            tracker_to_use = tracker_arg or "bytetrack.yaml"
            if tracker_arg and not Path(tracker_arg).exists():
                print(f"Aviso: tracker '{tracker_arg}' não encontrado, usando 'bytetrack.yaml'.")
                tracker_to_use = "bytetrack.yaml"

            fps_hint = source_fps(source)
            stream = model.track(
                source=str(source),
                conf=conf,
                device=device,
                imgsz=imgsz,
                tracker=tracker_to_use,
                stream=True,
                verbose=False,
                persist=True,
            )
            for res in stream:
                frame = res.orig_img
                init_writer_if_needed(frame, fps_hint)
                frame_count += 1

                boxes = res.boxes
                if boxes is None:
                    continue
                xyxy = boxes.xyxy.cpu().numpy()
                scores = boxes.conf.cpu().numpy()
                cls_ids = boxes.cls.cpu().numpy()
                track_ids = boxes.id.cpu().numpy() if boxes.id is not None else [None] * len(xyxy)

                for bb, score, cls_id, tid in zip(xyxy, scores, cls_ids, track_ids):
                    x1, y1, x2, y2 = _expand_box(bb, frame.shape)
                    crop = frame[int(y1) : int(y2), int(x1) : int(x2)]
                    cache_key = int(tid) if tid is not None else None
                    if cache_key is not None and cache_key in track_text_cache:
                        text = track_text_cache[cache_key]
                    else:
                        text = run_plate_ocr(crop, device == "cuda", rec_model_dir=rec_model_dir)
                        if cache_key is not None:
                            track_text_cache[cache_key] = text
                    tlabel = f"#{int(tid)} " if tid is not None else ""
                    label = f"{tlabel}{text or 'plate'} ({score:.2f})"
                    _draw_box_label(frame, (x1, y1, x2, y2), label)

                if writer is not None:
                    writer.write(frame)
                if show:
                    cv2.imshow("Placas (q para sair)", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
        else:
            cap = cv2.VideoCapture(str(source))
            if not cap.isOpened():
                raise RuntimeError(f"Não foi possível abrir o vídeo: {source}")
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            if fps < 1e-3:
                fps = 30
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_count += 1
                init_writer_if_needed(frame, fps)

                preds = model.predict(
                    source=frame,
                    conf=conf,
                    device=device,
                    imgsz=imgsz,
                    verbose=False,
                )[0]

                boxes = preds.boxes.xyxy.cpu().numpy() if preds.boxes is not None else []
                scores = preds.boxes.conf.cpu().numpy() if preds.boxes is not None else []
                cls_ids = preds.boxes.cls.cpu().numpy() if preds.boxes is not None else []

                for bb, score, cls_id in zip(boxes, scores, cls_ids):
                    x1, y1, x2, y2 = _expand_box(bb, frame.shape)
                    crop = frame[int(y1) : int(y2), int(x1) : int(x2)]
                    text = run_plate_ocr(crop, device == "cuda", rec_model_dir=rec_model_dir)
                    label = f"{text or 'plate'} ({score:.2f})"
                    _draw_box_label(frame, (x1, y1, x2, y2), label)

                if writer is not None:
                    writer.write(frame)
                if show:
                    cv2.imshow("Placas (q para sair)", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
            cap.release()
    finally:
        if writer is not None:
            writer.release()
            print(f"Vídeo salvo em: {out_path} ({frame_count} frames)")
        if show:
            cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Treino YOLOv11 + OCR para placas.")
    parser.add_argument("--mode", choices=["train", "infer"], default="train")
    parser.add_argument("--weights", type=str, default="yolo11n.pt", help="Checkpoint para treinar/fazer inferência.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument(
        "--source",
        type=str,
        default="exemplo.jpg",
        help="Imagem, vídeo local ou stream (rtsp/rtmp/http).",
    )
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--show", action="store_true", help="Exibe janela com vídeo em tempo real.")
    parser.add_argument("--save-video", action="store_true", help="Salva o vídeo anotado com OCR.")
    parser.add_argument("--track", action="store_true", help="Usa rastreamento (ByteTrack) para manter IDs por objeto.")
    parser.add_argument(
        "--ocr-model-dir",
        type=str,
        default=None,
        help="Caminho para modelo PaddleOCR de LPR (rec_model_dir). Use det=False.",
    )
    return parser.parse_args()


def main() -> None:
    ensure_local_ultralytics_config()
    args = parse_args()

    def is_video_like(src: str) -> bool:
        lower = src.lower()
        if lower.startswith(("rtsp://", "rtmp://", "http://", "https://", "udp://", "tcp://")):
            return True
        ext = Path(lower).suffix
        return ext in {".mp4", ".avi", ".mov", ".mkv", ".m4v"}

    if args.mode == "train":
        best = train_detector(
            data_yaml=DATA_YAML,
            base_weights=args.weights,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
        )
        print(f"Treino finalizado. Melhor checkpoint: {best}")
    else:
        source_raw = args.source
        if is_video_like(source_raw):
            run_video_with_ocr(
                weights=Path(args.weights),
                source=source_raw,
                conf=args.conf,
                imgsz=args.imgsz,
                show=args.show,
                save=args.save_video,
                use_track=args.track,
                rec_model_dir=args.ocr_model_dir,
            )
        else:
            annotations = run_inference_with_ocr(
                weights=Path(args.weights),
                source=Path(source_raw),
                conf=args.conf,
                rec_model_dir=args.ocr_model_dir,
            )
            print("Detecções + OCR:")
            for ann in annotations:
                print(ann)


if __name__ == "__main__":
    main()
