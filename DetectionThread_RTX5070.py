import os
import time
from threading import Lock

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont

from PyQt6.QtCore import QThread, pyqtSignal, QUrl
from PyQt6.QtGui import QImage
import pygame

class SoundManager:
    def __init__(self):
        pygame.mixer.init()
        self.alarm = pygame.mixer.Sound("warning.mp3")
        pass
    def play_alarm(self):
        if not pygame.mixer.get_busy():
            self.alarm.play()


class DetectionThread(QThread):
    detection_ready = pyqtSignal(QImage)
    log_signal = pyqtSignal(str)

    def __init__(
        self,
        camera_thread,
        model_path="best_old_50.pt",
        parent=None,
        device="cuda:0",
        default_conf=0.5,

        # Профиль для RTX 5070: инференс отделён от частоты обновления UI.
        imgsz=640,
        infer_fps=15,
        iou=0.45,
        max_det=5,
        use_half=True,
        display_fps=15,
        display_max_width=1280,
        compile_model=False,

        # Отрисовка и диагностика
        fast_draw=True,
        diagnostic_logs=False,
    ):
        super().__init__(parent)

        self.camera_thread = camera_thread
        self._running = True

        self.imgsz = int(imgsz)
        self.infer_fps = float(infer_fps)
        self.infer_interval = 1.0 / self.infer_fps if self.infer_fps > 0 else 0.0
        self.iou = float(iou)
        self.max_det = int(max_det)
        self.use_half = bool(use_half)
        self.fast_draw = bool(fast_draw)
        self.diagnostic_logs = bool(diagnostic_logs)
        self.display_fps = max(1.0, float(display_fps))
        self.display_interval = 1.0 / self.display_fps
        self.display_max_width = max(640, int(display_max_width))
        self.compile_model = bool(compile_model)
        self.last_display_time = 0.0
        self.sound_manager = SoundManager()

        # Чтобы OpenCV не забивал CPU-потоки и не мешал PyQt.
        try:
            cv2.setNumThreads(1)
        except Exception:
            pass

        # Выбор устройства
        if str(device).startswith("cuda") and torch.cuda.is_available():
            self.device = device
            self.is_cuda = True

            # Диагностика GPU. Для RTX 5070 нужен современный PyTorch
            # с поддержкой архитектуры Blackwell/CUDA.
            try:
                gpu_index = int(str(device).split(":")[-1]) if ":" in str(device) else 0
                torch.cuda.set_device(gpu_index)
                props = torch.cuda.get_device_properties(gpu_index)
                self.log_signal.emit(
                    f"GPU: {props.name}, VRAM: {props.total_memory / (1024**3):.1f} GB, "
                    f"compute capability: {props.major}.{props.minor}"
                )
            except Exception as e:
                self.log_signal.emit(f"GPU info warning: {e}")

            try:
                torch.backends.cudnn.benchmark = True
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass
        else:
            self.device = "cpu"
            self.is_cuda = False
            self.use_half = False

        # Загрузка модели
        self.model = YOLO(model_path)

        try:
            self.model.to(self.device)
        except Exception as e:
            self.log_signal.emit(f"Model device warning: {e}")

        try:
            self.model.fuse()
        except Exception:
            pass

        # torch.compile может ускорить стабильный поток, но зависит от версии
        # PyTorch/Inductor. Поэтому по умолчанию выключен.
        if self.is_cuda and self.compile_model:
            try:
                self.model.model = torch.compile(
                    self.model.model,
                    mode="reduce-overhead",
                    fullgraph=False,
                )
                self.log_signal.emit("torch.compile включен")
            except Exception as e:
                self.log_signal.emit(f"torch.compile отключен: {e}")

        # Confidence
        self._conf_lock = Lock()
        self.conf = float(default_conf)

        # Логи
        self.last_log_time = 0.0
        self.last_detail_log_time = 0.0
        self.log_interval = 2.0
        self.prev_signature = None

        # Логика тревоги без озвучки
        self.alert_conf_threshold = 0.0
        self.alert_cooldown = 10.0
        self.last_alert_time = 0.0

        self.alert_streak_needed = 1
        self.weapon_streak = 0
        self.miss_streak = 0
        self.reset_announce_after_misses = 8
        self.weapon_announced = False

        # Названия классов, которые считаем оружием
        self.target_class_names = {
            self._normalize_name(x)
            for x in {
                "оружие",
                "Оружие ",
                "weapon",
                "gun",
                "pistol",
                "rifle",
                "knife",
            }
        }

        self.display_weapon_name = "Оружие"

        # Имена классов модели
        self.names = self._get_model_names()
        self.target_class_ids = self._find_target_class_ids()

        # Шрифт с поддержкой кириллицы
        self.font_path = self._find_cyrillic_font()
        self.box_font = ImageFont.truetype(self.font_path, 22)

        self.log_signal.emit(f"Устройство инференса: {self.device}")
        if self.is_cuda:
            try:
                self.log_signal.emit(f"CUDA: {torch.version.cuda}, PyTorch: {torch.__version__}")
            except Exception:
                pass
        self.log_signal.emit(f"Классы модели: {self.names}")
        self.log_signal.emit(f"Шрифт для кириллицы: {self.font_path}")

        if self.target_class_ids:
            self.log_signal.emit(f"ID целевых классов оружия: {self.target_class_ids}")
        else:
            self.log_signal.emit(
                "ВНИМАНИЕ: целевые классы оружия не найдены по именам. "
                "Фильтр classes отключен."
            )

        self._warmup_model()

    # -------------------------------------------------------------------------
    # Служебные методы
    # -------------------------------------------------------------------------

    def _normalize_name(self, name: str) -> str:
        return str(name).strip().lower()

    def _get_model_names(self) -> dict:
        names = getattr(self.model, "names", {}) or {}

        if isinstance(names, dict):
            return {int(k): str(v) for k, v in names.items()}

        if isinstance(names, list):
            return {i: str(v) for i, v in enumerate(names)}

        return {}

    def _find_target_class_ids(self):
        ids = []

        for cls_id, cls_name in self.names.items():
            if self._normalize_name(cls_name) in self.target_class_names:
                ids.append(int(cls_id))

        return ids if ids else None

    def _find_cyrillic_font(self) -> str:
        candidates = [
            # Windows
            r"C:\Windows\Fonts\arial.ttf",
            r"C:\Windows\Fonts\segoeui.ttf",
            r"C:\Windows\Fonts\tahoma.ttf",
            r"C:\Windows\Fonts\calibri.ttf",

            # Linux
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "/usr/share/fonts/truetype/freefont/FreeSans.ttf",

            # macOS
            "/System/Library/Fonts/Arial.ttf",
            "/Library/Fonts/Arial.ttf",
        ]

        for path in candidates:
            if os.path.exists(path):
                return path

        raise FileNotFoundError(
            "Не найден .ttf-шрифт с поддержкой кириллицы. "
            "На Windows обычно подходит C:\\Windows\\Fonts\\arial.ttf"
        )

    def _warmup_model(self):
        try:
            dummy = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)

            with torch.inference_mode():
                # Два прогрева убирают задержку первых реальных кадров.
                _ = self._predict(dummy, conf=max(0.01, self._get_conf()))
                if self.is_cuda:
                    _ = self._predict(dummy, conf=max(0.01, self._get_conf()))
                    torch.cuda.synchronize()

            self.log_signal.emit("Модель прогрета")
        except Exception as e:
            self.log_signal.emit(f"Warmup warning: {e}")

    # -------------------------------------------------------------------------
    # Confidence
    # -------------------------------------------------------------------------

    def _get_conf(self) -> float:
        with self._conf_lock:
            return float(self.conf)

    def _set_conf(self, value: float):
        with self._conf_lock:
            self.conf = float(value)

    def set_confidence(self, new_conf: float):
        self._set_conf(new_conf)

    # -------------------------------------------------------------------------
    # YOLO-инференс
    # -------------------------------------------------------------------------

    def _predict(self, frame, conf: float):
        base_kwargs = dict(
            source=frame,
            conf=conf,
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.device,
            max_det=self.max_det,
            verbose=False,
            augment=False,
            rect=True,
            stream=False,
            retina_masks=False,
        )

        # Если модель содержит отдельный класс оружия — ищем только его.
        # Это уменьшает лишнюю постобработку.
        if self.target_class_ids:
            base_kwargs["classes"] = self.target_class_ids

        # FP16 для CUDA. На GTX 1660 обычно можно оставить включенным.
        if self.is_cuda and self.use_half:
            try:
                return self.model.predict(**base_kwargs, half=True)
            except Exception as e:
                if self.diagnostic_logs:
                    self.log_signal.emit(f"FP16 недоступен, fallback в FP32: {e}")

        return self.model.predict(**base_kwargs)

    def _result_to_detections(self, result):
        boxes = result.boxes
        detections = []

        weapon_found = False
        weapon_max_conf = 0.0
        detected_set = set()
        detected_names = []

        if boxes is None or len(boxes) == 0:
            return detections, weapon_found, weapon_max_conf, detected_set, detected_names

        cls_ids = boxes.cls.detach().cpu().numpy().astype(int)
        confs = boxes.conf.detach().cpu().numpy().astype(float)
        xyxys = boxes.xyxy.detach().cpu().numpy().astype(int)

        for cls_id, det_conf, xyxy in zip(cls_ids, confs, xyxys):
            original_cls_name = str(self.names.get(int(cls_id), str(cls_id))).strip()
            cls_name = self._normalize_name(original_cls_name)
            det_conf = float(det_conf)

            is_weapon = (
                cls_name in self.target_class_names
                and det_conf >= self.alert_conf_threshold
            )

            if is_weapon:
                display_name = self.display_weapon_name
                weapon_found = True
                weapon_max_conf = max(weapon_max_conf, det_conf)
            else:
                display_name = original_cls_name

            detected_set.add(display_name)
            detected_names.append(f"{display_name}:{det_conf:.2f}")

            x1, y1, x2, y2 = [int(v) for v in xyxy]

            detections.append(
                {
                    "cls_id": int(cls_id),
                    "name": cls_name,
                    "display_name": display_name,
                    "conf": det_conf,
                    "xyxy": (x1, y1, x2, y2),
                    "is_weapon": is_weapon,
                }
            )
            self.sound_manager.play_alarm()
            
        return detections, weapon_found, weapon_max_conf, detected_set, detected_names

    # -------------------------------------------------------------------------
    # Быстрая отрисовка с кириллицей
    # -------------------------------------------------------------------------

    def _draw_fast(self, frame, detections):
        """
        Быстрая отрисовка вместо result.plot().
        Текст рисуется через Pillow, поэтому кириллица отображается нормально.
        """
        out = frame.copy()

        # Рамки рисуем через OpenCV
        for det in detections:
            x1, y1, x2, y2 = det["xyxy"]

            if det.get("is_weapon", False):
                color = (0, 0, 255)
            else:
                color = (0, 180, 255)

            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        if not detections:
            return out

        # Подписи рисуем через Pillow
        rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        draw = ImageDraw.Draw(pil_img)

        for det in detections:
            x1, y1, x2, y2 = det["xyxy"]
            label = f'{det["display_name"]} {det["conf"]:.2f}'

            if det.get("is_weapon", False):
                bg_rgb = (255, 0, 0)
            else:
                bg_rgb = (255, 180, 0)

            text_rgb = (255, 255, 255)

            bbox = draw.textbbox((0, 0), label, font=self.box_font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]

            pad_x = 6
            pad_y = 4

            text_x = max(int(x1), 0)
            text_y = int(y1) - text_h - pad_y * 2 - 4

            if text_y < 0:
                text_y = int(y1) + 4

            draw.rectangle(
                [
                    text_x,
                    text_y,
                    text_x + text_w + pad_x * 2,
                    text_y + text_h + pad_y * 2,
                ],
                fill=bg_rgb,
            )

            draw.text(
                (text_x + pad_x, text_y + pad_y),
                label,
                font=self.box_font,
                fill=text_rgb,
            )

        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    # -------------------------------------------------------------------------
    # Основной цикл
    # -------------------------------------------------------------------------

    def run(self):
        while self._running:
            loop_start = time.perf_counter()

            frame = self.camera_thread.get_last_frame_copy()

            if frame is None:
                self.msleep(20)
                continue

            frame = np.ascontiguousarray(frame)
            current_conf = self._get_conf()

            try:
                with torch.inference_mode():
                    results = self._predict(frame, conf=current_conf)
            except Exception as e:
                self.log_signal.emit(f"Detection error: {e}")
                self.msleep(100)
                continue

            result = results[0]

            (
                detections,
                weapon_found,
                weapon_max_conf,
                detected_set,
                detected_names,
            ) = self._result_to_detections(result)

            now = time.time()
            current_signature = tuple(sorted(detected_set))

            # Логируем только изменение найденных объектов или раз в несколько секунд
            if current_signature and (
                current_signature != self.prev_signature
                or now - self.last_log_time >= self.log_interval
            ):
                self.log_signal.emit(
                    f"Обнаружены объекты: {', '.join(current_signature)}"
                )
                self.prev_signature = current_signature
                self.last_log_time = now

            # Подробный лог только в диагностическом режиме
            if (
                self.diagnostic_logs
                and detected_names
                and now - self.last_detail_log_time >= self.log_interval
            ):
                self.log_signal.emit(f"Детекция: {', '.join(detected_names)}")
                self.last_detail_log_time = now

            # Тревога без озвучки — только лог
            if weapon_found:
                self.weapon_streak += 1
                self.miss_streak = 0

                if self.diagnostic_logs:
                    self.log_signal.emit(
                        f"Оружие найдено, conf={weapon_max_conf:.2f}, "
                        f"streak={self.weapon_streak}"
                    )

                if (
                    self.weapon_streak >= self.alert_streak_needed
                    and not self.weapon_announced
                    and now - self.last_alert_time >= self.alert_cooldown
                ):
                    self.last_alert_time = now
                    self.weapon_announced = True

                    self.sound_manager.play_alarm()
                    
                    self.log_signal.emit(
                        f"ТРЕВОГА: обнаружено оружие, conf={weapon_max_conf:.2f}"
                    )
            else:
                self.miss_streak += 1
                self.weapon_streak = 0

                if self.miss_streak >= self.reset_announce_after_misses:
                    self.weapon_announced = False

            # Отрисовка кадра
            try:
                if self.fast_draw:
                    annotated_frame = self._draw_fast(frame, detections)
                else:
                    annotated_frame = result.plot()
            except Exception as e:
                self.log_signal.emit(f"Draw error: {e}")
                annotated_frame = frame

            # Передача кадра в PyQt.
            # Ограничиваем FPS GUI, чтобы очередь сигналов Qt не накапливалась.
            now_display = time.perf_counter()
            if now_display - self.last_display_time >= self.display_interval:
                try:
                    display_frame = annotated_frame
                    h0, w0 = display_frame.shape[:2]

                    # Не передаём в GUI исходный 1080p/4K кадр без необходимости.
                    if w0 > self.display_max_width:
                        scale = self.display_max_width / float(w0)
                        new_w = self.display_max_width
                        new_h = max(1, int(h0 * scale))
                        display_frame = cv2.resize(
                            display_frame,
                            (new_w, new_h),
                            interpolation=cv2.INTER_AREA,
                        )

                    rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                    h, w, ch = rgb_frame.shape
                    bytes_per_line = ch * w

                    qimg = QImage(
                        rgb_frame.data,
                        w,
                        h,
                        bytes_per_line,
                        QImage.Format.Format_RGB888,
                    ).copy()

                    self.detection_ready.emit(qimg)
                    self.last_display_time = now_display

                except Exception as e:
                    self.log_signal.emit(f"QImage error: {e}")

            # Ограничение FPS инференса
            elapsed = time.perf_counter() - loop_start
            sleep_time = self.infer_interval - elapsed

            if sleep_time > 0:
                self.msleep(int(sleep_time * 1000))
            else:
                self.msleep(1)

    # -------------------------------------------------------------------------
    # Остановка
    # -------------------------------------------------------------------------

    def stop(self):
        self._running = False
        self.quit()
        self.wait()