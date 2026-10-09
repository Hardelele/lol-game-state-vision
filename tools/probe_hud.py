"""Зависит ли модель координат от маски, HUD, обрезки и разрешения.

Зачем. Модель, которая выучила карту, должна давать тот же ответ, если
на кадре закрыто или открыто чуть другое, если кадр обрезан иначе или
пришёл в другом разрешении. Модель, которая выучила раскладку экрана,
на этом ломается. Опыт родительской сессии показал, что CoordNet
(coord_model.py) — второго рода: медиана 89 ед. на входе как при обучении
и 280-950 ед., если поменять маску или обрезку на 8%. Этот инструмент —
постоянная версия того опыта, для моделей любого типа.

Как. Кадры невиданного ролика пересобираются прямо из mp4 (ffmpeg, 1
кадр/с — как при сборке датасета, tools/build_coords.py), из каждого
делаются варианты входа, и одна и та же модель отвечает на все. Варианты:

* контроль — сохранённые сцены датасета (должны совпасть со «как при
  обучении», иначе пересборка кадров неверна);
* как при обучении; без временных масок (чат, килфид, объявления); без
  маски вообще; чужой HUD — три панели с текстурой нижней полосы
  интерфейса в местах, которые при обучении были открыты; те же места
  чёрными; обрезка шире и уже на 8%;
* кусок кадра — случайный вырез 30-50% площади произвольных пропорций;
* другое разрешение — кадр 1280×720 и 960×540 вместо 1920×1080;
* сцена 4:3 — из плотной области берётся центральный кусок 4:3;
* весь кадр — синтетический UI: игры на входе нет, координаты не
  считаются, только доля кадров, отброшенных головой «игра / не игра».

Если у патчевой модели есть голова «игра / не игра» (FrameHead), для
каждого варианта печатается ещё доля отброшенных кадров: на игровых
вариантах это ложные отбрасывания.

Старая модель получает каждый вариант так, как его получила бы она:
картинка приводится к её входу 384×200. Патчевая модель (patch_model.py)
получает картинку в родном разрешении и её геометрию — где эта картинка
лежала относительно камеры (аффинное «пиксель → пиксель хранимой сцены»);
масштаб она нормирует сама через мир. Для куска кадра геометрия выреза
известна: это проверка «по куску определить место», а не «угадать, какой
это кусок».

Метрики камеры: медиана и среднее ошибки (игровые ед.), доля промахов
больше 1000 ед. Для патчевой модели ещё точность отдельного патча: медиана ошибки точки
патча и доля патчей ближе 500 ед. — по патчам, которые на ≥75% закрыты
видимой сценой (не маской, не панелью).

Пример:
    python tools/probe_hud.py 58w57eJ5Qks runs/coords/loo-58w57eJ5Qks/model.pt \
        runs/patches/cnn/model.pt --out runs/checks/probe_hud
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from paths import COORDS_DATA, CHECKS, DATA, DATASET
from mask_frames import crop_box, mask_boxes
from coord_model import CoordNet, GRID, expected_point
from train_coords import MAP_UNITS
from patch_model import load_patchnet, load_projection, scene_from_box, infer_images

LAYOUT = DATASET / "layouts" / "spectator-volibear-challenger.json"
W, H = 1920, 1080
OPAQUE_NAMES = ("scoreboard_and_objective_icons", "team_and_intro_left",
                "team_and_intro_right", "bottom_overlays_and_minimap")


def rel(x0, y0, x1, y1):
    return (int(x0 * W), int(y0 * H), int(x1 * W), int(y1 * H))


class Variants:
    """Варианты входа из одного полного кадра 1920×1080.

    Каждый вариант даёт картинку и её геометрию: прямоугольник полного
    кадра, из которого она получена, и размер, к которому её привели.
    Маска «видна сцена» строится теми же операциями, что и картинка.
    """

    def __init__(self, layout: dict):
        self.layout = layout
        self.all = mask_boxes((W, H), layout)
        names = list(layout["mask"])
        self.opaque = [b for n, b in zip(names, self.all) if n in OPAQUE_NAMES]
        self.crop = crop_box((W, H), layout, "dense")
        cx0, cy0, cx1, cy1 = self.crop
        self.store = (720, round(720 * (cy1 - cy0) / (cx1 - cx0)))
        # Чужой HUD: три панели там, где при обучении всегда была сцена.
        self.foreign = [rel(0.30, 0.105, 0.70, 0.165), rel(0.74, 0.28, 0.85, 0.52),
                        rel(0.17, 0.68, 0.62, 0.735)]
        pw, ph = int((cx1 - cx0) * 0.08), int((cy1 - cy0) * 0.08)
        self.zoom_out = (cx0 - pw, cy0 - ph, cx1 + pw, cy1 + ph)
        self.zoom_in = (cx0 + pw, cy0 + ph, cx1 - pw, cy1 - ph)
        w43 = round((cy1 - cy0) * 4 / 3)
        mx = (cx0 + cx1) // 2
        self.box43 = (mx - w43 // 2, cy0, mx - w43 // 2 + w43, cy1)
        self.names = {
            "train": "как при обучении",
            "no_temp": "без временных масок (чат, килфид, объявления)",
            "no_mask": "без маски вообще",
            "foreign_ui": "чужой HUD: 3 панели с UI-текстурой",
            "foreign_black": "те же 3 места, но чёрные",
            "zoom_out": "обрезка шире на 8%",
            "zoom_in": "обрезка уже на 8%",
            "piece": "кусок кадра 30-50% площади",
            "res720": "кадр 1280x720",
            "res540": "кадр 960x540",
            "aspect43": "сцена 4:3 (центр плотной области)",
            "ui_full": "весь кадр — синтетический UI (не игра)",
        }

    def _build(self, fr, boxes, crop, out_size, paste_ui=None, black_extra=None,
               hud=None):
        """Маска → панели → обрезка → масштаб; то же для маски «видна сцена»."""
        s = fr.copy()
        op = Image.new("L", fr.size, 255)
        for b in hud if hud is not None else self.opaque:
            op.paste(0, b)              # настоящий HUD трансляции — не сцена
        for b in boxes:
            s.paste((0, 0, 0), b)
            op.paste(0, b)
        if paste_ui:
            # Реальная текстура интерфейса: нижняя полоса того же кадра.
            src = fr.crop(rel(0.0, 0.80, 1.0, 1.0))
            for b in paste_ui:
                s.paste(src.resize((b[2] - b[0], b[3] - b[1])), b[:2])
                op.paste(0, b)
        for b in black_extra or []:
            s.paste((0, 0, 0), b)
            op.paste(0, b)
        img = s.crop(crop).resize(out_size, Image.LANCZOS)
        opm = np.asarray(op.crop(crop).resize(out_size, Image.NEAREST)) > 127
        return img, opm

    def make(self, key: str, fr: Image.Image, rng: np.random.Generator):
        """→ (картинка PIL, A «пиксель картинки → пиксель хранимой сцены», маска)."""
        st = self.store
        if key == "train":
            img, op = self._build(fr, self.all, self.crop, st)
            return img, np.eye(3), op
        if key == "no_temp":
            img, op = self._build(fr, self.opaque, self.crop, st)
            return img, np.eye(3), op
        if key == "no_mask":
            img, op = self._build(fr, [], self.crop, st)
            return img, np.eye(3), op
        if key == "foreign_ui":
            img, op = self._build(fr, self.all, self.crop, st, paste_ui=self.foreign)
            return img, np.eye(3), op
        if key == "foreign_black":
            img, op = self._build(fr, self.all, self.crop, st, black_extra=self.foreign)
            return img, np.eye(3), op
        if key in ("zoom_out", "zoom_in"):
            box = self.zoom_out if key == "zoom_out" else self.zoom_in
            img, op = self._build(fr, self.all, box, st)
            return img, scene_from_box(box, st, self.crop, st), op
        if key == "piece":
            img, op = self._build(fr, self.all, self.crop, st)
            sw, sh = st
            area = rng.uniform(0.30, 0.50) * sw * sh
            ar = math.exp(rng.uniform(math.log(0.5), math.log(3.0)))
            cw = int(min(sw, math.sqrt(area * ar)))
            ch = int(min(sh, area / cw))
            x0, y0 = int(rng.integers(0, sw - cw + 1)), int(rng.integers(0, sh - ch + 1))
            a = np.array([[1, 0, x0], [0, 1, y0], [0, 0, 1]], float)
            return img.crop((x0, y0, x0 + cw, y0 + ch)), a, op[y0:y0 + ch, x0:x0 + cw]
        if key in ("res720", "res540"):
            w2, h2 = (1280, 720) if key == "res720" else (960, 540)
            small = fr.resize((w2, h2), Image.LANCZOS)
            f = W / w2
            boxes = mask_boxes((w2, h2), self.layout)
            crop = crop_box((w2, h2), self.layout, "dense")
            size = (crop[2] - crop[0], crop[3] - crop[1])
            s = small.copy()
            op = Image.new("L", small.size, 255)
            for b in boxes:
                s.paste((0, 0, 0), b)
                op.paste(0, b)
            img = s.crop(crop)
            opm = np.asarray(op.crop(crop)) > 127
            box = tuple(v * f for v in crop)
            return img, scene_from_box(box, size, self.crop, st), opm
        if key == "aspect43":
            b = self.box43
            size = (round((b[2] - b[0]) * st[0] / (self.crop[2] - self.crop[0])), st[1])
            img, op = self._build(fr, self.all, b, size)
            return img, scene_from_box(b, size, self.crop, st), op
        if key == "ui_full":
            # Кадр, где игры нет вовсе: координаты на нём смысла не имеют,
            # проверяется только, отбрасывает ли его голова «игра / не игра».
            from train_patches import synth_ui
            ui = synth_ui(st[0], st[1], rng)
            return Image.fromarray(ui), np.eye(3), np.zeros((st[1], st[0]), bool)
        raise KeyError(key)


def frames(video: Path):
    """Полные кадры 1 кадр/с — тот же поток, что при сборке датасета."""
    cmd = ["ffmpeg", "-v", "error", "-i", str(video), "-vf", "fps=1",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=W * H * 6)
    try:
        while True:
            buf = p.stdout.read(W * H * 3)
            if len(buf) < W * H * 3:
                break
            yield np.frombuffer(buf, np.uint8).reshape(H, W, 3)
    finally:
        if p.poll() is None:
            p.terminate()
        p.stdout.close()
        p.wait()


class OldModel:
    """CoordNet: картинка → вход 384×200 → ожидаемая точка тепловой карты."""

    kind = "coordnet"

    def __init__(self, path: Path, dev: str):
        ck = torch.load(path, map_location=dev, weights_only=False)
        self.net = CoordNet(grid=ck.get("grid", GRID)).to(dev)
        self.net.encoder.load_state_dict(ck["encoder"])
        self.net.head.load_state_dict(ck["head"])
        self.net.eval()
        self.size = tuple(ck["input_size"])
        self.grid = ck.get("grid", GRID)
        self.trained_on = ck.get("trained_on", [])
        self.dev = dev

    def tens(self, img: Image.Image) -> torch.Tensor:
        a = np.asarray(img.convert("RGB").resize(self.size, Image.BILINEAR),
                       dtype=np.float32) / 255.0
        return torch.from_numpy(np.ascontiguousarray(a.transpose(2, 0, 1) * 2 - 1))

    @torch.no_grad()
    def predict(self, imgs, mats, opens):
        out = []
        for j in range(0, len(imgs), 64):
            x = torch.stack([self.tens(im) for im in imgs[j:j + 64]]).to(self.dev)
            out.append(expected_point(self.net(x).float(), self.grid).cpu().numpy())
        return np.concatenate(out), None


class PatchModel:
    kind = "patch"

    def __init__(self, path: Path, dev: str):
        self.net = load_patchnet(path, dev)
        ck = torch.load(path, map_location="cpu", weights_only=False)
        self.trained_on = ck.get("trained_on", [])
        self.h0, _ = load_projection()
        self.dev = dev
        # Порог головы «игра / не игра»; None — у чекпоинта её нет.
        self.threshold = self.net.frame_threshold if self.net.frame is not None else None

    def predict(self, imgs, mats, opens):
        arr = [np.asarray(im.convert("RGB")) for im in imgs]
        cams, _, patches = infer_images(self.net, arr, mats, opens, self.h0, self.dev)
        return cams, patches


def load_any(path: Path, dev: str):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    return (PatchModel if ck.get("kind") == "patch" else OldModel)(path, dev)


def metrics(pred, gt, patches=None, threshold=None) -> dict:
    e = np.linalg.norm(pred - gt, axis=1) * MAP_UNITS
    e = np.where(np.isnan(e), 1e5, e)
    out = {"n": int(len(e)), "median": float(np.median(e)), "mean": float(e.mean()),
           "miss1000": float((e > 1000).mean())}
    if patches is not None:
        pe = np.concatenate([np.linalg.norm(q["pts"] - (t + q["off"]), axis=1)
                             for q, t in zip(patches, gt) if len(q["pts"])]
                            or [np.array([])]) * MAP_UNITS
        out["patch_n"] = int(len(pe))
        out["patch_median"] = float(np.median(pe)) if len(pe) else float("nan")
        out["patch_within500"] = float((pe < 500).mean()) if len(pe) else float("nan")
        g = [q.get("game") for q in patches]
        if g and g[0] is not None and threshold is not None:
            g = np.array(g, dtype=float)
            # Доля кадров, которые голова «игра / не игра» отбросила бы.
            out["rejected"] = float((g < threshold).mean())
            out["game_median"] = float(np.median(g))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("video", help="идентификатор ролика с data/videos/<id>.mp4")
    ap.add_argument("models", type=Path, nargs="+", help="чекпоинты любого типа")
    ap.add_argument("--step", type=int, default=2, help="брать каждый N-й кадр (1 кадр/с)")
    ap.add_argument("--min-quality", type=float, default=0.4)
    ap.add_argument("--variants", default="all",
                    help="через запятую; по умолчанию все")
    ap.add_argument("--chunk", type=int, default=48)
    ap.add_argument("--out", type=Path, default=CHECKS / "probe_hud")
    ap.add_argument("--tag", default="", help="суффикс имени JSON")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    layout = json.loads(LAYOUT.read_text(encoding="utf-8"))
    var = Variants(layout)
    keys = list(var.names) if args.variants == "all" else args.variants.split(",")
    rows = list(csv.DictReader((COORDS_DATA / f"{args.video}.csv").open(encoding="utf-8")))
    want = {int(r["idx"]): r for r in rows
            if float(r["quality"]) >= args.min_quality and int(r["idx"]) % args.step == 0}
    models = {str(p): load_any(p, dev) for p in args.models}
    for k, m in models.items():
        seen = " (ВИДЕЛА этот ролик при обучении!)" if args.video in m.trained_on else ""
        print(f"модель {k}: {m.kind}{seen}")

    h0, _ = load_projection()
    keep = None
    sample_dir = args.out / args.video
    sample_dir.mkdir(parents=True, exist_ok=True)
    preds = {k: {v: [] for v in ["control"] + keys} for k in models}
    pats = {k: {v: [] for v in ["control"] + keys} for k in models}
    gt = []
    buf = {v: ([], [], []) for v in ["control"] + keys}

    def flush():
        for v, (imgs, mats, ops) in buf.items():
            if not imgs:
                continue
            for k, m in models.items():
                p, pt = m.predict(imgs, mats, ops)
                preds[k][v].append(p)
                if pt is not None:
                    pats[k][v].extend(pt)
            imgs.clear(); mats.clear(); ops.clear()

    from train_patches import stored_keep
    keep = stored_keep(layout, store=var.store)
    saved = False
    n_buf = 0
    for i, fr in enumerate(frames(DATA / "videos" / f"{args.video}.mp4")):
        if i not in want:
            continue
        r = want[i]
        im = Image.fromarray(fr)
        rng = np.random.default_rng(i)
        for v in keys:
            img, a, op = var.make(v, im, rng)
            buf[v][0].append(img); buf[v][1].append(a); buf[v][2].append(op)
            if not saved and i > 600:
                img.save(sample_dir / f"{v}.jpg", quality=90)
        if i > 600:
            saved = True
        with Image.open(COORDS_DATA / r["scene"]) as s:
            buf["control"][0].append(s.convert("RGB"))
        buf["control"][1].append(np.eye(3)); buf["control"][2].append(keep)
        gt.append((float(r["cx"]), float(r["cy"])))
        n_buf += 1
        if n_buf >= args.chunk:
            flush()
            n_buf = 0
            print(f"  кадров {len(gt)} из {len(want)}", flush=True)
    flush()
    gt = np.array(gt)

    names = {"control": "сохранённые сцены из датасета (контроль)", **var.names}
    res = {}
    for k, m in models.items():
        print(f"\n{args.video}: {len(gt)} кадров, модель {k}")
        res[k] = {"kind": m.kind, "trained_on": m.trained_on, "variants": {}}
        for v in ["control"] + keys:
            p = np.concatenate(preds[k][v])
            mt = metrics(p, gt, pats[k][v] if m.kind == "patch" else None,
                         getattr(m, "threshold", None))
            if v == "ui_full":
                # Игры на входе нет — ошибка координат не считается.
                mt = {key: mt[key] for key in ("n", "rejected", "game_median") if key in mt}
                res[k]["variants"][v] = mt
                if "rejected" in mt:
                    print(f"  {names[v]:48s} отброшено как «не игра» "
                          f"{mt['rejected'] * 100:5.1f}%, медиана P(игра) {mt['game_median']:.3f}")
                continue
            res[k]["variants"][v] = mt
            line = (f"  {names[v]:48s} медиана {mt['median']:6.0f} ед.  среднее "
                    f"{mt['mean']:6.0f}  >1000 ед. {mt['miss1000'] * 100:5.1f}%")
            if "patch_median" in mt:
                line += (f"  | патч: медиана {mt['patch_median']:5.0f} ед., "
                         f"<500 ед. {mt['patch_within500'] * 100:4.1f}%")
            if "rejected" in mt:
                line += f"  | отброшено {mt['rejected'] * 100:4.1f}%"
            print(line)
    args.out.mkdir(parents=True, exist_ok=True)
    out = args.out / f"{args.video}{('-' + args.tag) if args.tag else ''}.json"
    out.write_text(json.dumps({"video": args.video, "step": args.step,
                               "frames": int(len(gt)), "variant_names": names,
                               "models": res}, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    print(f"\nсводка → {out}; примеры входов → {sample_dir}")


if __name__ == "__main__":
    main()
