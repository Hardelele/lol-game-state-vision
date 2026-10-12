"""Облик под плашкой → чемпион: признаки DINOv2 и выбор внутри состава.

Зачем. Имя над плашкой видно не всегда: вместо него состояние
(«STUNNED»), его закрывают эффекты, текст других плашек, край кадра.
Облик под плашкой виден почти всегда. Метки для обучения даёт само имя
над плашкой («HUD как учитель», docs/entity-model-plan.md, 2.3).

Признак — замороженный DINOv2 ViT-S/14 по вырезу облика 128×128
(lolvision/identity.py), увеличенному до 224×224: среднее патч-токенов
центрального окна 6×6 из 16×16 (строки 4–9, столбцы 5–10 — там, где под
плашкой стоит модель), 384 числа, L2-нормировано. Центр, а не CLS и не
среднее по всему вырезу: в вырезе много фона и соседей, и по ближайшему
соседу центр точнее (на трёх роликах: 86/97/77% против 85/97/76% у CLS и
83/97/74% у среднего). Над ним классификаторы:

* общий — линейная голова на чемпионах обучающих роликов
  (tools/train_appearance.py); на чужом ролике выбирает только среди пяти
  чемпионов команды плашки; чемпиона, которого в обучении не было, назвать
  не может;
* по матчу — прототипы из вырезов того же ролика, где имя прочитано
  (`MatchPrototypes`): скин, хрома и форма этого матча у прототипа те же,
  поэтому устойчивость к скинам получается сама собой. Работает с первых
  минут, когда у чемпиона набралось хоть несколько прочитанных имён.

Формы (Нидали, Элиза, Джейс, Гнар) и эволюция скинов — несколько
прототипов на чемпиона (ближайший, а не среднее), так что новая форма
узнаётся, как только для неё есть прочитанные имена.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from lolvision.scene_kind import _MEAN, _STD

IN = 224
FEAT = 384
CENTER = (slice(4, 10), slice(5, 11))   # окно патчей 6×6 из 16×16


@torch.no_grad()
def features(vit: torch.nn.Module, crops: np.ndarray, dev: str = "cuda",
             batch: int = 256) -> np.ndarray:
    """crops (N, 128, 128, 3) uint8 → (N, 384) float32, L2-нормировано."""
    mean = torch.tensor(_MEAN, device=dev).view(1, 3, 1, 1)
    std = torch.tensor(_STD, device=dev).view(1, 3, 1, 1)
    out = []
    for i in range(0, len(crops), batch):
        x = torch.from_numpy(np.ascontiguousarray(crops[i:i + batch])).to(dev)
        x = x.permute(0, 3, 1, 2).float() / 255
        x = F.interpolate(x, size=(IN, IN), mode="bilinear", align_corners=False)
        with torch.autocast("cuda", dtype=torch.float16, enabled=dev.startswith("cuda")):
            f = vit.forward_features((x - mean) / std)
        g = IN // 14
        p = f["x_norm_patchtokens"].reshape(len(x), g, g, -1)
        v = p[:, CENTER[0], CENTER[1]].mean((1, 2)).float()
        out.append(F.normalize(v, dim=1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, FEAT), np.float32)


class LinearHead:
    """Общий классификатор: softmax(W·f + b) по чемпионам обучения."""

    def __init__(self, ids: list[str], w: np.ndarray, b: np.ndarray):
        self.ids = list(ids)
        self.w = np.asarray(w, np.float32)
        self.b = np.asarray(b, np.float32)

    @classmethod
    def load(cls, path) -> LinearHead:
        z = np.load(path)
        return cls([str(s) for s in z["ids"]], z["w"], z["b"])

    def save(self, path) -> None:
        np.savez(path, ids=np.array(self.ids), w=self.w, b=self.b)

    def logits(self, f: np.ndarray) -> np.ndarray:
        return f @ self.w.T + self.b

    def pick(self, f: np.ndarray, candidates: list[str]) -> tuple[list[str], np.ndarray]:
        """Для каждого признака — лучший из `candidates` и вероятности по ним
        (N, len(candidates)). Кандидат, которого голова не знает, получает 0."""
        lg = self.logits(f)
        idx = [self.ids.index(c) if c in self.ids else -1 for c in candidates]
        sub = np.stack([lg[:, i] if i >= 0 else np.full(len(f), -1e9, np.float32)
                        for i in idx], 1)
        p = np.exp(sub - sub.max(1, keepdims=True))
        p /= p.sum(1, keepdims=True)
        return [candidates[j] for j in p.argmax(1)], p


class FeatureProjection:
    """Своё пространство облика: линейная проекция признаков DINOv2 (384 → d),
    обученная разводить чемпионов обучающих роликов (косинусный
    классификатор). Сама проекция чемпионов не называет — она для
    прототипов матча: в ней ближайшие соседи меньше смотрят на фон и
    освещение и больше на модель чемпиона; работает и для чемпионов,
    которых в обучении не было."""

    def __init__(self, w: np.ndarray):
        self.w = np.asarray(w, np.float32)            # (384, d)

    def __call__(self, f: np.ndarray) -> np.ndarray:
        z = f @ self.w
        return z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-6)


class MatchPrototypes:
    """Прототипы облика внутри одного матча: вырезы с прочитанным именем."""

    def __init__(self, k: int = 1):
        self.k = k
        self.feats: list[np.ndarray] = []
        self.labels: list[str] = []

    def add(self, f: np.ndarray, label: str) -> None:
        self.feats.append(np.asarray(f, np.float32))
        self.labels.append(label)

    def scores(self, f: np.ndarray, candidates: list[str]) -> np.ndarray:
        """(N, len(candidates)): среднее сходство с k ближайшими прототипами
        кандидата (по умолчанию k = 1: при k = 5 частый чемпион — ведомый
        каналом — перетягивает редких); нет прототипов — −1."""
        out = np.full((len(f), len(candidates)), -1.0, np.float32)
        if not self.feats:
            return out
        bank = np.stack(self.feats)
        lab = np.array(self.labels)
        sims = f @ bank.T
        for j, c in enumerate(candidates):
            m = lab == c
            if m.any():
                s = np.sort(sims[:, m], 1)[:, ::-1][:, :self.k]
                out[:, j] = s.mean(1)
        return out
