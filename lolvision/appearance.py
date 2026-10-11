"""Облик под плашкой → чемпион: признаки DINOv2 и выбор внутри состава.

Зачем. Имя над плашкой видно не всегда: вместо него состояние
(«STUNNED»), его закрывают эффекты, текст других плашек, край кадра.
Облик под плашкой виден почти всегда. Метки для обучения даёт само имя
над плашкой («HUD как учитель», docs/entity-model-plan.md, 2.3).

Признак — замороженный DINOv2 ViT-S/14 по вырезу облика 128×128
(lolvision/identity.py), увеличенному до 224×224: CLS-токен и среднее
патч-токенов, 768 чисел, L2-нормировано. Над ним два классификатора:

* общий — линейная голова на чемпионах обучающих роликов
  (tools/train_appearance.py); на чужом ролике выбирает только среди пяти
  чемпионов команды плашки; чемпиона, которого в обучении не было, назвать
  не может;
* по матчу — прототипы из вырезов того же ролика, где имя прочитано
  (`MatchPrototypes`): скин, хрома и форма этого матча у прототипа те же,
  поэтому устойчивость к скинам получается сама собой. Работает с первых
  минут, когда у чемпиона набралось хоть несколько прочитанных имён.

Формы (Нидали, Элиза, Джейс, Гнар) и эволюция скинов — несколько
прототипов на чемпиона (k ближайших, а не среднее), так что новая форма
узнаётся, как только для неё есть прочитанные имена.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from lolvision.scene_kind import _MEAN, _STD

IN = 224
FEAT = 768


@torch.no_grad()
def features(vit: torch.nn.Module, crops: np.ndarray, dev: str = "cuda",
             batch: int = 256) -> np.ndarray:
    """crops (N, 128, 128, 3) uint8 → (N, 768) float32, L2-нормировано."""
    mean = torch.tensor(_MEAN, device=dev).view(1, 3, 1, 1)
    std = torch.tensor(_STD, device=dev).view(1, 3, 1, 1)
    out = []
    for i in range(0, len(crops), batch):
        x = torch.from_numpy(np.ascontiguousarray(crops[i:i + batch])).to(dev)
        x = x.permute(0, 3, 1, 2).float() / 255
        x = F.interpolate(x, size=(IN, IN), mode="bilinear", align_corners=False)
        with torch.autocast("cuda", dtype=torch.float16, enabled=dev.startswith("cuda")):
            f = vit.forward_features((x - mean) / std)
        v = torch.cat([f["x_norm_clstoken"], f["x_norm_patchtokens"].mean(1)], 1).float()
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


class MatchPrototypes:
    """Прототипы облика внутри одного матча: вырезы с прочитанным именем."""

    def __init__(self, k: int = 5):
        self.k = k
        self.feats: list[np.ndarray] = []
        self.labels: list[str] = []

    def add(self, f: np.ndarray, label: str) -> None:
        self.feats.append(np.asarray(f, np.float32))
        self.labels.append(label)

    def scores(self, f: np.ndarray, candidates: list[str]) -> np.ndarray:
        """(N, len(candidates)): среднее сходство с k ближайшими прототипами
        кандидата; нет прототипов — −1."""
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
