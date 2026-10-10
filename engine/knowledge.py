"""Knowledge base: what has worked, per kind of source. This is the 'learning' part.

- Every tournament result is recorded per source category (e.g. live/sd/heavy/low) and per
  content type, so later runs on similar footage start from proven strategies.
- Trial models are promoted when they beat the incumbents, retired when they keep losing.
- User feedback (ratings / preferred candidate) nudges the judge's metric weights, so the
  automatic judge gradually agrees with human taste.
"""
from __future__ import annotations

import time

from .common import DATA_DIR, load_json, save_json
from .judge import DEFAULT_WEIGHTS

KB_PATH = DATA_DIR / "knowledge.json"
PROMOTE_AFTER, RETIRE_AFTER = 2, 4


class Knowledge:
    def __init__(self) -> None:
        self.d = load_json(KB_PATH, {})
        self.d.setdefault("weights", dict(DEFAULT_WEIGHTS))
        for k, v in DEFAULT_WEIGHTS.items():
            self.d["weights"].setdefault(k, v)
        self.d.setdefault("stats", {})      # scope -> strategy_key -> {n, mean, wins, model}
        self.d.setdefault("models", {})     # model -> {trials, wins, deltas}
        self.d.setdefault("feedback", [])

    def save(self) -> None:
        save_json(KB_PATH, self.d)

    @property
    def weights(self) -> dict:
        return self.d["weights"]

    @staticmethod
    def scopes(category: str) -> list[str]:
        content, res, comp, noise = category.split("/")
        return [category, f"{content}/{res}/{comp}", f"{content}/{res}", content]

    # ---------------------------------------------------------------- priors for the planner
    def prior(self, category: str, key: str) -> tuple[float, int]:
        """Best available evidence for a strategy key: (mean score, n) from the most specific scope."""
        for sc in self.scopes(category):
            st = self.d["stats"].get(sc, {}).get(key)
            if st and st["n"] > 0:
                return st["mean"], st["n"]
        return float("nan"), 0

    def model_prior(self, category: str, model: str) -> tuple[float, int]:
        for sc in self.scopes(category):
            rows = [v for v in self.d["stats"].get(sc, {}).values() if v.get("model") == model]
            if rows:
                n = sum(r["n"] for r in rows)
                return max(r["mean"] for r in rows), n
        return float("nan"), 0

    def champion(self, category: str) -> str | None:
        for sc in self.scopes(category):
            st = self.d["stats"].get(sc)
            if st:
                best = max(st.items(), key=lambda kv: (kv[1]["wins"], kv[1]["mean"]))
                if best[1]["n"] >= 1:
                    return best[0]
        return None

    # ---------------------------------------------------------------- record results
    def record(self, category: str, results: list[dict], registry) -> list[str]:
        """results: [{key, model, score}] from one tournament (higher = better)."""
        events = []
        if not results:
            return events
        winner = max(results, key=lambda r: r["score"])
        for sc in self.scopes(category):
            table = self.d["stats"].setdefault(sc, {})
            for r in results:
                st = table.setdefault(r["key"], {"n": 0, "mean": 0.0, "wins": 0, "model": r["model"]})
                st["n"] += 1
                st["mean"] += (r["score"] - st["mean"]) / st["n"]
                st["wins"] += int(r is winner)

        # model lifecycle: compare each model's best entry to the best incumbent this round
        best_by_model: dict[str, float] = {}
        for r in results:
            best_by_model[r["model"]] = max(best_by_model.get(r["model"], -1e9), r["score"])
        incumbents = [s for m, s in best_by_model.items()
                      if registry.data["models"].get(m, {}).get("status") == "active"]
        ref = max(incumbents) if incumbents else max(best_by_model.values())
        for m, s in best_by_model.items():
            meta = registry.data["models"].get(m, {})
            mk = self.d["models"].setdefault(m, {"trials": 0, "wins": 0, "deltas": []})
            mk["trials"] += 1
            mk["wins"] += int(m == winner["model"])
            mk["deltas"] = (mk["deltas"] + [round(s - ref, 2)])[-10:]
            if meta.get("status") == "trial":
                recent = mk["deltas"][-RETIRE_AFTER:]
                if mk["wins"] >= 1 or (mk["trials"] >= PROMOTE_AFTER and
                                       sum(mk["deltas"]) / len(mk["deltas"]) > -1.5):
                    registry.set_status(m, "active", f"(won or matched incumbents: {mk['deltas']})")
                    events.append(f"trial model {m} PROMOTED")
                elif len(recent) >= RETIRE_AFTER and max(recent) < -5:
                    registry.set_status(m, "retired", f"(lost by >5 pts in {RETIRE_AFTER} trials)")
                    events.append(f"trial model {m} retired")
        self.save()
        return events

    # ---------------------------------------------------------------- human feedback
    def feedback(self, run: dict, preferred: str | None, rating: int | None, lr: float = 0.15) -> list[str]:
        """Align the judge with a human: if the user prefers a candidate the judge ranked lower,
        shift weight toward the metrics where the preferred one was better."""
        msgs = []
        cands = {c["id"]: c for c in run.get("candidates", [])}
        w = self.d["weights"]
        if preferred and preferred in cands:
            pref = cands[preferred]
            judged = max(cands.values(), key=lambda c: c["score"])
            if judged["id"] != pref["id"]:
                for k in w:
                    if k not in pref["components"] or k not in judged["components"]:
                        continue
                    diff = pref["components"][k] - judged["components"][k]
                    if k in ("blocking", "ringing", "colour"):
                        diff = -diff
                    w[k] = max(0.01, w[k] + lr * diff)
                total = sum(w.values())
                for k in w:
                    w[k] = round(w[k] / total, 4)
                msgs.append("judge weights updated: " + ", ".join(f"{k}={v:.3f}" for k, v in w.items()))
            # human preference counts as a strong win for that strategy
            for sc in self.scopes(run["category"]):
                st = self.d["stats"].setdefault(sc, {}).setdefault(
                    pref["key"], {"n": 0, "mean": pref["score"], "wins": 0, "model": pref["model"]})
                st["wins"] += 3
            msgs.append(f"recorded human preference for {pref['label']}")
        if rating is not None:
            chosen = run.get("chosen_key")
            for sc in self.scopes(run["category"]):
                st = self.d["stats"].get(sc, {}).get(chosen)
                if st:
                    st["mean"] += (rating - 3) * 2.0   # 1..5 stars -> -4..+4 points
            msgs.append(f"rating {rating}/5 applied to {chosen}")
        self.d["feedback"].append({"run": run["id"], "preferred": preferred, "rating": rating,
                                   "time": time.strftime("%Y-%m-%d %H:%M")})
        self.save()
        return msgs
