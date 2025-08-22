#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Simplified GRPO for GRAIL

Uses ONLY dataset-provided columns; no trajectory parsing or title lookups.

Prompt always renders:
  • PROFILE: from viewer_profile_sentence if present, else synthesized from columns (age, q26, q29, pid1/ideo1, q31, freq_youtube, college)
  • CURRENTLY WATCHING: current_video_title + current_video_id
  • HISTORY (prior only, most recent first): from watched_detailed_json with [watched/total] seconds
  • OPTIONS: from slate_items_json (names preferred, fallback to ids)

Core columns expected:
  state_text               (not required; prompt is built from columns)
  current_video_id         (gold)
  current_video_title      (name for current)
  watched_vids_json        (list of ids in order)
  watched_detailed_json    (list of {id,title,watch_seconds,total_length,...})
  slate_items_json         (list of {title,id})
  n_options                (int; number of choices)
  viewer_profile_sentence  (optional; we synthesize a fallback if missing)

Optional GAIL shaping (env):
  GAIL_USE=1/0 (default 1)
  GAIL_DISC_MODEL=distilbert-base-uncased
  GAIL_LR=2e-5
  GAIL_ALPHA=1.0
  GAIL_DEVICE=cuda:0|cpu (default auto)
"""

from __future__ import annotations
import os, sys, re, json, logging
from typing import Any, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import datasets
import transformers
from transformers import set_seed
from trl import TrlParser, ModelConfig, get_peft_config
from trl.trainer.grpo_trainer import GRPOTrainer

# --- Open-R1 plumbing ---
_TREES = "/n/fs/similarity/trees/src"
if _TREES not in sys.path:
    sys.path.insert(0, _TREES)

from open_r1.configs import GRPOConfig, GRPOScriptArguments
from open_r1.utils import get_dataset, get_model, get_tokenizer
from open_r1.rewards import get_reward_funcs

logger = logging.getLogger(__name__)

# ---------- tiny helpers ----------
ANS_RE   = re.compile(r"(?si)<answer>\s*([^<\n]+?)\s*</answer>")
IDX_ONLY = re.compile(r'^\s*(?:option\s*)?(\d+)\s*$', re.I)

def _completion_text(x: Any) -> str:
    if isinstance(x, str): return x
    if isinstance(x, dict): return str(x.get("content", "")).strip()
    if isinstance(x, list) and x:
        for m in reversed(x):
            if isinstance(m, dict) and "content" in m:
                c = str(m.get("content","")).strip()
                if c: return c
        try: return " ".join(str(m.get("content","")).strip() for m in x if isinstance(m, dict))
        except Exception: pass
    return str(x)

def _parse_index_from_answer_block(text: str) -> Optional[int]:
    m = ANS_RE.search(text or "")
    s = (m.group(1).strip() if m else (text or "").strip())
    m2 = IDX_ONLY.match(s)
    if not m2: return None
    try: return int(m2.group(1))
    except Exception: return None

def _canon(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (s or "").lower().strip())

def _as_list_json(x: Any, default="[]") -> list:
    try:
        if isinstance(x, str): return json.loads(x or default)
        if isinstance(x, list): return x
    except Exception:
        pass
    return []

def _load_slate_items(ex: dict) -> List[dict]:
    arr = _as_list_json(ex.get("slate_items_json"))
    out = []
    for it in arr:
        if not isinstance(it, dict): continue
        t = (it.get("title") or "").strip()
        v = (it.get("id") or "").strip()
        if t or v: out.append({"title": t, "id": v})
    return out

def _gold_index_from_items(gold: str, items: List[dict]) -> int:
    gold = (gold or "").strip()
    if not gold or not items: return -1
    for i, it in enumerate(items, 1):
        if gold == it.get("id", ""): return i
    gc = _canon(gold)
    if gc:
        for i, it in enumerate(items, 1):
            if gc == _canon(it.get("title", "")): return i
    return -1

# ---------- synthesize a viewer one-liner if missing ----------
def _synthesize_viewer_sentence(ex: dict) -> str:
    bits: List[str] = []
    # age
    age = ex.get("age")
    try:
        age_i = int(age) if age not in (None, "", "nan") else None
    except Exception:
        age_i = None
    if isinstance(age_i, int) and age_i > 0:
        bits.append(f"{age_i}-year-old")
    # gender (q26)
    gender = str(ex.get("q26") or "").strip().lower()
    if gender in {"man", "male"}: bits.append("man")
    elif gender in {"woman", "female"}: bits.append("woman")
    elif gender: bits.append(gender.title())
    # race (q29)
    race = str(ex.get("q29") or "").strip()
    if race and race.lower() != "nan": bits.append(race)
    # party/ideo (pid1 / ideo1 preferred)
    pid1  = str(ex.get("pid1") or "").strip()
    ideo1 = str(ex.get("ideo1") or "").strip()
    if pid1 and pid1.lower() != "nan":
        if ideo1 and ideo1.lower() != "nan":
            bits.append(f"{pid1} {ideo1}".lower())
        else:
            bits.append(pid1)
    elif ideo1 and ideo1.lower() != "nan":
        bits.append(ideo1.lower())
    # income (q31)
    inc = str(ex.get("q31") or "").strip()
    if inc and inc.lower() != "nan": bits.append(inc)
    # education (college -> True/False object; or text)
    college = str(ex.get("college") or "").strip().lower()
    if college in {"true","1","yes","y"}: bits.append("college-educated")
    # youtube frequency
    fy = str(ex.get("freq_youtube") or "").strip()
    if fy in {"0","1","2","3","4","5"}:
        fmap = {"0":"rarely","1":"occasionally","2":"a few times a month","3":"weekly","4":"several times a week","5":"daily"}
        bits.append(f"watches YouTube {fmap[fy]}")
    # fallback if empty
    s = ", ".join(b for b in bits if b)
    return s if s else "(no profile provided)"

# ---------- build the user prompt purely from columns ----------
def _build_user_prompt_from_columns(ex: dict, max_hist: int = 12) -> str:
    lines: List[str] = []

    # PROFILE
    viewer = (ex.get("viewer_profile_sentence") or "").strip()
    if not viewer:
        viewer = _synthesize_viewer_sentence(ex)
    lines.append("PROFILE:")
    lines.append(viewer)

    # CURRENTLY WATCHING
    cvt  = (ex.get("current_video_title") or "").strip()
    cvid = (ex.get("current_video_id") or "").strip()
    if cvt or cvid:
        lines.append("\nCURRENTLY WATCHING:")
        lines.append(f"{cvt or '(untitled)'}{(' — id: '+cvid) if cvid else ''}")

    # HISTORY (prior only, most recent first)
    vids = _as_list_json(ex.get("watched_vids_json"))
    det  = _as_list_json(ex.get("watched_detailed_json"))
    cur_idx = -1
    if cvid and vids:
        try: cur_idx = vids.index(cvid)
        except ValueError: cur_idx = len(vids)
    prior = det[:max(0, cur_idx)] if isinstance(det, list) else []

    if prior:
        def _sec(x):
            try: return f"{int(round(float(x)))}s"
            except Exception: return "?"
        lines.append("\nHISTORY (most recent first):")
        recent = list(reversed(prior))[:max_hist if max_hist and max_hist > 0 else len(prior)]
        for r in recent:
            name = (r.get("title") or r.get("id") or "(untitled)").strip()
            ws, tl = _sec(r.get("watch_seconds")), _sec(r.get("total_length"))
            lines.append(f"- [{ws}/{tl}] {name}")

    # OPTIONS
    items = _load_slate_items(ex)
    lines.append("\nOPTIONS:")
    if items:
        for i, it in enumerate(items, 1):
            name = (it.get("title") or it.get("id") or "(untitled)").strip()
            lines.append(f"{i}. {name}")
    else:
        lines.append("(no options provided)")

    return "\n".join(lines)

# ---------- simple discriminator for optional GAIL ----------
class OnlineDiscriminator:
    def __init__(self, model_name: str, device: torch.device, lr: float = 2e-5):
        from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
        self.tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        cfg = AutoConfig.from_pretrained(model_name, num_labels=2)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, config=cfg)
        self.model.to(device).train()
        self.opt = optim.AdamW(self.model.parameters(), lr=lr)
        self.device = device

    @torch.no_grad()
    def prob_positive(self, texts: List[str]) -> np.ndarray:
        batch = self.tok(texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to(self.device)
        logits = self.model(**batch).logits
        return logits.softmax(dim=-1)[:, 1].detach().cpu().numpy()

    def train_batch(self, texts: List[str], labels: List[int]) -> Optional[float]:
        if not texts: return None
        batch = self.tok(texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to(self.device)
        y = torch.tensor(labels, dtype=torch.long, device=self.device)
        out = self.model(**batch, labels=y)
        loss = out.loss
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.opt.step()
        return float(loss.detach().cpu().item())

def _render_disc_text(viewer: str, state_text: str,
                      slate_items: List[dict], action_surface: str, action_id: Optional[str]) -> str:
    names = [f"{i}. {(it.get('title') or it.get('id') or '(untitled)')}" for i, it in enumerate(slate_items, 1)]
    ids   = [f"{i}. {(it.get('id') or '(none)')}" for i, it in enumerate(slate_items, 1)]
    parts = [
        f"VIEWER: {viewer or '(none)'}",
        f"STATE:\n{state_text or '(none)'}",
        "SLATE (names):",
        *(names if names else ["(none)"]),
        "SLATE_IDS:",
        *(ids if ids else ["(none)"]),
        f"ACTION_NAME: {action_surface or '(none)'}",
        f"ACTION_ID: {action_id or '(none)'}",
    ]
    return "\n".join(parts)

def make_gail_reward_fn(disc: Optional[OnlineDiscriminator], alpha: float = 1.0):
    def _reward(completions, answer, **kw):
        if disc is None: return [0.0] * len(completions)
        viewer  = kw.get("viewer_profile") or ""
        state   = kw.get("state_text") or ""
        items   = kw.get("slate_items") or []
        def _aslist(x, n): return x if isinstance(x, list) else [x]*n
        n = len(completions)
        viewerL = _aslist(viewer, n); stateL = _aslist(state, n); itemsL = _aslist(items, n)

        texts = []
        for comp, v, s, its in zip(completions, viewerL, stateL, itemsL):
            mi = _parse_index_from_answer_block(_completion_text(comp))
            choice = its[mi-1] if (isinstance(mi, int) and 1 <= mi <= len(its)) else {"title":"", "id":None}
            surface = (choice.get("title") or choice.get("id") or "")
            texts.append(_render_disc_text(v or "", s or "", its or [], surface, choice.get("id")))
        probs = disc.prob_positive(texts)

        out = []
        for comp, its, pr in zip(completions, itemsL, probs):
            mi = _parse_index_from_answer_block(_completion_text(comp))
            ok = isinstance(mi, int) and 1 <= mi <= len(its)
            out.append(float(np.clip(pr if ok else 0.0, 0.0, 1.0)) * alpha)
        return out
    return _reward

# ---------- map one row -> conversation dict ----------
def _make_conversation_simple(ex: dict, system_prompt: Optional[str], max_hist: int = 12) -> dict:
    user_msg = _build_user_prompt_from_columns(ex, max_hist=max_hist)

    sys_msg = system_prompt or (
        "You are choosing EXACTLY ONE item from a short slate for a specific viewer.\n"
        "Think briefly in <think>…</think>, then output ONLY the option NUMBER (1..N) inside <answer>…</answer>.\n"
        "Format (STRICT): <think>…</think><answer>3</answer>"
    )

    items = _load_slate_items(ex)
    nopts = int(ex.get("n_options") or len(items) or 0)

    gold_id = str(ex.get("current_video_id") or "").strip()
    gidx    = _gold_index_from_items(gold_id, items)

    return {
        "prompt": [{"role":"system","content":sys_msg},{"role":"user","content":user_msg}],
        "answer": gold_id,
        "gold_index": gidx,
        "n_options": nopts,
        "viewer_profile": str(ex.get("viewer_profile_sentence") or _synthesize_viewer_sentence(ex)),
        "state_text": user_msg,
        "slate_items": items,
        "slate_text":  str(ex.get("slate_text") or ""),
        "task": "GRAIL",
        "is_replay": False, "accuracy": 0.0, "mix_group_id": -1, "mix_copy_idx": -1,
    }

# ---------- rewards ----------
def index_accuracy_reward(completions, answer, **kw) -> List[float]:
    gold_index = kw.get("gold_index") or [-1] * len(completions)
    if isinstance(gold_index, int): gold_index = [gold_index] * len(completions)
    outs = []
    for comp, g in zip(completions, gold_index):
        mi = _parse_index_from_answer_block(_completion_text(comp))
        outs.append(1.0 if (mi is not None and int(g) == mi and g > 0) else 0.0)
    return outs

# ---------- main ----------
def main(script_args, training_args, model_args):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    set_seed(training_args.seed)

    raw = get_dataset(script_args)
    tok = get_tokenizer(model_args, training_args)

    # Keep rows with gold AND at least one option
    def _ok(ex):
        gold = str(ex.get("current_video_id") or "").strip()
        if not gold: return False
        try: nopt = int(ex.get("n_options") or 0)
        except Exception: nopt = 0
        if nopt > 0: return True
        # fallback: parse slate items to check
        return len(_load_slate_items(ex)) > 0

    raw = raw.filter(_ok)

    # Build conversation from columns only
    max_hist = int(os.environ.get("GRAIL_MAX_HISTORY", "12") or "12")
    ds = raw.map(lambda ex: _make_conversation_simple(ex, training_args.system_prompt, max_hist=max_hist),
                 load_from_cache_file=False)

    # Keep only what trainer/rewards need
    keep_cols = {
        "prompt","answer","gold_index","n_options",
        "viewer_profile","state_text","slate_text","slate_items",
        "task","is_replay","accuracy","mix_group_id","mix_copy_idx"
    }
    for split in list(ds.keys()):
        drop = [c for c in ds[split].column_names if c not in keep_cols]
        ds[split] = ds[split].remove_columns(drop)

    # Rewards: YAML + optional GAIL + index accuracy
    reward_fns = []
    try:
        reward_fns = get_reward_funcs(script_args, ref_model=None, tokenizer=tok)
    except Exception as e:
        print(f"[warn] get_reward_funcs failed: {e}")

    use_gail   = os.environ.get("GAIL_USE", "1") != "0"
    gail_alpha = float(os.environ.get("GAIL_ALPHA", "1.0"))
    disc = None
    if use_gail:
        dev = torch.device(os.environ.get("GAIL_DEVICE", "cuda:0" if torch.cuda.is_available() else "cpu"))
        disc_model = os.environ.get("GAIL_DISC_MODEL", "distilbert-base-uncased")
        disc_lr    = float(os.environ.get("GAIL_LR", "2e-5"))
        disc = OnlineDiscriminator(disc_model, dev, lr=disc_lr)
        reward_fns.append(make_gail_reward_fn(disc, alpha=gail_alpha))

    # Ensure index accuracy is present
    names = {getattr(fn, "__name__", "") for fn in reward_fns}
    if "index_accuracy_reward" not in names:
        reward_fns.append(index_accuracy_reward)

    # Normalize weights (uniform if not provided)
    ws = list(map(float, getattr(training_args, "reward_weights", [1.0] * len(reward_fns))))
    if len(ws) != len(reward_fns): ws = [1.0] * len(reward_fns)
    s = sum(max(0.0, w) for w in ws) or 1.0
    training_args.reward_weights = [max(0.0, w)/s for w in ws]
    print(f"[grpo] rewards={len(reward_fns)} weights={training_args.reward_weights}")

    # Model + trainer
    model = get_model(model_args, training_args)
    model.generation_config.do_sample = True
    model.generation_config.return_dict_in_generate = True
    model.config.return_dict_in_generate = True

    train_split = script_args.dataset_train_split
    eval_ds = None
    if getattr(training_args, "do_eval", False) and script_args.dataset_test_split in ds:
        full = ds[script_args.dataset_test_split]
        n_keep = max(1, int(0.1 * len(full)))
        eval_ds = full.shuffle(seed=training_args.seed).select(range(n_keep))

    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        reward_funcs=reward_fns,
        train_dataset=ds[train_split],
        eval_dataset=eval_ds,
        peft_config=get_peft_config(model_args),
        processing_class=tok,
    )

    # Train/Eval/Save
    from transformers.trainer_utils import get_last_checkpoint
    last = (training_args.resume_from_checkpoint
            or (get_last_checkpoint(training_args.output_dir) if os.path.isdir(training_args.output_dir) else None))
    train_result = trainer.train(resume_from_checkpoint=last)
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()
    trainer.save_model(training_args.output_dir)

    if getattr(training_args, "do_eval", False):
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if getattr(training_args, "push_to_hub", False):
        trainer.push_to_hub(dataset_name=script_args.dataset_name, tags=["open-r1"])

if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
