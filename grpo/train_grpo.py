"""GRPO after SFT on MolCoT-Paired-4K: Qwen3.5-4B-Base started from the Original-CoT SFT adapter, the Curated-CoT
SFT adapter, or the base model without SFT (fresh LoRA).

The SFT stage is the epoch-4 LoRA adapter written by sft/train_sft.py (checkpoint-844 of
runs/qwen3.5-4b-base/<variant>/run_<id>/model). GRPO continues training that adapter with the same prompt and
chat template; all arms get identical GRPO data and rewards, so only the initialisation differs. The GRPO data
are the training descriptions with their reference SMILES (no CoT is used).

Settings shared with SFT (LoRA adapter r=16/alpha=32/dropout 0.05, optimiser, cosine schedule with 3%
warmup, gradient clipping 0.3, 16 records per optimizer step, max length 1280, seeds, evaluation
decoding) are copied from the SFT runs; GRPO-specific settings are TRL 1.13 defaults (8 generations
per prompt, beta 0, DAPO loss, clip epsilon 0.2, rollout temperature 1.0).

Reward = 1.0 * EM + 0.25 * format (4 : 1 ratio of correctness to strict format), identical for every arm:
  EM      1 if the strict answer (last closed <answer> tag inside the model's own first turn) is the
          reference molecule after RDKit canonicalisation, else 0;
  format  1 for the SFT target format (reasoning, then exactly one <answer>...</answer> ending the
          response, no invented turn), 0.5 for a closed answer tag with a format defect, 0 otherwise.
Validity and answer rate are logged with weight 0. Deliberately not used: a length reward (it would favour the
longer CoT style of one condition), a validity reward, a repetition penalty (rollouts would differ from the
evaluation decoding), and masking of truncated completions (the base model's completions are all truncated at
the start, so the no-SFT arm would get no gradient).

Evaluation runs in the same process on the colocated vLLM engine, before SFT (base weights), after
SFT, and every --eval-every GRPO steps (default 210 = at the end of each epoch; the strict pass@1/2/4/8/16
of each evaluation are the unbiased estimates from the 16 samples), with the decoding of train_sft.py: greedy plus 16 samples
(temperature 1.0, top-p 0.9, top-k 50), at most 768 new tokens, stop at <|endoftext|>.

Run in the GRPO environment (requirements-grpo.txt: vLLM 0.29, TRL 1.13, PEFT, RDKit); see grpo/run_grpo.sh.
"""
import argparse
import json
import math
import os
import random
import re
import time
from functools import lru_cache
from pathlib import Path

import torch
from datasets import Dataset
from rdkit import RDLogger
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import GRPOConfig, GRPOTrainer
from trl.generation import vllm_generation as trl_vllm_generation

RDLogger.DisableLog("rdApp.*")

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
BASE_MODEL = "Qwen/Qwen3.5-4B-Base"
SFT_CHECKPOINT = "checkpoint-844"  # end of epoch 4 (211 optimizer steps per epoch)

# ---------------------------------------------------------------------------------------------
# Prompt and scoring, copied verbatim from sft/train_sft.py (stored rule) and
# eval/score_predictions.py (strict rule).
# ---------------------------------------------------------------------------------------------
PROMPT = """You are an expert in chemistry. Given the following molecular description, write its SMILES representation step-by-step. Analyze the main structure, then the substructures, then confirm other molecular properties. Your final answer must be exactly in the form <answer>{{smile}}</answer>.

Molecular description:
{explain}"""
ANSWER_RE = re.compile(r"<\s*answer\s*>\s*(.*?)\s*<\s*/\s*answer\s*>", re.IGNORECASE | re.DOTALL)
TURN_RE = re.compile(r"^(?:user|assistant|system)[ \t]*$", re.MULTILINE)


def extract_answer(text: str) -> str:
    matches = ANSWER_RE.findall(text or "")
    candidate = matches[-1].strip() if matches else ""
    if not candidate and text:
        nonempty = [line.strip() for line in text.splitlines() if line.strip()]
        candidate = nonempty[-1] if nonempty else ""
    candidate = candidate.replace("```smiles", "").replace("```", "").strip()
    return candidate.splitlines()[0].strip() if candidate else ""


@lru_cache(maxsize=None)
def canonicalize(smiles: str):
    from rdkit import Chem

    if not smiles:
        return None
    molecule = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(molecule, canonical=True) if molecule else None


@lru_cache(maxsize=None)
def tanimoto(smiles_a: str, smiles_b: str) -> float:
    from rdkit import Chem, DataStructs
    from rdkit.Chem import rdFingerprintGenerator

    if not smiles_a or not smiles_b:
        return 0.0
    molecule_a, molecule_b = Chem.MolFromSmiles(smiles_a), Chem.MolFromSmiles(smiles_b)
    if molecule_a is None or molecule_b is None:
        return 0.0
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2)
    return float(DataStructs.TanimotoSimilarity(generator.GetFingerprint(molecule_a), generator.GetFingerprint(molecule_b)))


def own_turn(generation):
    """The generation up to the first invented chat turn, and whether one exists."""
    boundary = TURN_RE.search(generation or "")
    if boundary is None:
        return generation or "", False
    return generation[: boundary.start()], True


def strict_answer(turn_text):
    """Content of the last closed <answer> tag, or None when there is no tag."""
    matches = ANSWER_RE.findall(turn_text)
    if not matches:
        return None
    candidate = matches[-1].strip()
    candidate = candidate.replace("```smiles", "").replace("```", "").strip()
    return candidate.splitlines()[0].strip() if candidate else ""


def score(generation: str, reference_canonical: str) -> dict:
    """Stored (train_sft.py) and strict reading of one generation."""
    stored_prediction = extract_answer(generation)
    stored_canonical = canonicalize(stored_prediction)
    turn_text, runs_on = own_turn(generation)
    answer = strict_answer(turn_text)
    strict_canonical = canonicalize(answer) if answer is not None else None
    return {
        "generation": generation,
        "prediction": stored_prediction,
        "canonical_prediction": stored_canonical,
        "valid": stored_canonical is not None,
        "exact_match": bool(stored_canonical is not None and stored_canonical == reference_canonical),
        "similarity": tanimoto(stored_canonical, reference_canonical) if stored_canonical else 0.0,
        "runon": runs_on,
        "strict_answered": answer is not None,
        "strict_canonical": strict_canonical,
        "strict_valid": strict_canonical is not None,
        "strict_em": bool(strict_canonical is not None and strict_canonical == reference_canonical),
        "strict_similarity": tanimoto(strict_canonical, reference_canonical) if strict_canonical else 0.0,
    }


# ---------------------------------------------------------------------------------------------
# Rewards (TRL passes the dataset column `reference` as a keyword argument)
# ---------------------------------------------------------------------------------------------
def _strict_canonical(completion: str):
    answer = strict_answer(own_turn(completion)[0])
    return None if answer is None else canonicalize(answer)


def exact_match_reward(prompts, completions, reference, **kwargs):
    """EM reward (weight 1.0): 1 if the strict answer is the reference molecule after RDKit
    canonicalisation, else 0. Same rule as the strict EM of the paper."""
    return [float(_strict_canonical(c) == r and r is not None) for c, r in zip(completions, reference)]


def format_score(completion: str) -> float:
    """Format of the SFT targets: step-by-step reasoning, then exactly one <answer>SMILES</answer> that ends
    the response, inside the model's own turn.
      1.0  no invented chat turn, exactly one closed answer tag, non-empty reasoning before it,
           nothing but whitespace after it;
      0.5  a closed answer tag inside the own turn, but one of the other conditions fails
           (several tags, text after the tag, no reasoning, or an invented turn after the answer);
      0.0  no closed answer tag inside the own turn (this includes completions cut at 768 tokens)."""
    turn, runs_on = own_turn(completion)
    matches = list(ANSWER_RE.finditer(turn))
    if not matches:
        return 0.0
    clean = (not runs_on and len(matches) == 1 and turn[: matches[0].start()].strip() != ""
             and turn[matches[-1].end():].strip() == "")
    return 1.0 if clean else 0.5


def format_reward(prompts, completions, reference, **kwargs):
    """Format reward (weight 0.25, i.e. EM : format = 4 : 1)."""
    return [format_score(c) for c in completions]


def valid_reward(prompts, completions, reference, **kwargs):  # logged with weight 0
    return [float(_strict_canonical(c) is not None) for c in completions]


def answered_reward(prompts, completions, reference, **kwargs):  # logged with weight 0
    return [float(strict_answer(own_turn(c)[0]) is not None) for c in completions]


# ---------------------------------------------------------------------------------------------
# vLLM weight names. The SFT adapter was trained on the text-only model that
# AutoModelForCausalLM builds from the multimodal Qwen3.5 checkpoint, whose parameters are
# called model.layers.* / model.embed_tokens / model.norm. The colocated vLLM engine runs
# Qwen3_5ForConditionalGeneration and expects the checkpoint names model.language_model.*.
# ---------------------------------------------------------------------------------------------
_trl_fix_name = trl_vllm_generation.VLLMGeneration._fix_param_name_to_vllm


def _fix_name_for_qwen35(self, name, extra_prefixes=None):
    name = _trl_fix_name(self, name, extra_prefixes)
    if name.startswith("model.") and not name.startswith(("model.language_model.", "model.visual.")):
        name = "model.language_model." + name[len("model."):]
    return name


trl_vllm_generation.VLLMGeneration._fix_param_name_to_vllm = _fix_name_for_qwen35

# TRL starts the colocated engine with seed = process index (0 for every run), so rollout sampling
# would not depend on the run. Use the run's seed instead (set in Run.main before the trainer).
VLLM_SEED = None
_trl_llm_class = trl_vllm_generation.LLM


def _seeded_llm(*args, **kwargs):
    if VLLM_SEED is not None:
        kwargs["seed"] = VLLM_SEED
    # TRL sizes the engine for the rollout batch (128). Each evaluation decodes 422 x 17 sequences,
    # so allow more concurrent sequences; the rollout batch itself is unchanged.
    kwargs["max_num_seqs"] = max(kwargs.get("max_num_seqs") or 0, 512)
    return _trl_llm_class(*args, **kwargs)


trl_vllm_generation.LLM = _seeded_llm


def vllm_model(vllm_generation):
    return vllm_generation.llm.llm_engine.model_executor.driver_worker.model_runner.model


def _iter_base_params(vllm_generation):
    """The base model's weights with the LoRA adapter NOT applied (the model before SFT)."""
    for name, param in vllm_generation.model.named_parameters():
        if "lora_" in name or "modules_to_save" in name or "original_module" in name:
            continue
        name = name.removeprefix("base_model.model.").replace(".base_layer", "")
        yield vllm_generation._fix_param_name_to_vllm(name), param.data


def push_weights_checked(vllm_generation, base_only=False):
    """sync_weights(), but count what vLLM actually loaded: an unmapped name is skipped silently.
    base_only=True pushes the base weights without the adapter (evaluation before SFT)."""
    target = vllm_model(vllm_generation)
    pushed, loaded, unloaded = 0, 0, []
    source = _iter_base_params(vllm_generation) if base_only else vllm_generation._iter_named_params()
    for name, param in source:
        pushed += 1
        result = target.load_weights([(name, param)])
        if result:
            loaded += 1
        else:
            unloaded.append(name)
    vllm_generation.llm.reset_prefix_cache()
    return {"pushed": pushed, "loaded": loaded, "not_loaded_examples": unloaded[:5], "not_loaded": len(unloaded)}


# ---------------------------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------------------------
def read_jsonl(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def render_prompt(explain, tokenizer):
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": PROMPT.format(explain=explain)}], tokenize=False, add_generation_prompt=True
    )


def unbiased_pass_at_k(n, c, k):
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


# ---------------------------------------------------------------------------------------------
# Evaluation on the colocated vLLM engine
# ---------------------------------------------------------------------------------------------
def evaluate(vllm_generation, tokenizer, test_rows, args, tag, out_dir, stage, base_only=False, extra=None):
    from vllm import SamplingParams

    started = time.time()
    with torch.no_grad():
        sync = push_weights_checked(vllm_generation, base_only=base_only)
    if sync["loaded"] < sync["pushed"] - 1:   # the tied lm_head may be skipped by vLLM
        raise RuntimeError(f"vLLM did not load the pushed weights: {sync}")
    llm = vllm_generation.llm
    eos = tokenizer.eos_token_id
    prompts = [render_prompt(row["explain"], tokenizer) for row in test_rows]
    greedy_params = SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens, stop_token_ids=[eos])
    sampled_params = SamplingParams(
        n=args.best_of_n, temperature=1.0, top_p=0.9, top_k=50, max_tokens=args.max_new_tokens,
        stop_token_ids=[eos], seed=args.eval_seed,
    )
    greedy_out = llm.generate(prompts, greedy_params, use_tqdm=False)
    sampled_out = llm.generate(prompts, sampled_params, use_tqdm=False)

    predictions, rows = [], []
    for index, (row, g, s) in enumerate(zip(test_rows, greedy_out, sampled_out)):
        reference = canonicalize(row["smile"])
        greedy = score(g.outputs[0].text, reference)
        greedy["n_tokens"] = len(g.outputs[0].token_ids)
        greedy["truncated"] = g.outputs[0].finish_reason == "length"
        sampled = []
        for o in s.outputs:
            candidate = score(o.text, reference)
            candidate["n_tokens"] = len(o.token_ids)
            candidate["truncated"] = o.finish_reason == "length"
            sampled.append(candidate)
        predictions.append({"index": index, "explain": row["explain"], "reference": row["smile"],
                            "canonical_reference": reference, "greedy": greedy, "sampled": sampled})
        rows.append((greedy, sampled))

    n = len(rows)
    mean = lambda values: 100.0 * sum(values) / len(values)
    metrics = {"tag": tag, "stage": stage, "n_records": n, "weight_sync": sync, **(extra or {})}
    for rule, em, valid, sim in (("strict", "strict_em", "strict_valid", "strict_similarity"),
                                 ("stored", "exact_match", "valid", "similarity")):
        metrics[f"{rule}_greedy_em"] = mean([g[em] for g, _ in rows])
        metrics[f"{rule}_greedy_valid"] = mean([g[valid] for g, _ in rows])
        metrics[f"{rule}_greedy_similarity"] = mean([g[sim] for g, _ in rows])
        metrics[f"{rule}_sampled_pass1_em"] = mean([sum(c[em] for c in s) / len(s) for _, s in rows])
        for k in (1, 2, 4, 8, 16):
            metrics[f"{rule}_pass_at_{k}"] = mean([unbiased_pass_at_k(len(s), sum(c[em] for c in s), k) for _, s in rows])
    metrics["strict_greedy_answered"] = mean([g["strict_answered"] for g, _ in rows])
    metrics["greedy_truncated"] = mean([g["truncated"] for g, _ in rows])
    metrics["greedy_runon"] = mean([g["runon"] for g, _ in rows])
    metrics["greedy_mean_tokens"] = sum(g["n_tokens"] for g, _ in rows) / n
    metrics["sampled_mean_tokens"] = sum(c["n_tokens"] for _, s in rows for c in s) / (n * args.best_of_n)
    metrics["eval_seconds"] = time.time() - started

    (out_dir / f"test_predictions_{tag}.json").write_text(
        json.dumps({"metrics": metrics, "predictions": predictions}, ensure_ascii=False))
    with open(out_dir / "eval_metrics.jsonl", "a", encoding="utf-8") as handle:
        handle.write(json.dumps(metrics) + "\n")
    print(f"[eval {tag}] strict greedy EM {metrics['strict_greedy_em']:.1f}  pass@16 {metrics['strict_pass_at_16']:.1f}  "
          f"answered {metrics['strict_greedy_answered']:.1f}  truncated {metrics['greedy_truncated']:.1f}  "
          f"weights {sync['loaded']}/{sync['pushed']}  {metrics['eval_seconds']:.0f}s", flush=True)
    return metrics


STAGES = {0: "before SFT (base model)", 1: "after SFT (epoch 4)", 2: "after GRPO epoch 1", 3: "after GRPO epoch 2"}


def wandb_log_eval(metrics, global_step):
    """Test metrics against test/stage (0 before SFT, 1 after SFT, 2/3 after GRPO epoch 1/2), the same
    x-axis in every run and arm; train/global_step is logged too."""
    import wandb

    if wandb.run is None:
        return
    values = {f"test/{k}": v for k, v in metrics.items()
              if isinstance(v, (int, float)) and not isinstance(v, bool) and k != "stage"}
    wandb.define_metric("test/stage")
    for key in values:
        wandb.define_metric(key, step_metric="test/stage")
    wandb.log({**values, "test/stage": metrics["stage"], "train/global_step": global_step})


class PeriodicEvaluation(TrainerCallback):
    """Evaluate on the test set every `every` optimizer steps and at the last step (the epoch ends are
    included when `every` divides the steps per epoch); send the before/after-SFT evaluations to wandb
    once the WandbCallback has opened the run (it runs before this callback)."""

    def __init__(self, run, every):
        self.run = run
        self.every = every

    def on_train_begin(self, args, state, control, **kwargs):
        if self.run.wandb_enabled:
            for metrics in self.run.initial_metrics:
                wandb_log_eval(metrics, 0)

    def on_step_end(self, args, state, control, **kwargs):
        step = state.global_step
        if step % self.every and step != state.max_steps:
            return
        epoch = step / self.run.steps_per_epoch
        # stage axis: 0 before SFT, 1 after SFT, 1 + GRPO epochs so far (2 and 3 at the epoch ends)
        metrics = self.run.evaluate(f"grpo_step_{step:04d}", stage=1 + epoch,
                                    extra={"grpo_step": step, "grpo_epoch": epoch})
        if self.run.wandb_enabled:
            wandb_log_eval(metrics, step)


# ---------------------------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--variant", choices=("curated", "original", "base"), required=True,
                   help="SFT adapter to start from; base = Qwen3.5-4B-Base without SFT and a fresh LoRA")
    p.add_argument("--run-id", type=int, required=True)
    p.add_argument("--output-dir", default=str(REPO / "runs_grpo"))
    p.add_argument("--data-dir", default=str(REPO / "data"), help="folder with train_data.jsonl and test_data.jsonl")
    p.add_argument("--sft-runs-dir", default=str(REPO / "runs" / "qwen3.5-4b-base"),
                   help="SFT runs of Qwen3.5-4B-Base written by sft/run_sft.sh (<variant>/run_<id>/model/checkpoint-844)")
    p.add_argument("--epochs", type=float, default=2.0)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--warmup-ratio", type=float, default=0.03, help="as in SFT (25 of 844 steps)")
    p.add_argument("--num-generations", type=int, default=8)
    p.add_argument("--prompts-per-step", type=int, default=16, help="as the SFT batch: 16 records per optimizer step")
    p.add_argument("--micro-batch", type=int, default=16, help="completions per forward/backward pass")
    p.add_argument("--max-new-tokens", type=int, default=768)
    p.add_argument("--best-of-n", type=int, default=16)
    p.add_argument("--vllm-gpu-memory", type=float, default=0.3)
    p.add_argument("--limit-train", type=int, default=-1)
    p.add_argument("--limit-test", type=int, default=-1)
    p.add_argument("--max-steps", type=int, default=-1)
    p.add_argument("--eval-every", type=int, default=210, help="test-set evaluation every N GRPO steps (210 = once per epoch; the last step is always evaluated)")
    p.add_argument("--skip-initial-eval", action="store_true")
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--tag", default="", help="suffix of the run directory (smoke tests)")
    return p.parse_args()


class Run:
    def __init__(self, args):
        self.args = args
        self.seed = 1000 + args.run_id           # the SFT seed of the same run id
        args.eval_seed = 2000 + args.run_id      # the SFT evaluation seed of the same run id
        self.sft_adapter = (None if args.variant == "base"
                            else Path(args.sft_runs_dir) / args.variant / f"run_{args.run_id:03d}" / "model" / SFT_CHECKPOINT)
        self.has_sft = args.variant != "base"
        self.out_dir = Path(args.output_dir) / "qwen3.5-4b-base" / args.variant / f"run_{args.run_id:03d}{args.tag}"
        self.wandb_enabled = not args.no_wandb
        self.trainer = None
        self.steps_per_epoch = None
        self.initial_metrics = []

    def evaluate(self, tag, stage, base_only=False, extra=None):
        return evaluate(self.trainer.vllm_generation, self.tokenizer, self.test_rows, self.args, tag, self.out_dir,
                        stage, base_only=base_only, extra=extra)

    def main(self):
        args = self.args
        if (self.out_dir / "final_metrics.json").exists():
            print(f"{self.out_dir} is finished; nothing to do.")
            return
        lock = self.out_dir / "RUNNING"
        if lock.exists():
            pid = int(lock.read_text().split()[0])
            try:
                os.kill(pid, 0)
                print(f"{self.out_dir} is being trained by pid {pid}; nothing to do.")
                return
            except ProcessLookupError:
                pass
        if self.out_dir.exists():
            stamp = time.strftime("%Y%m%d_%H%M%S")
            self.out_dir.rename(self.out_dir.with_name(self.out_dir.name + f".incomplete_{stamp}"))
        self.out_dir.mkdir(parents=True)
        lock.write_text(f"{os.getpid()} {time.strftime('%F %T')} GPU {os.environ.get('CUDA_VISIBLE_DEVICES')}\n")
        global VLLM_SEED
        VLLM_SEED = self.seed
        random.seed(self.seed)
        torch.manual_seed(self.seed)

        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
        tokenizer.padding_side = "left"
        self.tokenizer = tokenizer

        train_rows = read_jsonl(Path(args.data_dir) / "train_data.jsonl")
        self.test_rows = read_jsonl(Path(args.data_dir) / "test_data.jsonl")
        if args.limit_train > 0:
            train_rows = train_rows[: args.limit_train]
        if args.limit_test > 0:
            self.test_rows = self.test_rows[: args.limit_test]
        dataset = Dataset.from_list([
            {"prompt": render_prompt(row["explain"], tokenizer), "reference": canonicalize(row["smile"])}
            for row in train_rows
        ])

        # wandb: one group per condition, read by wandb.init inside the HF WandbCallback.
        os.environ.setdefault("WANDB_RUN_GROUP", f"grpo-4b-{args.variant}{args.tag}")
        os.environ.setdefault("WANDB_JOB_TYPE", "smoke" if args.tag else "grpo")
        os.environ.setdefault("WANDB_TAGS", f"grpo,qwen3.5-4b,{args.variant},run{args.run_id:03d}")

        model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.bfloat16, device_map={"": 0}, trust_remote_code=True)
        if self.sft_adapter is None:
            # No SFT: a fresh LoRA with the SFT configuration (train_sft.py: r=16, alpha=2r, dropout 0.05,
            # all linear layers of the text model), initialised from the run seed. B = 0, so the
            # starting policy is the base model.
            torch.manual_seed(self.seed)
            model = get_peft_model(model, LoraConfig(
                r=16, lora_alpha=32, lora_dropout=0.05, bias="none", target_modules="all-linear",
                exclude_modules=r".*(vision_tower|vision_encoder|multi_modal_projector|multimodal_projector|audio_tower).*",
                task_type="CAUSAL_LM"))
        else:
            model = PeftModel.from_pretrained(model, str(self.sft_adapter), is_trainable=True)
        model.print_trainable_parameters()

        completions_per_step = args.prompts_per_step * args.num_generations
        grad_accum = completions_per_step // args.micro_batch
        steps_per_epoch = math.ceil(len(dataset) / args.prompts_per_step)
        self.steps_per_epoch = steps_per_epoch
        total_steps = args.max_steps if args.max_steps > 0 else math.ceil(steps_per_epoch * args.epochs)
        warmup_steps = math.ceil(args.warmup_ratio * total_steps)
        config = GRPOConfig(
            output_dir=str(self.out_dir / "checkpoints"),
            run_name=f"grpo-4b-{args.variant}-run{args.run_id:03d}{args.tag}",
            seed=self.seed,
            data_seed=self.seed,                   # as SFT: seed = data_seed = 1000 + run id
            learning_rate=args.lr,
            # Optimisation settings shared with SFT are copied from the SFT runs
            # (their training_args.bin): cosine schedule, 3% warmup, fused AdamW,
            # no weight decay, gradient clipping at 0.3, LoRA dropout 0.05 kept on.
            lr_scheduler_type="cosine",
            warmup_steps=warmup_steps,
            optim="adamw_torch_fused",
            weight_decay=0.0,
            max_grad_norm=0.3,
            num_train_epochs=args.epochs,
            max_steps=args.max_steps,
            per_device_train_batch_size=args.micro_batch,
            gradient_accumulation_steps=grad_accum,
            steps_per_generation=grad_accum,       # one rollout batch per optimizer step (on-policy)
            num_generations=args.num_generations,
            max_completion_length=args.max_new_tokens,
            temperature=1.0,
            top_p=1.0,
            beta=0.0,
            loss_type="dapo",
            scale_rewards="group",
            disable_dropout=False,
            reward_weights=[1.0, 0.25, 0.0, 0.0],   # EM, format, (validity, answered: logged only)
            use_vllm=True,
            vllm_mode="colocate",
            vllm_gpu_memory_utilization=args.vllm_gpu_memory,
            vllm_max_model_length=1280,
            gradient_checkpointing=True,
            bf16=True,
            logging_steps=1,
            save_strategy="epoch",
            save_only_model=True,
            report_to=[] if args.no_wandb else ["wandb"],
        )
        self.trainer = GRPOTrainer(
            model=model,
            reward_funcs=[exact_match_reward, format_reward, valid_reward, answered_reward],
            args=config,
            train_dataset=dataset,
            processing_class=tokenizer,
            callbacks=[PeriodicEvaluation(self, args.eval_every)],
        )
        run_config = {
            "variant": args.variant, "run_id": args.run_id, "seed": self.seed, "data_seed": self.seed,
            "vllm_seed": VLLM_SEED, "eval_seed": args.eval_seed,
            "base_model": BASE_MODEL, "sft_adapter": str(self.sft_adapter) if self.sft_adapter else None,
            "lora_target_modules": sorted(model.peft_config["default"].target_modules), "train_records": len(dataset),
            "test_records": len(self.test_rows), "completions_per_step": completions_per_step,
            "grad_accum": grad_accum, "steps_per_epoch": steps_per_epoch, "total_steps": total_steps,
            "warmup_steps": warmup_steps, **{k: v for k, v in vars(args).items()},
            "grpo_config": {k: str(v) for k, v in config.to_dict().items()},
        }
        (self.out_dir / "run_config.json").write_text(json.dumps(run_config, indent=1))

        started = time.time()
        if not args.skip_initial_eval:
            # Stage 0: the base model (adapter not applied) = before SFT. Stage 1: the SFT model.
            self.initial_metrics.append(self.evaluate("before_sft", 0, base_only=True))
            if self.has_sft:
                self.initial_metrics.append(self.evaluate("after_sft", 1))
        self.trainer.train()
        self.trainer.save_model(str(self.out_dir / "adapter_final"))
        evals = [json.loads(line) for line in (self.out_dir / "eval_metrics.jsonl").read_text().splitlines() if line.strip()]
        final = dict(evals[-1])
        final["wall_seconds"] = time.time() - started
        (self.out_dir / "train_log_history.json").write_text(json.dumps(self.trainer.state.log_history))
        (self.out_dir / "final_metrics.json").write_text(json.dumps(final, indent=1))
        lock.unlink(missing_ok=True)
        if self.wandb_enabled:
            self.upload_to_wandb(evals, run_config)

    def upload_to_wandb(self, evals, run_config):
        """Summary (SFT vs final GRPO), a table with one row per evaluation, and the result files."""
        import wandb

        if wandb.run is None:
            return
        wandb.config.update({"grpo": run_config}, allow_val_change=True)
        columns = ["stage", "tag", "strict_greedy_em", "strict_greedy_valid", "strict_greedy_similarity", "strict_sampled_pass1_em",
                   "strict_pass_at_1", "strict_pass_at_2", "strict_pass_at_4", "strict_pass_at_8", "strict_pass_at_16",
                   "strict_greedy_answered", "greedy_truncated", "greedy_mean_tokens"]
        wandb.log({"test/evaluations": wandb.Table(columns=columns, data=[[e.get(c) for c in columns] for e in evals])})
        by_tag = {e["tag"]: e for e in evals}
        final_tag = evals[-1]["tag"]
        for metric in columns[2:]:
            for tag, e in by_tag.items():
                wandb.run.summary[f"{tag}/{metric}"] = e.get(metric)
            for start in ("before_sft", "after_sft"):
                if start in by_tag:
                    wandb.run.summary[f"delta_grpo_minus_{start.replace('_sft', '')}_sft/{metric}"] = by_tag[final_tag][metric] - by_tag[start][metric]
        artifact = wandb.Artifact(f"grpo-4b-{self.args.variant}-run{self.args.run_id:03d}{self.args.tag}-results", type="results")
        for name in ["run_config.json", "eval_metrics.jsonl", "final_metrics.json", *sorted(p.name for p in self.out_dir.glob("test_predictions_*.json"))]:
            artifact.add_file(str(self.out_dir / name))
        wandb.log_artifact(artifact)
        wandb.finish()


if __name__ == "__main__":
    Run(parse_args()).main()
