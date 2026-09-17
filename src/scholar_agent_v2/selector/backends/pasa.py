from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

from ...candidate_pool.models import CandidatePaper
from ...query_understanding.schema import QueryAnalysisResult


PASA_SELECTOR_PROMPT = """You are an elite researcher in the field of AI, conducting research on {user_query}. Evaluate whether the following paper fully satisfies the detailed requirements of the user query and provide your reasoning. Ensure that your decision and reasoning are consistent.

Searched Paper:
Title: {title}
Abstract: {abstract}

User Query: {user_query}

Output format: Decision: True/False
Reason:... 
Decision:"""


@dataclass(frozen=True)
class PasaPriorConfig:
    model_path: str
    batch_size: int = 8
    max_length: int = 1536
    score_mode: Literal["full_vocab", "binary_normalized"] = "full_vocab"
    trust_remote_code: bool = False
    use_chat_template: bool = False


class PasaPriorScorer:
    """Optional GPU paper-level relevance prior based on PaSa's one-token True score idea."""

    def __init__(self, config: PasaPriorConfig) -> None:
        self.config = config
        self._model = None
        self._tokenizer = None
        self._torch = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "PaSa prior requires optional GPU dependencies: torch, transformers, accelerate"
            ) from exc

        model_path = str(Path(self.config.model_path).expanduser())
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            padding_side="left",
            trust_remote_code=self.config.trust_remote_code,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=self.config.trust_remote_code,
        )
        model.eval()

        self._torch = torch
        self._tokenizer = tokenizer
        self._model = model

    def _single_token_ids(self, variants: Sequence[str]) -> list[int]:
        assert self._tokenizer is not None
        ids: list[int] = []
        for variant in variants:
            encoded = self._tokenizer.encode(variant, add_special_tokens=False)
            if len(encoded) == 1 and encoded[0] not in ids:
                ids.append(encoded[0])
        return ids

    def _render_prompt(self, analysis: QueryAnalysisResult, candidate: CandidatePaper) -> str:
        prompt = PASA_SELECTOR_PROMPT.format(
            user_query=analysis.original_query,
            title=candidate.title or "",
            abstract=candidate.abstract or "",
        ).strip()
        if self.config.use_chat_template:
            assert self._tokenizer is not None
            return self._tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        return prompt

    def score(
        self,
        analysis: QueryAnalysisResult,
        candidates: Sequence[CandidatePaper],
    ) -> tuple[dict[str, float], float]:
        if not candidates:
            return {}, 0.0
        self._ensure_loaded()
        assert self._model is not None and self._tokenizer is not None and self._torch is not None

        started = time.perf_counter()
        prompts = [self._render_prompt(analysis, candidate) for candidate in candidates]
        true_ids = self._single_token_ids(["True", " True", "true", " true"])
        false_ids = self._single_token_ids(["False", " False", "false", " false"])
        if not true_ids:
            raise RuntimeError("Tokenizer has no single-token representation for True; PaSa 1-token scoring is incompatible.")

        scores: list[float] = []
        input_device = self._model.get_input_embeddings().weight.device

        for start in range(0, len(prompts), self.config.batch_size):
            batch_prompts = prompts[start : start + self.config.batch_size]
            encoded = self._tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config.max_length,
            )
            encoded = {key: value.to(input_device) for key, value in encoded.items()}

            with self._torch.inference_mode():
                outputs = self._model.generate(
                    **encoded,
                    max_new_tokens=1,
                    output_scores=True,
                    return_dict_in_generate=True,
                    do_sample=False,
                    pad_token_id=self._tokenizer.pad_token_id,
                )
                probabilities = outputs.scores[0].softmax(dim=-1)
                true_prob = probabilities[:, true_ids].sum(dim=-1)
                if self.config.score_mode == "binary_normalized" and false_ids:
                    false_prob = probabilities[:, false_ids].sum(dim=-1)
                    denom = (true_prob + false_prob).clamp_min(1e-12)
                    batch_scores = true_prob / denom
                else:
                    batch_scores = true_prob
                scores.extend(batch_scores.detach().float().cpu().tolist())

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return {
            candidate.candidate_id: max(0.0, min(1.0, float(score)))
            for candidate, score in zip(candidates, scores)
        }, elapsed_ms


__all__ = ["PASA_SELECTOR_PROMPT", "PasaPriorConfig", "PasaPriorScorer"]
