from __future__ import annotations

import json
import re
from typing import List

from rag_agent.core.dto import Citation, Claim
from rag_agent.core.ports import VerifierPort
from rag_agent.core.errors import VerifierError
from rag_agent.adapters import register_verifier, LLM_REG
from rag_agent.settings import settings

from rag_agent.core.logging_setup import get_logger
logger = get_logger(__name__)

def _strip_code_fences(s: str) -> str:
    # Remove ```json ... ``` or ``` ... ``` fences if present
    s = s.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```$", "", s)
    return s.strip()


def _extract_json_array(s: str) -> str | None:
    """Return the substring between the first '[' and the last ']'. """
    try:
        start = s.index("[")
        end = s.rindex("]") + 1
        return s[start:end]
    except ValueError:
        return None


@register_verifier("llm-check")
class LLVerify(VerifierPort):
    """
    Provider-agnostic verifier:
      - extract_claims: asks the LLM to return JSON claims pointing to allowed citation_ids.
      - score: returns a single float in [0,1] for (claim, evidence).
    """

    def __init__(self):
        llm_cls = LLM_REG[settings.model_provider]
        self.llm = llm_cls()
        # Soft cap to keep extraction bounded; can move to settings later.
        self.max_claims = getattr(settings, "verifier_max_claims", 5)
        logger.info("verifier:init", extra={"provider": settings.model_provider,
                                            "max_claims": self.max_claims})

    # ───────────────────────── extract_claims ─────────────────────────
    def extract_claims(self, answer_text: str, citations: List[Citation]) -> List[Claim]:
        allowed_ids = [c.chunk_id for c in citations]

        schema_hint = (
            "Return ONLY a JSON array of objects with keys: id, text, citation_ids.\n"
            "Each citation_ids entry MUST be chosen from this allowed list "
            f"(no others): {allowed_ids}\n"
            f"At most {self.max_claims} claims. Each claim text ≤ 160 characters; remove filler.\n"
            "No prose, no markdown, JSON only."
        )
        prompt = (
            "Extract the key factual claims from the assistant answer."
            "Remain spartan with how you reply, only claims that are clearly not speculative should be included."
            "Group related words into concise claims that a human could verify.\n\n"
            f"{schema_hint}\n\n"
            f"Assistant answer:\n{answer_text}\n"
        )

        raw = self.llm.complete(prompt, max_tokens=512)

        # Parse with a couple of gentle repairs to handle phi-3 style outputs
        cleaned = _strip_code_fences(raw)
        json_text = _extract_json_array(cleaned) or cleaned

        try:
            data = json.loads(json_text)
        except Exception:
            # One retry: ask the model to fix into strict JSON (still provider-agnostic).
            fix_prompt = (
                "Fix the following into VALID JSON that strictly matches this schema:\n"
                "[{id: string, text: string, citation_ids: string[]}]\n"
                "Return ONLY the JSON array, no extra text.\n\n"
                f"INPUT:\n{raw}"
            )
            fixed = self.llm.complete(fix_prompt, max_tokens=400)
            fixed = _strip_code_fences(fixed)
            json_text = _extract_json_array(fixed) or fixed
            try:
                data = json.loads(json_text)
            except Exception as e:
                # Last-resort fallback: one claim (the whole answer) with all citations.
                return [
                    Claim(id="clm_1", text=answer_text, citation_ids=[])
                ]

        # Validate & normalize
        claims: List[Claim] = []
        seen_ids = set()
        for i, item in enumerate(data[: self.max_claims], start=1):
            try:
                text = str(item.get("text", "")).strip()
                # Ensure citation_ids are valid & unique per claim
                raw_ids = item.get("citation_ids", [])
                if not isinstance(raw_ids, list):
                    raw_ids = []
                cids = [cid for cid in raw_ids if cid in allowed_ids]
                cids = list(dict.fromkeys(cids))  # dedupe, preserve order

                if not text:
                    continue

                cid = item.get("id")
                if not isinstance(cid, str) or not cid.strip():
                    cid = f"clm_{i}"
                if cid in seen_ids:
                    cid = f"clm_{i}_{len(seen_ids)}"
                seen_ids.add(cid)

                claims.append(Claim(id=cid, text=text, citation_ids=cids))
            except Exception:
                # Skip malformed entries quietly; we already have a fallback above
                continue

        if not claims:
            # Fallback if everything got filtered out
            claims = [Claim(id="clm_1", text=answer_text, citation_ids=[])]

        return claims

    # ───────────────────────────── score ─────────────────────────────
    def score(self, claim: str, evidence: str) -> float:
        """
        Return a float in [0,1] indicating how well `evidence` supports `claim`.
        Robust to chatty models: we parse <score>…</score>, then JSON, then first float.
        """
        prompt = (
            "You are a strict fact-checker.\n"
            "Given a CLAIM and EVIDENCE, output a single support score between 0 and 1.\n"
            "Rules:\n"
            "- 1.0 means fully supported by the evidence.\n"
            "- 0.0 means contradicted or unsupported.\n"
            "- Output ONLY one number wrapped in <score>…</score>.\n\n"
            f"CLAIM: {claim}\n"
            f"EVIDENCE: {evidence}\n\n"
            "Answer format:\n"
            "<score>0.00</score>"
        )

        gen_kwargs = dict(
            max_tokens=getattr(settings, "verifier_max_tokens_score", 16),
            temperature=getattr(settings, "verifier_temperature", 0.0),
            top_p=getattr(settings, "verifier_top_p", 1.0),
        )

        try:
            # Try with generation controls first
            try:
                resp = self.llm.complete(prompt, **gen_kwargs)
            except TypeError:
                # Some adapters may not accept these kwargs; retry minimal
                resp = self.llm.complete(prompt)

            text = str(resp).strip()

            # 1) Prefer <score>…</score>
            m = re.search(r"<\s*score\s*>\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*<\s*/\s*score\s*>", text)
            if m:
                val = float(m.group(1))
                return max(0.0, min(1.0, val))

            # 2) Try JSON like {"score": 0.87}
            #    (helpful if a model ignores tags)
            mj = re.search(r"\{[^}]*\}", text, flags=re.DOTALL)
            if mj:
                try:
                    obj = json.loads(mj.group(0))
                    if isinstance(obj, dict) and "score" in obj:
                        val = float(obj["score"])
                        return max(0.0, min(1.0, val))
                except Exception:
                    pass

            # 3) Fall back to the first float-like number
            m2 = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
            if not m2:
                # Last resort: unknown format
                raise VerifierError(f"Non-numeric verifier output: {text!r}")
            val = float(m2.group(0))
            return max(0.0, min(1.0, val))

        except Exception as e:
            # We bubble a real error so the caller can decide (e.g., set None or 0.0)
            raise VerifierError(f"Could not score claim: {e}") from e
