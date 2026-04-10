from __future__ import annotations

import uuid
from typing import Any, Literal, cast

from rapidfuzz import fuzz

from .documents.generator import DocumentGenerator
from .models import (
    Action,
    ActionResult,
    ActionType,
    AnnotatePayload,
    ComparePayload,
    DocumentView,
    ExtractPayload,
    Observation,
    QueryPayload,
    Reward,
    SubmitPayload,
    SummarisePayload,
)
from .reward.reward_engine import RewardContext, RewardEngine
from .state import Annotation, ConsistencyFlag, EnvState, QueryRecord, StateManager
from .tasks.registry import TaskRegistry, TaskSpec

SeverityLevel = Literal["INFO", "WARNING", "ERROR"]
TaskName = Literal["ICF_EXTRACTION", "PROTOCOL_ICF_CONSISTENCY", "SAP_AE_RECONCILIATION"]


class PharmaTrialsEnv:
    def __init__(self) -> None:
        self._registry = TaskRegistry.default()
        self._state_manager = StateManager()
        self._reward_engine = RewardEngine()
        self._current_task_spec: TaskSpec | None = None

    def reset(self, task_id: str | None = None, seed: int | None = None) -> Observation:
        task_spec = self._registry.get(task_id)
        resolved_seed = seed if seed is not None else int(uuid.uuid4().int % 100000)
        task_summaries = self._registry.summaries()

        generator = DocumentGenerator(seed=resolved_seed, task_spec=task_spec)
        documents, metadata = generator.generate()
        ground_truth = task_spec.grader.generate_ground_truth(documents, metadata)

        state = EnvState(
            task_id=task_spec.task_id,
            task_name=task_spec.task_name,
            seed=resolved_seed,
            step_number=0,
            max_steps=task_spec.max_steps,
            tasks=task_summaries,
            documents=documents,
            done=False,
            episode_id=str(uuid.uuid4()),
            metadata={
                "seed": resolved_seed,
                "protocol_number": metadata.get("protocol_number"),
                "available_tasks": task_summaries,
                "available_task_count": len(task_summaries),
            },
            ground_truth=ground_truth,
            required_items=list(task_spec.required_output_items),
        )

        self._state_manager.replace(state)
        self._current_task_spec = task_spec
        return self._build_observation(last_action_result=None)

    def step(self, action: Action) -> tuple[Observation, Reward, bool, dict[str, Any]]:
        if self._current_task_spec is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        state = self._state_manager.state
        task_spec = self._current_task_spec

        if state.done:
            obs = self._build_observation(
                ActionResult(
                    action_type=action.action_type.value,
                    success=False,
                    output={},
                    error_message="Episode already finished.",
                )
            )
            reward = self._reward_engine.compute(state, task_spec, RewardContext(step_penalty=0.05))
            return obs, reward, True, self._info(reward)

        state.step_number += 1

        if action.action_type not in task_spec.allowed_actions:
            return self._handle_invalid_action(
                action,
                f"Action {action.action_type.value} is not allowed for task {task_spec.task_id}.",
            )

        handler_map = {
            ActionType.EXTRACT: self._handle_extract,
            ActionType.COMPARE: self._handle_compare,
            ActionType.SUMMARISE: self._handle_summarise,
            ActionType.ANNOTATE: self._handle_annotate,
            ActionType.QUERY: self._handle_query,
            ActionType.SUBMIT: self._handle_submit,
        }

        try:
            action_result, ctx = handler_map[action.action_type](action)
            state.invalid_action_streak = (
                0 if action_result.success else state.invalid_action_streak + 1
            )
        except (
            KeyError,
            TypeError,
            ValueError,
            RuntimeError,
        ) as exc:  # pragma: no cover
            return self._handle_invalid_action(action, f"Unhandled action error: {exc}")

        repeated_penalty = self._repeat_action_penalty(action)
        ctx.step_penalty += repeated_penalty

        if state.invalid_action_streak >= 3:
            state.done = True
            completeness = self._reward_engine.completeness(state)
            forced_score = completeness * 0.5
            reward = Reward(
                total=max(0.0, min(1.0, forced_score)),
                accuracy=0.0,
                completeness=completeness,
                regulatory_alignment=0.0,
                efficiency=0.0,
                step_penalty=0.15,
                weights=task_spec.reward_weights,
            )
            obs = self._build_observation(action_result)
            return obs, reward, True, self._info(reward)

        if state.step_number >= state.max_steps and action.action_type != ActionType.SUBMIT:
            state.done = True

        reward = self._reward_engine.compute(state, task_spec, ctx)
        obs = self._build_observation(action_result)
        return obs, reward, state.done, self._info(reward)

    def state(self) -> EnvState:
        return self._state_manager.state.model_copy(deep=True)

    def task_summaries(self) -> list[dict[str, object]]:
        return self._registry.summaries()

    def _handle_extract(self, action: Action) -> tuple[ActionResult, RewardContext]:
        payload = action.payload
        if not isinstance(payload, ExtractPayload):
            raise ValueError("Invalid EXTRACT payload")

        state = self._state_manager.state
        doc = state.documents.get(payload.doc_id)
        if doc is None:
            return (
                ActionResult(
                    action_type="EXTRACT",
                    success=False,
                    output={},
                    error_message="Unknown doc_id.",
                ),
                RewardContext(step_penalty=0.05),
            )

        extracted: dict[str, Any] = {}
        for field in payload.fields:
            val = self._extract_field_from_document(doc, field)
            if val is not None:
                extracted[field] = val
                state.addressed_items.add(field)

        action_id = f"extract_{state.step_number}"
        state.extraction_outputs[action_id] = extracted

        accuracy = self._reward_engine.extraction_accuracy(extracted, state.ground_truth)
        reg = self._reward_engine.regulatory_alignment_from_payload(extracted)
        result = ActionResult(
            action_type="EXTRACT",
            success=True,
            output={"extracted_fields": extracted},
            error_message=None,
        )
        return result, RewardContext(accuracy=accuracy, regulatory_alignment=reg)

    def _handle_compare(self, action: Action) -> tuple[ActionResult, RewardContext]:
        payload = action.payload
        if not isinstance(payload, ComparePayload):
            raise ValueError("Invalid COMPARE payload")

        state = self._state_manager.state
        docs = self._get_compare_documents(payload)
        if docs is None:
            return (
                ActionResult(
                    action_type="COMPARE",
                    success=False,
                    output={},
                    error_message="Unknown compare document.",
                ),
                RewardContext(step_penalty=0.05),
            )
        doc_a, doc_b = docs
        inconsistencies = self._collect_inconsistencies(payload, doc_a, doc_b)
        f1 = self._compare_f1(inconsistencies, state.ground_truth)

        reg = self._reward_engine.regulatory_alignment_from_payload(
            {"inconsistencies": inconsistencies}
        )
        result = ActionResult(
            action_type="COMPARE",
            success=True,
            output={"inconsistencies": inconsistencies},
            error_message=None,
        )
        return result, RewardContext(accuracy=f1, regulatory_alignment=reg)

    def _get_compare_documents(
        self, payload: ComparePayload
    ) -> tuple[DocumentView, DocumentView] | None:
        state = self._state_manager.state
        doc_a = state.documents.get(payload.doc_id_a)
        doc_b = state.documents.get(payload.doc_id_b)
        if doc_a is None or doc_b is None:
            return None
        return doc_a, doc_b

    def _collect_inconsistencies(
        self, payload: ComparePayload, doc_a: DocumentView, doc_b: DocumentView
    ) -> list[dict[str, Any]]:
        state = self._state_manager.state
        truth_items = state.ground_truth.get("inconsistencies", [])
        inconsistencies: list[dict[str, Any]] = []
        for field in payload.comparison_fields:
            values = self._extract_compare_values(doc_a, doc_b, field)
            if values is None:
                continue
            a_val, b_val = values
            truth_match = next((row for row in truth_items if str(row.get("field")) == field), None)
            inconsistency = self._build_inconsistency_record(
                field=field,
                values=(a_val, b_val),
                truth_match=truth_match,
                payload=payload,
            )
            inconsistencies.append(inconsistency)
            self._record_consistency_flag((doc_a, doc_b), field, (a_val, b_val), inconsistency)
            state.addressed_items.add(field)
        return inconsistencies

    def _extract_compare_values(
        self, doc_a: DocumentView, doc_b: DocumentView, field: str
    ) -> tuple[Any, Any] | None:
        a_val = self._extract_field_from_document(doc_a, field)
        b_val = self._extract_field_from_document(doc_b, field)
        if a_val is None or b_val is None:
            return None
        if str(a_val) == str(b_val):
            return None
        return a_val, b_val

    def _build_inconsistency_record(
        self,
        field: str,
        values: tuple[Any, Any],
        truth_match: dict[str, Any] | None,
        payload: ComparePayload,
    ) -> dict[str, str]:
        a_val, b_val = values
        default_severity = self._default_compare_severity(field)
        row = truth_match or {}
        severity = str(row.get("severity", default_severity)).upper()
        return {
            "field": field,
            "doc_a_value": str(a_val),
            "doc_b_value": str(b_val),
            "section_in_protocol": str(
                row.get("section_in_protocol") or payload.section_hint_a or "Study Design"
            ),
            "section_in_icf": str(
                row.get("section_in_icf") or payload.section_hint_b or "Study Procedures"
            ),
            "severity": severity,
            "regulatory_basis": str(
                row.get("regulatory_basis") or "ICH E6(R3) consistency requirement"
            ),
        }

    def _default_compare_severity(self, field: str) -> str:
        if field in {
            "num_visits",
            "total_duration_weeks",
            "dose_mg",
            "route_of_administration",
        }:
            return "ERROR"
        return "WARNING"

    def _record_consistency_flag(
        self,
        doc_pair: tuple[DocumentView, DocumentView],
        field: str,
        values: tuple[Any, Any],
        inconsistency: dict[str, str],
    ) -> None:
        state = self._state_manager.state
        doc_a, doc_b = doc_pair
        a_val, b_val = values
        severity_value = cast(SeverityLevel, inconsistency["severity"])
        state.consistency_flags.append(
            ConsistencyFlag(
                doc_id_a=doc_a.doc_id,
                doc_id_b=doc_b.doc_id,
                field=field,
                value_a=str(a_val),
                value_b=str(b_val),
                is_inconsistent=True,
                severity=severity_value,
            )
        )

    def _compare_f1(
        self, inconsistencies: list[dict[str, Any]], ground_truth: dict[str, Any]
    ) -> float:
        truth_set = {
            (i["field"], i["severity"].upper()) for i in ground_truth.get("inconsistencies", [])
        }
        pred_set = {(i["field"], i["severity"].upper()) for i in inconsistencies}
        if not pred_set:
            return 0.0
        intersection = len(pred_set & truth_set)
        precision = intersection / len(pred_set)
        recall = intersection / max(1, len(truth_set))
        if (precision + recall) == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    def _handle_summarise(self, action: Action) -> tuple[ActionResult, RewardContext]:
        payload = action.payload
        if not isinstance(payload, SummarisePayload):
            raise ValueError("Invalid SUMMARISE payload")

        state = self._state_manager.state
        doc = state.documents.get(payload.doc_id)
        if doc is None:
            return (
                ActionResult(
                    action_type="SUMMARISE",
                    success=False,
                    output={},
                    error_message="Unknown doc_id.",
                ),
                RewardContext(step_penalty=0.05),
            )

        summary = " ".join(doc.content.split()[: payload.max_words])
        result = ActionResult(
            action_type="SUMMARISE",
            success=True,
            output={"summary": summary, "focus_areas": payload.focus_areas},
            error_message=None,
        )
        reg = self._reward_engine.regulatory_alignment_from_payload({"summary": summary})
        return result, RewardContext(accuracy=0.2, regulatory_alignment=reg)

    def _handle_annotate(self, action: Action) -> tuple[ActionResult, RewardContext]:
        payload = action.payload
        if not isinstance(payload, AnnotatePayload):
            raise ValueError("Invalid ANNOTATE payload")

        state = self._state_manager.state
        doc = state.documents.get(payload.doc_id)
        if doc is None:
            return (
                ActionResult(
                    action_type="ANNOTATE",
                    success=False,
                    output={},
                    error_message="Unknown doc_id.",
                ),
                RewardContext(step_penalty=0.05),
            )

        state.annotations.append(
            Annotation(
                doc_id=payload.doc_id,
                section=payload.section,
                label=payload.label,
                note=payload.note,
                severity=payload.severity,
                step_number=state.step_number,
            )
        )
        reg = self._reward_engine.regulatory_alignment_from_payload(payload.model_dump())
        return (
            ActionResult(
                action_type="ANNOTATE",
                success=True,
                output={"annotation_saved": True},
                error_message=None,
            ),
            RewardContext(accuracy=0.1, regulatory_alignment=reg),
        )

    def _handle_query(self, action: Action) -> tuple[ActionResult, RewardContext]:
        payload = action.payload
        if not isinstance(payload, QueryPayload):
            raise ValueError("Invalid QUERY payload")

        state = self._state_manager.state
        doc = state.documents.get(payload.doc_id)
        if doc is None:
            return (
                ActionResult(
                    action_type="QUERY",
                    success=False,
                    output={},
                    error_message="Unknown doc_id.",
                ),
                RewardContext(step_penalty=0.05),
            )

        answer = self._query_document(doc, payload.question)
        state.query_history.append(
            QueryRecord(
                doc_id=payload.doc_id,
                question=payload.question,
                answer=answer,
                step_number=state.step_number,
            )
        )
        reg = self._reward_engine.regulatory_alignment_from_payload(
            {"question": payload.question, "answer": answer}
        )
        return (
            ActionResult(
                action_type="QUERY",
                success=True,
                output={"answer": answer},
                error_message=None,
            ),
            RewardContext(accuracy=0.05, regulatory_alignment=reg),
        )

    def _handle_submit(self, action: Action) -> tuple[ActionResult, RewardContext]:
        payload = action.payload
        if not isinstance(payload, SubmitPayload):
            raise ValueError("Invalid SUBMIT payload")

        state = self._state_manager.state
        spec = self._require_current_task_spec()

        query_penalty = 0.02 * len([q for q in state.query_history if not q.consumed])

        score_detail = spec.grader.score(payload.model_dump(), state.ground_truth)
        score = float(score_detail.get("score", 0.0))
        reg = self._reward_engine.regulatory_alignment_from_payload(payload.answer)

        state.done = True

        return (
            ActionResult(
                action_type="SUBMIT",
                success=True,
                output={"grader": score_detail, "confidence": payload.confidence},
                error_message=None,
            ),
            RewardContext(
                accuracy=score,
                regulatory_alignment=reg,
                step_penalty=query_penalty,
                is_submit=True,
            ),
        )

    def _handle_invalid_action(
        self, action: Action, reason: str
    ) -> tuple[Observation, Reward, bool, dict[str, Any]]:
        state = self._state_manager.state
        spec = self._require_current_task_spec()
        state.invalid_action_streak += 1
        ctx = RewardContext(step_penalty=0.05)
        action_result = ActionResult(
            action_type=action.action_type.value,
            success=False,
            output={},
            error_message=reason,
        )

        if state.invalid_action_streak >= 3:
            state.done = True
            completeness = self._reward_engine.completeness(state)
            forced_score = completeness * 0.5
            reward = Reward(
                total=max(0.0, min(1.0, forced_score)),
                accuracy=0.0,
                completeness=completeness,
                regulatory_alignment=0.0,
                efficiency=0.0,
                step_penalty=0.15,
                weights=spec.reward_weights,
            )
        else:
            reward = self._reward_engine.compute(state, spec, ctx)

        obs = self._build_observation(action_result)
        return obs, reward, state.done, self._info(reward)

    def _extract_field_from_document(self, doc: DocumentView, field: str) -> Any | None:
        aliases = {
            "study_title": "Title",
            "sponsor_name": "Sponsor",
            "protocol_number": "Protocol Number",
            "compound_name": "Compound Name",
            "dose_mg": "Dose (mg)",
            "route_of_administration": "Route of Administration",
            "risk_list": "Risk List",
            "primary_endpoint": "Primary Endpoint",
            "dosing_frequency": "Dosing Frequency",
            "num_visits": "Number of Visits",
            "total_duration_weeks": "Total Duration (weeks)",
            "primary_indication": "Primary Indication",
            "irb_name": "IRB Name",
            "irb_phone": "IRB Phone",
            "compensation_amount_usd": "Compensation Amount (USD)",
            "analysis_population": "Analysis Population",
            "meddra_coding_level": "MedDRA Coding Level",
            "ctcae_version": "CTCAE Version",
            "sae_definition": "SAE Definition",
        }
        label = aliases.get(field, field)
        for line in doc.content.splitlines():
            if line.lower().startswith(label.lower() + ":"):
                value = line.split(":", 1)[1].strip()
                if field in {
                    "dose_mg",
                    "num_visits",
                    "total_duration_weeks",
                    "compensation_amount_usd",
                    "severity_grade_reported",
                }:
                    digits = "".join(ch for ch in value if ch.isdigit() or ch == ".")
                    if digits:
                        return float(digits) if "." in digits else int(digits)
                return value
        return None

    def _query_document(self, doc: DocumentView, question: str) -> str:
        q = question.lower()
        best_line = "No direct answer found."
        best_score = 0.0
        for line in doc.content.splitlines():
            score = fuzz.partial_ratio(q, line.lower())
            if score > best_score:
                best_score = score
                best_line = line.strip()
        return best_line

    def _repeat_action_penalty(self, action: Action) -> float:
        state = self._state_manager.state
        doc_key = self._action_doc_key(action)
        fingerprint = f"{action.action_type.value}|{doc_key}"
        history = state.action_count_by_fingerprint.setdefault(fingerprint, [])
        penalty = 0.0
        if doc_key is not None and history and (state.step_number - history[-1]) <= 3:
            penalty = 0.03
        history.append(state.step_number)

        if action.action_type in {ActionType.EXTRACT, ActionType.COMPARE}:
            for query in state.query_history:
                if not query.consumed:
                    query.consumed = True
        return penalty

    def _action_doc_key(self, action: Action) -> str | None:
        payload = action.payload
        if hasattr(payload, "doc_id"):
            return str(getattr(payload, "doc_id"))
        if hasattr(payload, "doc_id_a") and hasattr(payload, "doc_id_b"):
            left = str(getattr(payload, "doc_id_a"))
            right = str(getattr(payload, "doc_id_b"))
            return "|".join(sorted([left, right]))
        return None

    def _build_observation(self, last_action_result: ActionResult | None) -> Observation:
        state = self._state_manager.state
        task_spec = self._require_current_task_spec()
        running = max(
            0.0,
            min(
                1.0,
                state.reward_components.episode_reward_sum / max(1, state.max_steps),
            ),
        )

        protocol_number = state.metadata.get("protocol_number", "unknown")
        instruction = task_spec.instruction_template.format(
            max_steps=state.max_steps,
            protocol_number=protocol_number,
        )

        return Observation(
            task_id=state.task_id,
            task_name=cast(TaskName, state.task_name),
            step_number=state.step_number,
            max_steps=state.max_steps,
            documents=state.documents,
            last_action_result=last_action_result,
            task_instruction=instruction,
            partial_score=running,
            done=state.done,
            metadata={
                "episode_id": state.episode_id,
                "seed": state.seed,
                "available_tasks": state.tasks,
                "available_task_count": len(state.tasks),
            },
        )

    def _info(self, reward: Reward) -> dict[str, Any]:
        state = self._state_manager.state
        return {
            "step_number": state.step_number,
            "done": state.done,
            "reward_total": reward.total,
            "invalid_action_streak": state.invalid_action_streak,
            "addressed_items": sorted(state.addressed_items),
        }

    def _require_current_task_spec(self) -> TaskSpec:
        spec = self._current_task_spec
        if spec is None:
            raise RuntimeError("Task spec not initialized. Call reset() first.")
        return spec
