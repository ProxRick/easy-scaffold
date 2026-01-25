# src/easy_scaffold/workflows/agents/cot_improvement_workflow.py
import logging
from typing import Any, Dict, List, Optional

from easy_scaffold.db.pydantic_models import RunLog
from easy_scaffold.workflows.base import AbstractWorkflow
from easy_scaffold.workflows.workflow_models import (
    CoTImprovementPayload,
    WorkItem,
)

logger = logging.getLogger(__name__)


class CoTImprovementWorkflow(AbstractWorkflow[CoTImprovementPayload]):
    """
    Workflow that improves Chain of Thought (CoT) by:
    1. Detecting and rewriting looping sections
    2. Normalizing steps to single behavior per step
    3. Adding a conclusive paragraph (only on full success)
    """

    def _format_steps_for_detection(self, steps: List[str]) -> str:
        """Format steps for loop detection prompt."""
        formatted = []
        for i, step in enumerate(steps):
            formatted.append(f"Step {i}: {step}")
        return "\n\n".join(formatted)

    def _format_labeled_steps_for_normalization(
        self, labeled_steps: List[Any]
    ) -> str:
        """Format labeled steps for normalization prompt."""
        formatted = []
        for step in labeled_steps:
            # Handle both dict and object access
            if isinstance(step, dict):
                step_index = step.get("step_index", 0)
                step_content = step.get("step_content", "")
                behaviors = step.get("behaviors", [])
            else:
                step_index = getattr(step, "step_index", 0)
                step_content = getattr(step, "step_content", "")
                behaviors = getattr(step, "behaviors", [])
            
            behaviors_str = ", ".join(behaviors) if behaviors else "None"
            formatted.append(
                f"Step number {step_index}\n{step_content},\nLabel: {behaviors_str}."
            )
        return "\n\n".join(formatted)

    async def _detect_and_rewrite_loops(
        self,
        steps: List[str],
        problem_statement: str,
        run_log: RunLog,
        document_id: str,
        workflow_name: str,
    ) -> Optional[List[str]]:
        """
        Detect looping behavior and rewrite if found.
        Returns improved steps or None if rewriting failed.
        """
        chunk_size = self._config.get("chunking", {}).get("chunk_size", 100)
        loop_detection_window = self._config.get("chunking", {}).get(
            "loop_detection_window", chunk_size
        )
        max_iterations = self._config.get("loop_detection", {}).get(
            "max_iterations", 5
        )
        verification_attempts = self._config.get("loop_detection", {}).get(
            "verification_attempts", 3
        )

        # Prepare steps for detection (use last window_size steps)
        if len(steps) > loop_detection_window:
            steps_for_detection = steps[-loop_detection_window:]
            detection_start_offset = len(steps) - loop_detection_window
        else:
            steps_for_detection = steps
            detection_start_offset = 0

        formatted_steps = self._format_steps_for_detection(steps_for_detection)

        # Detect loop
        detection_context = {
            "problem_statement": problem_statement,
            "cot_steps": formatted_steps,
        }

        detection_result = await self._execute_stage(
            "LoopDetection",
            detection_context,
            run_log,
            document_id,
            workflow_name,
        )

        detection_outputs = detection_result.get("outputs", {})
        has_loop = detection_outputs.get("loop_detection_has_loop", False)
        loop_start_index_relative = detection_outputs.get(
            "loop_detection_start_index"
        )

        if not has_loop or loop_start_index_relative is None:
            logger.info("No loop detected. Proceeding with original steps.")
            return steps

        # Calculate absolute loop start index
        loop_start_absolute = detection_start_offset + loop_start_index_relative
        cot_before = steps[:loop_start_absolute]
        cot_loop = steps[loop_start_absolute:]

        logger.info(
            f"Loop detected starting at step {loop_start_absolute}. "
            f"Attempting to rewrite {len(cot_loop)} steps."
        )

        # Iterative rewriting with verification (following imo_agent.py pattern)
        for attempt in range(max_iterations):
            rewriting_context = {
                "problem_statement": problem_statement,
                "cot_before_loop": "\n\n".join(cot_before),
                "cot_loop_section": "\n\n".join(cot_loop),
            }

            rewriting_result = await self._execute_stage(
                "LoopRewriting",
                rewriting_context,
                run_log,
                document_id,
                workflow_name,
            )

            rewritten_section = rewriting_result.get("outputs", {}).get(
                "rewritten_section", ""
            )

            if not rewritten_section:
                logger.warning(
                    f"LoopRewriting returned empty section on attempt {attempt + 1}"
                )
                continue

            # Attempt k consecutive verifications
            verification_results = []
            for k_attempt in range(verification_attempts):
                verification_context = {
                    "original_loop_section": "\n\n".join(cot_loop),
                    "rewritten_section": rewritten_section,
                }

                verification_result = await self._execute_stage(
                    "LoopFidelityVerification",
                    verification_context,
                    run_log,
                    document_id,
                    workflow_name,
                )

                verification_outputs = verification_result.get("outputs", {})
                is_faithful = verification_outputs.get("fidelity_is_faithful", False)
                verification_results.append(is_faithful)

                # If any verification fails, break and restart rewriting
                if not is_faithful:
                    logger.info(
                        f"Fidelity verification failed on attempt {k_attempt + 1}/{verification_attempts}. "
                        f"Restarting rewriting."
                    )
                    break

            # Check if we got k consecutive successes
            if (
                len(verification_results) == verification_attempts
                and all(verification_results)
            ):
                # Success! Use rewritten section
                rewritten_steps = rewritten_section.split("\n\n")
                # Filter out empty strings
                rewritten_steps = [s for s in rewritten_steps if s.strip()]
                improved_steps = cot_before + rewritten_steps
                logger.info(
                    f"Loop rewriting succeeded after {attempt + 1} attempts. "
                    f"Rewrote {len(cot_loop)} steps into {len(rewritten_steps)} steps."
                )
                return improved_steps

        # Failed after max_iterations
        logger.warning(
            f"Loop rewriting failed after {max_iterations} attempts. Skipping document."
        )
        return None

    async def _label_steps_chunked(
        self,
        steps: List[str],
        problem_statement: str,
        run_log: RunLog,
        document_id: str,
        workflow_name: str,
    ) -> List[Any]:
        """Label steps in chunks and merge results."""
        chunk_size = self._config.get("chunking", {}).get("chunk_size", 100)
        all_labeled_steps = []

        for chunk_start in range(0, len(steps), chunk_size):
            chunk_end = min(chunk_start + chunk_size, len(steps))
            chunk_steps = steps[chunk_start:chunk_end]

            formatted_chunk = self._format_steps_for_detection(chunk_steps)

            labeling_context = {
                "problem_statement": problem_statement,
                "cot_steps": formatted_chunk,
            }

            labeling_result = await self._execute_stage(
                "MultiBehaviorLabeling",
                labeling_context,
                run_log,
                document_id,
                workflow_name,
            )

            labeled_chunk = labeling_result.get("outputs", {}).get(
                "labeled_steps", []
            )

            # Adjust step_index to account for chunk offset
            adjusted_chunk = []
            for step in labeled_chunk:
                if isinstance(step, dict):
                    step_copy = step.copy()
                    step_copy["step_index"] = chunk_start + step.get("step_index", 0)
                    adjusted_chunk.append(step_copy)
                else:
                    # Create a copy to avoid mutating the original
                    step.step_index = chunk_start + step.step_index
                    adjusted_chunk.append(step)
            
            all_labeled_steps.extend(adjusted_chunk)

        # Verify we have the same number of steps
        if len(all_labeled_steps) != len(steps):
            logger.error(
                f"Labeling mismatch: input {len(steps)} steps, output {len(all_labeled_steps)} steps"
            )
            raise ValueError("Labeling output step count mismatch")

        return all_labeled_steps

    async def _normalize_steps(
        self,
        labeled_steps: List[Any],
        problem_statement: str,
        run_log: RunLog,
        document_id: str,
        workflow_name: str,
    ) -> Optional[str]:
        """
        Normalize steps to single behavior per step.
        Returns normalized CoT string or None if normalization failed.
        """
        max_iterations = self._config.get("step_normalization", {}).get(
            "max_iterations", 5
        )
        verification_attempts = self._config.get("step_normalization", {}).get(
            "verification_attempts", 3
        )
        chunk_size = self._config.get("chunking", {}).get("chunk_size", 100)
        overlap_size = self._config.get("chunking", {}).get(
            "normalization_overlap", 10
        )

        # Default message for first attempt or when no previous failures
        default_feedback = (
            "This is the first normalization attempt (or no previous verification failures). "
            "Focus on creating truly atomic steps where each step has exactly one behavior. "
            "When in doubt, split steps rather than merge them. Steps longer than 2-3 sentences "
            "likely need to be split. Ensure the label matches the PRIMARY and ONLY behavior present."
        )

        # Format original labeled steps for first attempt (will be used as previous_normalized_steps)
        formatted_original = self._format_labeled_steps_for_normalization(
            labeled_steps
        )

        # For now, process full CoT if under threshold, otherwise use simple chunking
        # TODO: Implement sliding window chunking for better boundary handling
        if len(labeled_steps) <= chunk_size * 2:
            # Process full CoT
            previous_feedback = []  # Collect failed verification reasoning
            previous_normalized_steps_formatted = formatted_original  # Start with original labeled steps

            # Iterative normalization with verification
            for attempt in range(max_iterations):
                # Prepare feedback message
                if previous_feedback:
                    feedback_message = (
                        "Previous verification attempts identified the following issues with your normalization. "
                        "Please address these specific problems:\n\n"
                        + "\n\n---\n\n".join(previous_feedback[-3:])  # Last 3 failures
                    )
                else:
                    feedback_message = default_feedback

                normalization_context = {
                    "problem_statement": problem_statement,
                    "previous_normalized_steps": previous_normalized_steps_formatted,
                    "previous_verification_feedback": feedback_message,
                }
                normalization_result = await self._execute_stage(
                    "StepNormalization",
                    normalization_context,
                    run_log,
                    document_id,
                    workflow_name,
                )

                normalized_steps = normalization_result.get("outputs", {}).get(
                    "normalized_steps", []
                )

                if not normalized_steps:
                    logger.warning(
                        f"StepNormalization returned empty on attempt {attempt + 1}"
                    )
                    continue

                # Format normalized steps with labels for next iteration and verification
                normalized_formatted = []
                for i, s in enumerate(normalized_steps):
                    # Handle both dict and object access
                    if isinstance(s, dict):
                        content = s.get("content", "")
                        behavior = s.get("behavior", "")
                    else:
                        content = getattr(s, "content", "")
                        behavior = getattr(s, "behavior", "")
                    # Handle empty behavior (should show "None" to match original format)
                    behavior_str = behavior if behavior else "None"
                    normalized_formatted.append(
                        f"Step number {i}\n{content},\nLabel: {behavior_str}."
                    )
                normalized_formatted_str = "\n\n".join(normalized_formatted)

                # Attempt k consecutive verifications
                verification_results = []
                for k_attempt in range(verification_attempts):
                    # Format for verification (compare original labeled steps vs normalized)
                    original_formatted = self._format_labeled_steps_for_normalization(
                        labeled_steps
                    )
                    normalized_for_verification = "\n\n".join(
                        [
                            f"Step {i}: {s.content}\nBehavior: {s.behavior}"
                            for i, s in enumerate(normalized_steps)
                        ]
                    )

                    verification_context = {
                        "original_labeled_steps": original_formatted,
                        "normalized_steps": normalized_formatted,
                    }

                    verification_result = await self._execute_stage(
                        "NormalizationVerification",
                        verification_context,
                        run_log,
                        document_id,
                        workflow_name,
                    )

                    verification_outputs = verification_result.get("outputs", {})
                    adheres = verification_outputs.get(
                        "normalization_adheres_to_standards", False
                    )
                    verification_results.append(adheres)

                    # If any verification fails, collect feedback and store normalized steps for next attempt
                    if not adheres:
                        reasoning = verification_outputs.get(
                            "normalization_reasoning", ""
                        )
                        if reasoning:
                            previous_feedback.append(reasoning)
                        # Store the formatted normalized steps for next iteration
                        previous_normalized_steps_formatted = normalized_formatted_str
                        logger.info(
                            f"Normalization verification failed on attempt {k_attempt + 1}/{verification_attempts}. "
                            f"Collected feedback and stored normalized steps. Restarting normalization."
                        )
                        break

                # Check if we got k consecutive successes
                if (
                    len(verification_results) == verification_attempts
                    and all(verification_results)
                ):
                    # Success! Join normalized steps
                    normalized_cot = "\n\n".join(
                        [step.content for step in normalized_steps]
                    )
                    logger.info(
                        f"Step normalization succeeded after {attempt + 1} attempts. "
                        f"Normalized {len(labeled_steps)} steps into {len(normalized_steps)} steps."
                    )
                    return normalized_cot
                
                # If we didn't get k consecutive successes but haven't broken yet,
                # update previous_normalized_steps_formatted for next iteration
                if len(verification_results) < verification_attempts:
                    # This shouldn't happen if break logic is correct, but safety check
                    previous_normalized_steps_formatted = normalized_formatted_str

            # Failed after max_iterations
            logger.warning(
                f"Step normalization failed after {max_iterations} attempts. Skipping document."
            )
            return None
        else:
            # TODO: Implement chunked normalization with sliding window
            logger.warning(
                f"CoT too long ({len(labeled_steps)} steps). Chunked normalization not yet implemented. Skipping."
            )
            return None

    async def _add_conclusive_paragraph(
        self,
        normalized_cot: str,
        problem_statement: str,
        run_log: RunLog,
        document_id: str,
        workflow_name: str,
    ) -> str:
        """Add conclusive paragraph to normalized CoT."""
        conclusion_context = {
            "problem_statement": problem_statement,
            "generator_cot": normalized_cot,
        }

        conclusion_result = await self._execute_stage(
            "CoTConclusion",
            conclusion_context,
            run_log,
            document_id,
            workflow_name,
        )

        conclusive_paragraph = conclusion_result.get("outputs", {}).get(
            "conclusive_paragraph", ""
        ).strip()

        if conclusive_paragraph:
            final_cot = normalized_cot.rstrip() + "\n\n" + conclusive_paragraph
            logger.info(
                f"Conclusive paragraph added. Length: {len(conclusive_paragraph)}"
            )
        else:
            logger.warning("CoTConclusion returned empty paragraph.")
            final_cot = normalized_cot

        return final_cot

    async def _run(
        self,
        work_item: WorkItem[CoTImprovementPayload],
        run_log: RunLog,
        document_id: str,
        workflow_name: str,
    ) -> Dict[str, Any]:
        """
        Execute the CoT improvement workflow:
        1. Split CoT into steps
        2. Detect and rewrite loops
        3. Label steps with behaviors
        4. Normalize steps to single behavior per step
        5. Add conclusive paragraph (only on full success)
        """
        payload = work_item.payload
        context: Dict[str, Any] = {
            "problem_id": payload.problem_id,
            "original_cot": payload.generator_cot,
        }

        # Step 1: Split CoT into steps
        steps = payload.generator_cot.split("\n\n")
        # Filter out empty strings
        steps = [s for s in steps if s.strip()]
        logger.info(f"Split CoT into {len(steps)} steps")

        # Step 2: Loop Detection & Rewriting
        improved_steps = await self._detect_and_rewrite_loops(
            steps,
            payload.problem_statement,
            run_log,
            document_id,
            workflow_name,
        )
        if improved_steps is None:
            # Loop rewriting failed - skip document
            context["status"] = "skipped"
            context["skip_reason"] = "loop_rewriting_failed"
            return context

        # Step 3: Behavior Labeling
        if len(improved_steps) > self._config.get("chunking", {}).get(
            "chunk_size", 100
        ):
            labeled_steps = await self._label_steps_chunked(
                improved_steps,
                payload.problem_statement,
                run_log,
                document_id,
                workflow_name,
            )
        else:
            # Process full CoT
            formatted_steps = self._format_steps_for_detection(improved_steps)
            labeling_context = {
                "problem_statement": payload.problem_statement,
                "cot_steps": formatted_steps,
            }
            labeling_result = await self._execute_stage(
                "MultiBehaviorLabeling",
                labeling_context,
                run_log,
                document_id,
                workflow_name,
            )
            labeled_steps_raw = labeling_result.get("outputs", {}).get(
                "labeled_steps", []
            )

            # Verify step count matches
            if len(labeled_steps_raw) != len(improved_steps):
                logger.error(
                    f"Labeling mismatch: input {len(improved_steps)} steps, output {len(labeled_steps_raw)} steps"
                )
                context["status"] = "skipped"
                context["skip_reason"] = "labeling_step_count_mismatch"
                return context

            # Convert to list of CoTStepBehaviorLabel objects if needed
            # The output_mapping should return the list directly
            labeled_steps = labeled_steps_raw

        # Step 4: Step Normalization
        normalized_cot = await self._normalize_steps(
            labeled_steps,
            payload.problem_statement,
            run_log,
            document_id,
            workflow_name,
        )
        if normalized_cot is None:
            # Step normalization failed - skip document
            context["status"] = "skipped"
            context["skip_reason"] = "step_normalization_failed"
            return context

        # Step 5: Add conclusive paragraph (only on full success)
        final_cot = await self._add_conclusive_paragraph(
            normalized_cot,
            payload.problem_statement,
            run_log,
            document_id,
            workflow_name,
        )

        context.update(
            {
                "status": "completed",
                "improved_cot": final_cot,
            }
        )

        logger.info(
            f"CoT improvement completed successfully. "
            f"Original: {len(payload.generator_cot)} chars, "
            f"Improved: {len(final_cot)} chars"
        )

        return context



