# src/easy_scaffold/workflows/agents/generator_critique_agent.py
import asyncio
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from easy_scaffold.common.custom_exceptions import EmptyResponseException, StageExecutionError
from easy_scaffold.db.pydantic_models import RunLog
from easy_scaffold.workflows.base import AbstractWorkflow
from easy_scaffold.workflows.error_handler import WorkflowErrorHandler
from easy_scaffold.workflows.workflow_models import (
    GeneratorCritiquePayload,
    SolutionDetection,
    WorkItem,
)

logger = logging.getLogger(__name__)


class GeneratorCritiqueWorkflow(AbstractWorkflow[GeneratorCritiquePayload]):
    """
    Iterative generator-critique workflow with chunked generation and intermediate verification.
    """
    
    def _load_language_patterns(self) -> Dict[str, Any]:
        """Load language patterns from JSON file."""
        patterns_file = Path("data/cot_label_analysis/language_patterns.json")
        if patterns_file.exists():
            try:
                with open(patterns_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load language patterns: {e}")
                return {}
        else:
            logger.warning("Language patterns file not found, using empty dict")
            return {}
    
    def _format_language_patterns(self, patterns: Dict[str, Any], pattern_names: Optional[List[str]] = None) -> str:
        """
        Format language patterns as a readable string instead of JSON.
        
        Args:
            patterns: Full language patterns dict
            pattern_names: Optional list of pattern names to include (filters patterns)
            
        Returns:
            Formatted string with language pattern information
        """
        if not patterns:
            return "No language patterns available."
        
        # Filter patterns if specific names provided
        if pattern_names:
            filtered_patterns = {name: patterns[name] for name in pattern_names if name in patterns}
        else:
            filtered_patterns = patterns
        
        if not filtered_patterns:
            return "No matching language patterns found."
        
        lines = []
        lines.append("=" * 80)
        lines.append("LANGUAGE PATTERNS REFERENCE")
        lines.append("=" * 80)
        lines.append("")
        
        for pattern_name, pattern_data in filtered_patterns.items():
            lines.append(f"## {pattern_name}")
            lines.append("")
            
            # Summary
            if "summary" in pattern_data:
                lines.append(f"**Summary**: {pattern_data['summary']}")
                lines.append("")
            
            # Tone
            if "tone_description" in pattern_data:
                lines.append(f"**Tone**: {pattern_data['tone_description']}")
                lines.append("")
            
            # Common phrases
            if "common_phrases" in pattern_data and pattern_data["common_phrases"]:
                lines.append("**Common Phrases**:")
                for phrase in pattern_data["common_phrases"]:
                    lines.append(f"  - \"{phrase}\"")
                lines.append("")
            
            # Key patterns
            if "key_patterns" in pattern_data and pattern_data["key_patterns"]:
                lines.append("**Key Patterns**:")
                for pattern in pattern_data["key_patterns"]:
                    lines.append(f"  - {pattern}")
                lines.append("")
            
            # Structural patterns
            if "structural_patterns" in pattern_data and pattern_data["structural_patterns"]:
                lines.append("**Structural Patterns**:")
                for struct_pattern in pattern_data["structural_patterns"]:
                    lines.append(f"  - {struct_pattern}")
                lines.append("")
            
            # Examples (limit to 2-3 for token efficiency)
            if "examples" in pattern_data and pattern_data["examples"]:
                lines.append("**Examples**:")
                for i, example in enumerate(pattern_data["examples"][:3], 1):  # Limit to 3 examples
                    lines.append(f"  {i}. {example}")
                lines.append("")
            
            lines.append("")
        
        return "\n".join(lines)
    
    def _extract_cot_and_solution(self, full_text: str) -> tuple[str, str]:
        """
        Extract Chain of Thought (CoT) and solution from generator output.
        Looks for thinking block markers (e.g., `</think>`).
        """
        thinking_end_patterns = [
            r'`</think>`',
            r'`</think>`',
            r'</think>',
            r'</think>',
        ]
        
        cot_text = full_text
        solution_text = ""
        
        for pattern in thinking_end_patterns:
            match = re.search(pattern, full_text, re.IGNORECASE)
            if match:
                end_pos = match.end()
                start_pos = match.start()
                # CoT is everything before the tag
                cot_text = full_text[:start_pos].strip()
                # Solution is everything after the tag
                solution_text = full_text[end_pos:].strip()
                return cot_text, solution_text
        
        # If no tag found, return full text as CoT (assuming incomplete)
        return cot_text, ""

    def _clean_step_content(self, text: str) -> str:
        """
        Clean specific step numbering artifacts from the generator's output.
        Removes 'Step N:', 'N.', or similar prefixes to prevent double labeling.
        """
        if not text or not isinstance(text, str):
            return ""
        # Remove "Step X:" or "Step X." pattern
        cleaned = re.sub(r'^(?:Step\s+\d+[:.]?|\d+\.|-)\s*', '', text.strip(), flags=re.IGNORECASE)
        return cleaned or ""  # Ensure we never return None

    def _count_actual_steps(self, cot_lines: List[str]) -> int:
        """
        Count actual steps (non-empty lines) in CoT.
        Empty strings are paragraph breaks, not steps.
        
        Args:
            cot_lines: List of CoT lines (may contain empty strings)
            
        Returns:
            Number of actual steps (non-empty lines)
        """
        return sum(1 for line in cot_lines if line and line.strip())
    
    def _get_step_slice_index(self, cot_lines: List[str], target_step: int) -> int:
        """
        Find the index in cot_lines that corresponds to keeping the first 'target_step' steps.
        Steps are non-empty lines (empty strings are paragraph breaks, not steps).
        Returns the index (exclusive) to slice up to, i.e., cot_lines[:index] keeps target_step steps.
        
        Args:
            cot_lines: List of CoT lines (may contain empty strings)
            target_step: 1-based step number (keep steps 1 through this step, inclusive)
            
        Returns:
            Exclusive slice index (0-based) to use with cot_lines[:index]
        """
        if target_step <= 0:
            return 0
        
        step_count = 0
        for i, line in enumerate(cot_lines):
            if line and line.strip():  # Non-empty line = a step
                step_count += 1
                if step_count == target_step:
                    return i + 1  # Include this step (exclusive slice index)
        
        # If we didn't find enough steps, return the full length
        return len(cot_lines)
    
    def _split_cot_by_double_newlines(self, cot_text: str) -> List[str]:
        """
        Split CoT text by double newlines to recover steps.
        This is the inverse of _join_cot_with_double_newlines.
        
        Use this for:
        - Resuming from stored CoT (which was joined with \n\n)
        - Processing shortened CoT (which uses \n\n separators)
        
        Do NOT use this for streaming generator output (use \n splitting instead).
        
        Args:
            cot_text: CoT text with steps separated by \n\n
            
        Returns:
            List of CoT steps (non-empty strings)
        """
        if not cot_text or not cot_text.strip():
            return []
        # Split by double newlines and filter empty strings
        steps = [step.strip() for step in cot_text.split('\n\n') if step.strip()]
        return steps
    
    def _join_cot_with_double_newlines(self, cot_lines: List[str]) -> str:
        """
        Join CoT lines preserving double newlines between paragraphs.
        Filters out empty/whitespace-only lines and joins with double newlines.
        
        This is the standard format for storing CoT. Steps are separated by \n\n.
        Empty strings in cot_lines are filtered out (they're just processing markers
        for paragraph breaks during streaming, but \n\n already represents the break).
        
        Args:
            cot_lines: List of CoT steps (may contain empty strings for paragraph breaks)
            
        Returns:
            CoT text with steps separated by \n\n (empty strings are filtered out)
        """
        if not cot_lines:
            return ""
        # Filter out empty strings and None values
        valid_lines = [line for line in cot_lines if line and line.strip()]
        return '\n\n'.join(valid_lines)

    def _sanitize_cot_list(self, cot_lines: List[str]) -> List[str]:
        """
        Sanitize CoT list by removing any closing thinking tokens (</think>).
        This ensures the thinking block remains open for hint injection.
        """
        if not cot_lines:
            return []
            
        sanitized = []
        # Pattern for </think> with optional backticks
        thinking_end_pattern = r'`?</think>`?'
        
        for line in cot_lines:
            if line is None:
                continue
            if line == "":
                sanitized.append("")
                continue
                
            # Remove </think> and anything after it from the line
            match = re.search(thinking_end_pattern, line, re.IGNORECASE)
            if match:
                # Keep content before the tag
                clean_line = line[:match.start()].strip()
                # Only add if content remains (or if it was a meaningful step that is now empty, 
                # we might want to skip it to avoid empty steps)
                if clean_line:
                    sanitized.append(clean_line)
            else:
                sanitized.append(line)
        
        return sanitized
    
    async def _shorten_cot_if_needed(
        self,
        cot_history: List[str],
        threshold: int,
        shortening_factor: str,
        context: Dict[str, Any],
        run_log: RunLog,
        document_id: str,
        workflow_name: str,
    ) -> Tuple[List[str], bool]:
        """
        Shorten CoT if it exceeds token threshold.
        
        Args:
            cot_history: List of CoT lines (may include empty strings for paragraph breaks)
            threshold: Token count threshold to trigger shortening
            shortening_factor: Target reduction factor (e.g., "90%")
            context: Workflow context dict
            run_log: Run log for stage execution
            document_id: Document ID
            workflow_name: Workflow name
            
        Returns:
            Tuple of (updated_cot_history, was_shortened)
            - updated_cot_history: Shortened CoT history if shortened, original otherwise
            - was_shortened: True if shortening occurred, False otherwise
        """
        # Reconstruct CoT text
        current_cot_text = self._join_cot_with_double_newlines(cot_history)
        full_partial = "<think>\n" + current_cot_text
        
        # Get model profile for token counting
        stage_cfg = self._stage_factory._stage_configs.get("GeneratorContinuation")
        if not stage_cfg:
            stage_cfg = self._stage_factory._stage_configs.get("Generator")
        if not stage_cfg:
            logger.warning("Could not find stage config for token counting. Skipping shortening check.")
            return cot_history, False
        
        model_profile = stage_cfg.model_profile or "default"
        
        # Count tokens
        current_tokens_count = self._stage_factory._llm_client.count_tokens(full_partial, model_profile)
        
        # Check threshold
        if current_tokens_count <= threshold:
            logger.debug(f"CoT token count {current_tokens_count} <= threshold {threshold}. No shortening needed.")
            return cot_history, False
        
        # Exceeds threshold - shorten
        logger.info(
            f"CoT length {current_tokens_count} tokens ({len(full_partial)} chars) "
            f"exceeds threshold {threshold} tokens. "
            f"Running Shortener with factor {shortening_factor}."
        )
        
        shortener_context = {
            "partial_solution": full_partial,
            "shortening_factor": shortening_factor
        }
        
        logger.debug(f"Shortener context keys: {list(shortener_context.keys())}")
        logger.debug(f"Shortener context partial_solution length: {len(shortener_context['partial_solution'])}")
        logger.debug(f"Shortener context shortening_factor: {shortener_context['shortening_factor']}")
        
        try:
            shortener_result = await self._execute_stage(
                "CoTShortener", shortener_context, run_log, document_id, workflow_name
            )
            shortened_cot = shortener_result.get("outputs", {}).get("shortened_cot", "")
            
            # Strip any meta-commentary that might have been included
            if shortened_cot:
                meta_patterns = [
                    r"^(?:I\s+need\s+to\s+condense.*?\.\s*)+",
                    r"^(?:Here'?s?\s+the\s+condensed.*?:\s*)+",
                    r"^(?:The\s+condensed.*?:\s*)+",
                    r"^(?:Here\s+is\s+the\s+condensed.*?:\s*)+",
                ]
                for pattern in meta_patterns:
                    shortened_cot = re.sub(pattern, "", shortened_cot, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL)
                shortened_cot = shortened_cot.strip()
            
            if shortened_cot and len(shortened_cot) < len(full_partial):
                logger.info(f"Shortener reduced CoT from {len(full_partial)} to {len(shortened_cot)} chars.")
                # Split by double newlines to match CoT format (steps separated by \n\n)
                # The shortener returns formatted text with \n\n separators
                steps = self._split_cot_by_double_newlines(shortened_cot)
                # Clean step content and filter empty
                new_cot_history = []
                for step in steps:
                    cleaned = self._clean_step_content(step)
                    if cleaned:
                        new_cot_history.append(cleaned)
                return new_cot_history, True
            else:
                logger.warning("Shortener failed to reduce length or returned empty. Continuing with original.")
                return cot_history, False
        except Exception as e:
            logger.error(f"Shortener stage failed: {e}")
            return cot_history, False
    
    def _cut_cot_at_index(self, cot_history: List[str], last_useful_step_index: int) -> str:
        """
        Cut CoT at the last useful step index (1-based).
        Steps are non-empty lines (empty strings are paragraph breaks, not steps).
        
        Args:
            cot_history: List of CoT steps (lines, may contain empty strings for paragraph breaks)
            last_useful_step_index: 1-based step number (keep steps 1 through this step, inclusive)
            
        Returns:
            CoT string cut at the index (preserving paragraph breaks)
        """
        if not cot_history or last_useful_step_index < 1:
            return ""
        
        # Use _get_step_slice_index to find the correct slice point
        # This counts actual steps (non-empty lines), not list positions
        slice_index = self._get_step_slice_index(cot_history, last_useful_step_index)
        cut_cot_lines = cot_history[:slice_index]
        
        # Join preserving paragraph breaks (empty strings become double newlines)
        return self._join_cot_with_double_newlines(cut_cot_lines)
    
    def _count_verified_steps(self, step_status: Optional[Dict[str, str]]) -> int:
        """Count the number of steps marked as 'correct'."""
        if not step_status:
            return 0
        return sum(1 for status in step_status.values() if status.lower() == "correct")

    def _is_progress_better(
        self,
        current_step_status: Dict[str, str],
        max_verified_count: int,
    ) -> bool:
        """
        Check if current progress is strictly better than max verified count.
        Using Parallel Progress logic: more correct steps = better.
        """
        current_count = self._count_verified_steps(current_step_status)
        return current_count > max_verified_count

    def _adjust_hint_level(
        self,
        max_verified_count: int,
        current_step_status: Dict[str, str],
        current_hint_level: int,
    ) -> int:
        """
        Adjusts the hint level based on Parallel Progress (total correct steps).
        
        Logic:
        - If total correct steps > max previous: Decrease hint level by 1 (min 0).
        - If no increase in correct steps: Increase hint level (capped at 4).
        """
        # Check if progress was made
        progress_made = self._is_progress_better(current_step_status, max_verified_count)
        
        if progress_made:
            # Verified count increased! Decrease hint level smoothly.
            return max(0, current_hint_level - 1)
        else:
            # No new verified steps. Increase hint level.
            return min(4, current_hint_level + 1)
    
    async def _run_new_critique_pipeline(
        self,
        cot_history: List[str],
        final_answer: str,
        context: Dict[str, Any],
        run_log: RunLog,
        document_id: str,
        workflow_name: str,
        previous_progress_pointer: Optional[Dict[str, Any]],
        hint_level: int,
        iteration_count: int,
        error_handler: WorkflowErrorHandler,
    ) -> Tuple[bool, Optional[str], Dict[str, Any]]:
        """
        Run the new critique pipeline: Verifier -> ProgressAssessor -> HintGenerator -> Translator.
        
        Returns:
            (should_accept, translated_hint, updated_context)
            - should_accept: True if solution is correct and should be accepted
            - translated_hint: Translated hint to inject (if any)
            - updated_context: Updated context dict with detection results
        """
        # Prepare CoT text (raw + step-labeled)
        cot_text = self._join_cot_with_double_newlines(cot_history)
        labeled_cot_text = self._label_cot_steps(cot_history)
        
        # Stage 1: Verifier (analyzes final answer only, with verification loop)
        verifier_context = {
            "problem_statement": context["problem_statement"],
            "reference_solution": context["reference_solution"],
            "solution_analysis": context.get("solution_analysis", ""),
            "generator_answer": final_answer,
        }
        
        num_attempts = self._config.get("critique_verification_attempts", 1)
        all_attempts_correct = True
        verifier_outputs = None
        
        for attempt_idx in range(num_attempts):
            logger.info(f"Running Verifier (Attempt {attempt_idx + 1}/{num_attempts})")
            try:
                verifier_result = await self._execute_stage_with_error_handling(
                    "Verifier", verifier_context, run_log, document_id, workflow_name, error_handler
                )
                outputs = verifier_result.get("outputs", {})
                is_correct = outputs.get("detection_correctness", False)
            except StageExecutionError as err:
                logger.error(f"Verifier failed: {err}")
                # Use fallback
                fallback_correctness = error_handler.get_fallback_value("Verifier", "correctness")
                if fallback_correctness is not None:
                    is_correct = fallback_correctness
                    outputs = {"detection_correctness": is_correct}
                    logger.warning(f"Using fallback correctness: {is_correct}")
                else:
                    raise err # Re-raise if no fallback

            if not is_correct:
                all_attempts_correct = False
                verifier_outputs = outputs  # Keep failing result
                logger.info(f"Verifier attempt {attempt_idx + 1} found errors.")
                break
            
            verifier_outputs = outputs  # Keep this result
        
        if all_attempts_correct:
            # Solution verified correct
            logger.info(f"Solution verified correct across {num_attempts} attempts!")
            context["status"] = "completed"
            context["solution_found"] = True
            context["detection_explanation"] = verifier_outputs.get("detection_explanation", "")
            context["generator_cot"] = cot_text
            context["generator_answer"] = final_answer
            if iteration_count == 1:
                context["already_correct"] = True
                logger.info("Solution was already correct on first attempt.")
            return True, None, context
        
        # Solution incorrect - proceed with fixing
        errors = verifier_outputs.get("detection_errors", [])
        contradiction_errors = verifier_outputs.get("detection_contradiction_errors", [])
        explanation = verifier_outputs.get("detection_explanation", "")
        
        # Convert ProofError objects to dicts for JSON serialization
        errors_dict = [
            {
                "type": err.type if hasattr(err, 'type') else err.get("type", ""),
                "description": err.description if hasattr(err, 'description') else err.get("description", ""),
                "location": err.location if hasattr(err, 'location') else err.get("location", "")
            }
            for err in errors
        ] if errors else []
        
        contradiction_errors_dict = [
            {
                "type": err.type if hasattr(err, 'type') else err.get("type", ""),
                "description": err.description if hasattr(err, 'description') else err.get("description", ""),
                "location": err.location if hasattr(err, 'location') else err.get("location", "")
            }
            for err in contradiction_errors
        ] if contradiction_errors else []
        
        # Format unified error report (errors + contradiction_errors) as string
        errors_report_text = self._format_errors(errors, contradiction_errors)
        
        # Stage 2: Progress Assessment (compare full CoT to solution analysis)
        logger.info("Assessing progress with full CoT")
        
        progress_context = {
            "problem_statement": context["problem_statement"],
            "solution_analysis": context.get("solution_analysis", ""),
            # Full CoT, labeled as Step 1/2/3... so ProgressAssessor can return a stable 1-based index
            "generator_cot": labeled_cot_text,
            "generator_answer": final_answer,
            "errors_report": errors_report_text,  # Formatted string
        }
        
        try:
            progress_result = await self._execute_stage_with_error_handling(
                "ProgressAssessor", progress_context, run_log, document_id, workflow_name, error_handler
            )
            progress_outputs = progress_result.get("outputs", {})
            progress_assessment = progress_outputs.get("progress_assessment")
        except StageExecutionError as err:
            logger.error(f"ProgressAssessor failed: {err}")
            fallback_pointer = error_handler.get_fallback_value("ProgressAssessor", "progress_pointer")
            if fallback_pointer is not None:
                logger.warning(f"Using fallback progress pointer: {fallback_pointer}")
                # Construct fallback assessment
                actual_step_count = self._count_actual_steps(cot_history) if cot_history else 1
                progress_assessment = {
                    "progress_pointer": fallback_pointer,
                    "remaining_work_summary": "Continue with the correct approach (Fallback)",
                    "last_useful_step_index": actual_step_count,
                }
            else:
                raise err
        
        # Convert to dict if BaseModel
        if progress_assessment is not None and not isinstance(progress_assessment, dict):
            if hasattr(progress_assessment, "model_dump"):
                progress_assessment = progress_assessment.model_dump()
        
        # Validate progress assessment
        if not progress_assessment or not isinstance(progress_assessment, dict):
            logger.warning("ProgressAssessor returned invalid output. Using fallback.")
            # Fallback: use actual step count (not list length)
            actual_step_count = self._count_actual_steps(cot_history) if cot_history else 1
            progress_assessment = {
                "progress_pointer": {"main_step": 0, "substep": 0},
                "remaining_work_summary": "Complete the proof",
                "last_useful_step_index": actual_step_count,
            }
        
        # Extract progress pointer and last useful step index
        progress_pointer = progress_assessment.get("progress_pointer", {})
        if not isinstance(progress_pointer, dict):
            progress_pointer = {"main_step": 0, "substep": 0}
        
        last_useful_step_index = progress_assessment.get("last_useful_step_index", 1)
        if not isinstance(last_useful_step_index, int) or last_useful_step_index < 1:
            logger.warning(f"Invalid last_useful_step_index: {last_useful_step_index}. Using 1.")
            last_useful_step_index = 1
        
        # Get max_verified_count from context
        max_verified_count = context.get("max_verified_count", 0)
        
        # Get step_status for current progress assessment
        # Convert list of StepStatusItem objects back to dict for easier usage
        step_status_list = progress_assessment.get("step_status", [])
        current_step_status = {}
        if isinstance(step_status_list, list):
            for item in step_status_list:
                # Handle both dict (from raw JSON) and object (from Pydantic)
                if isinstance(item, dict):
                    step_id = item.get("step_id")
                    status = item.get("status")
                else:
                    step_id = getattr(item, "step_id", None)
                    status = getattr(item, "status", None)
                
                if step_id and status:
                    current_step_status[step_id] = status
        elif isinstance(step_status_list, dict):
             # Fallback if somehow it's still a dict (legacy or raw)
             current_step_status = step_status_list
        
        # Adjust hint level based on verified step count (Parallel Progress)
        new_hint_level = self._adjust_hint_level(max_verified_count, current_step_status, hint_level)
        
        current_verified_count = self._count_verified_steps(current_step_status)
        if new_hint_level != hint_level:
            logger.info(
                f"Hint level adjusted: {hint_level} -> {new_hint_level} "
                f"(verified count: {max_verified_count} -> {current_verified_count})"
            )
        
        # Update max_verified_count if current is better
        if current_verified_count > max_verified_count:
            max_verified_count = current_verified_count
            logger.info(f"New max verified step count set: {max_verified_count}")

        # Update context with new hint level and progress
        context["hint_level"] = new_hint_level
        context["current_progress_pointer"] = progress_pointer  # Store for next iteration
        context["max_verified_count"] = max_verified_count # Store max progress (count)
        
        # Code Logic: Cut CoT at last_useful_step_index
        generator_cot_up_to_index = self._cut_cot_at_index(cot_history, last_useful_step_index)
        slice_index = self._get_step_slice_index(cot_history, last_useful_step_index)
        logger.info(f"Cut CoT at step {last_useful_step_index}: kept {last_useful_step_index} steps (slice index: {slice_index})")
        
        # Stage 3: Hint Generation
        logger.info(f"Generating hint for hint_level={new_hint_level}, progress_pointer={progress_pointer}")
        
        hint_context = {
            "problem_statement": context["problem_statement"],
            "reference_solution": context["reference_solution"],
            "solution_analysis": context.get("solution_analysis", ""),
            # Provide labeled CoT so step indices (last_useful_step_index) are unambiguous
            "generator_cot_up_to_index": self._label_cot_steps(self._split_cot_by_double_newlines(generator_cot_up_to_index)),
            "generator_answer": final_answer,
            "progress_assessment": self._format_progress_assessment(progress_assessment),
            "step_status": progress_assessment.get("step_status", {}),
            "hint_level": new_hint_level,
            "errors_report": errors_report_text,  # Formatted string
        }
        
        try:
            hint_result = await self._execute_stage_with_error_handling(
                "HintGenerator", hint_context, run_log, document_id, workflow_name, error_handler
            )
            generated_hint = hint_result.get("outputs", {}).get("hint", "")
        except StageExecutionError as err:
            logger.error(f"HintGenerator failed: {err}")
            generated_hint = error_handler.get_fallback_value("HintGenerator", "hint")
            if generated_hint:
                logger.warning(f"Using fallback hint: {generated_hint}")
            else:
                raise err
        
        if not generated_hint:
            logger.warning("HintGenerator returned empty hint. Using fallback.")
            generated_hint = "Continue with the correct approach."
        
        # Stage 4: Translator (translate hint to generator's style)
        logger.info("Translating hint to generator's style")
        
        # Load and format behavioral patterns
        language_patterns = self._load_language_patterns()
        behavioral_patterns_text = self._format_language_patterns(language_patterns) if language_patterns else "No language patterns available."
        
        translator_context = {
            "problem_statement": context["problem_statement"],
            "generated_hint": generated_hint,
            # Back-compat: some stage configs historically mapped generated_hint <- "hint"
            "hint": generated_hint,
            # Translator uses this for *style*, so keep it raw (no injected Step prefixes).
            "generator_cot_up_to_index": generator_cot_up_to_index,
            "behavioral_patterns": behavioral_patterns_text,  # Formatted string
        }
        
        try:
            translator_result = await self._execute_stage_with_error_handling(
                "Translator", translator_context, run_log, document_id, workflow_name, error_handler
            )
            translator_outputs = translator_result.get("outputs", {})
            
            # New Text-Only Handling: Output is just a raw string
            translated_hint = translator_outputs.get("translated_hint", "")
        except StageExecutionError as err:
            logger.error(f"Translator failed: {err}")
            # Translator is optional, fallback to generated_hint
            translated_hint = generated_hint
            logger.warning("Translator failed, falling back to raw generated hint.")
            translator_outputs = {}
        if not translated_hint:
             # Fallback to generated_hint if empty
             translated_hint = generated_hint
        
        # Split into steps manually by double newlines
        translated_hint_steps = [s.strip() for s in translated_hint.split('\n\n') if s.strip()]
        
        # We no longer get behavioral_chain or chain_reason from text output
        behavioral_chain = []
        chain_reason = "Internal (Text Mode)"
        
        context["translated_hint_steps"] = translated_hint_steps
        
        if not translated_hint:
            logger.warning("Translator returned empty hint. Using generated hint.")
            translated_hint = generated_hint
        
        # Update context with results
        context["detection_explanation"] = explanation
        context["detection_errors"] = errors_dict
        context["detection_contradiction_errors"] = contradiction_errors_dict
        context["progress_assessment"] = progress_assessment
        context["progress_pointer"] = progress_pointer
        context["last_useful_step_index"] = last_useful_step_index
        context["behavioral_chain"] = behavioral_chain
        context["chain_reason"] = chain_reason
        context["reset_triggered"] = False
        
        return False, translated_hint, context
    
    async def _run_hint_generation_pipeline(
        self,
        cot_history: List[str],
        final_answer: str,
        context: Dict[str, Any],
        run_log: RunLog,
        document_id: str,
        workflow_name: str,
        previous_progress_pointer: Optional[Dict[str, Any]],
        hint_level: int,
        iteration_count: int,
        error_handler: WorkflowErrorHandler,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Generate hint when generator is stuck (retries exhausted).
        Skips Verifier, goes directly to ProgressAssessor -> HintGenerator -> Translator.
        
        Returns:
            (translated_hint, updated_context)
        """
        logger.info("Generator stuck (retries exhausted). Skipping Verifier and generating hint directly.")
        
        # Prepare CoT text (raw + step-labeled for ProgressAssessor)
        cot_text = self._join_cot_with_double_newlines(cot_history)
        labeled_cot_text = self._label_cot_steps(cot_history)
        
        # Format error report (empty since we skipped Verifier)
        errors_report_text = "No errors detected. Generator needs more guidance to proceed."
        
        # Stage 1: Progress Assessment (compare full CoT to solution analysis)
        logger.info("Assessing progress with full CoT")
        
        progress_context = {
            "problem_statement": context["problem_statement"],
            "solution_analysis": context.get("solution_analysis", ""),
            "generator_cot": labeled_cot_text,  # Full CoT labeled Step 1/2/3...
            "generator_answer": final_answer,
            "errors_report": errors_report_text,  # Formatted string
        }
        
        try:
            progress_result = await self._execute_stage_with_error_handling(
                "ProgressAssessor", progress_context, run_log, document_id, workflow_name, error_handler
            )
            progress_outputs = progress_result.get("outputs", {})
            progress_assessment = progress_outputs.get("progress_assessment")
        except StageExecutionError as err:
            logger.error(f"ProgressAssessor failed: {err}")
            fallback_pointer = error_handler.get_fallback_value("ProgressAssessor", "progress_pointer")
            if fallback_pointer is not None:
                logger.warning(f"Using fallback progress pointer: {fallback_pointer}")
                # Construct fallback assessment
                actual_step_count = self._count_actual_steps(cot_history) if cot_history else 1
                progress_assessment = {
                    "progress_pointer": fallback_pointer,
                    "remaining_work_summary": "Continue with the correct approach (Fallback)",
                    "last_useful_step_index": actual_step_count,
                }
            else:
                raise err
        
        # Convert to dict if BaseModel
        if progress_assessment is not None and not isinstance(progress_assessment, dict):
            if hasattr(progress_assessment, "model_dump"):
                progress_assessment = progress_assessment.model_dump()
        
        # Validate progress assessment
        if not progress_assessment or not isinstance(progress_assessment, dict):
            logger.warning("ProgressAssessor returned invalid output. Using fallback.")
            # Fallback: use actual step count (not list length)
            actual_step_count = self._count_actual_steps(cot_history) if cot_history else 1
            progress_assessment = {
                "progress_pointer": {"main_step": 0, "substep": 0},
                "remaining_work_summary": "Complete the proof",
                "last_useful_step_index": actual_step_count,
            }
        
        # Extract progress pointer and last useful step index
        progress_pointer = progress_assessment.get("progress_pointer", {})
        if not isinstance(progress_pointer, dict):
            progress_pointer = {"main_step": 0, "substep": 0}
        
        # Use actual step count for fallback (not list length, which includes empty strings)
        actual_step_count = self._count_actual_steps(cot_history) if cot_history else 1
        last_useful_step_index = progress_assessment.get("last_useful_step_index", actual_step_count)
        if not isinstance(last_useful_step_index, int) or last_useful_step_index < 1:
            last_useful_step_index = actual_step_count
        
        # Get max_verified_count from context
        max_verified_count = context.get("max_verified_count", 0)
        
        # Get step_status for current progress assessment
        # Convert list of StepStatusItem objects back to dict for easier usage
        step_status_list = progress_assessment.get("step_status", [])
        current_step_status = {}
        if isinstance(step_status_list, list):
            for item in step_status_list:
                # Handle both dict (from raw JSON) and object (from Pydantic)
                if isinstance(item, dict):
                    step_id = item.get("step_id")
                    status = item.get("status")
                else:
                    step_id = getattr(item, "step_id", None)
                    status = getattr(item, "status", None)
                
                if step_id and status:
                    current_step_status[step_id] = status
        elif isinstance(step_status_list, dict):
             # Fallback if somehow it's still a dict (legacy or raw)
             current_step_status = step_status_list
        
        # Adjust hint level based on verified step count
        new_hint_level = self._adjust_hint_level(max_verified_count, current_step_status, hint_level)
        
        current_verified_count = self._count_verified_steps(current_step_status)
        if new_hint_level != hint_level:
            logger.info(
                f"Hint level adjusted: {hint_level} -> {new_hint_level} "
                f"(verified count: {max_verified_count} -> {current_verified_count})"
            )
        
        # Update max_verified_count if current is better
        if current_verified_count > max_verified_count:
            max_verified_count = current_verified_count
            logger.info(f"New max verified step count set: {max_verified_count}")
        
        # Update context
        updated_context = {
            "hint_level": new_hint_level,
            "current_progress_pointer": progress_pointer,
            "max_verified_count": max_verified_count,
            "last_useful_step_index": last_useful_step_index,
        }
        
        # Code Logic: Cut CoT at last_useful_step_index
        generator_cot_up_to_index = self._cut_cot_at_index(cot_history, last_useful_step_index)
        slice_index = self._get_step_slice_index(cot_history, last_useful_step_index)
        logger.info(f"Cut CoT at step {last_useful_step_index}: kept {last_useful_step_index} steps (slice index: {slice_index})")
        
        # Stage 2: Hint Generation
        logger.info(f"Generating hint for stuck generator, hint_level={new_hint_level}, progress_pointer={progress_pointer}")
        
        hint_context = {
            "problem_statement": context["problem_statement"],
            "reference_solution": context["reference_solution"],
            "solution_analysis": context.get("solution_analysis", ""),
            "generator_cot_up_to_index": self._label_cot_steps(self._split_cot_by_double_newlines(generator_cot_up_to_index)),
            "generator_answer": final_answer,
            "progress_assessment": self._format_progress_assessment(progress_assessment),
            "step_status": current_step_status,  # Use the processed dict
            "hint_level": new_hint_level,
            "errors_report": errors_report_text,  # Formatted string
        }
        
        try:
            hint_result = await self._execute_stage_with_error_handling(
                "HintGenerator", hint_context, run_log, document_id, workflow_name, error_handler
            )
            generated_hint = hint_result.get("outputs", {}).get("hint", "")
        except StageExecutionError as err:
            logger.error(f"HintGenerator failed: {err}")
            generated_hint = error_handler.get_fallback_value("HintGenerator", "hint")
            if generated_hint:
                logger.warning(f"Using fallback hint: {generated_hint}")
            else:
                raise err
        
        if not generated_hint:
            logger.warning("HintGenerator returned empty hint. Using fallback.")
            generated_hint = "Continue with the correct approach."
        
        # Stage 3: Translator (translate hint to generator's style)
        logger.info("Translating hint to generator's style")
        
        # Load and format behavioral patterns
        language_patterns = self._load_language_patterns()
        behavioral_patterns_text = self._format_language_patterns(language_patterns) if language_patterns else "No language patterns available."
        
        translator_context = {
            "problem_statement": context["problem_statement"],
            "generated_hint": generated_hint,
            # Back-compat: some stage configs historically mapped generated_hint <- "hint"
            "hint": generated_hint,
            # Translator uses this for style matching; keep raw.
            "generator_cot_up_to_index": generator_cot_up_to_index,
            "behavioral_patterns": behavioral_patterns_text,  # Formatted string
        }
        
        translator_result = await self._execute_stage(
            "Translator", translator_context, run_log, document_id, workflow_name
        )
        translator_outputs = translator_result.get("outputs", {})
        translated_hint_steps = translator_outputs.get("translated_hint_steps", [])
        if isinstance(translated_hint_steps, list):
            translated_hint_steps = [str(s).strip() for s in translated_hint_steps if str(s).strip()]
        else:
            translated_hint_steps = []
        translated_hint = "\n\n".join(translated_hint_steps) if translated_hint_steps else translator_outputs.get("translated_hint", generated_hint)  # Fallback to original if translation fails
        
        if not translated_hint:
            logger.warning("Translator returned empty hint. Using generated hint.")
            translated_hint = generated_hint
        
        # Update context
        updated_context = context.copy()
        updated_context["hint_level"] = new_hint_level
        updated_context["current_progress_pointer"] = progress_pointer
        updated_context["last_useful_step_index"] = last_useful_step_index
        updated_context["progress_assessment"] = progress_assessment
        updated_context["translated_hint_steps"] = translated_hint_steps
        
        return translated_hint, updated_context
    
    def _append_steps_list_to_cot(self, cot_history: List[str], steps: List[str]) -> None:
        """
        Append a list of step-paragraphs to CoT history, enforcing step boundaries.

        This guarantees that each step becomes its own CoT step separated by a paragraph break
        (represented internally as empty strings and serialized as \n\n).
        """
        if not steps:
            return
        cleaned_steps = [str(s).strip() for s in steps if str(s).strip()]
        if not cleaned_steps:
            return
        cot_history.append("")
        for i, step in enumerate(cleaned_steps):
            cot_history.append(step)
            if i < len(cleaned_steps) - 1:
                cot_history.append("")

    def _append_multi_paragraph_text_to_cot(self, cot_history: List[str], text: str) -> None:
        """
        Append multi-paragraph text to CoT history, splitting by \\n\\n.
        Each paragraph becomes a separate step, with empty strings between for paragraph breaks.
        
        Args:
            cot_history: List of CoT steps (strings, empty strings represent paragraph breaks)
            text: Text to append (may contain \\n\\n separators)
        """
        if not text or not text.strip():
            return
        
        # Split by double newlines and filter empty paragraphs
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        if not paragraphs:
            return
        
        # Add paragraph break before first paragraph
        cot_history.append("")
        
        # Append each paragraph as a separate step
        for i, para in enumerate(paragraphs):
            cot_history.append(para)
            # Add empty string between paragraphs (except after last)
            if i < len(paragraphs) - 1:
                cot_history.append("")
    
    def _label_cot_steps(self, cot_lines: List[str]) -> str:
        """
        Label CoT lines with step numbers for critique.
        Checks for existing step numbers first - only adds labels if not already present.
        Preserves paragraph breaks (empty strings become double newlines).
        """
        return self._label_partial_cot_steps(cot_lines, start_step=1)
    
    def _label_partial_cot_steps(self, cot_lines: List[str], start_step: int = 1) -> str:
        """
        Label partial CoT lines with step numbers starting from a given offset.
        Useful for labeling partial CoTs (e.g., up to injection point, or between certain steps).
        
        Args:
            cot_lines: List of CoT lines (may contain empty strings for paragraph breaks)
            start_step: Starting step number (default 1). For partial CoTs, this should be
                       the original step number where the partial CoT begins.
        
        Returns:
            Labeled CoT string with step numbers adjusted to start from start_step.
        """
        if not cot_lines:
            return ""
        
        # Check if steps are already labeled (look for "Step X:" pattern)
        import re
        first_non_empty = next((line for line in cot_lines if line.strip()), "")
        if first_non_empty and re.match(r'^\s*Step\s+\d+\s*:', first_non_empty, re.IGNORECASE):
            # Already labeled - need to renumber starting from start_step
            labeled_lines = []
            step_num = start_step
            for line in cot_lines:
                if line == "":
                    # Empty string represents paragraph break - preserve
                    labeled_lines.append("")
                else:
                    # Remove existing step label if present
                    cleaned_line = re.sub(r'^\s*Step\s+\d+\s*:\s*\n?', '', line, flags=re.IGNORECASE)
                    labeled_lines.append(f"Step {step_num}:\n{cleaned_line}")
                    step_num += 1
        else:
            # Not labeled - add step numbers starting from start_step
            labeled_lines = []
            step_num = start_step
            for line in cot_lines:
                if line == "":
                    # Empty string represents paragraph break - preserve as double newline
                    labeled_lines.append("")
                else:
                    labeled_lines.append(f"Step {step_num}:\n{line}")
                    step_num += 1
        
        # Join with double newlines preserved
        return self._join_cot_with_double_newlines(labeled_lines)
    
    def _format_progress_assessment(self, progress_assessment: Dict[str, Any]) -> str:
        """
        Format ProgressAssessment dict as a readable string with indentation.
        Uses progress_pointer instead of established_facts/steps.
        
        Args:
            progress_assessment: Dict containing progress assessment fields
            
        Returns:
            Formatted string with clear structure and indentation
        """
        if not progress_assessment or not isinstance(progress_assessment, dict):
            return "No progress assessment available."
        
        lines = []
        lines.append("PROGRESS ASSESSMENT")
        lines.append("")
        
        # Progress Pointer
        progress_pointer = progress_assessment.get("progress_pointer", {})
        if progress_pointer:
            main_step = progress_pointer.get("main_step", 0)
            substep = progress_pointer.get("substep", 0)
            lines.append("**Progress Pointer:**")
            lines.append(f"  - Main Step: {main_step}")
            lines.append(f"  - Substep: {substep}")
            lines.append(f"  - Position: Main Step {main_step}, Sub-step {substep}")
            lines.append("")
        
        # Progress Metrics
        remaining_work = progress_assessment.get("remaining_work_summary", "")
        last_useful_step_index = progress_assessment.get("last_useful_step_index", 0)
        
        lines.append("**Progress Metrics:**")
        lines.append(f"  - Last Useful Step Index: {last_useful_step_index}")
        if remaining_work:
            lines.append(f"  - Remaining Work: {remaining_work}")
        
        # Step Status (Swiss Cheese Map)
        step_status = progress_assessment.get("step_status")
        if step_status:
            lines.append("")
            lines.append("Step Status (Verified Coverage):")
            
            # Handle list of objects (if raw from API) or dict (if processed)
            status_map = {}
            if isinstance(step_status, list):
                for item in step_status:
                    if isinstance(item, dict):
                        sid = item.get("step_id")
                        stat = item.get("status")
                        if sid and stat:
                            status_map[sid] = stat
                    else:
                        # Pydantic object
                        sid = getattr(item, "step_id", None)
                        stat = getattr(item, "status", None)
                        if sid and stat:
                            status_map[sid] = stat
            elif isinstance(step_status, dict):
                status_map = step_status
            
            if status_map:
                # Sort by step ID naturally (assuming 1.1, 1.2, 2.1 format)
                try:
                    sorted_keys = sorted(status_map.keys(), key=lambda x: [int(p) for p in x.split('.') if p.isdigit()])
                except ValueError:
                    sorted_keys = sorted(status_map.keys())
                
                for key in sorted_keys:
                    status = status_map[key]
                    lines.append(f"  - {key}: {status}")
            else:
                lines.append("  - No steps reported.")
        
        lines.append("")
        
        return "\n".join(lines)
    
    def _format_errors(self, errors: List[Any], contradiction_errors: List[Any]) -> str:
        """
        Format errors and contradiction errors into a unified error report string.
        
        Args:
            errors: List of ProofError objects or dicts from regular error detection
            contradiction_errors: List of ProofError objects or dicts from contradiction detection
            
        Returns:
            Formatted string with clear error report, noting these are from the final answer
        """
        lines = []
        lines.append("ERROR REPORT (from Generator's Final Answer)")
        lines.append("")
        lines.append("**Note**: These errors were identified in the generator's final solution/answer, not in the Chain of Thought.")
        lines.append("")
        
        # Regular Errors
        if errors:
            lines.append("**Errors Found:**")
            for i, err in enumerate(errors, 1):
                err_type = err.type if hasattr(err, 'type') else err.get("type", "Unknown")
                err_desc = err.description if hasattr(err, 'description') else err.get("description", "")
                err_loc = err.location if hasattr(err, 'location') else err.get("location", "")
                lines.append(f"  {i}. **{err_type}**")
                lines.append(f"     Description: {err_desc}")
                lines.append(f"     Location in final answer: {err_loc}")
                lines.append("")
        
        # Contradiction Errors
        if contradiction_errors:
            lines.append("**Contradiction Errors:**")
            for i, err in enumerate(contradiction_errors, 1):
                err_type = err.type if hasattr(err, 'type') else err.get("type", "Unknown")
                err_desc = err.description if hasattr(err, 'description') else err.get("description", "")
                err_loc = err.location if hasattr(err, 'location') else err.get("location", "")
                lines.append(f"  {i}. **{err_type}**")
                lines.append(f"     Description: {err_desc}")
                lines.append(f"     Location in final answer: {err_loc}")
                lines.append("")
        
        if not errors and not contradiction_errors:
            lines.append("No errors found.")
            lines.append("")
        
        return "\n".join(lines)
    
    def _check_red_flags(
        self,
        new_chunk: str,
        finish_reason: str,
        is_thinking_complete: bool,
        consecutive_short_chunks: int,
        short_output_threshold: int,
        consecutive_threshold: int,
    ) -> bool:
        """
        Check if red flags are present that suggest solution might be complete.
        
        Args:
            new_chunk: Newly generated chunk
            finish_reason: Finish reason from generation ("stop", "length", etc.)
            is_thinking_complete: Current thinking completion status
            consecutive_short_chunks: Number of consecutive short chunks
            short_output_threshold: Token threshold for "short" chunks
            consecutive_threshold: Number of consecutive short chunks to trigger
            
        Returns:
            True if red flags detected, False otherwise
        """
        # Red flag 1: Model finished but no closing token found
        if finish_reason == "stop" and not is_thinking_complete:
            return True
        
        # Red flag 2: Multiple consecutive short chunks
        if consecutive_short_chunks >= consecutive_threshold:
            return True
        
        return False
    
    def _detect_alternative_tokens(
        self,
        text: str,
        alternative_tokens: List[str],
    ) -> Optional[Tuple[str, int]]:
        """
        Detect alternative closing tokens in text.
        
        Args:
            text: Full text to search
            alternative_tokens: List of alternative token patterns to search for
            
        Returns:
            Tuple of (token_found, position) if found, None otherwise
            Position is character index where token starts
        """
        if not text or not alternative_tokens:
            return None
        
        text_length = len(text)
        found_tokens = []
        
        for token in alternative_tokens:
            # Search for token (case-insensitive)
            pattern = re.escape(token)
            matches = list(re.finditer(pattern, text, re.IGNORECASE))
            
            for match in matches:
                position = match.start()
                # Check if token is in last 20% of text (likely closing token, not mid-reasoning)
                if position > 0.8 * text_length:
                    found_tokens.append((token, position))
        
        # Return first token found (closest to end)
        if found_tokens:
            # Sort by position (descending) to get token closest to end
            found_tokens.sort(key=lambda x: x[1], reverse=True)
            return found_tokens[0]
        
        return None
    
    def _split_into_paragraphs(self, text: str) -> List[str]:
        """
        Split text into paragraphs by double newlines.
        
        Args:
            text: Text to split
            
        Returns:
            List of paragraphs (non-empty strings)
        """
        if not text or not text.strip():
            return []
        
        # Split by double newlines and filter empty paragraphs
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        return paragraphs
    
    def _format_paragraphs_for_llm(self, paragraphs: List[str]) -> str:
        """
        Format paragraphs with "Paragraph i:" prefixes for LLM analysis.
        
        Args:
            paragraphs: List of paragraph strings
            
        Returns:
            Formatted string with numbered paragraphs
        """
        if not paragraphs:
            return ""
        
        formatted_lines = []
        for i, para in enumerate(paragraphs, 1):
            formatted_lines.append(f"Paragraph {i}: {para}")
        
        return "\n\n".join(formatted_lines)
    
    def _get_paragraph_slice_index(self, paragraphs: List[str], target_paragraph: int) -> int:
        """
        Find the index in paragraphs list that corresponds to keeping the first 'target_paragraph' paragraphs.
        
        Args:
            paragraphs: List of paragraph strings
            target_paragraph: 1-based paragraph number (keep paragraphs 1 through this, inclusive)
            
        Returns:
            Exclusive slice index (0-based) to use with paragraphs[:index]
        """
        if target_paragraph <= 0:
            return 0
        
        if target_paragraph > len(paragraphs):
            return len(paragraphs)
        
        return target_paragraph  # Simple: paragraphs are 1-indexed, list is 0-indexed
    
    def _cut_text_at_paragraph_index(
        self,
        paragraphs: List[str],
        solution_start_paragraph_index: int,
    ) -> Tuple[List[str], str]:
        """
        Cut text at paragraph index, separating CoT from solution.
        
        Args:
            paragraphs: List of paragraph strings
            solution_start_paragraph_index: 1-based paragraph number where solution starts
            
        Returns:
            Tuple of (cot_paragraphs, solution_text)
            - cot_paragraphs: List of paragraphs before solution (CoT)
            - solution_text: Solution paragraphs joined with \n\n
        """
        slice_index = self._get_paragraph_slice_index(paragraphs, solution_start_paragraph_index - 1)
        cot_paragraphs = paragraphs[:slice_index]
        solution_paragraphs = paragraphs[slice_index:]
        solution_text = "\n\n".join(solution_paragraphs)
        return cot_paragraphs, solution_text
    
    async def _detect_solution_with_llm(
        self,
        paragraphs_text: str,
        problem_statement: str,
        context: Dict[str, Any],
        run_log: RunLog,
        document_id: str,
        workflow_name: str,
    ) -> Optional[SolutionDetection]:
        """
        Use LLM to detect if solution exists after CoT using structured output.
        
        Args:
            paragraphs_text: Formatted paragraphs text with "Paragraph i:" prefixes
            problem_statement: Problem statement for context
            context: Workflow context dict
            run_log: Run log for stage execution
            document_id: Document ID
            workflow_name: Workflow name
            
        Returns:
            SolutionDetection object if detection successful, None if LLM call failed
        """
        try:
            detector_context = {
                "paragraphs_text": paragraphs_text,
                "problem_statement": problem_statement,
            }
            
            detector_result = await self._execute_stage(
                "SolutionDetector", detector_context, run_log, document_id, workflow_name
            )
            
            outputs = detector_result.get("outputs", {})
            solution_detection = outputs.get("solution_detection")
            
            # Convert to SolutionDetection if needed
            if solution_detection is not None and not isinstance(solution_detection, SolutionDetection):
                if isinstance(solution_detection, dict):
                    solution_detection = SolutionDetection(**solution_detection)
                elif hasattr(solution_detection, "model_dump"):
                    solution_detection = SolutionDetection(**solution_detection.model_dump())
            
            if solution_detection and isinstance(solution_detection, SolutionDetection):
                return solution_detection
            else:
                logger.warning("SolutionDetector returned invalid output.")
                return None
                
        except Exception as e:
            logger.error(f"SolutionDetector stage failed: {e}")
            return None

    async def _execute_stage_with_error_handling(
        self,
        stage_name: str,
        context: Dict[str, Any],
        run_log: RunLog,
        document_id: str,
        workflow_name: str,
        error_handler: WorkflowErrorHandler,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Executes a stage with retry logic based on error classification.
        """
        attempt = 0
        while True:
            attempt += 1
            try:
                return await self._execute_stage(
                    stage_name, context, run_log, document_id, workflow_name, overrides=overrides
                )
            except Exception as e:
                error_type = error_handler.classify_error(e)
                max_retries, cooldown = error_handler.get_retry_settings(error_type)
                
                if attempt <= max_retries:
                    logger.warning(
                        f"Stage {stage_name} failed (attempt {attempt}/{max_retries + 1}). "
                        f"Retrying in {cooldown}s. Error: {e}"
                    )
                    await asyncio.sleep(cooldown)
                    continue
                else:
                    # Re-raise to let caller handle critical failure or fallback
                    raise e

    def _detect_and_handle_loop(
        self, 
        cot_history: List[str], 
        buffer: str, 
        new_chunk: str,
        min_length: int = 200
    ) -> Tuple[bool, List[str], str, str]:
        """
        Detects exact-ngram repetitions in the generated content.
        Returns: (is_detected, new_cot_history, new_buffer, new_chunk)
        """
        # Reconstruct full text for analysis
        current_cot_text = self._join_cot_with_double_newlines(cot_history)
        
        # Reconstruct the exact text state: History + Buffer
        prefix = current_cot_text
        if buffer:
            # buffer is usually appended with a newline if current_cot_text exists
            prefix += ("\n" + buffer if prefix else buffer)
        
        full_text = prefix + new_chunk
        
        if len(full_text) < min_length * 2:
            return False, cot_history, buffer, new_chunk
            
        # We look for a suffix of full_text that appears earlier
        # Query length should be at least min_length
        # We search for the suffix of length 'query_len' in the text
        
        # Use a significant portion of the new content + some context for the query
        # But ensure we don't pick a query larger than half the text
        query_len = max(len(new_chunk), min_length)
        query_len = min(query_len, len(full_text) // 2)
        
        if query_len < min_length:
             return False, cot_history, buffer, new_chunk

        query = full_text[-query_len:]
        
        # Search for this query in the text strictly before the suffix start
        # search_region includes the suffix, so rfind will find the suffix itself at len-query_len
        # We want to find a previous occurrence.
        # So we search in full_text[:-1] to exclude the very last position
        
        search_region = full_text[:-1]
        match_pos = search_region.rfind(query)
        
        if match_pos != -1:
            # Found a match. Now extend backwards to find the full length of the repetition.
            suffix_start = len(full_text) - query_len
            match_start = match_pos
            
            current_match_len = query_len
            
            # Extend backwards
            while (match_start > 0 and suffix_start > 0 and 
                   full_text[match_start - 1] == full_text[suffix_start - 1]):
                match_start -= 1
                suffix_start -= 1
                current_match_len += 1
                
            if current_match_len >= min_length:
                # Loop detected!
                logger.warning(
                    f"Loop detected! Repeated sequence length: {current_match_len} chars. "
                    f"Cutting at index {suffix_start} and appending </think>."
                )
                
                # Construct the valid text
                valid_text = full_text[:suffix_start]
                
                # Rebuild cot_history from valid_text
                new_history = self._split_cot_by_double_newlines(valid_text)
                
                # Return modified state:
                # - New history reflecting the cut
                # - Empty buffer
                # - new_chunk as </think> to trigger completion
                return True, new_history, "", "</think>"
                
        return False, cot_history, buffer, new_chunk

    async def _run(
        self,
        work_item: WorkItem[GeneratorCritiquePayload],
        run_log: RunLog,
        document_id: str,
        workflow_name: str,
    ) -> Dict[str, Any]:
        payload = work_item.payload
        context: Dict[str, Any] = payload.model_dump()
        context["run_log"] = run_log
        context["solution_found"] = False  # Track if a correct solution was found
        context["already_correct"] = False  # Track if the first attempt was already correct

        # Config
        print("--------------------------------")
        print("Agent Config:")
        print(self._config)
        print("--------------------------------")
        max_iterations = self._config.get("max_iterations", 10)
        chunk_size = self._config.get("chunk_size", 2048)
        intermediate_check_interval = self._config.get("intermediate_check_interval", 10) # steps
        cot_shortening_threshold = self._config.get("cot_shortening_threshold", 25000)
        cot_shortening_factor = self._config.get("cot_shortening_factor", 0.8)

        # Initialize error handler
        error_handling_config = self._config.get("error_handling")
        error_handler = WorkflowErrorHandler(error_handling_config)

        # Run SolutionAnalyzer once (cached in context for subsequent iterations)
        if "solution_analysis" not in context:
            logger.info("Running SolutionAnalyzer to break down reference solution")
            solution_analyzer_context = {
                "problem_statement": context["problem_statement"],
                "reference_solution": context["reference_solution"],
            }
            try:
                solution_analyzer_result = await self._execute_stage_with_error_handling(
                    "SolutionAnalyzer", solution_analyzer_context, run_log, document_id, workflow_name, error_handler
                )
                solution_analysis = solution_analyzer_result.get("outputs", {}).get("solution_analysis", "")
                context["solution_analysis"] = solution_analysis  # Store as string
                logger.info("Solution analysis completed and cached")
            except StageExecutionError as err:
                # Critical stage failed - preserve state and fail workflow
                logger.error(f"SolutionAnalyzer failed: {err}")
                context["status"] = "failed"
                context["error"] = f"SolutionAnalyzer failed: {str(err)}"
                context["error_type"] = error_handler.classify_error(err).value
                context["failed_stage"] = "SolutionAnalyzer"
                return context
        else:
            logger.info("Using cached solution analysis")

        # State initialization with resumption support
        if payload.existing_cot:
            logger.info(f"Resuming workflow from previous run. Loaded {len(payload.existing_cot)} chars of CoT.")
            logger.debug(f"Existing CoT first 200 chars: {repr(payload.existing_cot[:200])}")
            # Split by double newlines to match how CoT was stored (via _join_cot_with_double_newlines)
            # This recovers the steps (empty strings were filtered out during join, so we don't need them here)
            cot_history = self._split_cot_by_double_newlines(payload.existing_cot)
            logger.debug(f"After split: {len(cot_history)} steps, first few steps: {cot_history[:3] if cot_history else 'EMPTY'}")
                
            # Ensure hint_level is an int, defaulting to 0 if None
            hint_level = payload.existing_hint_level if payload.existing_hint_level is not None else 0
            # Load previous_progress_pointer from context if available (for hint level adjustment)
            previous_progress_pointer = context.get("previous_progress_pointer")
            # Load max_progress_pointer from context if available
            max_progress_pointer = context.get("max_progress_pointer")
            
            # Overthinking retry mechanism
            overthinking_retry_count = 0  # Reset on resumption
            
            iteration_count = 0
            is_thinking_complete = False
            buffer = ""
            final_answer = ""
        else:
            cot_history: List[str] = [] # List of complete reasoning steps (lines)
            buffer: str = "" # Incomplete line buffer
            final_answer: str = ""
            is_thinking_complete: bool = False
            iteration_count: int = 0
            hint_level: int = 0
            previous_progress_pointer: Optional[Dict[str, Any]] = None  # Track previous progress pointer for hint level adjustment
            max_progress_pointer: Optional[Dict[str, Any]] = None # Track max progress pointer reached so far
            
            # Overthinking retry mechanism
            overthinking_retry_count: int = 0
        
        # Track whether hints were just injected (for token suppression)
        hints_just_injected: bool = False
        
        # Solution detection state tracking
        consecutive_short_chunks: int = 0
        last_chunks: List[str] = []  # Keep last 2-3 chunks for repetition detection
        max_last_chunks: int = 3

        logger.info(f"Starting chunked generator-critique workflow for {payload.problem_id}")
        logger.info(f"Initial CoT steps count: {len(cot_history)}")

        while iteration_count < max_iterations:
            # Check if we are done (thinking complete AND verified correct)
            # We handle final verification inside the loop to allow restarting
            
            iteration_count += 1
            
            # Prepare input for generator
            # We reconstruct the 'partial_solution' from history + buffer
            # Filter out None values to prevent join errors
            cot_history = [line for line in cot_history if line is not None]
            
            # Join lines - preserve double newlines between paragraphs
            # Empty strings in cot_history represent paragraph breaks (\n\n)
            current_cot_text = self._join_cot_with_double_newlines(cot_history)
            
            logger.info(
                f"Reconstructing partial_solution: "
                f"iteration={iteration_count}, "
                f"cot_history_lines={len(cot_history)}, "
                f"current_cot_text_length={len(current_cot_text)}, "
                f"is_thinking_complete={is_thinking_complete}, "
                f"buffer_length={len(buffer)}, "
                f"first_100_chars_of_cot={repr(current_cot_text[:100]) if current_cot_text else 'EMPTY'}, "
                f"cot_history_sample={repr(cot_history[:3]) if cot_history else 'EMPTY'}"
            )
            
            if is_thinking_complete:
                # Answer Phase: partial_solution includes CoT + separator + answer so far
                # We use a standard separator since we stripped the original one
                full_partial = current_cot_text + "\n\n</think>\n\n" + final_answer
            else:
                # Thinking Phase
                if buffer:
                    current_cot_text += "\n" + buffer if current_cot_text else buffer
                # Preserve all content
                full_partial = current_cot_text
                # CRITICAL: Ensure it ends with newline to signal continuation to clients.py
                if full_partial and not full_partial.endswith("\n"):
                    full_partial += "\n"
            
            # Ensure the partial solution starts with <think>
            # Check if we need to add the prefix
            if not full_partial or not full_partial.strip():
                # If empty or only whitespace, start fresh with thinking token
                full_partial = "<think>\n"
            elif not full_partial.strip().startswith("<think>"):
                # Has content but doesn't start with token - add prefix
                # Preserve original content, just add prefix (don't strip - preserve formatting)
                full_partial = "<think>\n" + full_partial
            
            logger.info(
                f"Setting partial_solution: length={len(full_partial)}, "
                f"first_200_chars={repr(full_partial[:200]) if full_partial else 'EMPTY'}, "
                f"last_100_chars={repr(full_partial[-100:]) if len(full_partial) > 100 else repr(full_partial)}"
            )
            
            if not full_partial or not full_partial.strip():
                logger.warning(
                    f"WARNING: partial_solution is empty! "
                    f"cot_history has {len(cot_history)} lines, "
                    f"current_cot_text length={len(current_cot_text)}"
                )
            
            # Check 1: Shorten partial_solution BEFORE GeneratorContinuation if it exceeds threshold
            # This ensures we never send an oversized input to the model
            cot_shortening_threshold = self._config.get("cot_shortening_threshold", 25000)
            cot_shortening_factor = self._config.get("cot_shortening_factor", "90%")
            
            stage_cfg_for_count = self._stage_factory._stage_configs.get("GeneratorContinuation")
            if not stage_cfg_for_count:
                stage_cfg_for_count = self._stage_factory._stage_configs.get("Generator")

            print("stage_cfg_for_count", stage_cfg_for_count)

            
            if full_partial and stage_cfg_for_count:
                model_profile = stage_cfg_for_count.model_profile or "default"
                partial_tokens = self._stage_factory._llm_client.count_tokens(full_partial, model_profile)

                print("partial_tokens", partial_tokens)
                
                if partial_tokens > cot_shortening_threshold:
                    logger.info(
                        f"Pre-Generation Shortening: partial_solution length {partial_tokens} tokens exceeds threshold {cot_shortening_threshold}. "
                        f"Running Shortener before GeneratorContinuation (iteration {iteration_count})."
                    )
                    # Extract CoT portion from partial_solution for shortening
                    # Remove prefix and any answer portion
                    cot_only = full_partial
                    if cot_only.startswith("<think>\n"):
                        cot_only = cot_only[len("<think>\n"):]
                    if "</think>" in cot_only:
                        cot_only = cot_only.split("</think>")[0]
                    
                    # Split back to list for shortening
                    cot_history_for_shortening = self._split_cot_by_double_newlines(cot_only)
                    cot_history, was_shortened = await self._shorten_cot_if_needed(
                        cot_history_for_shortening,
                        cot_shortening_threshold,
                        cot_shortening_factor,
                        context,
                        run_log,
                        document_id,
                        workflow_name,
                    )
                    if was_shortened:
                        logger.info(f"Pre-Generation Shortening complete. New CoT steps: {len(cot_history)}")
                        # Rebuild partial_solution with shortened CoT
                        current_cot_text = self._join_cot_with_double_newlines(cot_history)
                        if is_thinking_complete:
                            full_partial = current_cot_text + "\n\n</think>\n\n" + final_answer
                        else:
                            if buffer:
                                current_cot_text += "\n" + buffer if current_cot_text else buffer
                            full_partial = current_cot_text
                            if full_partial and not full_partial.endswith("\n"):
                                full_partial += "\n"
                            if not full_partial.strip().startswith("<think>"):
                                full_partial = "<think>\n" + full_partial
            
            context["partial_solution"] = full_partial
            
            # Shorten CoT during Thinking Phase if threshold exceeded
            # (Answer Phase shortening happens after injection point cut)
            if not is_thinking_complete:
                cot_shortening_threshold = self._config.get("cot_shortening_threshold", 25000)
                cot_shortening_factor = self._config.get("cot_shortening_factor", "90%")
                
                # Get model profile for token counting
                stage_name_for_profile = "Generator" if iteration_count == 1 and not full_partial else "GeneratorContinuation"
                stage_cfg = self._stage_factory._stage_configs.get(stage_name_for_profile)
                
                if stage_cfg:
                    model_profile = stage_cfg.model_profile or "default"
                    current_tokens_count = self._stage_factory._llm_client.count_tokens(full_partial, model_profile)
                    
                    print("current_tokens_count", current_tokens_count)
                    
                    if current_tokens_count > cot_shortening_threshold:
                        overthinking_retry_limit = self._config.get("overthinking_retry_limit", 5)
                        
                        # Try retry mechanism first (truncate last step) before using shortener
                        if overthinking_retry_count < overthinking_retry_limit:
                            logger.info(
                                f"Generator failed to generate closing token. CoT length {current_tokens_count} tokens "
                                f"exceeds threshold {cot_shortening_threshold} tokens. "
                                f"Retrying by truncating last step (retry {overthinking_retry_count + 1}/{overthinking_retry_limit})."
                            )
                            
                            # Truncate last step to allow model to self-correct
                            if cot_history:
                                logger.info(f"Truncating last step (Step {len(cot_history)}) to allow model to self-correct.")
                                cot_history.pop()
                            
                            # Clear buffer and reset state for retry
                            buffer = ""
                            overthinking_retry_count += 1
                            
                            # Reconstruct partial_solution after truncation
                            current_cot_text = self._join_cot_with_double_newlines(cot_history)
                            full_partial = current_cot_text
                            if full_partial and not full_partial.endswith("\n"):
                                full_partial += "\n"
                            # Ensure prefix
                            if not full_partial.strip().startswith("<think>"):
                                full_partial = "<think>\n" + full_partial
                            context["partial_solution"] = full_partial
                            
                            # Continue loop to retry generation
                            continue
                        else:
                            # Retries exhausted - generator stuck, needs more hints
                            logger.info(
                                f"Retry limit ({overthinking_retry_limit}) reached. Generator failed to generate closing token "
                                f"after {overthinking_retry_limit} retries. CoT length {current_tokens_count} tokens "
                                f"exceeds threshold {cot_shortening_threshold} tokens. Generator stuck, adding more hints."
                            )
                            
                            # Call hint generation pipeline (skip Verifier)
                            previous_progress_pointer = context.get("previous_progress_pointer")
                            translated_hint, updated_context = await self._run_hint_generation_pipeline(
                                cot_history,
                                final_answer,
                                context,
                                run_log,
                                document_id,
                                workflow_name,
                                previous_progress_pointer,
                                hint_level,
                                iteration_count,
                                error_handler
                            )
                            
                            # Update context with results
                            context.update(updated_context)
                            
                            # Get last_useful_step_index from context
                            # Fallback to actual step count (not list length, which includes empty strings)
                            actual_step_count = self._count_actual_steps(cot_history) if cot_history else 1
                            last_useful_step_index = updated_context.get("last_useful_step_index", actual_step_count)
                            if last_useful_step_index < 1:
                                last_useful_step_index = actual_step_count
                            
                            # Cut CoT at last_useful_step_index (1-based step number)
                            # Use _get_step_slice_index to account for empty strings (paragraph breaks)
                            slice_index = self._get_step_slice_index(cot_history, last_useful_step_index)
                            cot_history = cot_history[:slice_index]
                            
                            # Sanitize CoT to remove any closing thinking tokens
                            cot_history = self._sanitize_cot_list(cot_history)
                            
                            # Append translated hint and restart
                            translated_steps = updated_context.get("translated_hint_steps", [])
                            if isinstance(translated_steps, list) and translated_steps:
                                self._append_steps_list_to_cot(cot_history, translated_steps)
                                logger.info(f"Appended translated hint steps to CoT (cut at step {last_useful_step_index})")
                            elif translated_hint:
                                self._append_multi_paragraph_text_to_cot(cot_history, translated_hint)
                                logger.info(f"Appended translated hint to CoT (cut at step {last_useful_step_index})")
                            
                            # Set flag to suppress </think> token in next generation
                            hints_just_injected = True
                            
                            buffer = ""
                            previous_progress_pointer = updated_context.get("current_progress_pointer")
                            context["previous_progress_pointer"] = previous_progress_pointer
                            hint_level = updated_context.get("hint_level", hint_level)
                            is_thinking_complete = False
                            final_answer = ""
                            overthinking_retry_count = 0  # Reset for new attempt with hints
                            
                            # Continue loop to regenerate with new hints
                            continue
            
            # Select Stage
            stage_name = "Generator" if iteration_count == 1 and not full_partial else "GeneratorContinuation"
            
            logger.info(
                f"Generating chunk {iteration_count} (Stage: {stage_name}), "
                f"partial_solution_length={len(full_partial)}, "
                f"chunk_size={chunk_size}"
            )
            logger.debug(
                f"About to execute stage '{stage_name}' with context keys: {list(context.keys())}, "
                f"partial_solution preview: {repr(full_partial[:200])}"
            )
            
            # Build overrides dict
            overrides = {"max_tokens": chunk_size}
            
            # Add token suppression if hints were just injected
            silence_tokens = self._config.get("silence_redacted_reasoning_tokens", 0)
            if hints_just_injected and silence_tokens > 0:
                # Suppress both closing and opening tokens to force content generation
                # Also suppress pivoting words (both cases) to force model to follow the hint
                overrides["_suppress_token"] = [
                    "</think>", "<think>",
                    "Wait", "wait",
                    "But", "but",
                    "However", "however",
                    "Alternatively", "alternatively",
                    "Actually", "actually",
                    "Hold", "hold",
                    "Maybe", "maybe",
                    "Instead", "instead",
                    "No", "no",
                    "Oops", "oops",
                    "Ah", "ah"
                ]
                overrides["_suppress_for_tokens"] = silence_tokens
                logger.info(
                    f"Suppressing </think>, <think> and pivoting tokens for {silence_tokens} tokens "
                    f"after hint injection (stage: {stage_name})"
                )
            
            try:
                result = await self._execute_stage_with_error_handling(
                    stage_name, context, run_log, document_id, workflow_name, error_handler,
                    overrides=overrides
                )
                
                # Reset flag after first generation call that uses it
                if hints_just_injected:
                    hints_just_injected = False
            except StageExecutionError as err:
                # Critical stage failed - preserve state and fail workflow
                logger.error(
                    f"Generator stage '{stage_name}' failed after error handling for {payload.problem_id} "
                    f"(iteration {iteration_count}): {err}"
                )
                # Preserve state before failing
                context["status"] = "failed"
                context["error"] = f"Generator stage failed: {str(err)}"
                context["error_type"] = error_handler.classify_error(err).value
                context["failed_stage"] = stage_name
                context["generator_cot"] = self._join_cot_with_double_newlines(cot_history) if cot_history else ""
                context["generator_answer"] = final_answer
                context["iteration_count"] = iteration_count
                context["hint_level"] = hint_level
                return context
            
            new_chunk = result.get("outputs", {}).get("generator_solution", "")
            finish_reason = result.get("finish_reason")

            # --- Loop Detection Logic ---
            if not is_thinking_complete:
                loop_config = self._config.get("loop_detection", {})
                if loop_config.get("enabled", False):
                    min_loop_len = loop_config.get("min_length", 200)
                    loop_detected, cot_history, buffer, new_chunk = self._detect_and_handle_loop(
                        cot_history, buffer, new_chunk, min_loop_len
                    )
                    if loop_detected:
                        logger.warning("Thinking Loop detected! Truncated and appended </think>.")

            if is_thinking_complete:
                # Append to answer
                final_answer += new_chunk
                
                if finish_reason == "stop":
                    # Answer is complete. Run 4-stage critique pipeline.
                    logger.info("Answer generation complete (EOS). Running 4-stage critique pipeline.")
                    
                    # Get previous progress pointer from context
                    previous_progress_pointer = context.get("previous_progress_pointer")
                    
                    should_accept, translated_hint, updated_context = await self._run_new_critique_pipeline(
                        cot_history, final_answer, context, run_log, document_id, workflow_name,
                        previous_progress_pointer, hint_level, iteration_count, error_handler
                    )
                    context.update(updated_context)
                    
                    if should_accept:
                        break  # Success!
                    else:
                        # Solution incorrect - append hint and restart
                        logger.info("Solution verified INCORRECT. Appending hint and restarting.")
                        
                        # Get last_useful_step_index from context (set by pipeline)
                        # Fallback to actual step count (not list length, which includes empty strings)
                        actual_step_count = self._count_actual_steps(cot_history) if cot_history else 1
                        last_useful_step_index = updated_context.get("last_useful_step_index", actual_step_count)
                        if last_useful_step_index < 1:
                            last_useful_step_index = actual_step_count
                        
                        # Cut CoT at last_useful_step_index (1-based step number)
                        # Use _get_step_slice_index to account for empty strings (paragraph breaks)
                        slice_index = self._get_step_slice_index(cot_history, last_useful_step_index)
                        cot_history = cot_history[:slice_index]
                        
                        translated_steps = updated_context.get("translated_hint_steps", [])
                        if isinstance(translated_steps, list) and translated_steps:
                            # Append translated hint steps to cut CoT
                            self._append_steps_list_to_cot(cot_history, translated_steps)
                            logger.info(f"Appended translated hint steps to CoT (cut at step {last_useful_step_index})")
                        elif translated_hint:
                            # Append translated hint to cut CoT
                            self._append_multi_paragraph_text_to_cot(cot_history, translated_hint)
                            logger.info(f"Appended translated hint to CoT (cut at step {last_useful_step_index})")
                        
                        # Set flag to suppress </think> token in next generation
                        hints_just_injected = True
                        
                        # Shorten CoT if needed after cutting and hint injection
                        cot_shortening_threshold = self._config.get("cot_shortening_threshold", 25000)
                        cot_shortening_factor = self._config.get("cot_shortening_factor", "90%")
                        cot_history, was_shortened = await self._shorten_cot_if_needed(
                            cot_history,
                            cot_shortening_threshold,
                            cot_shortening_factor,
                            context,
                            run_log,
                            document_id,
                            workflow_name,
                        )
                        if was_shortened:
                            logger.info(f"CoT shortened after hint injection (Answer Phase). New CoT steps: {len(cot_history)}")
                        
                        # Update state for next iteration
                        buffer = ""
                        previous_progress_pointer = updated_context.get("current_progress_pointer")
                        context["previous_progress_pointer"] = previous_progress_pointer
                        hint_level = updated_context.get("hint_level", hint_level)
                        is_thinking_complete = False
                        final_answer = ""
                        overthinking_retry_count = 0  # Reset retry count after hint injection (Answer Phase)
                else:
                    logger.info("Answer incomplete (max tokens). Continuing generation.")
                    continue

            else:
                # Thinking Phase
                # Check for thinking completion in this chunk
                cot_part, solution_part = self._extract_cot_and_solution(new_chunk)
                
                if "</think>" in new_chunk or "`</think>`" in new_chunk:
                    is_thinking_complete = True
                    overthinking_retry_count = 0  # Reset retry count when thinking completes
                    
                    full_new_cot = buffer + cot_part
                    # Preserve empty strings (they represent \n\n paragraph breaks)
                    new_steps = []
                    for line in full_new_cot.split('\n'):
                        if line == "":
                            # Empty string represents paragraph break - preserve it
                            new_steps.append("")
                        elif line.strip():
                            # Non-empty line with content - clean and add
                            cleaned = self._clean_step_content(line)
                            if cleaned:  # Only add if cleaning didn't make it empty
                                new_steps.append(cleaned)
                    cot_history.extend(new_steps)
                    buffer = "" 
                    
                    # Check 2: Shorten combined CoT AFTER GeneratorContinuation if it exceeds threshold
                    cot_shortening_threshold = self._config.get("cot_shortening_threshold", 25000)
                    cot_shortening_factor = self._config.get("cot_shortening_factor", "90%")
                    
                    stage_cfg_for_count = self._stage_factory._stage_configs.get("GeneratorContinuation")
                    if not stage_cfg_for_count:
                        stage_cfg_for_count = self._stage_factory._stage_configs.get("Generator")
                    
                    if cot_history and stage_cfg_for_count:
                        combined_cot_text = self._join_cot_with_double_newlines(cot_history)
                        combined_partial = "<think>\n" + combined_cot_text
                        model_profile = stage_cfg_for_count.model_profile or "default"
                        combined_tokens = self._stage_factory._llm_client.count_tokens(combined_partial, model_profile)
                        
                        if combined_tokens > cot_shortening_threshold:
                            logger.info(
                                f"Post-Generation Shortening: Combined CoT length {combined_tokens} tokens exceeds threshold {cot_shortening_threshold}. "
                                f"Running Shortener after GeneratorContinuation (iteration {iteration_count})."
                            )
                            cot_history, was_shortened = await self._shorten_cot_if_needed(
                                cot_history,
                                cot_shortening_threshold,
                                cot_shortening_factor,
                                context,
                                run_log,
                                document_id,
                                workflow_name,
                            )
                            if was_shortened:
                                logger.info(f"Post-Generation Shortening complete. New CoT steps: {len(cot_history)}")
                    
                    final_answer = solution_part
                    logger.info(f"Thinking complete. Answer part length: {len(final_answer)}")
                    
                    if finish_reason == "stop":
                        # Answer generation complete in same chunk as </think>. Run 4-stage critique pipeline.
                        logger.info("Answer generation complete (EOS) in same chunk as </think>. Running 4-stage critique pipeline.")
                        
                        # Get previous progress pointer from context
                        previous_progress_pointer = context.get("previous_progress_pointer")
                        
                        should_accept, translated_hint, updated_context = await self._run_new_critique_pipeline(
                            cot_history, final_answer, context, run_log, document_id, workflow_name,
                            previous_progress_pointer, hint_level, iteration_count, error_handler
                        )
                        context.update(updated_context)
                        
                        if should_accept:
                            break  # Success!
                        else:
                            # Solution incorrect - append hint and restart
                            logger.info("Solution verified INCORRECT. Appending hint and restarting.")
                            
                            # Get last_useful_step_index from context (set by pipeline)
                            # Use actual step count for fallback (not list length)
                            actual_step_count = self._count_actual_steps(cot_history) if cot_history else 1
                            last_useful_step_index = updated_context.get("last_useful_step_index", actual_step_count)
                            if last_useful_step_index < 1:
                                last_useful_step_index = actual_step_count
                            
                            # Cut CoT at last_useful_step_index (1-based step number)
                            # Use _get_step_slice_index to account for empty strings (paragraph breaks)
                            slice_index = self._get_step_slice_index(cot_history, last_useful_step_index)
                            cot_history = cot_history[:slice_index]
                            
                            # Sanitize CoT to remove any closing thinking tokens
                            cot_history = self._sanitize_cot_list(cot_history)
                            
                            translated_steps = updated_context.get("translated_hint_steps", [])
                            if isinstance(translated_steps, list) and translated_steps:
                                # Append translated hint steps to cut CoT
                                self._append_steps_list_to_cot(cot_history, translated_steps)
                                logger.info(f"Appended translated hint steps to CoT (cut at step {last_useful_step_index})")
                            elif translated_hint:
                                # Append translated hint to cut CoT
                                self._append_multi_paragraph_text_to_cot(cot_history, translated_hint)
                                logger.info(f"Appended translated hint to CoT (cut at step {last_useful_step_index})")
                            
                            # Set flag to suppress </think> token in next generation
                            hints_just_injected = True
                            
                            # Shorten CoT if needed after cutting and hint injection
                            cot_shortening_threshold = self._config.get("cot_shortening_threshold", 25000)
                            cot_shortening_factor = self._config.get("cot_shortening_factor", "90%")
                            cot_history, was_shortened = await self._shorten_cot_if_needed(
                                cot_history,
                                cot_shortening_threshold,
                                cot_shortening_factor,
                                context,
                                run_log,
                                document_id,
                                workflow_name,
                            )
                            if was_shortened:
                                logger.info(f"CoT shortened after hint injection (Thinking Phase). New CoT steps: {len(cot_history)}")
                            
                            # Update state for next iteration
                            buffer = ""
                            previous_progress_pointer = updated_context.get("current_progress_pointer")
                            context["previous_progress_pointer"] = previous_progress_pointer
                            hint_level = updated_context.get("hint_level", hint_level)
                            is_thinking_complete = False
                            final_answer = ""
                            overthinking_retry_count = 0  # Reset retry count after hint injection (Thinking Phase)
                    else:
                        logger.info("Answer incomplete (max tokens) after </think>. Continuing generation.")
                        continue
                else:
                    # Still thinking
                    full_new_text = buffer + new_chunk
                    lines = full_new_text.split('\n')
                    
                    if len(lines) > 0:
                        # Preserve empty strings (they represent \n\n paragraph breaks)
                        # Only filter out lines that are None or have content but are just whitespace
                        new_steps = []
                        for l in lines[:-1]:  # Exclude last line (incomplete, will be buffered)
                            if l is None:
                                continue
                            elif l == "":
                                # Empty string represents paragraph break - preserve it
                                new_steps.append("")
                            elif l.strip():
                                # Non-empty line with content - clean and add
                                cleaned = self._clean_step_content(l)
                                if cleaned:  # Only add if cleaning didn't make it empty
                                    new_steps.append(cleaned)
                        # Drop last incomplete line (as per original logic)
                        buffer = lines[-1] if len(lines) > 0 else ""
                        cot_history.extend(new_steps)
                    
                    # Check 3: Shorten combined CoT AFTER GeneratorContinuation (still thinking case)
                    cot_shortening_threshold = self._config.get("cot_shortening_threshold", 25000)
                    cot_shortening_factor = self._config.get("cot_shortening_factor", "90%")
                    
                    stage_cfg_for_count = self._stage_factory._stage_configs.get("GeneratorContinuation")
                    if not stage_cfg_for_count:
                        stage_cfg_for_count = self._stage_factory._stage_configs.get("Generator")
                    
                    if cot_history and stage_cfg_for_count:
                        combined_cot_text = self._join_cot_with_double_newlines(cot_history)
                        combined_partial = "<think>\n" + combined_cot_text
                        model_profile = stage_cfg_for_count.model_profile or "default"
                        combined_tokens = self._stage_factory._llm_client.count_tokens(combined_partial, model_profile)
                        
                        if combined_tokens > cot_shortening_threshold:
                            logger.info(
                                f"Post-Generation Shortening (Thinking): Combined CoT length {combined_tokens} tokens exceeds threshold {cot_shortening_threshold}. "
                                f"Running Shortener after GeneratorContinuation (iteration {iteration_count})."
                            )
                            cot_history, was_shortened = await self._shorten_cot_if_needed(
                                cot_history,
                                cot_shortening_threshold,
                                cot_shortening_factor,
                                context,
                                run_log,
                                document_id,
                                workflow_name,
                            )
                            if was_shortened:
                                logger.info(f"Post-Generation Shortening (Thinking) complete. New CoT steps: {len(cot_history)}")
                    
                    logger.info(f"Chunk generated. Current CoT steps: {len(cot_history)}")
                    
                    # Solution Detection: Check for red flags
                    solution_detection_config = self._config.get("solution_detection", {})
                    if solution_detection_config.get("enabled", True) and not is_thinking_complete:
                        # Update state tracking
                        chunk_tokens = self._stage_factory._llm_client.count_tokens(new_chunk, model_profile)
                        short_output_threshold = solution_detection_config.get("short_output_threshold", 30)
                        
                        if chunk_tokens < short_output_threshold:
                            consecutive_short_chunks += 1
                        else:
                            consecutive_short_chunks = 0
                        
                        # Keep last N chunks for repetition detection
                        last_chunks.append(new_chunk)
                        if len(last_chunks) > max_last_chunks:
                            last_chunks.pop(0)
                        
                        # Check red flags
                        consecutive_threshold = solution_detection_config.get("consecutive_short_chunks_threshold", 2)
                        red_flags = self._check_red_flags(
                            new_chunk, finish_reason, is_thinking_complete,
                            consecutive_short_chunks, short_output_threshold, consecutive_threshold
                        )
                        
                        if red_flags:
                            logger.info("Red flags detected: Checking for solution completion...")
                            
                            # Get full text (buffer + new_chunk + cot_history)
                            full_cot_text = self._join_cot_with_double_newlines(cot_history)
                            full_text = full_cot_text + "\n\n" + buffer + new_chunk if buffer else full_cot_text + "\n\n" + new_chunk
                            
                            # Step 1: Check for alternative tokens
                            alternative_tokens = solution_detection_config.get("alternative_tokens", [])
                            token_result = self._detect_alternative_tokens(full_text, alternative_tokens)
                            
                            if token_result:
                                token_found, token_position = token_result
                                logger.info(f"Alternative token '{token_found}' found at position {token_position}. Extracting solution...")
                                
                                # Extract CoT and solution at token position
                                cot_part = full_text[:token_position].strip()
                                solution_part = full_text[token_position + len(token_found):].strip()
                                
                                # Convert CoT part back to list format (split by \n\n, then by \n for steps)
                                # This matches how cot_history is structured
                                cot_history_new = []
                                cot_paragraphs = self._split_into_paragraphs(cot_part)
                                for para in cot_paragraphs:
                                    # Split paragraph into lines and add as steps
                                    para_lines = para.split('\n')
                                    for line in para_lines:
                                        if line.strip():
                                            cleaned = self._clean_step_content(line)
                                            if cleaned:
                                                cot_history_new.append(cleaned)
                                    # Add paragraph break between paragraphs
                                    if cot_history_new:  # Only if we have content
                                        cot_history_new.append("")
                                
                                # Remove trailing empty string if present
                                if cot_history_new and cot_history_new[-1] == "":
                                    cot_history_new.pop()
                                
                                # Update state
                                cot_history = cot_history_new
                                final_answer = solution_part
                                is_thinking_complete = True
                                buffer = ""
                                consecutive_short_chunks = 0
                                last_chunks = []
                                
                                logger.info(f"Solution extracted via alternative token. Answer length: {len(final_answer)}")
                                continue  # Skip to next iteration to process answer
                            
                            # Step 2: No token found - use LLM detection with structured output
                            else:
                                logger.info("No alternative token found. Using LLM to detect solution...")
                                
                                # Split into paragraphs
                                paragraphs = self._split_into_paragraphs(full_text)
                                
                                if len(paragraphs) > 0:
                                    # Format for LLM
                                    paragraphs_text = self._format_paragraphs_for_llm(paragraphs)
                                    
                                    # Call LLM detection (returns SolutionDetection object)
                                    solution_detection = await self._detect_solution_with_llm(
                                        paragraphs_text,
                                        context["problem_statement"],
                                        context,
                                        run_log,
                                        document_id,
                                        workflow_name,
                                    )
                                    
                                    if solution_detection and solution_detection.has_solution:
                                        solution_start_para = solution_detection.solution_start_paragraph_index
                                        logger.info(
                                            f"LLM detected solution starting at paragraph {solution_start_para}. "
                                            f"Reasoning: {solution_detection.reasoning}"
                                        )
                                        
                                        # Validate paragraph index
                                        if solution_start_para > len(paragraphs):
                                            logger.warning(
                                                f"Invalid paragraph index {solution_start_para} (max {len(paragraphs)}). "
                                                f"Using last paragraph as fallback."
                                            )
                                            solution_start_para = len(paragraphs)
                                        
                                        # Cut at paragraph index
                                        cot_paragraphs, solution_text = self._cut_text_at_paragraph_index(
                                            paragraphs, solution_start_para
                                        )
                                        
                                        # Convert CoT paragraphs back to list format (split by \n for steps)
                                        # This matches how cot_history is structured
                                        cot_history_new = []
                                        for para in cot_paragraphs:
                                            # Split paragraph into lines and add as steps
                                            para_lines = para.split('\n')
                                            for line in para_lines:
                                                if line.strip():
                                                    cleaned = self._clean_step_content(line)
                                                    if cleaned:
                                                        cot_history_new.append(cleaned)
                                            # Add paragraph break between paragraphs
                                            if cot_history_new:  # Only if we have content
                                                cot_history_new.append("")
                                        
                                        # Remove trailing empty string if present
                                        if cot_history_new and cot_history_new[-1] == "":
                                            cot_history_new.pop()
                                        
                                        # Update state
                                        cot_history = cot_history_new
                                        final_answer = solution_text
                                        is_thinking_complete = True
                                        buffer = ""
                                        consecutive_short_chunks = 0
                                        last_chunks = []
                                        
                                        logger.info(f"Solution extracted via LLM detection. Answer length: {len(final_answer)}")
                                        continue  # Skip to next iteration to process answer
                                    elif solution_detection:
                                        logger.info(f"LLM detected no solution. Reasoning: {solution_detection.reasoning}")
                                    else:
                                        logger.warning("LLM detection failed. Continuing generation...")
                                else:
                                    logger.warning("No paragraphs found in text. Skipping LLM detection.")
                    
                    # CRITICAL CHECK: If model finished (stop) but no closing token found,
                    # Handle it programmatically (truncate and retry) without calling Critique
                    if finish_reason == "stop" and not is_thinking_complete:
                        logger.warning("Model finished generation but no </think> token found. Handling missing token error programmatically.")
                        
                        # Truncate the last step, assuming the error happened at the end
                        # (e.g. trailed off or hallucinated structure without closing)
                        if cot_history:
                            logger.info(f"Truncating last step (Step {len(cot_history)}) due to missing closing token.")
                            cot_history.pop() 
                        
                        # Do NOT append a text hint to cot_history as it would appear as the assistant's own thought.
                        # Simply truncating and retrying is often sufficient for the model to self-correct formatting.
                        
                        # Reset state for retry
                        buffer = ""
                        is_thinking_complete = False
                        final_answer = ""
                        
                        # Increase hint level slightly (internal tracking)
                        hint_level = min(4, hint_level + 1)
                        
                        continue  # Restart loop to regenerate
                    
                    # No intermediate critique - just continue generating chunks until </think> token
                    logger.info("Continuing generation (no intermediate critique).")


        # Final Check (if we exited loop due to max iterations)
        if iteration_count >= max_iterations and context.get("status") != "completed":
            context["status"] = "failed"
            context["generator_cot"] = self._join_cot_with_double_newlines(cot_history)
            context["generator_answer"] = final_answer

        context["hint_level"] = hint_level  # Ensure hint_level is saved for resumption
        context["iteration_count"] = iteration_count
        return context


