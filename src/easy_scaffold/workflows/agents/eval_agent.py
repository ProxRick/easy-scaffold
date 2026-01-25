# src/easy_scaffold/workflows/agents/eval_agent.py
import asyncio
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

try:
    from sympy import sympify, N, Rational, Float
    from sympy.core.sympify import SympifyError
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False

from easy_scaffold.db.pydantic_models import RunLog
from easy_scaffold.workflows.base import AbstractWorkflow
from easy_scaffold.workflows.workflow_models import EvalPayload, WorkItem

logger = logging.getLogger(__name__)


class EvalWorkflow(AbstractWorkflow[EvalPayload]):
    """
    Workflow for evaluating model performance.
    Supports:
    - Model-specific results storage
    - Task-specific evaluation types (math vs open-ended)
    - Pass@k metrics via incremental sampling
    - Caching of generated responses
    """

    def _validate_initial_cot_string(self, initial_string: str) -> bool:
        """
        Validate initial CoT string for potential issues.
        
        Args:
            initial_string: The initial CoT string to validate
            
        Returns:
            True if valid, False if warnings issued
        """
        if not initial_string:
            return True
        
        # Check for closing thinking tags that might cause issues
        closing_tags = ["</think>", "`</think>`"]
        found_tags = [tag for tag in closing_tags if tag in initial_string]
        
        if found_tags:
            logger.warning(
                f"Initial CoT string contains closing tags: {found_tags}. "
                f"This may cause issues with thinking block extraction. "
                f"Consider removing these tags from the initial string."
            )
            return False
        
        return True
    
    def _extract_generated_content(self, full_output: str, initial_string: str) -> str:
        """
        Extract only the newly generated content, removing the initial_string prefix.
        
        When using completion_mode with initial CoT string, the model output includes
        the initial_string + newly generated content. This method extracts only the new content.
        
        Args:
            full_output: Full model output including initial_string
            initial_string: The initial CoT string that was prefilled
            
        Returns:
            Only the newly generated content (after initial_string)
        """
        if not initial_string:
            return full_output
        
        # Strategy 1: Check if output starts with initial_string
        if full_output.startswith(initial_string):
            generated_content = full_output[len(initial_string):].strip()
            logger.debug(f"Extracted generated content (starts with initial_string): {len(generated_content)} chars")
            return generated_content
        
        # Strategy 2: Find initial_string in output (might be embedded)
        idx = full_output.find(initial_string)
        if idx >= 0:
            generated_content = full_output[idx + len(initial_string):].strip()
            logger.debug(f"Extracted generated content (found initial_string at position {idx}): {len(generated_content)} chars")
            return generated_content
        
        # Strategy 3: Couldn't find initial_string - return full output
        # This shouldn't happen in normal operation, but handle gracefully
        logger.warning(
            f"Could not find initial_string in output. "
            f"Initial string length: {len(initial_string)}, "
            f"Output length: {len(full_output)}. "
            f"Returning full output."
        )
        return full_output
    
    def _extract_response_from_thinking_model(self, full_output: str) -> Tuple[str, bool]:
        """
        Extract response from thinking model output.
        Thinking models output CoT in <think>...</think> blocks.
        This method extracts everything after the closing tag as the response.
        
        Args:
            full_output: Full model output including CoT and response
            
        Returns:
            Tuple of (response_text, has_valid_closing_tag)
            - response_text: Response text (everything after closing thinking tag), or empty if no tag found
            - has_valid_closing_tag: True if closing tag was found, False otherwise
        """
        if not full_output:
            return "", False
        
        # Patterns for thinking block closing tags (try most specific first)
        # Support both backtick-wrapped and plain patterns
        thinking_end_patterns = [
            r'`</think>`',   # Backtick-wrapped (most specific)
            r'</think>',     # Plain tag
        ]
        
        # Find the last occurrence of any closing tag
        last_match = None
        last_pos = -1
        
        for pattern in thinking_end_patterns:
            matches = list(re.finditer(pattern, full_output, re.IGNORECASE))
            if matches:
                # Get the last match
                match = matches[-1]
                if match.end() > last_pos:
                    last_match = match
                    last_pos = match.end()
        
        if last_match:
            # Extract everything after the closing tag
            response = full_output[last_pos:].strip()
            logger.debug(f"Extracted response from thinking model (CoT length: {last_pos}, Response length: {len(response)})")
            return response, True
        
        # No thinking tag found - return empty response to indicate invalid format
        logger.warning(
            f"No thinking block closing tag found in output. "
            f"Output length: {len(full_output)} chars. "
            f"This indicates an incomplete or malformed generation. "
            f"Returning empty response to trigger default scoring."
        )
        return "", False

    def _extract_boxed_answer(self, text: str) -> Optional[str]:
        """
        Extract content inside \boxed{...} with proper handling of nested braces.
        Uses balanced brace matching to handle cases like \boxed{\dfrac{652}{3}}.
        """
        if not text:
            return None
        
        # Find all \boxed{ occurrences
        boxed_pattern = r'\\boxed\{'
        matches = list(re.finditer(boxed_pattern, text))
        
        if not matches:
            # Try without backslash
            boxed_pattern = r'boxed\{'
            matches = list(re.finditer(boxed_pattern, text, re.IGNORECASE))
        
        if not matches:
            return None
        
        # For each match, find the matching closing brace using balanced brace counting
        for match in matches:
            start_pos = match.end()  # Position after opening brace
            brace_count = 1
            pos = start_pos
            
            while pos < len(text) and brace_count > 0:
                if text[pos] == '{':
                    brace_count += 1
                elif text[pos] == '}':
                    brace_count -= 1
                pos += 1
            
            if brace_count == 0:
                # Found matching closing brace
                content = text[start_pos:pos-1].strip()  # pos-1 to exclude closing brace
                if content:
                    return content
        
        return None

    def _normalize_answer(self, answer: str) -> str:
        """
        Normalize answer string for comparison.
        - Strips whitespace
        - Removes spaces
        - Removes leading zeros from integers (e.g., "025" -> "25")
        """
        if not answer:
            return ""
        
        normalized = answer.strip().replace(" ", "")
        
        # Remove leading zeros from pure integer strings (but preserve "0")
        # Pattern: starts with optional sign, then zeros, then digits
        match = re.match(r'^([-+]?)0+(\d+)$', normalized)
        if match:
            sign = match.group(1)
            digits = match.group(2)
            if digits:  # If there are digits after zeros
                normalized = sign + digits
            else:
                # All zeros, keep as "0"
                normalized = "0"
        
        return normalized

    def _symbolic_check(self, generated_answer: str, ground_truth: str) -> bool:
        """
        Check if generated answer matches ground truth using symbolic math comparison.
        
        First tries string normalization (handles leading zeros, whitespace).
        Then tries numeric comparison using SymPy for mathematical equivalence
        (handles fractions, decimals, scientific notation, etc.).
        
        Args:
            generated_answer: The answer extracted from model output
            ground_truth: The expected correct answer
            
        Returns:
            True if answers are mathematically equivalent, False otherwise
        """
        # Step 1: Try normalized string comparison (fast path)
        norm_gen = self._normalize_answer(generated_answer)
        norm_gt = self._normalize_answer(ground_truth)
        if norm_gen == norm_gt:
            return True
        
        # Step 2: Try numeric comparison using SymPy (handles mathematical equivalence)
        if SYMPY_AVAILABLE:
            try:
                # Try to parse both as mathematical expressions
                gen_expr = sympify(norm_gen, evaluate=False)
                gt_expr = sympify(norm_gt, evaluate=False)
                
                # Compare numerically (handles fractions, decimals, etc.)
                # Use high precision to avoid floating point issues
                gen_val = N(gen_expr, 50)
                gt_val = N(gt_expr, 50)
                
                # Check if they're numerically equal
                if abs(float(gen_val - gt_val)) < 1e-10:
                    logger.debug(f"Mathematical equivalence: '{generated_answer}' == '{ground_truth}'")
                    return True
                    
            except (SympifyError, ValueError, TypeError) as e:
                # Parsing failed - fall back to string comparison
                logger.debug(f"Could not parse as math expressions: {e}. Using string comparison.")
                pass
            except (AttributeError, NotImplementedError) as e:
                # SymPy cannot evaluate (e.g., FiniteSet, unsupported types) - re-raise to trigger judge fallback
                logger.debug(
                    f"SymPy cannot evaluate expressions (e.g., contains FiniteSet or unsupported types): {e}. "
                    f"Will fall back to judge evaluation. Generated: '{generated_answer}', Ground truth: '{ground_truth}'"
                )
                # Re-raise so caller knows to fall back to judge
                raise
        
        # Step 3: Fallback to normalized string comparison
        return norm_gen == norm_gt

    async def _generate_samples(
        self,
        context: Dict[str, Any],
        run_log: RunLog,
        document_id: str,
        workflow_name: str,
        num_needed: int,
    ) -> List[str]:
        """Generate N new solution samples."""
        from easy_scaffold.common.custom_exceptions import EmptyResponseException
        
        # Check if completion_mode with initial CoT is enabled
        completion_config = self._config.get("completion_mode", {})
        use_completion_mode = completion_config.get("enabled", False)
        initial_cot_string = completion_config.get("initial_cot_string", "").strip()
        
        # Validate initial string if provided
        if use_completion_mode and initial_cot_string:
            self._validate_initial_cot_string(initial_cot_string)
        
        # Determine stage name and prepare context
        if use_completion_mode and initial_cot_string:
            # Use GeneratorContinuation with partial_solution
            stage_name = "GeneratorContinuation"
            # Construct partial_solution with <think> prefix (added automatically)
            # The initial_string should NOT include <think> prefix
            # Match the format used in generator_critique_agent.py
            partial_solution = f"<think>\n{initial_cot_string}"
            # Ensure it ends with newline for proper continuation
            if not partial_solution.endswith("\n"):
                partial_solution += "\n"
            
            logger.info(
                f"Using completion_mode with initial CoT string. "
                f"Initial string length: {len(initial_cot_string)} chars, "
                f"Partial solution length: {len(partial_solution)} chars"
            )
            
            # Create context copy with partial_solution
            generation_context = context.copy()
            generation_context["partial_solution"] = partial_solution
        else:
            # Use regular Generator stage
            stage_name = "Generator"
            generation_context = context
            if use_completion_mode and not initial_cot_string:
                logger.warning(
                    "completion_mode.enabled is True but initial_cot_string is empty. "
                    "Falling back to regular Generator stage."
                )
        
        new_samples = []
        
        # Parallel generation
        tasks = []
        for i in range(num_needed):
            tasks.append(
                self._execute_stage(
                    stage_name, generation_context, run_log, document_id, workflow_name
                )
            )
        
        # Use return_exceptions=True to handle individual failures gracefully
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for idx, res in enumerate(results):
            # Check if this result is an exception
            if isinstance(res, Exception):
                # Check if it's an EmptyResponseException (directly or as cause)
                is_empty_response = (
                    isinstance(res, EmptyResponseException) or
                    "Empty response" in str(res) or
                    "EmptyResponseException" in type(res).__name__ or
                    (hasattr(res, "__cause__") and isinstance(res.__cause__, EmptyResponseException))
                )
                
                if is_empty_response:
                    logger.error(
                        f"Document {document_id}: Generator sample {idx + 1}/{num_needed} "
                        f"returned empty response after max retries. Skipping this sample. "
                        f"Error: {res}"
                    )
                    new_samples.append("") # Add empty string to maintain count
                else:
                    # Re-raise unexpected exceptions
                    logger.error(
                        f"Document {document_id}: Generator sample {idx + 1}/{num_needed} "
                        f"failed with unexpected error: {type(res).__name__}: {res}"
                    )
                    raise res
            else:
                # Normal result processing
                full_output = res.get("outputs", {}).get("generator_solution", "")
                
                # Extract generated content if using completion_mode
                if use_completion_mode and initial_cot_string and full_output:
                    # Remove initial_string from output to get only newly generated content
                    output = self._extract_generated_content(full_output, initial_cot_string)
                    logger.debug(
                        f"Sample {idx + 1}: Extracted {len(output)} chars from {len(full_output)} chars "
                        f"(removed {len(initial_cot_string)} char initial_string)"
                    )
                else:
                    output = full_output
                
                if output:
                    new_samples.append(output)
                else:
                    logger.warning(
                        f"Document {document_id}: Generator sample {idx + 1}/{num_needed} "
                        f"produced empty output during sampling."
                    )
                    new_samples.append("") # Keep empty string to maintain count or handle failure
                
        return new_samples

    async def _evaluate_sample(
        self,
        sample: str,
        context: Dict[str, Any],
        task_type: str,
        judge_stage_name: str,
        ground_truth: Optional[str],
        run_log: RunLog,
        document_id: str,
        workflow_name: str,
    ) -> Dict[str, Any]:
        """Evaluate a single solution sample."""
        if not sample:
            return {"correct": False, "reasoning": "Empty generation", "method": "basic_check"}

        # Extract response from thinking model (if applicable)
        # This removes CoT and keeps only the final response for judging
        response_text, has_valid_closing_tag = self._extract_response_from_thinking_model(sample)
        
        # If no valid closing tag found, return default failure values without sending to judge
        # This prevents sending long loopy outputs that cause empty responses from Gemini
        if not has_valid_closing_tag:
            logger.warning(
                f"Sample missing </think> closing tag. "
                f"Returning default failure values without judge evaluation. "
                f"Sample length: {len(sample)} chars"
            )
            
            if task_type == "grader":
                return {
                    "score": 0,
                    "classification": "Incomplete",
                    "reasoning": "Generation missing </think> closing tag - incomplete or malformed output",
                    "designed_marking_scheme": {},
                    "feedback": {},
                    "method": "missing_closing_tag"
                }
            else:
                return {
                    "correct": False,
                    "reasoning": "Generation missing </think> closing tag - incomplete or malformed output",
                    "method": "missing_closing_tag"
                }
        
        has_thinking = response_text != sample.strip()  # Check if extraction occurred
        
        if has_thinking:
            logger.debug(f"Extracted response from thinking model (original length: {len(sample)}, response length: {len(response_text)})")
        else:
            logger.debug("No thinking block detected, using full output as response")

        # Case 1: Math Problem - Extract and Compare Final Answer
        if task_type == "math_correctness":
            # Step 1: Try to extract boxed answer from response (not CoT)
            extracted = self._extract_boxed_answer(response_text)
            
            if extracted:
                # Step 2a: Extraction succeeded - compare directly to ground truth
                if ground_truth:
                    try:
                        if self._symbolic_check(extracted, ground_truth):
                            return {
                                "correct": True, 
                                "reasoning": f"Answer match: extracted '{extracted}' matches ground truth '{ground_truth}'",
                                "extracted_answer": extracted,
                                "method": "symbolic_match"
                            }
                        else:
                            # SymPy evaluated and they don't match
                            return {
                                "correct": False,
                                "reasoning": f"Answer mismatch: extracted '{extracted}' does not match ground truth '{ground_truth}'",
                                "extracted_answer": extracted,
                                "method": "symbolic_check"
                            }
                    except (AttributeError, NotImplementedError) as e:
                        # SymPy cannot evaluate (e.g., FiniteSet, unsupported types) - fall back to judge
                        logger.info(
                            f"SymPy evaluation failed for extracted '{extracted}' vs ground truth '{ground_truth}': {e}. "
                            f"Falling back to judge evaluation."
                        )
                        # Fall through to judge evaluation below
                    except Exception as e:
                        # Other unexpected errors - log and fall back to judge
                        logger.warning(
                            f"Unexpected error in symbolic check for '{extracted}' vs '{ground_truth}': {e}. "
                            f"Falling back to judge evaluation."
                        )
                        # Fall through to judge evaluation below
                else:
                    # No ground truth but we extracted an answer - can't verify
                    return {
                        "correct": False,
                        "reasoning": f"Extracted answer '{extracted}' but no ground truth provided for comparison",
                        "extracted_answer": extracted,
                        "method": "no_ground_truth"
                    }
            
            # Step 2b: Extraction failed OR SymPy evaluation failed - fallback to judge
            # Judge will: (1) check if solution has final answer, (2) if yes, compare to ground truth
            logger.info(f"No boxed answer found, using judge to extract final answer and compare to ground truth")
            judge_context = context.copy()
            judge_context["generator_solution"] = response_text  # Use extracted response, not full CoT
            judge_context["ground_truth_answer"] = ground_truth  # May be None, judge handles it
            
            judge_result = await self._execute_stage(
                judge_stage_name, judge_context, run_log, document_id, workflow_name
            )
            outputs = judge_result.get("outputs", {})
            result = {
                "correct": outputs.get("correct", False),
                "reasoning": outputs.get("reasoning", "No reasoning provided"),
                "extracted_answer": None,
                "method": "llm_judge"
            }
            # Store full output if thinking model was used
            if has_thinking:
                result["generator_solution_full"] = sample
            return result

        # Case 2: Proof Correctness - Rigor-based evaluation against reference solution
        elif task_type == "proof_correctness":
            # Proof checker requires reference solution for comparison
            reference_solution = context.get("reference_solution")
            if not reference_solution:
                logger.warning("Proof correctness task requires reference_solution, but none provided")
                return {
                    "correct": False,
                    "reasoning": "Proof correctness evaluation requires a reference solution for comparison",
                    "method": "missing_reference"
                }
            
            # Use ProofCheckerJudge to evaluate rigor and correctness
            logger.info("Using proof checker judge to evaluate solution rigor against reference")
            judge_context = context.copy()
            judge_context["generator_solution"] = response_text  # Use extracted response, not full CoT
            judge_context["reference_solution"] = reference_solution
            
            judge_result = await self._execute_stage(
                judge_stage_name, judge_context, run_log, document_id, workflow_name
            )
            outputs = judge_result.get("outputs", {})
            errors = outputs.get("errors", [])
            
            # Convert ProofError Pydantic objects to dictionaries for MongoDB storage
            errors_dict = []
            if errors and isinstance(errors, list):
                for error in errors:
                    if hasattr(error, "model_dump"):
                        # Pydantic model - convert to dict
                        errors_dict.append(error.model_dump())
                    elif isinstance(error, dict):
                        # Already a dictionary
                        errors_dict.append(error)
                    else:
                        # Fallback: try to convert to dict
                        errors_dict.append({
                            "type": getattr(error, "type", str(error)),
                            "description": getattr(error, "description", ""),
                            "location": getattr(error, "location", "")
                        })
            
            result = {
                "correct": outputs.get("correct", False),
                "reasoning": outputs.get("reasoning", "No reasoning provided"),
                "errors": errors_dict,  # List of error dictionaries
                "method": "proof_checker_judge"
            }
            # Store full output if thinking model was used
            if has_thinking:
                result["generator_solution_full"] = sample
            return result

        # Case 3: Grader - Auto-rubric generation and scoring (0-7 scale)
        elif task_type == "grader":
            # Grader requires reference solution for rubric generation
            reference_solution = context.get("reference_solution")
            if not reference_solution:
                logger.warning("Grader task requires reference_solution, but none provided")
                return {
                    "score": 0,
                    "reasoning": "Grader evaluation requires a reference solution for rubric generation",
                    "method": "missing_reference"
                }
            
            # Use GraderJudge to generate rubric and score solution
            logger.info("Using grader judge to generate rubric and score solution")
            judge_context = context.copy()
            judge_context["generator_solution"] = response_text  # Use extracted response, not full CoT
            judge_context["reference_solution"] = reference_solution
            
            judge_result = await self._execute_stage(
                judge_stage_name, judge_context, run_log, document_id, workflow_name
            )
            outputs = judge_result.get("outputs", {})
            
            # Extract score from overall_assessment
            # Note: output_mapping maps model's "overall_assessment" field to "grade_result" key in context
            # So we access it via "grade_result" here, but it contains the overall_assessment object
            overall_assessment = outputs.get("grade_result", {})
            
            # Handle Pydantic model instance
            if hasattr(overall_assessment, "model_dump"):
                assessment_dict = overall_assessment.model_dump()
                score = assessment_dict.get("score", 0)
                classification = assessment_dict.get("classification", "Unknown")
                logger.debug(f"Extracted score from Pydantic model: score={score}, classification={classification}")
            elif isinstance(overall_assessment, dict):
                score = overall_assessment.get("score", 0)
                classification = overall_assessment.get("classification", "Unknown")
                logger.debug(f"Extracted score from dict: score={score}, classification={classification}")
            else:
                # Fallback: try to get attributes directly
                score = getattr(overall_assessment, "score", 0) if hasattr(overall_assessment, "score") else 0
                classification = getattr(overall_assessment, "classification", "Unknown") if hasattr(overall_assessment, "classification") else "Unknown"
                logger.warning(f"overall_assessment is neither dict nor Pydantic model (type: {type(overall_assessment)}). Using fallback extraction: score={score}")
            
            # Handle string scores (LLM might return "7" instead of 7)
            if isinstance(score, str):
                try:
                    score = int(score)
                except (ValueError, TypeError):
                    logger.warning(f"Could not parse score '{score}', defaulting to 0")
                    score = 0
            elif not isinstance(score, int):
                logger.warning(f"Score is not int or string (type: {type(score)}, value: {score}), defaulting to 0")
                score = 0
            
            # Clamp score to 0-7 range
            score = max(0, min(7, score))
            logger.info(f"Final extracted score: {score}/7 (classification: {classification})")
            designed_marking_scheme = outputs.get("designed_marking_scheme", {})
            feedback = outputs.get("feedback", {})
            
            # Convert Pydantic objects to dictionaries for MongoDB storage
            if hasattr(designed_marking_scheme, "model_dump"):
                designed_marking_scheme = designed_marking_scheme.model_dump()
            if hasattr(feedback, "model_dump"):
                feedback = feedback.model_dump()
            
            result = {
                "score": score,
                "classification": classification,
                "reasoning": outputs.get("reasoning", "No reasoning provided"),
                "designed_marking_scheme": designed_marking_scheme,
                "feedback": feedback,
                "method": "grader_judge"
            }
            # Store full output if thinking model was used
            if has_thinking:
                result["generator_solution_full"] = sample
            return result

        # Case 4: Open Ended (no ground truth expected)
        else:
            judge_context = context.copy()
            judge_context["generator_solution"] = response_text  # Use extracted response, not full CoT
            
            judge_result = await self._execute_stage(
                judge_stage_name, judge_context, run_log, document_id, workflow_name
            )
            outputs = judge_result.get("outputs", {})
            result = {
                "correct": outputs.get("correct", False),
                "reasoning": outputs.get("reasoning", "No reasoning provided"),
                "method": "llm_judge"
            }
            # Store full output if thinking model was used
            if has_thinking:
                result["generator_solution_full"] = sample
            return result

    async def _run(
        self,
        work_item: WorkItem[EvalPayload],
        run_log: RunLog,
        document_id: str,
        workflow_name: str,
    ) -> Dict[str, Any]:
        payload = work_item.payload
        context: Dict[str, Any] = payload.model_dump()
        context["run_log"] = run_log

        # 1. Configuration
        task_type = self._config.get("task_type", "math_correctness")
        judge_stage_name = self._config.get("judge_stage_name", "MathJudge")
        num_samples = int(self._config.get("num_samples", 1))
        
        # Get target model name from global LLM config (passed via app)
        # Assuming app passes 'llm_default_model' in _config or similar mechanism
        # If not directly available, we might rely on the fact that 'Generator' stage
        # uses the default profile which points to the model.
        # However, for saving results, we need the KEY.
        # We will use a runtime override or config value 'target_model_name'.
        target_model_name = self._config.get("target_model_name", "unknown_model")

        logger.info(f"Starting Eval ({task_type}) for {target_model_name} on {payload.problem_id}")

        # 2. Cache Loading & Incremental Generation
        cached_responses_map = payload.cached_responses or {}
        existing_samples = cached_responses_map.get(target_model_name, [])
        
        num_existing = len(existing_samples)
        num_needed = max(0, num_samples - num_existing)
        
        logger.info(f"Found {num_existing} cached samples. Generating {num_needed} more.")
        
        new_samples = []
        if num_needed > 0:
            new_samples = await self._generate_samples(
                context, run_log, document_id, workflow_name, num_needed
            )
        
        all_samples = existing_samples + new_samples
        
        # 3. Evaluation Loop
        eval_results = []
        
        # Parallel evaluation of all samples
        eval_tasks = []
        for sample in all_samples:
            eval_tasks.append(
                self._evaluate_sample(
                    sample, 
                    context, 
                    task_type, 
                    judge_stage_name, 
                    payload.ground_truth_answer,
                    run_log, 
                    document_id, 
                    workflow_name
                )
            )
            
        eval_results = await asyncio.gather(*eval_tasks)
        
        # 4. Aggregation
        if task_type == "grader":
            # Calculate average score for grader task type
            scores = []
            for r in eval_results:
                score = r.get("score", 0)
                # Handle both int and string scores
                if isinstance(score, str):
                    try:
                        score = int(score)
                    except (ValueError, TypeError):
                        score = 0
                elif not isinstance(score, int):
                    score = 0
                # Clamp to 0-7 range
                score = max(0, min(7, score))
                scores.append(score)
            
            average_score = sum(scores) / len(scores) if scores else 0.0
            num_perfect = sum(1 for s in scores if s == 7)
            
            final_output = {
                "average_score": average_score,
                "total_samples": len(all_samples),
                "scores": scores,
                "num_perfect": num_perfect,
                "samples_results": eval_results,
                "task_type": task_type
            }
            
            logger.info(
                f"Eval Complete. Average Score: {average_score:.2f}/7 "
                f"({num_perfect}/{len(all_samples)} perfect scores)"
            )
        else:
            # Calculate pass rate for other task types (math_correctness, proof_correctness, open_ended)
            num_correct = sum(1 for r in eval_results if r.get("correct"))
            pass_rate = num_correct / len(all_samples) if all_samples else 0.0
            
            final_output = {
                "pass_rate": pass_rate,
                "total_samples": len(all_samples),
                "num_correct": num_correct,
                "samples_results": eval_results,
                "task_type": task_type
            }
            
            logger.info(f"Eval Complete. Pass Rate: {pass_rate:.2f} ({num_correct}/{len(all_samples)})")

        # 5. Output Construction
        # We return specific keys that match the bindings
        return {
            "status": "completed",
            "eval_output": final_output,
            "generated_responses": all_samples
        }



