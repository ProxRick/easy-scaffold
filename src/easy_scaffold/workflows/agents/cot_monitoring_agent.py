# src/easy_scaffold/workflows/agents/cot_monitoring_agent.py
import logging
import re
from typing import Any, Dict, List, Optional, Union

from easy_scaffold.db.pydantic_models import RunLog
from easy_scaffold.workflows.base import AbstractWorkflow
from easy_scaffold.workflows.workflow_models import CoTMonitoringPayload, WorkItem

logger = logging.getLogger(__name__)


class CoTMonitoringWorkflow(AbstractWorkflow[CoTMonitoringPayload]):
    """
    Workflow for monitoring and analyzing Chain of Thought (CoT) behavioral patterns.
    
    Supports both:
    - Single CoT/solution inputs (S1 training samples)
    - List of CoT/solution inputs (eval results with multiple responses)
    
    Extracts CoT from solutions (if </think> token exists) and classifies
    each step into one of 10 behavioral categories.
    """

    def _clean_step_content(self, text: str) -> str:
        """
        Clean specific step numbering artifacts from the CoT.
        Removes 'Step N:', 'N.', or similar prefixes to prevent double labeling.
        """
        if not text or not isinstance(text, str):
            return ""
        # Remove "Step X:" or "Step X." pattern
        cleaned = re.sub(r'^(?:Step\s+\d+[:.]?|\d+\.|-)\s*', '', text.strip(), flags=re.IGNORECASE)
        return cleaned or ""

    def _join_cot_with_double_newlines(self, cot_lines: List[str]) -> str:
        """
        Join CoT lines preserving double newlines between paragraphs.
        Empty strings in the list represent paragraph breaks (\n\n).
        """
        if not cot_lines:
            return ""
        return '\n'.join(cot_lines)

    def _label_cot_steps(self, cot_lines: List[str]) -> str:
        """
        Label CoT lines with step numbers for monitoring.
        Preserves paragraph breaks (empty strings become double newlines).
        """
        if not cot_lines:
            return ""
        
        labeled_lines = []
        step_num = 1
        for line in cot_lines:
            if line == "":
                # Empty string represents paragraph break - preserve as double newline
                labeled_lines.append("")
            else:
                labeled_lines.append(f"Step {step_num}:\n{line}")
                step_num += 1
        
        # Join with double newlines preserved
        return self._join_cot_with_double_newlines(labeled_lines)

    def _chunk_labeled_cot(self, labeled_cot: str, max_steps: int) -> List[str]:
        """
        Split labeled CoT into chunks that don't exceed max_steps.
        Preserves step boundaries - never splits in the middle of a step.
        
        Args:
            labeled_cot: Labeled CoT text with "Step N:" prefixes
            max_steps: Maximum number of steps per chunk
            
        Returns:
            List of CoT chunks, each containing complete steps
        """
        if not labeled_cot:
            return []
        
        # Count total steps first
        step_pattern = re.compile(r'^Step\s+\d+:', re.MULTILINE)
        total_steps = len(step_pattern.findall(labeled_cot))
        
        # If total steps <= max_steps, return as single chunk
        if total_steps <= max_steps:
            return [labeled_cot]
        
        # Split by step boundaries (Step N:)
        step_boundary_pattern = re.compile(r'(Step\s+\d+:)', re.MULTILINE)
        chunks = []
        current_chunk = ""
        current_step_count = 0
        
        # Find all step boundaries
        matches = list(step_boundary_pattern.finditer(labeled_cot))
        if not matches:
            # No step markers found, return as single chunk
            return [labeled_cot]
        
        # Group steps into chunks
        for i, match in enumerate(matches):
            step_start = match.start()
            step_end = matches[i + 1].start() if i + 1 < len(matches) else len(labeled_cot)
            step_content = labeled_cot[step_start:step_end]
            
            # Check if adding this step would exceed max_steps
            if current_step_count > 0 and current_step_count >= max_steps:
                # Save current chunk and start new one
                chunks.append(current_chunk.strip())
                current_chunk = step_content
                current_step_count = 1
            else:
                # Add step to current chunk
                current_chunk += step_content
                current_step_count += 1
        
        # Add final chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks

    def _prepare_cot_for_monitoring(self, cot_text: str) -> str:
        """
        Prepare CoT text for monitoring by splitting into paragraphs and labeling steps.
        Splits by double newlines (\n\n) to treat paragraphs as logical steps.
        
        Args:
            cot_text: Raw CoT text string
            
        Returns:
            Labeled CoT text with "Step 1:", "Step 2:" prefixes
        """
        if not cot_text or not isinstance(cot_text, str):
            logger.warning("_prepare_cot_for_monitoring: Empty or invalid cot_text input")
            return ""
        
        logger.debug(f"_prepare_cot_for_monitoring: Input CoT length: {len(cot_text)} chars")
        logger.debug(f"_prepare_cot_for_monitoring: Input CoT first 200 chars: {repr(cot_text[:200])}")
        
        # Split by double newlines to get paragraphs (logical steps)
        # This matches the user's approach: split by \n\n to get ~100 steps from a 122-step CoT
        paragraphs = cot_text.split("\n\n")
        logger.info(f"_prepare_cot_for_monitoring: Split by \\n\\n resulted in {len(paragraphs)} paragraphs")
        
        # Clean and filter paragraphs
        cleaned_paragraphs = []
        for idx, paragraph in enumerate(paragraphs):
            if paragraph.strip():
                # Non-empty paragraph with content - clean and add
                cleaned = self._clean_step_content(paragraph.strip())
                if cleaned:  # Only add if cleaning didn't make it empty
                    cleaned_paragraphs.append(cleaned)
                else:
                    logger.debug(f"_prepare_cot_for_monitoring: Paragraph {idx + 1} became empty after cleaning")
            else:
                logger.debug(f"_prepare_cot_for_monitoring: Paragraph {idx + 1} is empty/whitespace, skipping")
        
        logger.info(f"_prepare_cot_for_monitoring: After filtering, {len(cleaned_paragraphs)} paragraphs remain")
        
        # Label each paragraph as a step
        labeled_parts = []
        for i, paragraph in enumerate(cleaned_paragraphs, start=1):
            labeled_parts.append(f"Step {i}:\n{paragraph}")
        
        # Join with double newlines to preserve paragraph structure
        labeled_cot = "\n\n".join(labeled_parts)
        
        logger.info(f"_prepare_cot_for_monitoring: Final labeled CoT has {len(cleaned_paragraphs)} steps, length: {len(labeled_cot)} chars")
        
        return labeled_cot

    def _extract_cot_from_solution(self, solution_text: str) -> str:
        """
        Extract CoT from solution text.
        - If `</think>` exists: return everything before it
        - Otherwise: return entire text (assume it's pure CoT)
        
        Args:
            solution_text: Solution text that may contain CoT before </think> token
            
        Returns:
            Extracted CoT string
        """
        if not solution_text or not isinstance(solution_text, str):
            return ""
        
        # Patterns for thinking block closing tags (matching eval_agent pattern)
        thinking_end_patterns = [
            r'`</think>`',   # Backtick-wrapped (most specific)
            r'</think>',     # Plain tag
        ]
        
        last_match = None
        last_pos = -1
        
        for pattern in thinking_end_patterns:
            matches = list(re.finditer(pattern, solution_text, re.IGNORECASE))
            if matches:
                match = matches[-1]
                if match.start() > last_pos:
                    last_match = match
                    last_pos = match.start()
        
        if last_match:
            cot = solution_text[:last_pos].strip()
            logger.info(f"_extract_cot_from_solution: Token found at position {last_pos}. Extracted CoT length: {len(cot)} chars, Solution length: {len(solution_text)} chars")
            if not cot:
                logger.warning(f"_extract_cot_from_solution: CoT extraction resulted in empty string (token found at position {last_pos})")
            else:
                logger.debug(f"_extract_cot_from_solution: Extracted CoT first 200 chars: {repr(cot[:200])}")
            return cot
        
        # No token found, assume entire text is CoT
        cot = solution_text.strip()
        if not cot:
            logger.warning("_extract_cot_from_solution: No </think> token found and solution text is empty/whitespace")
        else:
            logger.info(f"_extract_cot_from_solution: No </think> token found, using entire text as CoT (length: {len(cot)} chars)")
            logger.debug(f"_extract_cot_from_solution: Using entire text as CoT, first 200 chars: {repr(cot[:200])}")
        return cot

    def _normalize_input(self, solution_or_cot: Union[str, List[str]]) -> List[str]:
        """
        Normalize input to always return a list of CoT strings.
        - If string: extract CoT and return [cot]
        - If list: extract CoT from each item and return [cot1, cot2, ...]
        
        Args:
            solution_or_cot: Single CoT/solution string or list of them
            
        Returns:
            List of extracted CoT strings
        """
        if isinstance(solution_or_cot, str):
            return [self._extract_cot_from_solution(solution_or_cot)]
        elif isinstance(solution_or_cot, list):
            return [self._extract_cot_from_solution(item) for item in solution_or_cot if item]
        else:
            logger.warning(f"Unexpected input type: {type(solution_or_cot)}. Returning empty list.")
            return []

    async def _run(
        self,
        work_item: WorkItem[CoTMonitoringPayload],
        run_log: RunLog,
        document_id: str,
        workflow_name: str,
    ) -> Dict[str, Any]:
        """
        Main workflow execution:
        1. Extract payload
        2. Normalize input to list of CoTs
        3. Process each CoT through CoTMonitoring stage
        4. Aggregate results
        5. Return structured output
        """
        payload = work_item.payload
        context: Dict[str, Any] = payload.model_dump()
        context["run_log"] = run_log

        problem_id = context.get("problem_id", "unknown")
        solution_or_cot = context.get("solution_or_cot")
        # Get model_name from workflow config (runtime parameter) or payload, default to "unknown"
        model_name = self._config.get("model_name") or context.get("model_name", "unknown")
        
        logger.info(f"Starting CoT Monitoring for {problem_id} (model: {model_name})")

        if not solution_or_cot:
            logger.warning("No solution_or_cot provided")
            return {
                "status": "failed",
                "error": "No solution_or_cot provided",
                "steps": [],
                "total_steps": 0
            }

        # Normalize input to list of CoTs
        cot_list = self._normalize_input(solution_or_cot)
        
        if not cot_list:
            logger.warning("No valid CoTs extracted from input")
            return {
                "status": "failed",
                "error": "No valid CoTs extracted from input",
                "steps": [],
                "total_steps": 0
            }

        logger.info(f"Processing {len(cot_list)} CoT(s)")

        # Process each CoT
        results = []
        for idx, cot in enumerate(cot_list):
            if not cot or not cot.strip():
                logger.warning(f"Skipping empty CoT at index {idx}")
                results.append(None)
                continue
            
            try:
                logger.info(f"CoT {idx + 1}/{len(cot_list)}: Starting processing. Original CoT length: {len(cot)} chars")
                
                # Prepare CoT by labeling steps (like generator_critique_agent does)
                labeled_cot = self._prepare_cot_for_monitoring(cot)
                
                # Check if preparation resulted in empty CoT
                if not labeled_cot or not labeled_cot.strip():
                    logger.warning(
                        f"CoT {idx + 1}/{len(cot_list)}: Prepared CoT is empty after labeling. "
                        f"Original CoT length: {len(cot)} chars. Skipping this CoT."
                    )
                    results.append(None)
                    continue
                
                logger.debug(f"CoT {idx + 1}/{len(cot_list)}: Prepared CoT for monitoring: {len(labeled_cot)} chars, original: {len(cot)} chars")
                
                # Count steps in the labeled CoT (input steps)
                # Steps are labeled as "Step N:\n<content>", so count occurrences of "Step N:"
                step_pattern = re.compile(r'^Step\s+\d+:', re.MULTILINE)
                step_matches = step_pattern.findall(labeled_cot)
                input_step_count = len(step_matches)
                
                # CRITICAL LOGGING: Show parsing result
                logger.info(
                    f"CoT {idx + 1}/{len(cot_list)}: ===== PARSING RESULT ===== "
                    f"Found {input_step_count} steps after parsing. "
                    f"Labeled CoT length: {len(labeled_cot)} chars, Original CoT length: {len(cot)} chars"
                )
                
                # Show sample of labeled CoT for debugging
                if input_step_count > 0:
                    logger.debug(f"CoT {idx + 1}/{len(cot_list)}: First 3 steps sample:\n{labeled_cot[:500]}")
                else:
                    logger.warning(f"CoT {idx + 1}/{len(cot_list)}: No steps found! Labeled CoT sample:\n{repr(labeled_cot[:500])}")
                
                if input_step_count == 0:
                    logger.warning(
                        f"CoT {idx + 1}/{len(cot_list)}: No steps found in labeled CoT! "
                        f"Labeled CoT length: {len(labeled_cot)} chars. "
                        f"This may indicate an issue with step parsing."
                    )
                    # Log a sample of the labeled CoT for debugging
                    logger.debug(f"Sample of labeled CoT (first 500 chars): {repr(labeled_cot[:500])}")
                
                logger.info(
                    f"CoT {idx + 1}/{len(cot_list)}: Input has {input_step_count} labeled steps "
                    f"(CoT length: {len(labeled_cot)} chars)"
                )
                
                # Get configuration
                max_retries = self._config.get("max_retries_empty_response", 3)
                max_cot_steps = self._config.get("max_cot_steps", 100)  # Default 100 steps per chunk
                
                # Chunk the CoT if it exceeds max_cot_steps (before retry logic)
                cot_chunks = self._chunk_labeled_cot(labeled_cot, max_cot_steps)
                
                if len(cot_chunks) > 1:
                    logger.info(
                        f"CoT {idx + 1}/{len(cot_list)}: Split into {len(cot_chunks)} chunks "
                        f"(original: {input_step_count} steps, max per chunk: {max_cot_steps} steps)"
                    )
                
                # Process each chunk and merge results
                all_step_classifications = []
                
                for chunk_idx, cot_chunk in enumerate(cot_chunks):
                    chunk_info = f"CoT {idx + 1}/{len(cot_list)}, chunk {chunk_idx + 1}/{len(cot_chunks)}"
                    
                    # Call CoTMonitoring stage with retry logic for empty responses
                    step_classifications = []
                    output_step_count = 0
                    finish_reason = ""
                    
                    for attempt in range(max_retries + 1):  # +1 for initial attempt
                        try:
                            monitoring_result = await self._execute_stage(
                                "CoTMonitoring",
                                {
                                    "problem_statement": payload.problem_statement,
                                    "cot": cot_chunk
                                },
                                run_log, document_id, workflow_name
                            )
                            
                            outputs = monitoring_result.get("outputs", {})
                            step_classifications = outputs.get("cot_steps", [])
                            output_step_count = len(step_classifications) if step_classifications else 0
                            finish_reason = monitoring_result.get("finish_reason", "")
                            
                            # Check if we got a valid response
                            if output_step_count > 0:
                                # Success - got at least some classifications
                                break
                            else:
                                # Empty response - retry if we have attempts left
                                if attempt < max_retries:
                                    logger.warning(
                                        f"{chunk_info}: Empty response on attempt {attempt + 1}/{max_retries + 1}. "
                                        f"Retrying... (Finish reason: {finish_reason})"
                                    )
                                else:
                                    logger.error(
                                        f"{chunk_info}: Empty response after {max_retries + 1} attempts. "
                                        f"Giving up. (Finish reason: {finish_reason})"
                                    )
                                    # Mark as failed by setting step_classifications to None
                                    step_classifications = None
                                    break
                        except Exception as e:
                            # Retry on exceptions too if we have attempts left
                            if attempt < max_retries:
                                logger.warning(
                                    f"{chunk_info}: Exception on attempt {attempt + 1}/{max_retries + 1}: {e}. "
                                    f"Retrying..."
                                )
                            else:
                                logger.error(
                                    f"{chunk_info}: Exception after {max_retries + 1} attempts: {e}. "
                                    f"Giving up."
                                )
                                raise
                    
                    # Merge chunk results
                    if step_classifications is None:
                        # Failed chunk - mark entire CoT as failed
                        logger.error(f"{chunk_info}: Failed to process chunk, marking entire CoT as failed")
                        all_step_classifications = None
                        break
                    elif step_classifications:
                        # Convert to dicts and add to merged results
                        for step in step_classifications:
                            if hasattr(step, "model_dump"):
                                all_step_classifications.append(step.model_dump())
                            elif isinstance(step, dict):
                                all_step_classifications.append(step)
                            else:
                                logger.warning(f"{chunk_info}: Unexpected step type: {type(step)}")
                
                # Use merged results
                step_classifications = all_step_classifications
                output_step_count = len(step_classifications) if step_classifications else 0
                
                # Check for potential truncation or mismatch
                if step_classifications is None:
                    # Failed after all retries - mark as None (skipped)
                    logger.error(
                        f"CoT {idx + 1}/{len(cot_list)}: Failed to process after chunking and retries. "
                        f"Marking as skipped."
                    )
                    results.append(None)
                elif input_step_count != output_step_count:
                    logger.warning(
                        f"CoT {idx + 1}/{len(cot_list)}: Step count mismatch detected! "
                        f"Input has {input_step_count} labeled steps, but output has {output_step_count} step classifications. "
                        f"This may indicate incomplete processing or chunking issues."
                    )
                    # Still append results even if counts don't match
                    results.append(step_classifications)
                else:
                    logger.info(
                        f"CoT {idx + 1}/{len(cot_list)}: Step counts match! "
                        f"Both input and output have {output_step_count} steps."
                    )
                    # step_classifications is already a list of dicts (merged from chunks)
                    results.append(step_classifications)
                    
                logger.info(
                    f"CoT {idx + 1}/{len(cot_list)} processing complete: "
                    f"{output_step_count} step classifications returned"
                )
                
            except Exception as e:
                logger.error(f"Error processing CoT {idx + 1}/{len(cot_list)}: {e}", exc_info=True)
                results.append(None)

        # Determine output structure
        if len(results) == 1:
            # Single input → single result
            single_result = results[0] if results[0] is not None else []
            return {
                "status": "completed",
                "steps": single_result,
                "total_steps": len(single_result),
                "model_name": model_name
            }
        else:
            # List input → list of results
            valid_results = [r for r in results if r is not None]
            return {
                "status": "completed",
                "results": results,  # List of step classifications (may contain None for failed items)
                "total_cots_processed": len(valid_results),
                "total_cots_skipped": len([r for r in results if r is None]),
                "total_steps": sum(len(r) for r in valid_results),
                "model_name": model_name
            }



