# src/easy_scaffold/workflows/agents/solution_selection.py
import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple

from easy_scaffold.configs.pydantic_models import (
    RelaxedCompletenessEvaluation,
    ComprehensiveCompletenessCheck,
    SolutionFixerOutput,
    SolutionCompletenessEvaluation,
    BestOfKSelectionOutput,
)
from easy_scaffold.db.pydantic_models import RunLog
from easy_scaffold.workflows.base import AbstractWorkflow
from easy_scaffold.workflows.workflow_models import SolutionSelectionPayload, WorkItem

logger = logging.getLogger(__name__)


class SolutionSelectionWorkflow(AbstractWorkflow[SolutionSelectionPayload]):
    async def _run(
        self,
        work_item: WorkItem[SolutionSelectionPayload],
        run_log: RunLog,
        document_id: str,
        workflow_name: str,
    ) -> Dict[str, Any]:
        payload = work_item.payload
        context: Dict[str, Any] = payload.model_dump()
        context["run_log"] = run_log

        # Load posts array
        posts = payload.posts.copy() if payload.posts else []
        problem_statement = payload.problem_statement

        logger.info(f"Processing {len(posts)} posts for problem {payload.problem_id}")

        # Get workflow config parameters
        max_solutions = self._config.get("max_solutions", 30)
        top_k_candidates = self._config.get("top_k_candidates", 5)
        completeness_success_threshold = self._config.get("completeness_success_threshold", 3)
        max_fixing_iterations = self._config.get("max_fixing_iterations", 5)

        # Initialize variables that will be used later
        completeness_evaluations: Dict[int, Dict[str, Any]] = {}
        top_k: List[Tuple[int, Dict[str, Any], Dict[str, Any]]] = []
        limited_solutions: List[Tuple[int, Dict[str, Any]]] = []

        # Check for cached top_k_candidates
        cached_top_k = payload.cached_top_k_candidates
        if cached_top_k:
            logger.info(f"Loaded cached top_k_candidates with {len(cached_top_k)} candidates")
            
            # Reconstruct top_k list from cached data
            for cached_item in cached_top_k:
                post_idx = cached_item.get("post_index")
                if post_idx is not None and post_idx < len(posts):
                    post = posts[post_idx]
                    completeness_eval = cached_item.get("completeness_evaluation", {})
                    top_k.append((post_idx, post, completeness_eval))
                    completeness_evaluations[post_idx] = completeness_eval
            
            # Reconstruct limited_solutions for the return value
            # Sort posts by like_count to match original logic
            solution_posts: List[Tuple[int, Dict[str, Any]]] = []
            for idx, post in enumerate(posts):
                if post.get("is_solution") is True:
                    solution_text = post.get("text", "")
                    if solution_text:
                        solution_posts.append((idx, post))
            
            solution_posts.sort(
                key=lambda x: (
                    -x[1].get("like_count", 0),
                    x[0]
                )
            )
            limited_solutions = solution_posts[:max_solutions]
            
            logger.info(
                f"Using cached top_k_candidates: {len(top_k)} candidates"
            )
        else:
            # No cache - run full evaluation pipeline
            # Gather solution posts and sort by like_count, then post order
            solution_posts: List[Tuple[int, Dict[str, Any]]] = []  # (idx, post)
            for idx, post in enumerate(posts):
                if post.get("is_solution") is True:
                    solution_text = post.get("text", "")
                    if solution_text:
                        solution_posts.append((idx, post))

            logger.info(f"Found {len(solution_posts)} solution posts")

            if not solution_posts:
                return {
                    "status": "no_solutions",
                    "batch_completeness_evaluations": [],
                    "top_k_candidates": [],
                    "selected_solution": None,
                    "total_solutions": 0,
                    "evaluated_solutions_count": 0,
                    "top_k_count": 0,
                }

            # Sort by like_count (descending), then by post index (ascending) for equal likes
            solution_posts.sort(
                key=lambda x: (
                    -x[1].get("like_count", 0),  # Negative for descending
                    x[0]  # Post index for tie-breaking
                )
            )

            # Limit to max_solutions
            limited_solutions = solution_posts[:max_solutions]
            logger.info(
                f"Limited to top {len(limited_solutions)} solutions "
                f"(sorted by like_count, then post order)"
            )

            # Run parallel individual completeness evaluations (using cheaper model)
            logger.info(f"Running parallel completeness evaluation on {len(limited_solutions)} solutions")
            
            # Create tasks for parallel execution
            eval_tasks = []
            for idx, post in limited_solutions:
                solution_text = post.get("text", "")
                eval_context = {
                    "problem_statement": problem_statement,
                    "solution_text": solution_text,
                    "run_log": run_log,
                }
                eval_tasks.append(
                    self._execute_stage(
                        "SolutionCompletenessEvaluation",
                        eval_context,
                        run_log,
                        document_id,
                        workflow_name,
                    )
                )
            
            # Execute all evaluations in parallel
            try:
                results = await asyncio.gather(*eval_tasks, return_exceptions=True)
                
                # Process results and map to solutions
                for i, (idx, post) in enumerate(limited_solutions):
                    result = results[i]
                    
                    if isinstance(result, Exception):
                        logger.error(
                            f"Error evaluating solution (post {idx}): {result}",
                            exc_info=result
                        )
                        # Create default evaluation for failed cases
                        eval_data = {
                            "is_complete": False,
                            "completeness_score": 0.0,
                            "cleanliness_score": 0.0,
                            "has_errors": True,
                            "errors": [],
                            "reasoning": f"Evaluation failed: {str(result)}",
                        }
                    else:
                        # Extract evaluation from result
                        raw_output = result.get("raw_output")
                        if raw_output:
                            if hasattr(raw_output, "model_dump"):
                                eval_data = raw_output.model_dump()
                            elif isinstance(raw_output, dict):
                                eval_data = raw_output
                            else:
                                eval_data = {
                                    "is_complete": getattr(raw_output, "is_complete", False),
                                    "completeness_score": getattr(raw_output, "completeness_score", 0.0),
                                    "cleanliness_score": getattr(raw_output, "cleanliness_score", 0.0),
                                    "has_errors": getattr(raw_output, "has_errors", True),
                                    "errors": getattr(raw_output, "errors", []),
                                    "reasoning": getattr(raw_output, "reasoning", ""),
                                }
                        else:
                            logger.warning(f"No output from completeness evaluation for post {idx}")
                            eval_data = {
                                "is_complete": False,
                                "completeness_score": 0.0,
                                "cleanliness_score": 0.0,
                                "has_errors": True,
                                "errors": [],
                                "reasoning": "No evaluation output received",
                            }
                    
                    # Store evaluation (convert to format compatible with existing code)
                    # Convert SolutionCompletenessEvaluation format to RelaxedCompletenessEvaluation format
                    completeness_evaluations[idx] = {
                        "is_complete": eval_data.get("is_complete", False),
                        "completeness_score": eval_data.get("completeness_score", 0.0),
                        "cleanliness_score": eval_data.get("cleanliness_score", 0.0),
                        "has_errors": eval_data.get("has_errors", False),
                        "errors": eval_data.get("errors", []),
                        "reasoning": eval_data.get("reasoning", ""),
                        # Also store for backward compatibility
                        "is_legit_and_fixable": eval_data.get("is_complete", False) and not eval_data.get("has_errors", True),
                        "fixability_score": eval_data.get("completeness_score", 0.0),
                        "issues": [err.get("description", str(err)) if isinstance(err, dict) else str(err) for err in eval_data.get("errors", [])],
                    }
                
                logger.info(
                    f"Parallel completeness evaluation complete: "
                    f"{len(completeness_evaluations)} evaluations received"
                )
                
            except Exception as e:
                logger.error(f"Error in parallel completeness evaluation: {e}", exc_info=True)
                return {
                    "status": "completeness_evaluation_failed",
                    "batch_completeness_evaluations": [],
                    "top_k_candidates": [],
                    "selected_solution": None,
                    "total_solutions": len(limited_solutions),
                    "evaluated_solutions_count": 0,
                    "top_k_count": 0,
                }

        # Sort ALL evaluated solutions by combined score (completeness_score + cleanliness_score)
        all_evaluated_solutions: List[Tuple[int, Dict[str, Any], Dict[str, Any]]] = []
        for idx, post in limited_solutions:
            if idx in completeness_evaluations:
                eval_data = completeness_evaluations[idx]
                all_evaluated_solutions.append((idx, post, eval_data))
        
        # Sort by combined score (descending)
        all_evaluated_solutions.sort(
            key=lambda x: x[2].get("completeness_score", 0.0) + x[2].get("cleanliness_score", 0.0),
            reverse=True,
        )

        # Select top k candidates (always k solutions, even if scores are low)
        top_k = all_evaluated_solutions[:top_k_candidates]
        logger.info(
            f"Selected top {len(top_k)} candidates "
            f"(combined scores: {[eval_data.get('completeness_score', 0) + eval_data.get('cleanliness_score', 0) for _, _, eval_data in top_k]})"
        )

        if not top_k:
            logger.warning("No top k candidates available, cannot select solution")
            return {
                "status": "no_top_k_candidates",
                "batch_completeness_evaluations": [
                    {"post_index": idx, **completeness_evaluations.get(idx, {})}
                    for idx, _ in limited_solutions
                ],
                "top_k_candidates": [],
                "selected_solution": None,
                "total_solutions": len(limited_solutions),
                "evaluated_solutions_count": len(completeness_evaluations),
                "top_k_count": 0,
            }
        
        # Format solutions for best-of-k selection
        solutions_list = [post.get("text", "") for _, post, _ in top_k]
        solutions_text = self._format_solutions_for_batch(solutions_list)
        
        # Prepare completeness scores for context
        completeness_scores_text = "\n".join([
            f"Solution {i+1}: completeness_score={eval_data.get('completeness_score', 0):.2f}, "
            f"cleanliness_score={eval_data.get('cleanliness_score', 0):.2f}, "
            f"combined_score={eval_data.get('completeness_score', 0) + eval_data.get('cleanliness_score', 0):.2f}"
            for i, (_, _, eval_data) in enumerate(top_k)
        ])
        
        # Run best-of-k selection (single call with better model)
        logger.info(f"Running best-of-k selection on {len(top_k)} candidates")
        best_of_k_context = {
            "problem_statement": problem_statement,
            "solutions": solutions_text,
            "completeness_scores": completeness_scores_text,
            "run_log": run_log,
        }
        
        try:
            best_of_k_result = await self._execute_stage(
                "BestOfKSelection",
                best_of_k_context,
                run_log,
                document_id,
                workflow_name,
            )
            
            raw_best_of_k = best_of_k_result.get("raw_output")
            if raw_best_of_k:
                if hasattr(raw_best_of_k, "selected_index"):
                    selected_index = raw_best_of_k.selected_index
                    selection_reasoning = raw_best_of_k.reasoning if hasattr(raw_best_of_k, "reasoning") else ""
                    comparison_notes = raw_best_of_k.comparison_notes if hasattr(raw_best_of_k, "comparison_notes") else None
                elif isinstance(raw_best_of_k, dict):
                    selected_index = raw_best_of_k.get("selected_index", 0)
                    selection_reasoning = raw_best_of_k.get("reasoning", "")
                    comparison_notes = raw_best_of_k.get("comparison_notes")
                else:
                    selected_index = getattr(raw_best_of_k, "selected_index", 0)
                    selection_reasoning = getattr(raw_best_of_k, "reasoning", "")
                    comparison_notes = getattr(raw_best_of_k, "comparison_notes", None)
            else:
                logger.warning("No output from best-of-k selection, falling back to first solution")
                selected_index = 0
                selection_reasoning = "Fallback: no selection output received"
                comparison_notes = None
            
            # Validate selected_index is within range
            if selected_index < 0 or selected_index >= len(top_k):
                logger.warning(
                    f"Invalid selected_index {selected_index} (range: 0-{len(top_k)-1}), "
                    f"falling back to first solution"
                )
                selected_index = 0
                selection_reasoning = f"Fallback: invalid index {selected_index}"
            
            best_idx, best_post, best_completeness_eval = top_k[selected_index]
            logger.info(
                f"Selected best solution (post {best_idx}, index {selected_index}): "
                f"combined_score={best_completeness_eval.get('completeness_score', 0) + best_completeness_eval.get('cleanliness_score', 0):.2f}"
            )
            
        except Exception as e:
            logger.error(f"Error in best-of-k selection: {e}", exc_info=True)
            # Fallback: select first solution (highest combined score)
            logger.warning("Falling back to first solution (highest combined score)")
            best_idx, best_post, best_completeness_eval = top_k[0]
            selected_index = 0
            selection_reasoning = f"Fallback due to error: {str(e)}"
            comparison_notes = None
        
        original_solution = best_post.get("text", "")

        # Run iterative fixing loop for selected solution
        logger.info("Starting completeness check and fixing loop for selected solution...")
        current_solution = original_solution
        fixing_iterations = 0
        completeness_history: List[Dict[str, Any]] = []
        fixer_history: List[Dict[str, Any]] = []
        was_fixed = False
        consecutive_successes = 0

        try:
            for iteration in range(max_fixing_iterations):
                fixing_iterations += 1
                logger.info(f"Iteration {fixing_iterations}/{max_fixing_iterations}")

                # Run comprehensive completeness check
                check_context = {
                    "problem_statement": problem_statement,
                    "solution_text": current_solution,
                    "run_log": run_log,
                }

                check_result = await self._execute_stage(
                    "ComprehensiveCompletenessCheck",
                    check_context,
                    run_log,
                    document_id,
                    workflow_name,
                )

                raw_check = check_result.get("raw_output")
                if raw_check:
                    if hasattr(raw_check, "is_complete"):
                        check_data = {
                            "is_complete": raw_check.is_complete,
                            "completeness_score": raw_check.completeness_score,
                            "issues": raw_check.issues if hasattr(raw_check, "issues") else [],
                            "feedback": raw_check.feedback if hasattr(raw_check, "feedback") else "",
                        }
                    elif isinstance(raw_check, dict):
                        check_data = {
                            "is_complete": raw_check.get("is_complete", False),
                            "completeness_score": raw_check.get("completeness_score", 0.0),
                            "issues": raw_check.get("issues", []),
                            "feedback": raw_check.get("feedback", ""),
                        }
                    else:
                        check_data = {
                            "is_complete": False,
                            "completeness_score": 0.0,
                            "issues": [],
                            "feedback": "",
                        }
                else:
                    check_data = {
                        "is_complete": False,
                        "completeness_score": 0.0,
                        "issues": [],
                        "feedback": "",
                    }

                completeness_history.append(check_data)

                if check_data.get("is_complete", False):
                    consecutive_successes += 1
                    logger.info(
                        f"Completeness check passed ({consecutive_successes}/{completeness_success_threshold})"
                    )
                    if consecutive_successes >= completeness_success_threshold:
                        logger.info("Solution is complete!")
                        break
                else:
                    consecutive_successes = 0
                    # Fix the solution
                    logger.info("Completeness check failed, applying fixer...")
                    fixer_context = {
                        "problem_statement": problem_statement,
                        "solution_text": current_solution,
                        "completeness_feedback": check_data.get("feedback", ""),
                        "run_log": run_log,
                    }

                    fixer_result = await self._execute_stage(
                        "SolutionFixer",
                        fixer_context,
                        run_log,
                        document_id,
                        workflow_name,
                    )

                    raw_fixer = fixer_result.get("raw_output")
                    if raw_fixer:
                        if hasattr(raw_fixer, "fixed_solution"):
                            fixer_data = {
                                "fixed_solution": raw_fixer.fixed_solution,
                                "changes_made": raw_fixer.changes_made if hasattr(raw_fixer, "changes_made") else "",
                                "reasoning": raw_fixer.reasoning if hasattr(raw_fixer, "reasoning") else "",
                            }
                        elif isinstance(raw_fixer, dict):
                            fixer_data = {
                                "fixed_solution": raw_fixer.get("fixed_solution", current_solution),
                                "changes_made": raw_fixer.get("changes_made", ""),
                                "reasoning": raw_fixer.get("reasoning", ""),
                            }
                        else:
                            fixer_data = {
                                "fixed_solution": current_solution,
                                "changes_made": "",
                                "reasoning": "",
                            }
                    else:
                        fixer_data = {
                            "fixed_solution": current_solution,
                            "changes_made": "",
                            "reasoning": "",
                        }

                    fixer_history.append(fixer_data)
                    current_solution = fixer_data.get("fixed_solution", current_solution)
                    was_fixed = True
                    logger.info(f"Applied fixer, solution updated (iteration {fixing_iterations})")

            # Check if we succeeded (after loop completes)
            if consecutive_successes < completeness_success_threshold:
                logger.warning(
                    f"Fixing loop failed after {fixing_iterations} iterations. "
                    f"Consecutive successes: {consecutive_successes}/{completeness_success_threshold}"
                )
                # Don't save anything if fixing failed
                return {
                    "status": "fixing_failed",
                    "batch_completeness_evaluations": [
                        {"post_index": idx, **completeness_evaluations.get(idx, {})}
                        for idx, _ in limited_solutions
                    ],
                    "top_k_candidates": [
                        {
                            "post_index": idx,
                            "solution_text": post.get("text", ""),
                            "completeness_evaluation": completeness_eval,
                        }
                        for idx, post, completeness_eval in top_k
                    ],
                    "selected_solution": None,
                    "total_solutions": len(limited_solutions),
                    "evaluated_solutions_count": len(completeness_evaluations),
                    "top_k_count": len(top_k),
                }

            final_solution = current_solution

        except Exception as e:
            logger.error(f"Error in completeness check or fixing loop: {e}", exc_info=True)
            return {
                "status": "check_or_fixing_failed",
                "batch_completeness_evaluations": [
                    {"post_index": idx, **completeness_evaluations.get(idx, {})}
                    for idx, _ in limited_solutions
                ],
                "top_k_candidates": [
                    {
                        "post_index": idx,
                        "solution_text": post.get("text", ""),
                        "completeness_evaluation": completeness_eval,
                    }
                    for idx, post, completeness_eval in top_k
                ],
                "selected_solution": None,
                "total_solutions": len(limited_solutions),
                "evaluated_solutions_count": len(completeness_evaluations),
                "top_k_count": len(top_k),
            }

        # Get final completeness check (last one in history)
        final_completeness_check = completeness_history[-1] if completeness_history else {
            "is_complete": False,
            "completeness_score": 0.0,
            "issues": [],
            "feedback": "",
        }

        # Build selected solution
        selected_solution = {
            "post_index": best_idx,
            "original_solution": original_solution,
            "final_solution": final_solution,
            "was_fixed": was_fixed,
            "fixing_iterations": fixing_iterations,
            "completeness_evaluations": completeness_history,
            "fixer_outputs": fixer_history,
            "final_completeness_evaluation": final_completeness_check,
            "completeness_evaluation": best_completeness_eval,
            "best_of_k_selection": {
                "selected_index": selected_index,
                "reasoning": selection_reasoning,
                "comparison_notes": comparison_notes,
            },
        }

        logger.info(
            f"Selected solution complete: post {best_idx}, "
            f"was_fixed={was_fixed}, iterations={fixing_iterations}"
        )

        return {
            "status": "completed",
            "batch_completeness_evaluations": [
                {"post_index": idx, **completeness_evaluations.get(idx, {})}
                for idx, _ in limited_solutions
            ],
            "top_k_candidates": [
                {
                    "post_index": idx,
                    "solution_text": post.get("text", ""),
                    "completeness_evaluation": completeness_eval,
                }
                for idx, post, completeness_eval in top_k
            ],
            "selected_solution": selected_solution,
            "total_solutions": len(limited_solutions),
            "evaluated_solutions_count": len(completeness_evaluations),
            "top_k_count": len(top_k),
        }

    def _format_solutions_for_batch(self, solutions: List[str]) -> str:
        """Format solutions as a numbered list with clear labels for batch evaluation."""
        formatted = []
        for i, solution in enumerate(solutions, 1):
            formatted.append(
                f"--- Solution {i} of {len(solutions)} ---\n"
                f"{solution}\n"
                f"--- End of Solution {i} ---\n"
            )
        return "\n".join(formatted)


