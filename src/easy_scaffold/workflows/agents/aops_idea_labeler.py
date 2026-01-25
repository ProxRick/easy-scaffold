# src/easy_scaffold/workflows/agents/aops_idea_labeler.py
import logging
from typing import Any, Dict, List

from easy_scaffold.db.pydantic_models import RunLog
from easy_scaffold.workflows.base import AbstractWorkflow
from easy_scaffold.workflows.workflow_models import AoPSIdeaLabelerPayload, WorkItem

logger = logging.getLogger(__name__)


class AoPSIdeaLabelerWorkflow(AbstractWorkflow[AoPSIdeaLabelerPayload]):
    async def _run(
        self,
        work_item: WorkItem[AoPSIdeaLabelerPayload],
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

        # Filter solution posts and track which ones need processing
        solution_posts_to_process: List[tuple[int, Dict[str, Any]]] = []
        for idx, post in enumerate(posts):
            if post.get("is_solution") is True:
                # Check if already processed
                if "idea_labeler_output" not in post:
                    solution_posts_to_process.append((idx, post))
                else:
                    logger.debug(f"Post at index {idx} already has idea_labeler_output, skipping")

        logger.info(f"Found {len(solution_posts_to_process)} solution posts to process")

        # Process each solution post
        updated_posts = posts.copy()
        processed_count = 0
        failed_count = 0

        for idx, post in solution_posts_to_process:
            try:
                solution_text = post.get("text", "")
                if not solution_text:
                    logger.warning(f"Post at index {idx} has no text field, skipping")
                    failed_count += 1
                    continue

                logger.info(f"Processing solution post at index {idx}")

                # Build context for the stage
                stage_context = {
                    "problem_statement": problem_statement,
                    "solution": solution_text,
                    "run_log": run_log,
                }

                # Execute IdeaLabeler stage
                result = await self._execute_stage(
                    "IdeaLabeler", stage_context, run_log, document_id, workflow_name
                )

                # Extract structured output from raw_output (Pydantic model)
                idea_output = None
                raw = result.get("raw_output")
                
                if raw:
                    # Extract from Pydantic model
                    if hasattr(raw, "main_idea"):
                        idea_output = {
                            "main_idea": raw.main_idea,
                            "proof_sketch": raw.proof_sketch,
                            "tags": list(raw.tags) if hasattr(raw, "tags") else [],
                        }
                    elif isinstance(raw, dict):
                        idea_output = {
                            "main_idea": raw.get("main_idea", ""),
                            "proof_sketch": raw.get("proof_sketch", ""),
                            "tags": raw.get("tags", []),
                        }

                if idea_output and idea_output.get("main_idea"):
                    # Add idea_labeler_output to the post
                    updated_posts[idx] = {**post, "idea_labeler_output": idea_output}
                    processed_count += 1
                    logger.info(f"Successfully processed post at index {idx}")
                else:
                    logger.warning(f"Failed to extract idea_labeler_output for post at index {idx}")
                    failed_count += 1

            except Exception as e:
                logger.error(f"Error processing post at index {idx}: {e}", exc_info=True)
                failed_count += 1
                # Continue with other posts even if one fails
                continue

        logger.info(
            f"Completed processing: {processed_count} succeeded, {failed_count} failed out of {len(solution_posts_to_process)} solution posts"
        )

        # Return updated posts array
        return {
            "status": "completed" if processed_count > 0 else "no_posts_processed",
            "posts": updated_posts,
            "processed_count": processed_count,
            "failed_count": failed_count,
        }



