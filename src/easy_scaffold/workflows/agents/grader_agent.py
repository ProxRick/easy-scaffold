# src/easy_scaffold/workflows/agents/grader_agent.py
import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from easy_scaffold.configs.pydantic_models import (
    ErrorItem,
    GradingResultOutput,
    GraderJudgeOutput,
    SolutionClusterItem,
    SolutionClusteringOutput,
    SimilarityMatchOutput,
    OverallAssessment,
)
from easy_scaffold.db.pydantic_models import RunLog
from easy_scaffold.workflows.base import AbstractWorkflow
from easy_scaffold.workflows.workflow_models import GraderPayload, WorkItem

logger = logging.getLogger(__name__)


class GraderAgentWorkflow(AbstractWorkflow[GraderPayload]):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Parse workflow config with defaults for backward compatibility
        # Support both flattened (new) and nested (old) formats
        if "workflow_config" in self._config:
            # Old nested format
            workflow_config = self._config.get("workflow_config", {})
            self._variant = workflow_config.get("variant", "5_stage")
            self._stage_selection = workflow_config.get("stages", {})
            self._grader_name = workflow_config.get("grader_name", "default")
            self._top_k_solutions = workflow_config.get("top_k_solutions", None)
        else:
            # New flattened format
            self._variant = self._config.get("variant", "5_stage")
            self._stage_selection = self._config.get("stages", {})
            self._grader_name = self._config.get("grader_name", "default")
            self._top_k_solutions = self._config.get("top_k_solutions", None)
        
        # Default stage names for backward compatibility
        if not self._stage_selection:
            self._stage_selection = {
                "solution_analysis": "SolutionAnalysis",
                "rubric_generation": "RubricGeneration",
                "grading": "Grading",
            }
        else:
            # Ensure all required stages are specified
            if "grading" not in self._stage_selection:
                self._stage_selection["grading"] = "Grading"
            if self._variant == "5_stage":
                if "solution_analysis" not in self._stage_selection:
                    self._stage_selection["solution_analysis"] = "SolutionAnalysis"
                if "rubric_generation" not in self._stage_selection:
                    self._stage_selection["rubric_generation"] = "RubricGeneration"
        
        logger.info(f"Using workflow variant: {self._variant}")
        logger.info(f"Stage selection: {self._stage_selection}")
        logger.info(f"Grader name: {self._grader_name}")
        logger.info(f"Top K solutions for clustering: {self._top_k_solutions if self._top_k_solutions else 'all'}")

    async def _run(
        self,
        work_item: WorkItem[GraderPayload],
        run_log: RunLog,
        document_id: str,
        workflow_name: str,
    ) -> Dict[str, Any]:
        """Main entry point that routes to variant-specific execution."""
        if self._variant == "1_stage":
            return await self._run_1_stage(work_item, run_log, document_id, workflow_name)
        elif self._variant == "3_stage":
            return await self._run_3_stage(work_item, run_log, document_id, workflow_name)
        elif self._variant == "5_stage":
            return await self._run_5_stage(work_item, run_log, document_id, workflow_name)
        else:
            raise ValueError(f"Unknown workflow variant: {self._variant}")

    async def _run_1_stage(
        self,
        work_item: WorkItem[GraderPayload],
        run_log: RunLog,
        document_id: str,
        workflow_name: str,
    ) -> Dict[str, Any]:
        """1-stage variant: Direct grading only."""
        payload = work_item.payload
        context: Dict[str, Any] = payload.model_dump()
        context["run_log"] = run_log

        given_solutions = payload.given_solutions.copy()
        solutions_to_grade = []

        for solution_id, solution_data in given_solutions.items():
            is_graded = self._is_solution_graded(solution_data, self._grader_name)
            logger.debug(f"Solution {solution_id} graded check result: {is_graded} for grader {self._grader_name}")
            if is_graded:
                logger.info(f"Solution {solution_id} already graded by {self._grader_name}, skipping...")
                continue
            solutions_to_grade.append((solution_id, solution_data))

        logger.info(f"Processing {len(solutions_to_grade)} solutions to grade (1-stage variant)")

        grading_stage_name = self._stage_selection["grading"]

        for solution_id, solution_data in solutions_to_grade:
            logger.info(f"Grading solution {solution_id} with grader {self._grader_name}...")
            current_student_solution = solution_data.get("solution", "")
            context["current_student_solution"] = current_student_solution

            # Direct grading without clustering, similarity, analysis, or rubric
            result = await self._execute_stage(
                grading_stage_name, context, run_log, document_id, workflow_name
            )
            raw_output = result.get("raw_output")

            grading_result = self._extract_grading_result(raw_output, context)
            # Store under grading_results[grader_name]
            if "grading_results" not in given_solutions[solution_id]:
                given_solutions[solution_id]["grading_results"] = {}
            given_solutions[solution_id]["grading_results"][self._grader_name] = grading_result.model_dump()

        return {
            "status": "completed",
            "given_solutions": given_solutions,
        }

    async def _run_3_stage(
        self,
        work_item: WorkItem[GraderPayload],
        run_log: RunLog,
        document_id: str,
        workflow_name: str,
    ) -> Dict[str, Any]:
        """3-stage variant: Clustering → Similarity → Grading."""
        payload = work_item.payload
        context: Dict[str, Any] = payload.model_dump()
        context["run_log"] = run_log

        # Load cached clustering if available (check new path first, fallback to old path for backward compatibility)
        expert_clusters = None
        posts_in_order = []  # For post_id lookup
        
        # Try new path first (grader_name-specific)
        if payload.cached_clustering:
            expert_clusters = payload.cached_clustering
            logger.info(f"Loaded cached clustering from grader-specific path with {len(expert_clusters)} clusters")
        else:
            # Backward compatibility: check old path (only if grader_name is "default" to avoid conflicts)
            # Note: This requires manual document lookup, so we'll skip it for now
            # Old cached data will need to be regenerated or migrated
            logger.debug("No cached clustering found at grader-specific path")

        new_clustering = False
        top_k_indices_to_cache = None

        # Solution Clustering (if not cached)
        if expert_clusters is None:
            # Best-of-K filtering phase
            logger.info("Running best-of-K filtering for clustering...")
            filtered_posts, top_k_indices_to_cache = await self._filter_top_k_solutions(
                payload.forum_posts,
                payload.problem_statement,
                run_log,
                document_id,
                workflow_name,
                cached_top_k_indices=payload.cached_top_k_indices,
            )
            
            # Format reference solutions with filtered posts
            reference_solutions_text, posts_in_order = self._format_reference_solutions(
                payload.forum_posts, filtered_posts=filtered_posts
            )
            context["reference_solutions_text"] = reference_solutions_text
            
            logger.info(f"Running solution clustering on {len(posts_in_order)} solutions...")
            result = await self._execute_stage(
                "SolutionClustering", context, run_log, document_id, workflow_name
            )
            expert_clusters = self._extract_clusters(result.get("raw_output"), context)
            context["expert_clusters"] = expert_clusters
            new_clustering = True
            logger.info(f"Generated clustering with {len(expert_clusters)} clusters")
        else:
            context["expert_clusters"] = expert_clusters
            # For cached clustering, we still need posts_in_order for lookup
            # Format all solutions to get the order (won't be used for clustering, just lookup)
            _, posts_in_order = self._format_reference_solutions(payload.forum_posts)

        given_solutions = payload.given_solutions.copy()
        solutions_to_grade = []

        for solution_id, solution_data in given_solutions.items():
            is_graded = self._is_solution_graded(solution_data, self._grader_name)
            logger.debug(f"Solution {solution_id} graded check result: {is_graded} for grader {self._grader_name}")
            if is_graded:
                logger.info(f"Solution {solution_id} already graded by {self._grader_name}, skipping...")
                continue
            solutions_to_grade.append((solution_id, solution_data))

        logger.info(f"Processing {len(solutions_to_grade)} solutions to grade (3-stage variant)")

        grading_stage_name = self._stage_selection["grading"]

        # Prepare shared context for all solutions
        expert_clusters_json = self._clusters_to_json_string(expert_clusters, posts_in_order)
        base_context = context.copy()
        base_context["expert_clusters_json"] = expert_clusters_json

        # Process solutions in parallel with concurrency limit
        max_concurrent_solutions = 5
        semaphore = asyncio.Semaphore(max_concurrent_solutions)

        async def process_single_solution(solution_id: str, solution_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
            """Process a single solution through the 3-stage pipeline."""
            async with semaphore:
                try:
                    solution_context = base_context.copy()
                    current_student_solution = solution_data.get("solution", "")
                    solution_context["current_student_solution"] = current_student_solution

                    logger.info(f"Grading solution {solution_id} with grader {self._grader_name}...")

                    # Similarity Assessment
                    result = await self._execute_stage(
                        "SimilarityAssessment", solution_context, run_log, document_id, workflow_name
                    )
                    similarity_match = self._extract_similarity_match(result.get("raw_output"), solution_context)

                    cluster_id = similarity_match.closest_rep_id
                    representative_solution = self._get_representative_solution(
                        expert_clusters, cluster_id, posts_in_order
                    )

                    if not representative_solution:
                        logger.error(
                            f"Could not find representative solution for cluster {cluster_id} (solution {solution_id})"
                        )
                        return None

                    solution_context["representative_solution"] = representative_solution

                    # Grading (no analysis or rubric)
                    result = await self._execute_stage(
                        grading_stage_name, solution_context, run_log, document_id, workflow_name
                    )
                    raw_output = result.get("raw_output")

                    grading_result = self._extract_grading_result(raw_output, solution_context)
                    return {
                        "solution_id": solution_id,
                        "grading_result": grading_result
                    }
                except Exception as e:
                    logger.error(f"Error processing solution {solution_id}: {e}", exc_info=True)
                    return None

        # Process all solutions in parallel
        tasks = [
            process_single_solution(solution_id, solution_data)
            for solution_id, solution_data in solutions_to_grade
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Store results
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Exception in solution processing: {result}", exc_info=True)
                continue
            if result is None:
                continue
            
            solution_id = result["solution_id"]
            grading_result = result["grading_result"]
            if "grading_results" not in given_solutions[solution_id]:
                given_solutions[solution_id]["grading_results"] = {}
            given_solutions[solution_id]["grading_results"][self._grader_name] = grading_result.model_dump()

        final_context: Dict[str, Any] = {"status": "completed", "given_solutions": given_solutions}
        if new_clustering:
            final_context["clustering"] = expert_clusters
        # Cache top_k_indices if we generated new ones (3-stage workflow)
        if top_k_indices_to_cache is not None:
            final_context["top_k_indices"] = top_k_indices_to_cache

        return final_context

    async def _run_5_stage(
        self,
        work_item: WorkItem[GraderPayload],
        run_log: RunLog,
        document_id: str,
        workflow_name: str,
    ) -> Dict[str, Any]:
        """5-stage variant: Clustering → Similarity → Analysis → Rubric → Grading."""
        payload = work_item.payload
        context: Dict[str, Any] = payload.model_dump()
        context["run_log"] = run_log

        # Load cached data if available (check new grader_name-specific paths)
        expert_clusters = None
        cached_analysis = {}
        cached_rubrics = {}
        posts_in_order = []  # For post_id lookup

        if payload.cached_clustering:
            expert_clusters = payload.cached_clustering
            logger.info(f"Loaded cached clustering from grader-specific path with {len(expert_clusters)} clusters")
        else:
            logger.debug("No cached clustering found at grader-specific path")

        if payload.cached_analysis:
            cached_analysis = payload.cached_analysis
            logger.info(f"Loaded cached analysis from grader-specific path for {len(cached_analysis)} clusters")
        else:
            logger.debug("No cached analysis found at grader-specific path")

        if payload.cached_rubrics:
            cached_rubrics = payload.cached_rubrics
            logger.info(f"Loaded cached rubrics from grader-specific path for {len(cached_rubrics)} clusters")
        else:
            logger.debug("No cached rubrics found at grader-specific path")

        # Track newly generated data for saving
        new_clustering = False
        new_analysis: Dict[str, str] = {}
        new_rubrics: Dict[str, str] = {}
        top_k_indices_to_cache = None

        # Solution Clustering (if not cached)
        if expert_clusters is None:
            # Best-of-K filtering phase
            logger.info("Running best-of-K filtering for clustering...")
            filtered_posts, top_k_indices_to_cache = await self._filter_top_k_solutions(
                payload.forum_posts,
                payload.problem_statement,
                run_log,
                document_id,
                workflow_name,
                cached_top_k_indices=payload.cached_top_k_indices,
            )
            
            # Format reference solutions with filtered posts
            reference_solutions_text, posts_in_order = self._format_reference_solutions(
                payload.forum_posts, filtered_posts=filtered_posts
            )
            context["reference_solutions_text"] = reference_solutions_text
            
            logger.info(f"Running solution clustering on {len(posts_in_order)} solutions...")
            result = await self._execute_stage(
                "SolutionClustering", context, run_log, document_id, workflow_name
            )
            expert_clusters = self._extract_clusters(result.get("raw_output"), context)
            context["expert_clusters"] = expert_clusters
            new_clustering = True
            logger.info(f"Generated clustering with {len(expert_clusters)} clusters")
        else:
            context["expert_clusters"] = expert_clusters
            # For cached clustering, we still need posts_in_order for lookup
            # Format all solutions to get the order (won't be used for clustering, just lookup)
            _, posts_in_order = self._format_reference_solutions(payload.forum_posts)

        # Process each given solution
        given_solutions = payload.given_solutions.copy()
        solutions_to_grade = []

        for solution_id, solution_data in given_solutions.items():
            is_graded = self._is_solution_graded(solution_data, self._grader_name)
            logger.debug(f"Solution {solution_id} graded check result: {is_graded} for grader {self._grader_name}")
            if is_graded:
                logger.info(f"Solution {solution_id} already graded by {self._grader_name}, skipping...")
                continue
            solutions_to_grade.append((solution_id, solution_data))

        logger.info(f"Processing {len(solutions_to_grade)} solutions to grade (5-stage variant)")

        analysis_stage_name = self._stage_selection["solution_analysis"]
        rubric_stage_name = self._stage_selection["rubric_generation"]
        grading_stage_name = self._stage_selection["grading"]

        # Prepare shared context for all solutions
        expert_clusters_json = self._clusters_to_json_string(expert_clusters, posts_in_order)
        base_context = context.copy()
        base_context["expert_clusters_json"] = expert_clusters_json

        # Process solutions in parallel with concurrency limit
        # Use semaphore to limit concurrent API calls (avoid overwhelming rate limits)
        max_concurrent_solutions = 5  # Process up to 5 solutions concurrently
        semaphore = asyncio.Semaphore(max_concurrent_solutions)
        
        # Shared locks for cluster-level caching (analysis/rubrics)
        analysis_lock = asyncio.Lock()
        rubric_lock = asyncio.Lock()

        async def process_single_solution(solution_id: str, solution_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
            """Process a single solution through the 5-stage pipeline."""
            async with semaphore:
                try:
                    solution_context = base_context.copy()
                    current_student_solution = solution_data.get("solution", "")
                    solution_context["current_student_solution"] = current_student_solution

                    logger.info(f"Grading solution {solution_id}...")

                    # Similarity Assessment
                    result = await self._execute_stage(
                        "SimilarityAssessment", solution_context, run_log, document_id, workflow_name
                    )
                    similarity_match = self._extract_similarity_match(result.get("raw_output"), solution_context)

                    cluster_id = similarity_match.closest_rep_id
                    representative_solution = self._get_representative_solution(
                        expert_clusters, cluster_id, posts_in_order
                    )

                    if not representative_solution:
                        logger.error(
                            f"Could not find representative solution for cluster {cluster_id} (solution {solution_id})"
                        )
                        return None

                    solution_context["representative_solution"] = representative_solution
                    solution_context["cluster_id"] = cluster_id

                    # Solution Analysis (per cluster, with caching)
                    # Use shared lock to prevent duplicate analysis generation for the same cluster
                    analysis_text = cached_analysis.get(cluster_id)
                    if analysis_text is None:
                        # Check again after acquiring lock (double-check pattern)
                        async with analysis_lock:
                            analysis_text = cached_analysis.get(cluster_id)
                            if analysis_text is None:
                                logger.info(f"Generating analysis for cluster {cluster_id}...")
                                result = await self._execute_stage(
                                    analysis_stage_name, solution_context, run_log, document_id, workflow_name
                                )
                                analysis_text = solution_context.get("solution_analysis_text", "")
                                if analysis_text:
                                    new_analysis[cluster_id] = analysis_text
                                    cached_analysis[cluster_id] = analysis_text
                    else:
                        logger.debug(f"Using cached analysis for cluster {cluster_id}")
                    
                    solution_context["solution_analysis_text"] = cached_analysis.get(cluster_id, "")

                    # Rubric Generation (per cluster, with caching)
                    rubric_text = cached_rubrics.get(cluster_id)
                    if rubric_text is None:
                        # Check again after acquiring lock (double-check pattern)
                        async with rubric_lock:
                            rubric_text = cached_rubrics.get(cluster_id)
                            if rubric_text is None:
                                logger.info(f"Generating rubric for cluster {cluster_id}...")
                                result = await self._execute_stage(
                                    rubric_stage_name, solution_context, run_log, document_id, workflow_name
                                )
                                rubric_text = solution_context.get("rubric_text", "")
                                if rubric_text:
                                    new_rubrics[cluster_id] = rubric_text
                                    cached_rubrics[cluster_id] = rubric_text
                    else:
                        logger.debug(f"Using cached rubric for cluster {cluster_id}")
                    
                    solution_context["rubric_text"] = cached_rubrics.get(cluster_id, "")

                    # Grading
                    logger.info(f"Grading solution {solution_id} with grader {self._grader_name}...")
                    result = await self._execute_stage(
                        grading_stage_name, solution_context, run_log, document_id, workflow_name
                    )
                    raw_output = result.get("raw_output")

                    grading_result = self._extract_grading_result(raw_output, solution_context)
                    return {
                        "solution_id": solution_id,
                        "grading_result": grading_result
                    }
                except Exception as e:
                    logger.error(f"Error processing solution {solution_id}: {e}", exc_info=True)
                    return None

        # Process all solutions in parallel
        tasks = [
            process_single_solution(solution_id, solution_data)
            for solution_id, solution_data in solutions_to_grade
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Store results
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Exception in solution processing: {result}", exc_info=True)
                continue
            if result is None:
                continue
            
            solution_id = result["solution_id"]
            grading_result = result["grading_result"]
            if "grading_results" not in given_solutions[solution_id]:
                given_solutions[solution_id]["grading_results"] = {}
            given_solutions[solution_id]["grading_results"][self._grader_name] = grading_result.model_dump()

        # Build final output dict
        final_context: Dict[str, Any] = {"status": "completed"}

        # Only include newly generated clustering
        if new_clustering:
            final_context["clustering"] = expert_clusters

        # Only include newly generated analysis (nested dict)
        if new_analysis:
            final_context["analysis"] = new_analysis

        # Only include newly generated rubrics (nested dict)
        if new_rubrics:
            final_context["rubrics"] = new_rubrics

        # Cache top_k_indices if we generated new ones
        if top_k_indices_to_cache is not None:
            final_context["top_k_indices"] = top_k_indices_to_cache

        # Always include updated given_solutions
        final_context["given_solutions"] = given_solutions

        return final_context

    def _format_reference_solutions(
        self, posts: List[Dict[str, Any]], filtered_posts: Optional[List[Tuple[int, Dict[str, Any]]]] = None
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Format forum posts into reference solutions string for clustering.
        
        Args:
            posts: All forum posts
            filtered_posts: Optional list of (index, post) tuples if pre-filtered (for best-of-K)
        
        Returns:
            Tuple of (formatted_solutions_string, list_of_posts_in_order)
            The list_of_posts_in_order is used for post_id lookup later
        """
        if filtered_posts is None:
            # Use all posts, filter for solutions
            solution_posts: List[Tuple[int, Dict[str, Any]]] = []
            for idx, post in enumerate(posts):
                if post.get("is_solution", False):
                    text = post.get("text", "")
                    if text:
                        solution_posts.append((idx, post))
        else:
            # Use pre-filtered posts
            solution_posts = filtered_posts
        
        if not solution_posts:
            return "", []
        
        # Format with clear delimiters (like solution_selection.py)
        formatted = []
        posts_in_order = []
        for i, (original_idx, post) in enumerate(solution_posts, 1):
            text = post.get("text", "")
            formatted.append(
                f"--- Solution {i} (Post ID: {original_idx}) of {len(solution_posts)} ---\n"
                f"{text}\n"
                f"--- End of Solution {i} ---\n"
            )
            posts_in_order.append(post)
        
        return "\n".join(formatted), posts_in_order

    def _is_solution_graded(self, solution_data: Dict[str, Any], grader_name: str) -> bool:
        """Check if solution already has grading result for the specific grader."""
        # Check new structure: grading_results[grader_name]
        grading_results = solution_data.get("grading_results", {})
        
        # Debug logging to understand why solutions are being skipped
        logger.debug(f"Checking if solution is graded by {grader_name}")
        if isinstance(grading_results, dict):
            logger.debug(f"grading_results keys: {list(grading_results.keys())}")
            grading_result = grading_results.get(grader_name)
            logger.debug(f"grading_results[{grader_name}] = {grading_result} (type: {type(grading_result)})")
            
            if grading_result:
                if isinstance(grading_result, dict):
                    # Check for standard format (GradingResultOutput)
                    has_overall = "overall_assessment" in grading_result
                    has_structure = "solution_structure_analysis" in grading_result
                    has_designed = "designed_marking_scheme" in grading_result
                    
                    logger.debug(f"grading_result keys: {list(grading_result.keys())}")
                    logger.debug(f"has overall_assessment: {has_overall}, has solution_structure_analysis: {has_structure}, has designed_marking_scheme: {has_designed}")
                    
                    if has_overall and has_structure:
                        # Additional validation: check if overall_assessment has a score
                        # A grading result without a score is incomplete and should be re-graded
                        overall = grading_result.get("overall_assessment", {})
                        has_score = isinstance(overall, dict) and "score" in overall
                        logger.debug(
                            f"Standard format check for {grader_name}: "
                            f"Keys: {list(grading_result.keys())}. "
                            f"Has score: {has_score}. "
                            f"overall_assessment type: {type(overall)}, keys: {list(overall.keys()) if isinstance(overall, dict) else 'N/A'}"
                        )
                        if has_score:
                            logger.info(f"Found complete standard grading result format for {grader_name}")
                            return True
                        else:
                            logger.warning(
                                f"Found incomplete grading result for {grader_name} (missing score). "
                                f"Will re-grade this solution."
                            )
                            return False
                    # Check for grader_judge format (GraderJudgeOutput)
                    if has_overall and has_designed:
                        # Additional validation: check if overall_assessment has a score
                        # A grading result without a score is incomplete and should be re-graded
                        overall = grading_result.get("overall_assessment", {})
                        has_score = isinstance(overall, dict) and "score" in overall
                        logger.debug(
                            f"Grader judge format check for {grader_name}: "
                            f"Keys: {list(grading_result.keys())}. "
                            f"Has score: {has_score}. "
                            f"overall_assessment type: {type(overall)}, keys: {list(overall.keys()) if isinstance(overall, dict) else 'N/A'}"
                        )
                        if has_score:
                            logger.info(f"Found complete grader_judge format for {grader_name}")
                            return True
                        else:
                            logger.warning(
                                f"Found incomplete grading result for {grader_name} (missing score). "
                                f"Will re-grade this solution."
                            )
                            return False
                    # Log if grading_result exists but doesn't match expected formats
                    logger.warning(
                        f"Found grading_result for {grader_name} but it doesn't match expected formats. "
                        f"Keys: {list(grading_result.keys())}. Will NOT skip solution."
                    )
                else:
                    logger.warning(
                        f"grading_results[{grader_name}] exists but is not a dict (type: {type(grading_result)}, value: {grading_result}). Will NOT skip solution."
                    )
            else:
                logger.debug(f"No grading_result found for {grader_name} in grading_results")
        else:
            logger.debug(f"grading_results is not a dict (type: {type(grading_results)})")
        
        # Backward compatibility: check old structure (grading_result) if grader_name is "default"
        if grader_name == "default":
            grading_result = solution_data.get("grading_result")
            if grading_result and isinstance(grading_result, dict):
                # Check for standard format (GradingResultOutput)
                if "overall_assessment" in grading_result and "solution_structure_analysis" in grading_result:
                    return True
                # Check for grader_judge format (GraderJudgeOutput)
                if "overall_assessment" in grading_result and "designed_marking_scheme" in grading_result:
                    return True

        return False

    def _get_representative_solution(
        self, clusters: List[Dict[str, Any]], cluster_id: str, posts_in_order: Optional[List[Dict[str, Any]]] = None
    ) -> Optional[str]:
        """
        Get representative solution for a given cluster ID.
        
        Args:
            clusters: List of cluster dicts
            cluster_id: Cluster ID to look up (e.g., "C1")
            posts_in_order: List of posts in the order they were sent to clustering (for post_id lookup)
        
        Returns:
            Representative solution text, or None if not found
        """
        for cluster in clusters:
            if isinstance(cluster, dict):
                if cluster.get("class_id") == cluster_id:
                    # Check if using new format (post_id) or old format (solution text)
                    if "representative_post_id" in cluster:
                        post_id = cluster.get("representative_post_id")
                        # post_id is 1-based in the formatted input
                        if posts_in_order and 1 <= post_id <= len(posts_in_order):
                            return posts_in_order[post_id - 1].get("text", "")
                        else:
                            logger.warning(f"Invalid post_id {post_id} for cluster {cluster_id}")
                            return None
                    elif "representative_solution" in cluster:
                        # Backward compatibility: old format with solution text
                        return cluster.get("representative_solution")
            elif hasattr(cluster, "class_id"):
                if cluster.class_id == cluster_id:
                    if hasattr(cluster, "representative_post_id"):
                        post_id = cluster.representative_post_id
                        if posts_in_order and 1 <= post_id <= len(posts_in_order):
                            return posts_in_order[post_id - 1].get("text", "")
                        else:
                            logger.warning(f"Invalid post_id {post_id} for cluster {cluster_id}")
                            return None
                    elif hasattr(cluster, "representative_solution"):
                        return cluster.representative_solution
        return None

    def _clusters_to_json_string(
        self, 
        clusters: List[Dict[str, Any]], 
        posts_in_order: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """
        Convert clusters list to JSON string for similarity assessment prompt.
        
        Args:
            clusters: List of cluster dicts (may have representative_post_id or representative_solution)
            posts_in_order: List of posts in the order they were sent to clustering (for post_id lookup)
        
        Returns:
            JSON string with clusters containing representative_solution text
        """
        # Convert to serializable format
        serializable_clusters = []
        for cluster in clusters:
            cluster_dict = None
            
            if isinstance(cluster, dict):
                cluster_dict = cluster.copy()
            elif hasattr(cluster, "model_dump"):
                cluster_dict = cluster.model_dump()
            else:
                # Fallback: convert to dict
                cluster_dict = {
                    "class_id": getattr(cluster, "class_id", ""),
                    "main_steps": getattr(cluster, "main_steps", []),
                }
                # Check for post_id or solution text
                if hasattr(cluster, "representative_post_id"):
                    cluster_dict["representative_post_id"] = getattr(cluster, "representative_post_id", None)
                elif hasattr(cluster, "representative_solution"):
                    cluster_dict["representative_solution"] = getattr(cluster, "representative_solution", "")
            
            # Convert representative_post_id to representative_solution if needed
            if cluster_dict and "representative_post_id" in cluster_dict:
                post_id = cluster_dict.get("representative_post_id")
                if posts_in_order and post_id is not None:
                    # post_id is 1-based (Solution 1, Solution 2, etc.)
                    # posts_in_order is 0-based list
                    if 1 <= post_id <= len(posts_in_order):
                        solution_text = posts_in_order[post_id - 1].get("text", "")
                        cluster_dict["representative_solution"] = solution_text
                        # Remove post_id from output (similarity assessment doesn't need it)
                        cluster_dict.pop("representative_post_id", None)
                    else:
                        logger.warning(
                            f"Invalid post_id {post_id} for cluster {cluster_dict.get('class_id', 'unknown')}, "
                            f"expected 1-{len(posts_in_order)}"
                        )
                        cluster_dict["representative_solution"] = ""
                else:
                    logger.warning(
                        f"Cannot resolve post_id {post_id} for cluster {cluster_dict.get('class_id', 'unknown')}: "
                        f"posts_in_order not available"
                    )
                    cluster_dict["representative_solution"] = ""
                    cluster_dict.pop("representative_post_id", None)
            
            # Ensure representative_solution exists (for backward compatibility with old format)
            # If cluster already has representative_solution (old format), keep it
            if cluster_dict and "representative_solution" not in cluster_dict:
                # This shouldn't happen if conversion worked, but provide fallback
                cluster_dict["representative_solution"] = ""
            
            serializable_clusters.append(cluster_dict)
        
        return json.dumps(serializable_clusters, indent=2)

    async def _filter_top_k_solutions(
        self,
        posts: List[Dict[str, Any]],
        problem_statement: str,
        run_log: RunLog,
        document_id: str,
        workflow_name: str,
        cached_top_k_indices: Optional[List[int]] = None,
    ) -> Tuple[List[Tuple[int, Dict[str, Any]]], Optional[List[int]]]:
        """
        Filter and rank solutions using completeness evaluation, then select top K.
        
        Args:
            posts: List of all forum posts
            problem_statement: Problem statement text
            run_log: Run log for logging
            document_id: Document ID
            workflow_name: Workflow name
            cached_top_k_indices: Optional cached list of post indices for top-k solutions
        
        Returns:
            Tuple of (list of (post_index, post) tuples for top K solutions, list of indices to cache)
        """
        if self._top_k_solutions is None:
            # No filtering - return all solutions
            solution_posts: List[Tuple[int, Dict[str, Any]]] = []
            for idx, post in enumerate(posts):
                if post.get("is_solution", False):
                    text = post.get("text", "")
                    if text:
                        solution_posts.append((idx, post))
            # Return all indices for caching (though not really needed when top_k_solutions is None)
            all_indices = [idx for idx, _ in solution_posts]
            return solution_posts, all_indices if cached_top_k_indices is None else None, None
        
        # Check if we have cached top-k indices
        if cached_top_k_indices:
            logger.info(f"Using cached top-k indices: {len(cached_top_k_indices)} solutions")
            # Validate indices and filter posts
            cached_solutions: List[Tuple[int, Dict[str, Any]]] = []
            for idx in cached_top_k_indices:
                if 0 <= idx < len(posts):
                    post = posts[idx]
                    if post.get("is_solution", False):
                        text = post.get("text", "")
                        if text:
                            cached_solutions.append((idx, post))
                else:
                    logger.warning(f"Invalid cached index {idx}, skipping")
            
            if len(cached_solutions) == len(cached_top_k_indices):
                logger.info(f"Successfully loaded {len(cached_solutions)} solutions from cache")
                return cached_solutions, None  # Return cached indices (no need to recache)
            else:
                logger.warning(
                    f"Cached indices validation failed: expected {len(cached_top_k_indices)}, "
                    f"got {len(cached_solutions)}. Re-evaluating..."
                )
                # Fall through to re-evaluation
        
        # Filter solutions
        solution_posts: List[Tuple[int, Dict[str, Any]]] = []
        for idx, post in enumerate(posts):
            if post.get("is_solution", False):
                text = post.get("text", "")
                if text:
                    solution_posts.append((idx, post))
        
        if not solution_posts:
            logger.warning("No solutions found for filtering")
            return [], None
        
        # Optionally sort by like_count first (if available)
        solution_posts.sort(
            key=lambda x: (
                -x[1].get("like_count", 0),  # Negative for descending
                x[0]  # Post index for tie-breaking
            )
        )
        
        # Limit to top_k_solutions before evaluation
        limited_solutions = solution_posts[:self._top_k_solutions]
        logger.info(f"Evaluating {len(limited_solutions)} solutions for best-of-K filtering")
        
        # Run parallel completeness evaluations
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
            
            # Process results and calculate scores
            scored_solutions: List[Tuple[int, Dict[str, Any], float]] = []
            for i, (idx, post) in enumerate(limited_solutions):
                result = results[i]
                
                if isinstance(result, Exception):
                    logger.warning(f"Error evaluating solution (post {idx}): {result}")
                    # Use default low score for failed evaluations
                    combined_score = 0.0
                else:
                    raw_output = result.get("raw_output")
                    if raw_output:
                        if hasattr(raw_output, "model_dump"):
                            eval_data = raw_output.model_dump()
                        elif isinstance(raw_output, dict):
                            eval_data = raw_output
                        else:
                            eval_data = {
                                "completeness_score": getattr(raw_output, "completeness_score", 0.0),
                                "cleanliness_score": getattr(raw_output, "cleanliness_score", 0.0),
                            }
                        completeness_score = eval_data.get("completeness_score", 0.0)
                        cleanliness_score = eval_data.get("cleanliness_score", 0.0)
                        combined_score = completeness_score + cleanliness_score
                    else:
                        combined_score = 0.0
                
                scored_solutions.append((idx, post, combined_score))
            
            # Sort by combined score (descending)
            scored_solutions.sort(key=lambda x: x[2], reverse=True)
            
            # Return top K (or all if less than K)
            top_k = scored_solutions[:self._top_k_solutions]
            logger.info(
                f"Selected top {len(top_k)} solutions "
                f"(scores: {[score for _, _, score in top_k[:5]]}{'...' if len(top_k) > 5 else ''})"
            )
            
            # Extract indices for caching
            top_k_indices = [idx for idx, _, _ in top_k]
            
            # Return list of (post_index, post) tuples and indices to cache
            return [(idx, post) for idx, post, _ in top_k], top_k_indices
            
        except Exception as e:
            logger.error(f"Error in best-of-K filtering: {e}", exc_info=True)
            # Fallback: return limited solutions without scoring
            logger.warning("Falling back to unsorted limited solutions")
            fallback_indices = [idx for idx, _ in limited_solutions]
            return limited_solutions, fallback_indices

    def _extract_clusters(
        self, raw_output: Any, context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Extract clusters from stage output (handles wrapped format)."""
        # Handle wrapped SolutionClusteringOutput format
        if isinstance(raw_output, SolutionClusteringOutput):
            clusters = raw_output.clusters
            return [item.model_dump() if hasattr(item, "model_dump") else item for item in clusters]
        elif isinstance(raw_output, dict) and "clusters" in raw_output:
            clusters = raw_output["clusters"]
            return [
                item.model_dump() if hasattr(item, "model_dump") else item
                for item in clusters
            ] if isinstance(clusters, list) else clusters
        elif isinstance(raw_output, list):
            # Backward compatibility: handle old list format
            return [
                item.model_dump() if hasattr(item, "model_dump") else item
                for item in raw_output
            ]
        else:
            # Fallback: try to get from context
            return context.get("expert_clusters", [])

    def _extract_similarity_match(
        self, raw_output: Any, context: Dict[str, Any]
    ) -> SimilarityMatchOutput:
        """Extract similarity match from stage output."""
        if isinstance(raw_output, SimilarityMatchOutput):
            return raw_output
        elif isinstance(raw_output, dict):
            return SimilarityMatchOutput(**raw_output)
        else:
            # Get from context
            closest_rep_id = context.get("closest_rep_id")
            if not closest_rep_id:
                raise ValueError("Failed to get similarity match")
            return SimilarityMatchOutput(
                closest_rep_id=closest_rep_id,
                justification=context.get("similarity_justification", ""),
            )

    def _extract_grading_result(
        self, raw_output: Any, context: Dict[str, Any]
    ) -> GradingResultOutput:
        """Extract and convert grading result from stage output."""
        # Check if it's a GraderJudgeOutput (new format)
        if isinstance(raw_output, GraderJudgeOutput):
            return self._convert_grader_judge_to_grading_result(raw_output)
        elif isinstance(raw_output, dict):
            # Check if it has grader_judge structure
            if "designed_marking_scheme" in raw_output and "overall_assessment" in raw_output:
                grader_judge = GraderJudgeOutput(**raw_output)
                return self._convert_grader_judge_to_grading_result(grader_judge)
            else:
                # Standard GradingResultOutput format
                return GradingResultOutput(**raw_output)
        elif isinstance(raw_output, GradingResultOutput):
            return raw_output
        else:
            # Try to construct from context fields (backward compatibility)
            return GradingResultOutput(
                overall_assessment=OverallAssessment(
                    score=context.get("grading_assessment", {}).get("score", 0),
                    rationale=context.get("grading_assessment", {}).get("rationale", ""),
                )
                if isinstance(context.get("grading_assessment"), dict)
                else context.get("grading_assessment", OverallAssessment(score=0, rationale="")),
                solution_structure_analysis=context.get("grading_structure_analysis", ""),
                substep_error_analysis=context.get("grading_error_analysis", []),
                cross_solution_consistency=context.get("grading_consistency", ""),
                error_propagation_analysis=context.get("grading_propagation", ""),
                rubric_milestone_assessment=context.get("grading_rubric_assessment", ""),
                clarity_structure_notation=context.get("grading_clarity", ""),
                constructive_feedback=context.get("grading_feedback", ""),
            )

    def _convert_grader_judge_to_grading_result(
        self, grader_judge: GraderJudgeOutput
    ) -> GradingResultOutput:
        """Convert GraderJudgeOutput to GradingResultOutput format for storage compatibility."""
        # Extract score and classification
        score = grader_judge.overall_assessment.score
        classification = grader_judge.overall_assessment.classification
        
        # Build rationale from classification and feedback
        rationale = f"{classification}. "
        if grader_judge.feedback.achieved_milestones:
            rationale += f"Achieved milestones: {', '.join(grader_judge.feedback.achieved_milestones)}. "
        if grader_judge.feedback.missed_milestones:
            rationale += f"Missed milestones: {', '.join(grader_judge.feedback.missed_milestones)}."
        
        # Convert errors to ErrorItem format
        error_items = []
        for error in grader_judge.feedback.errors:
            error_items.append(
                ErrorItem(
                    type=error.severity,
                    description=error.issue,
                    location=error.location,
                )
            )
        
        # Build rubric assessment from designed marking scheme
        rubric_text = f"Rubric Summary: {grader_judge.designed_marking_scheme.summary}\n"
        rubric_text += "Milestones:\n"
        for milestone in grader_judge.designed_marking_scheme.milestones:
            rubric_text += f"- {milestone.points} points: {milestone.description}\n"
        
        # Build feedback text
        feedback_text = f"Classification: {classification}\n"
        if grader_judge.feedback.achieved_milestones:
            feedback_text += f"Achieved: {', '.join(grader_judge.feedback.achieved_milestones)}\n"
        if grader_judge.feedback.missed_milestones:
            feedback_text += f"Missed: {', '.join(grader_judge.feedback.missed_milestones)}\n"
        if grader_judge.feedback.errors:
            feedback_text += "Errors:\n"
            for error in grader_judge.feedback.errors:
                feedback_text += f"- [{error.severity}] {error.issue} at: {error.location}\n"
        
        return GradingResultOutput(
            overall_assessment=OverallAssessment(score=score, rationale=rationale.strip()),
            solution_structure_analysis="",  # Not available in grader_judge format
            substep_error_analysis=error_items,
            cross_solution_consistency="",  # Not available in grader_judge format
            error_propagation_analysis="",  # Not available in grader_judge format
            rubric_milestone_assessment=rubric_text,
            clarity_structure_notation="",  # Not available in grader_judge format
            constructive_feedback=feedback_text,
        )



