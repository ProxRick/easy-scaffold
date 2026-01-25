from typing import Any, Dict, Generic, List, Literal, Optional, TypeVar, Union

from pydantic import BaseModel, Field, model_validator


class ProblemPayload(BaseModel):
    """Flat payload for problem-centric workflows."""
    problem_id: str
    statement: str


class SolutionPayload(BaseModel):
    """Flat payload for solution-centric workflows."""
    problem_id: str
    statement: str
    solution_id: str
    solution_text: str


class GraderPayload(BaseModel):
    """Payload for grader workflow."""
    problem_id: str
    problem_statement: str
    given_solutions: Dict[str, Dict[str, Any]]  # Keys: "S1", "S2", etc. Each value has metadata + "solution" field
    forum_posts: List[Dict[str, Any]]  # Each post has "is_solution" bool and "text" field
    # Optional cached data (will be None if not present in document)
    cached_clustering: Optional[List[Dict[str, Any]]] = None
    cached_analysis: Optional[Dict[str, str]] = None  # Dict mapping cluster_id -> analysis text
    cached_rubrics: Optional[Dict[str, str]] = None  # Dict mapping cluster_id -> rubric text
    cached_top_k_indices: Optional[List[int]] = None  # List of post indices for top-k solutions (global cache)


class CoTVerificationPayload(BaseModel):
    """Payload for CoT verification workflow."""
    problem_id: str
    problem_statement: str
    cot_trajectory: str
    attempted_solution: str
    # Optional cached data (will be None if not present in document)
    cached_cot_verification_report: Optional[str] = None
    verification_result: Optional[Dict[str, Any]] = None
    rigorous_fixer_output: Optional[Dict[str, Any]] = None
    rigorous_emulator_output: Optional[str] = None
    rigorous_fixed_cot: Optional[str] = None
    rigorous_fixed_attempt: Optional[str] = None


class AoPSIdeaLabelerPayload(BaseModel):
    """Payload for AoPS idea labeling workflow."""
    problem_id: str
    problem_statement: str
    posts: List[Dict[str, Any]]


class SolutionSelectionPayload(BaseModel):
    """Payload for solution selection workflow."""
    problem_id: str
    problem_statement: str
    posts: List[Dict[str, Any]]  # Forum posts array
    # Optional cached data (will be None if not present in document)
    cached_top_k_candidates: Optional[List[Dict[str, Any]]] = None


class IssueCatcherPayload(BaseModel):
    """Payload for issue catcher workflow - checks if solutions admit self-reported issues."""
    problem_id: str
    problem_statement: str
    solution: str


class GeneratorCritiquePayload(BaseModel):
    """Payload for generator-critique workflow."""
    problem_id: str
    problem_statement: str
    reference_solution: str
    # Optional fields for resuming from a previous run
    existing_cot: Optional[str] = None
    existing_hint: Optional[str] = None
    existing_mistake: Optional[str] = None
    existing_hint_level: Optional[int] = Field(default=0, description="Resume hint level, defaults to 0 if missing")


class BaselineGeneratorPayload(BaseModel):
    """Payload for baseline generator workflow (one-shot generation without critique)."""
    problem_id: str
    problem_statement: str
    reference_solution: Optional[str] = None  # Optional, for consistency with other workflows


class EvalPayload(BaseModel):
    """Payload for evaluation workflow with multi-sample support."""
    problem_id: str
    problem_statement: str
    reference_solution: Optional[str] = None
    ground_truth_answer: Optional[str] = None
    # Map of model_name -> List[solution_strings]
    cached_responses: Optional[Dict[str, List[str]]] = None


class SolutionVerificationPayload(BaseModel):
    """Payload for solution verification workflow - verifies existing solutions."""
    problem_id: str
    problem_statement: str
    solution_to_verify: str
    reference_solution: Optional[str] = None
    ground_truth_answer: Optional[str] = None


class CoTConclusionPayload(BaseModel):
    """Payload for CoT conclusion workflow - adds a conclusive paragraph to end of CoT."""
    problem_id: str
    problem_statement: str
    generator_cot: str


class CoTMonitoringPayload(BaseModel):
    """Payload for CoT monitoring workflow - analyzes behavioral patterns in Chain of Thought."""
    problem_id: str
    problem_statement: str
    solution_or_cot: Union[str, List[str]]  # Single CoT/solution OR list of them
    model_name: Optional[str] = None  # For eval results context (optional)


class CoTImprovementPayload(BaseModel):
    """Payload for CoT improvement workflow - detects loops, normalizes steps, and adds conclusion."""
    problem_id: str
    problem_statement: str
    generator_cot: str  # Original CoT to improve


class LoopDetectionResponse(BaseModel):
    """Response from loop detection stage."""
    has_loop: bool = Field(description="Whether a looping pattern exists")
    loop_start_index: Optional[int] = Field(
        default=None,
        description="Index of the step where looping behavior starts (0-indexed). Only set if has_loop=True"
    )
    reasoning: str = Field(
        description="Thorough explanation of why the loop pattern exists or doesn't exist, and why the specific start index was chosen"
    )


class LoopRewritingResponse(BaseModel):
    """Response from loop rewriting stage."""
    rewritten_section: str = Field(
        description="Rewritten section with paragraphs separated by \\n\\n. Repetitive checks consolidated into itemized markdown lists."
    )


class LoopFidelityVerification(BaseModel):
    """Response from loop fidelity verification stage."""
    is_faithful: bool = Field(description="Whether the rewritten section maintains fidelity to the original")
    reasoning: str = Field(description="Explanation of the fidelity assessment")


class CoTStepBehaviorLabel(BaseModel):
    """Represents a single step with its behavior labels."""
    step_index: int = Field(description="0-indexed position of the step in the CoT")
    step_content: str = Field(description="The actual content of the step")
    behaviors: List[str] = Field(
        description="List of behavior labels (can be empty, 0-3 behaviors). Valid behaviors: Knowledge Retrieval, Example Testing & Problem Engagement, Representation Shift, Subgoal Setting, Pattern Recognition, Hypothesis Generation, Forward Deduction, Backward Chaining, Verification, Backtracking"
    )
    reasoning: str = Field(description="Explanation of why these behaviors were assigned")


class CoTBehaviorLabelingResponse(BaseModel):
    """Response from multi-behavior labeling stage."""
    labeled_steps: List[CoTStepBehaviorLabel] = Field(
        description="List of all steps with their behavior labels. Must have same length as input steps."
    )


class NormalizedStep(BaseModel):
    """A normalized step with single behavior."""
    content: str = Field(description="The content of the normalized step")
    behavior: str = Field(
        description="Single behavior label for this step. Must be one of the 10 valid behaviors."
    )
    reasoning: str = Field(
        description="Explanation of why this step is separate and why it has this single behavior"
    )


class CoTStepNormalizationResponse(BaseModel):
    """Response from step normalization stage."""
    normalized_steps: List[NormalizedStep] = Field(
        description="List of normalized steps, each with a single behavior. Steps are in chronological order."
    )


class NormalizationVerification(BaseModel):
    """Response from normalization verification stage."""
    adheres_to_standards: bool = Field(
        description="Whether the normalization adheres to all standards: single behavior per step, mathematical content preserved, chronological order maintained"
    )
    reasoning: str = Field(description="Explanation of the verification assessment")


class JudgeResponse(BaseModel):
    """Structured output from a judge model."""
    reasoning: str
    correct: bool


class ProofError(BaseModel):
    """Represents an error found in a proof."""
    type: str = Field(description="Type of error. Must be one of: 'Proof by Example', 'Proposal Without Verification', 'Inventing Wrong Facts', 'Begging the Question (Circular Reasoning)', 'Solution by Trial-and-Error', 'Calculation Mistakes', 'Missing Closing Token / Hallucinated Content', 'Contradiction with Reference Solution'")
    description: str = Field(description="Detailed explanation of the error and why it affects the proof")
    location: str = Field(
        description=(
            "Where the error occurs in the generator's *final answer* (not CoT). "
            "Prefer a short identifying 5-gram snippet copied verbatim from the final answer, "
            "optionally with a brief locator (e.g., 'Near: <5-gram>')."
        )
    )


class ProofCheckerResponse(BaseModel):
    """Structured output from proof checker judge with error listing."""
    reasoning: str = Field(description="Comprehensive explanation of the proof evaluation")
    correct: bool = Field(description="True if the solution is correct, complete, and rigorous; False otherwise")
    errors: List[ProofError] = Field(
        default_factory=list,
        description="List of all errors, gaps, or fallacies found in the proof. Empty list if correct=True."
    )


class MarkingSchemeMilestone(BaseModel):
    """A single milestone in the marking scheme."""
    points: str = Field(description="Point value as integer or range (e.g., '1', '1-2', '2')")
    description: str = Field(description="The logical truth required (e.g., 'Proving f(x) is injective')")


class DesignedMarkingScheme(BaseModel):
    """The marking scheme designed by the grader."""
    summary: str = Field(description="Brief description of the logic path required")
    milestones: List[MarkingSchemeMilestone] = Field(description="List of milestones with point values")


class FeedbackError(BaseModel):
    """An error found in the student solution."""
    location: str = Field(description="Quote from text or location description")
    issue: str = Field(description="Specific logic error")
    severity: str = Field(description="Severity level: High, Medium, or Low")


class GraderFeedback(BaseModel):
    """Feedback on the student solution."""
    achieved_milestones: List[str] = Field(
        default_factory=list,
        description="List of milestones from the scheme that the student hit"
    )
    missed_milestones: List[str] = Field(
        default_factory=list,
        description="List of milestones missed"
    )
    errors: List[FeedbackError] = Field(
        default_factory=list,
        description="List of errors found in the solution"
    )


class GradeResult(BaseModel):
    """The grading result."""
    score: int = Field(description="Score from 0 to 7", ge=0, le=7)
    classification: str = Field(description="Classification of the solution (e.g., 'Substantial Progress')")


class GraderResponse(BaseModel):
    """Structured output from grader judge with rubric generation and scoring."""
    reasoning: str = Field(
        description="Internal monologue/scratchpad showing rubric design and analysis process"
    )
    designed_marking_scheme: DesignedMarkingScheme = Field(
        description="The 7-point marking scheme derived from the official solution"
    )
    overall_assessment: GradeResult = Field(description="The grade assigned to the student solution")
    feedback: GraderFeedback = Field(description="Detailed feedback on achieved/missed milestones and errors")


class CritiqueResponse(BaseModel):
    """Structured output from the critique stage."""
    explanation: str
    correctness: bool
    first_mistake_step: Optional[int] = None
    progress: bool
    hint: Optional[str] = None
    errors: List[ProofError] = Field(
        default_factory=list,
        description="List of all proof errors found in the solution. Empty list if correctness=True."
    )


class IssueDetectionResponse(BaseModel):
    """Response from issue detection stage - analyzes final answer only."""
    correctness: bool = Field(description="Whether the solution is correct")
    issue_type: Literal[
        "method_divergence", 
        "method_error", 
        "both", 
        "correct"
    ] = Field(description="Type of issue identified in the final answer")
    progress: bool = Field(description="Whether generator made progress from previous attempt")
    errors: List[ProofError] = Field(
        default_factory=list,
        description="List of errors found in the final answer (with locations in the answer)"
    )
    discrepancies: List[str] = Field(
        default_factory=list,
        description="List of discrepancies between generator solution and reference solution (structural/logical differences)"
    )
    contradiction_errors: List[ProofError] = Field(
        default_factory=list,
        description="List of errors detected by comparing contradictory claims between generator and reference solutions"
    )
    dead_end_rationale_needed: bool = Field(
        default=False,
        description="Whether dead-end rationale should be generated"
    )
    already_correct: bool = Field(
        default=False,
        description="True if first attempt is correct with reference approach"
    )
    explanation: str = Field(description="Brief reasoning for the classification")


class VerifierResponse(BaseModel):
    """Response from Verifier stage - checks generator's final answer for correctness."""

    correctness: bool = Field(description="Whether the generator's final answer is correct")
    errors: List[ProofError] = Field(
        default_factory=list,
        description="List of errors found in the generator's final answer (empty if correctness=True)",
    )
    contradiction_errors: List[ProofError] = Field(
        default_factory=list,
        description=(
            "List of contradiction-type errors found when comparing the generator's final answer "
            "against the reference solution (empty if none)."
        ),
    )
    explanation: str = Field(
        description="Concise explanation of why the final answer is correct/incorrect and what failed."
    )


class RationaleCoverage(BaseModel):
    """Defines the CoT range to summarize for dead-end rationale."""
    start_step: int = Field(description="Starting step index (inclusive)")
    end_step: int = Field(description="Ending step index (inclusive)")


class DiscrepancyLocation(BaseModel):
    """Maps a discrepancy to a CoT step."""
    discrepancy: str = Field(description="Description of the discrepancy")
    step: int = Field(description="CoT step number where the discrepancy originates")


class ContradictionErrorLocation(BaseModel):
    """Maps a contradiction error to a CoT step."""
    error: ProofError = Field(description="The contradiction error")
    step: int = Field(description="CoT step number where the contradiction error occurs")


class InjectionPointResponse(BaseModel):
    """Response from injection point detection stage - traces errors to CoT steps."""
    divergence_start_step: Optional[int] = Field(
        default=None,
        description="Earliest CoT step where generator starts heading towards divergent path"
    )
    commitment_step: Optional[int] = Field(
        default=None,
        description="CoT step where generator fully commits to divergent approach"
    )
    mistake_step: Optional[int] = Field(
        default=None,
        description="CoT step where error claim/calculation is made"
    )
    injection_step: int = Field(
        description="Final CoT step index where hint should be injected"
    )
    rationale_coverage: Optional[RationaleCoverage] = Field(
        default=None,
        description="CoT range to summarize for dead-end rationale (if needed)"
    )
    discrepancy_locations: List[DiscrepancyLocation] = Field(
        default_factory=list,
        description="List of discrepancy locations mapping discrepancies to CoT steps"
    )
    contradiction_error_locations: List[ContradictionErrorLocation] = Field(
        default_factory=list,
        description="List of contradiction error locations mapping contradiction errors to CoT steps"
    )
    
    @model_validator(mode='after')
    def validate_injection_step(self):
        """Ensure injection_step is always provided."""
        if self.injection_step is None or self.injection_step < 1:
            raise ValueError("injection_step must be a positive integer")
        return self


# Keep old model for backward compatibility during transition
class IssueClassificationResponse(BaseModel):
    """Legacy response model - kept for backward compatibility."""
    correctness: bool = Field(description="Whether the solution is correct")
    issue_type: Literal[
        "method_divergence", 
        "method_error", 
        "both", 
        "correct"
    ] = Field(description="Type of issue identified")
    divergence_start_step: Optional[int] = Field(
        default=None,
        description="Earliest step where generator starts heading towards divergent path (for method_divergence or both)"
    )
    commitment_step: Optional[int] = Field(
        default=None,
        description="Step where generator fully commits to divergent approach (for dead-end rationale generation)"
    )
    mistake_step: Optional[int] = Field(
        default=None,
        description="Step with calculation/logic error (for method_error)"
    )
    progress: bool = Field(description="Whether generator made progress from previous attempt")
    errors: List[ProofError] = Field(
        default_factory=list,
        description="List of committed proof errors (not exploration)"
    )
    dead_end_rationale_needed: bool = Field(
        default=False,
        description="Whether dead-end rationale should be generated"
    )
    already_correct: bool = Field(
        default=False,
        description="True if first attempt is correct with reference approach"
    )
    explanation: str = Field(description="Brief reasoning for the classification")
    
    @model_validator(mode='after')
    def validate_inject_step_available(self):
        """
        Validate that required step fields are present based on issue_type.
        This ensures we can always determine an injection point for incorrect solutions.
        """
        if not self.correctness:
            if self.issue_type == "method_divergence":
                if not (self.divergence_start_step or self.commitment_step):
                    raise ValueError(
                        f"method_divergence requires divergence_start_step or commitment_step, "
                        f"but both are None. This prevents determining an injection point."
                    )
            elif self.issue_type == "method_error":
                if not self.mistake_step:
                    raise ValueError(
                        f"method_error requires mistake_step, but it is None. "
                        f"This prevents determining an injection point."
                    )
            elif self.issue_type == "both":
                if not (self.divergence_start_step or self.commitment_step or self.mistake_step):
                    raise ValueError(
                        f"both requires at least one of divergence_start_step, commitment_step, or mistake_step, "
                        f"but all are None. This prevents determining an injection point."
                    )
        return self


class CoTStepClassification(BaseModel):
    """Represents a single step classification in CoT monitoring."""
    step_number: Union[int, str] = Field(description="Step number (1-based index or string identifier)")
    snippet: str = Field(description="First few words of the step for identification")
    label: str = Field(description="Behavioral category label (one of 10 categories)")
    reasoning: str = Field(description="Explanation for why this step was classified with this label")


class BehavioralPatternStep(BaseModel):
    """Represents a single step in a behavioral pattern chain."""
    pattern: str = Field(description="Behavioral pattern name (one of 10 labels)")
    high_level_content: str = Field(description="High-level description of what this step should accomplish")
    reasoning: str = Field(description="Why this pattern is used at this point in the chain")


class ChainPlan(BaseModel):
    """Plan for a behavioral pattern chain to guide hint generation."""
    chain: List[str] = Field(description="Ordered list of behavioral pattern names")
    steps: List[BehavioralPatternStep] = Field(description="Detailed steps with high-level content")
    overall_reasoning: str = Field(description="Overall reasoning for this chain selection")
    chain_length: int = Field(description="Length of chain (for validation)")


class ProgressPointer(BaseModel):
    """
    Pointer into the solution_analysis structure.

    main_step is 1-based. substep uses a decimal convention (e.g., 2.2 for substep 2.2).
    Use (0, 0) to indicate no established progress.
    """

    main_step: int = Field(ge=0, description="1-based main step index (0 if none)")
    substep: float = Field(ge=0, description="Substep index like 2.2 (0 if none)")


class StepStatusItem(BaseModel):
    """Represents the status of a single substep."""
    step_id: str = Field(description="Substep identifier (e.g., '1.1', '2.3')")
    status: Literal["correct", "incorrect", "missing"] = Field(description="Verification status of this step")


class ProgressAssessment(BaseModel):
    """
    Objective assessment of generator progress.

    - progress_pointer / progress_percentage are based on *final answer correctness* with a prefix rule
      over the ordered substeps in solution_analysis.
    - last_useful_step_index is based on the labeled CoT and indicates where to cut before injecting hints.
    """

    progress_pointer: ProgressPointer = Field(
        description=(
            "The last substep that is established correctly in the generator's final answer, "
            "using prefix-correctness over the ordered substeps in solution_analysis."
        )
    )
    remaining_work_summary: str = Field(
        description="Brief summary of what remains to be done (high-level, not detailed steps)"
    )
    last_useful_step_index: int = Field(
        ge=1,
        description=(
            "1-based index into the labeled CoT steps (Step 1, Step 2, ...) indicating the last "
            "useful step to keep before injecting a hint. Exclude pure rehashing at the end."
        ),
    )
    last_useful_step_reasoning: str = Field(
        description="Explanation for why this index was chosen, verifying it keeps all correct steps (including after errors)."
    )
    step_status: List[StepStatusItem] = Field(
        default_factory=list,
        description=(
            "List of statuses for all substeps. Used to track verified vs. structure coverage."
        ),
    )


class TranslatorResponse(BaseModel):
    """Response from Translator stage - stylistic rewrite only (no new mathematical content)."""

    translated_hint: str = Field(description="Translated hint as a single string (may contain \\n\\n).")
    translated_hint_steps: List[str] = Field(
        default_factory=list,
        description="Translated hint split into step-paragraph strings (each item is one step).",
    )
    behavioral_chain: List[str] = Field(
        default_factory=list,
        description="Chosen behavioral pattern chain (list of pattern labels).",
    )
    chain_reason: str = Field(description="Reason for choosing the behavioral chain.")


class SolutionDetection(BaseModel):
    """
    Detection result for whether a solution exists after Chain of Thought reasoning.
    
    - has_solution: Whether a solution section exists after the CoT
    - solution_start_paragraph_index: 1-based index into paragraphs indicating where solution starts
    - reasoning: Explanation for the detection decision
    """
    
    has_solution: bool = Field(
        description="True if a solution/final answer section exists after the Chain of Thought reasoning."
    )
    
    solution_start_paragraph_index: int = Field(
        ge=1,
        description=(
            "1-based index into the paragraphs (Paragraph 1, Paragraph 2, ...) indicating where "
            "the solution section begins. Only valid if has_solution=True. "
            "This is the first paragraph that contains solution content (formatted sections, final answers, etc.)."
        ),
    )
    
    reasoning: str = Field(
        description=(
            "Explanation for the detection decision. If has_solution=True, explain why this paragraph "
            "index marks the start of the solution. If has_solution=False, explain why no solution was detected."
        ),
    )


class CoTMonitoringResponse(BaseModel):
    """Structured output from CoT monitoring stage."""
    steps: List[CoTStepClassification] = Field(
        default_factory=list,
        description="List of step classifications in order"
    )


Payload = TypeVar("Payload")


class WorkItemMetadata(BaseModel):
    """Metadata associated with a unit of work."""

    document_id: Any


class WorkItem(BaseModel, Generic[Payload]):
    """A generic container representing a unit of work for a workflow."""

    payload: Payload
    metadata: WorkItemMetadata


class FieldBinding(BaseModel):
    from_: str = Field(alias="from")


class OutputBinding(BaseModel):
    to: str


class WorkflowBindings(BaseModel):
    inputs: Dict[str, FieldBinding]
    outputs: Dict[str, OutputBinding]


class WorkflowSpec(BaseModel):
    class_: str = Field(alias="class")
    payload_model: str


class QueryTemplate(BaseModel):
    collection: str
    template: Dict[str, object]


class WorkflowBindingConfig(BaseModel):
    workflow: WorkflowSpec
    query: QueryTemplate
    bindings: WorkflowBindings



