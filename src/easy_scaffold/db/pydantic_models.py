# src/easy_scaffold/db/pydantic_models.py
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class Problem(BaseModel):
    """Represents a problem document from the 'problems' collection."""
    post_url: str = Field(..., description="The URL of the problem post.")
    problem: str = Field(..., description="The text of the problem.")
    is_duplicate: bool = Field(..., description="Flag to indicate if the problem is a duplicate.")


class StageLog(BaseModel):
    """Log for a single stage within a workflow run."""
    stage_name: str = Field(..., description="The name of the stage.")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    inputs: Dict[str, Any] = Field(..., description="The inputs to the stage.")
    outputs: Dict[str, Any] = Field(..., description="The outputs from the stage.")
    status: str = Field(..., description="Status of the stage execution (e.g., 'completed', 'failed').")
    error_message: Optional[str] = Field(None, description="Error message if the stage failed.")
    token_stats: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Raw token usage details returned by the model call.",
    )


class RunLog(BaseModel):
    """Log for a single execution run of a workflow."""
    run_id: int = Field(..., description="The sequential ID of the run for a given problem.")
    workflow_name: str = Field(..., description="The name of the workflow being run.")
    
    # Model parameters for the run
    model_name: Optional[str] = Field(None, description="The name of the model used.")
    temperature: Optional[float] = Field(None, description="The temperature setting used.")
    max_tokens: Optional[int] = Field(None, description="The max_tokens setting used.")

    start_time: datetime = Field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None
    status: str = Field(..., description="Overall status of the run (e.g., 'running', 'completed', 'failed').")
    stages: List[StageLog] = Field(default_factory=list)


class SolverLog(BaseModel):
    """The top-level document in the logs collection for a given document."""
    document_id: str = Field(..., description="Identifier linking back to the source document.")
    runs: Dict[str, List[RunLog]] = Field(default_factory=dict, description="Workflow-specific run logs, keyed by workflow name.")


