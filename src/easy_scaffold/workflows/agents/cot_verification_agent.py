# src/easy_scaffold/workflows/agents/cot_verification_agent.py
import logging
from typing import Any, Dict

from easy_scaffold.configs.pydantic_models import AttemptVerificationOutput, CoTSolutionFixerOutput
from easy_scaffold.db.pydantic_models import RunLog
from easy_scaffold.workflows.base import AbstractWorkflow
from easy_scaffold.workflows.workflow_models import CoTVerificationPayload, WorkItem

logger = logging.getLogger(__name__)


class CoTVerificationAgentWorkflow(AbstractWorkflow[CoTVerificationPayload]):
    async def _run(
        self,
        work_item: WorkItem[CoTVerificationPayload],
        run_log: RunLog,
        document_id: str,
        workflow_name: str,
    ) -> Dict[str, Any]:
        payload = work_item.payload
        context: Dict[str, Any] = payload.model_dump()
        context["run_log"] = run_log

        # Load cached CoT verification report if available
        cot_verification_report = None
        if payload.cached_cot_verification_report:
            cot_verification_report = payload.cached_cot_verification_report
            logger.info("Loaded cached CoT verification report")
            context["cot_verification_report"] = cot_verification_report
        else:
            # Stage 1: Generate CoT verification report
            logger.info("Generating CoT verification report...")
            result = await self._execute_stage(
                "CoTAnalyzer", context, run_log, document_id, workflow_name
            )
            
            # Extract verification report from context (set via output_key)
            cot_verification_report = context.get("cot_verification_report", "")
            if not cot_verification_report:
                # Fallback: try to get from raw_output
                raw_output = result.get("raw_output")
                if isinstance(raw_output, str):
                    cot_verification_report = raw_output
                else:
                    logger.warning("Could not extract CoT verification report from stage output")
                    cot_verification_report = ""
            
            logger.info(f"Generated CoT verification report (length: {len(cot_verification_report)} chars)")

        # Stage 2: Generate attempt verification
        attempt_verification = None
        
        if payload.verification_result:
            # Check if payload has valid structure
            if "is_valid" in payload.verification_result:
                logger.info("Loaded cached attempt verification result")
                attempt_verification = AttemptVerificationOutput(
                    is_valid=payload.verification_result["is_valid"],
                    verification_report=payload.verification_result.get("verification_report", "")
                )
                
        if not attempt_verification:
            logger.info("Generating attempt verification...")
            result = await self._execute_stage(
                "AttemptVerificationGenerator", context, run_log, document_id, workflow_name
            )
            
            # Extract structured output
            raw_output = result.get("raw_output")
            
            if isinstance(raw_output, AttemptVerificationOutput):
                attempt_verification = raw_output
            elif isinstance(raw_output, dict):
                attempt_verification = AttemptVerificationOutput(**raw_output)
            else:
                # Try to get from context fields
                is_valid = context.get("is_valid", False)
                verification_report = context.get("attempt_verification_report", "")
                attempt_verification = AttemptVerificationOutput(
                    is_valid=is_valid,
                    verification_report=verification_report
                )
        
        # Update context with verification result so subsequent stages have access to 'is_valid' etc.
        context["is_valid"] = attempt_verification.is_valid
        context["attempt_verification_report"] = attempt_verification.verification_report
        
        logger.info(f"Attempt verification result: is_valid={attempt_verification.is_valid}")

        # Build final outputs for verification part
        verified_cot = payload.cot_trajectory + "\n\n" + cot_verification_report
        
        if attempt_verification.is_valid:
            verified_attempt = payload.attempted_solution
        else:
            verified_attempt = payload.attempted_solution + "\n\n" + attempt_verification.verification_report

        verification_result = {
            "is_valid": attempt_verification.is_valid,
            "verification_report": attempt_verification.verification_report
        }
        
        # --- Fixing and Emulation Logic ---
        
        rigorous_fixed_cot = verified_cot
        rigorous_fixed_attempt = verified_attempt
        fixer_output_data = payload.rigorous_fixer_output
        emulator_output_text = payload.rigorous_emulator_output

        if attempt_verification.is_valid:
             # If valid, fixed version is same as verified version
             pass
        else:
            # If invalid, try to fix
            
            # 1. Run SolutionFixer if not cached
            if not fixer_output_data:
                logger.info("Solution invalid. Running SolutionFixer...")
                result = await self._execute_stage(
                    "SolutionFixer", context, run_log, document_id, workflow_name
                )
                raw_output = result.get("raw_output")
                if isinstance(raw_output, CoTSolutionFixerOutput):
                    fixer_output = raw_output
                elif isinstance(raw_output, dict):
                    fixer_output = CoTSolutionFixerOutput(**raw_output)
                else:
                    # Fallback extraction from context
                    fixer_output = CoTSolutionFixerOutput(
                        is_fixable=context.get("is_fixable", False),
                        corrected_solution=context.get("corrected_solution", ""),
                        reasoning=context.get("reasoning", "")
                    )
                fixer_output_data = fixer_output.model_dump()
            else:
                logger.info("Loaded cached SolutionFixer output")
                # Ensure context has necessary fields for next stage
                context.update(fixer_output_data)

            # 2. Branch based on fixability
            if fixer_output_data.get("is_fixable", False):
                # If fixable, run CoTEmulator if not cached
                if not emulator_output_text:
                    logger.info("Solution is fixable. Running CoTEmulator...")
                    result = await self._execute_stage(
                        "CoTEmulator", context, run_log, document_id, workflow_name
                    )
                    emulator_output_text = context.get("emulator_output", "")
                    if not emulator_output_text:
                         raw_output = result.get("raw_output")
                         if isinstance(raw_output, str):
                             emulator_output_text = raw_output
                else:
                    logger.info("Loaded cached CoTEmulator output")

                # Construct fixed outputs for fixable case
                rigorous_fixed_cot = (
                    payload.cot_trajectory + "\n\n"
                    + cot_verification_report + "\n\n"
                    + emulator_output_text
                )
                rigorous_fixed_attempt = fixer_output_data.get("corrected_solution", "")
            else:
                # If not fixable
                logger.info("Solution is not fixable.")
                rigorous_fixed_cot = (
                    payload.cot_trajectory + "\n\n"
                    + cot_verification_report + "\n\n"
                    + fixer_output_data.get("reasoning", "")
                )
                # Keep the verified attempt (which is the original attempt) as we couldn't fix it
                rigorous_fixed_attempt = verified_attempt

        # Build final context dict
        final_context: Dict[str, Any] = {
            "status": "completed",
            "cot_verification_report": cot_verification_report,
            "verified_cot": verified_cot,
            "verified_attempt": verified_attempt,
            "verification_result": verification_result,
            "rigorous_fixer_output": fixer_output_data,
            "rigorous_emulator_output": emulator_output_text,
            "rigorous_fixed_cot": rigorous_fixed_cot,
            "rigorous_fixed_attempt": rigorous_fixed_attempt,
        }

        return final_context


