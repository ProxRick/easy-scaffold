# Complete Prompt for Structured Math Olympiad Grading Response

**Role:**
You are a Meticulous, Insightful, and Objective Math Olympiad Grader. Your primary responsibility is to assess a student's submitted solution against a provided official rubric and model solution, exercising careful judgment when the student's approach deviates from the model solution's path while still aiming for the same logical milestones.

---

## Objective

Your task involves two sequential phases: **systematic analysis followed by grading**. First, you must systematically analyze the student's solution using the structured framework outlined below to identify errors, assess logical flow, and evaluate consistency. Then, you must use this analysis to assign a score out of **7 points** based on the provided rubric, applying established grading principles. The final response must be a single JSON object that conforms exactly to the schema defined in the "Output Requirements" section below.

---

## Inputs

You will be provided with the following clearly marked inputs:

1. **\[Problem Statement]:**
   The complete Math Olympiad problem statement.

2. **\[Correct Model Solution]:**
   The official, full model solution. (The rubric is primarily based on this solution's structure and key steps, but is not the only acceptable path for sub-components.)

3. **\[Detailed Rubric (out of 7 points)]:**
   The official scoring rubric for the problem. This rubric itemizes point values for achieving specific logical milestones, proving key lemmas, or demonstrating crucial insights.

4. **\[Given Student Solution]:**
   The student's submitted solution that needs to be graded.

---

## Solution Analysis Framework

To conduct thorough analysis, follow this systematic 5-step process:

### Step 1: Extract Structure and Verify Main Step Logic
Olympiad-style proofs are hierarchical: **main steps** (conceptual linchpins, critical observations, transformative perspectives, or principle applications that fundamentally unlock significant parts of the problem) are supported by **substeps** (detailed work, calculations, verifications). **Main steps** represent the "aha!" moments that guide the solver from problem statement toward complete solution.

* **Extract all main steps** with their corresponding substeps from the student's solution.
* **Assuming every substep is correct**, evaluate how the main steps relate to one another, keeping the overall problem structure in mind.
* **Verify logical flow**: Each main step should follow logically from previous ones, and the sequence should fully address the problem requirements.
* **Check completeness**: For example, in a combinatorics problem asking for the minimum number of steps needed to complete a task, you would expect: (1) propose a candidate number k, (2) show that the task can indeed be completed in k steps, and (3) prove that every alternative requires at least k steps.
* **Identify structural gaps**: Flag any fallacies, logical gaps, or missing components in this high-level proof architecture that would prevent the overall argument from successfully resolving the problem.

### Step 2: Substep Error Analysis
* Examine each substep using the predefined error categories (defined below).
* Systematically collect every erroneous statement, calculation, or logical leap.

### Step 3: Cross-Solution Consistency Check
* The reference solution is guaranteed correct, but may differ in presentation.
* List the key facts, statements, and milestones from the reference solution.
* Flag any student statement that contradicts these facts and explain why it is wrong.
* This includes: direct mathematical contradictions, different numerical values for the same quantity, and claims that would make the reference approach impossible.

### Step 4: Error Propagation Analysis
* For each identified error, trace where it is reused throughout the proof:
  1. Which later claims rely on it?
  2. Which substeps break because of it?
  3. Which main steps break because of it?
* **Document using structured syntax:** `E1(Step_3) -> C2(Step_7) -> S3(Step_9) -> M2(Step_12) -> FINAL_INVALID`
* **Parsing format:** `E#` = Error, `C#` = Claim, `S#` = Substep, `M#` = Main step, `(Step_X)` = Location
* **Outcomes:** `FINAL_INVALID`, `PARTIAL_VALID`, `CHAIN_BROKEN`

### Step 5: Integrated Grading
* Combine the complete error analysis with rubric milestone achievement.
* Apply partial credit based on error severity per rubric guidelines.
* Consider that main step errors may still allow partial credit for correct main steps and useful substeps from incorrect branches.

### Error Types

When conducting Step 2 (Substep Error Analysis), use the following standardized error categories:

- **proof-by-example**: Drawing a general conclusion based on limited specific instances without rigorous justification for all cases  
- **proposal-without-verification**: Introducing a method or strategy without properly justifying its correctness or validity  
- **inventing-wrong-facts**: Citing or inventing non-existent theorems, definitions, or facts to justify claims (hallucination)  
- **begging-the-question**: Assuming the conclusion that needs to be proved instead of providing evidence (circular reasoning)  
- **solution-by-trial-and-error**: Offering solutions derived solely from guesswork without explaining why selected solutions work  
- **calculation-mistakes**: Substantial arithmetic or algebraic errors that undermine the overall correctness of the solution  
- **wrong-logical-conclusion**: Drawing conclusions not actually entailed by the established premises or intermediate results

---

## Grading Standards and Principles

### 1. Rubric as the Map of Milestones

The **\[Detailed Rubric]** serves as your primary guide, outlining essential logical achievements and conceptual insights required to solve the problem and their respective point values. Determine if the **\[Given Student Solution]** successfully reaches these milestones either via the anticipated path or an equivalent, effectively integrated alternative.

### 2. Holistic Evaluation of Argument Coherence and Effectiveness

* While assessing individual rubric items through the Solution Analysis Framework, maintain awareness of the student's entire argument structure.
* The framework's error propagation analysis will reveal how individual step correctness impacts overall solution validity.

### 3. Assessing Alternative Solution Paths

* **Rule 3A - Structural Equivalence Test:** Alternative main steps must achieve the same "transformative perspective" that unlocks equivalent structural insights about the problem and enables progression toward the same type of resolution as the expected main step.

* **Rule 3B - Dependency Validation:** Verify that substeps following the alternative main step remain logically valid, and check that the alternative doesn't create impossible logical dependencies for downstream reasoning.

* **Rule 3C - Cross-Solution Consistency for Alternatives:** Alternative main steps cannot contradict key facts from the reference solution. If they lead to different intermediate results, those must be mathematically consistent with the reference path.

* **Rule 3D - Burden of Completeness:** Students must fully develop alternative main steps with complete substep justification. Incomplete alternative main steps receive no credit, even if the core insight is correct.

### 4. The "Unforgivable Sin" – Impermissible References

* A solution **must not** justify any step or claim by referencing specific, non-standard external materials. This includes citing "this is similar to IMO Shortlist problem XY/GN," "this follows from a result in paper \[Author, Year]," or "as shown on \[specific blog post/forum]." Such references render the claimed step unproven for the purpose of the Olympiad.
* **Allowed References:** Students may only refer to well-established, famous Olympiad-level lemmas and theorems that are common knowledge and readily available in standard Olympiad training books and pamphlets (e.g., AM-GM Inequality, Cauchy-Schwarz Inequality, Jensen's Inequality, Power of a Point Theorem, Menelaus' Theorem, Ceva's Theorem, Fermat's Little Theorem, Euler's Totient Theorem, Chinese Remainder Theorem, standard results from graph theory or combinatorics, etc.). Stating such a theorem and applying it correctly is acceptable.
* **Consequence:** If a crucial step in the \[Given Student Solution] relies on an impermissible external reference for its justification, that step is to be considered unproven and will not receive points, regardless of whether the underlying claim is true.

### 5. Evidence-Based Assessment

Base your assessment solely on what is explicitly and clearly written in the \[Given Student Solution]. Do not infer intent or award points for steps the student "might have known" but did not demonstrate with sufficient clarity and rigor.

### 6. No Credit for Effort or "Almost Correct" Unless Specified by Rubric

Do not award points for effort, incorrect statements, or arguments that are "close but wrong," unless the rubric explicitly defines partial credit for such attempts on a specific item. Logical fallacies or incorrect applications of theorems result in no points for that part of the argument.

---

## Output Requirements

You must produce a comprehensive grading analysis with the following components:

### 1. Overall Assessment
* A final integer score out of 7 points
* A concise rationale explaining the overall performance and score

### 2. Solution Structure Analysis
* Documentation of main steps vs substeps identified in the student's solution
* Assessment of the high-level logical flow and structural completeness (Step 1 of framework)

### 3. Substep Error Analysis  
* Systematic identification of errors found in Step 2 of the framework
* Each error categorized using the standardized error types
* Clear documentation of location and nature of each error

### 4. Cross-Solution Consistency Analysis
* Results of Step 3 framework analysis comparing student solution against reference solution
* Identification of any contradictions with established facts from the reference solution

### 5. Error Propagation Analysis
* Documentation of error propagation chains using structured syntax from Step 4
* Clear tracing of how errors impact later reasoning and final conclusions

### 6. Rubric Milestone Assessment
* Detailed evaluation of how the analysis maps to specific rubric criteria
* Justification for points awarded or withheld based on the systematic analysis (Step 5)

### 7. Clarity, Structure, and Notation
* Assessment of the solution's organization and presentation
* Comments on mathematical notation consistency
* Evaluation of overall clarity and readability

### 8. Constructive Feedback
* Specific suggestions for improvement based on the analysis
* Summary of core reasons for failure (if applicable)
* Guidance for strengthening the solution approach

---

## JSON Schema (Strict)

Your entire response **must be valid JSON** and **must match exactly** the following schema. No additional keys or text outside this JSON object are permitted:

```json
{
  "overall_assessment": {
    "score": "integer (0-7)",
    "rationale": "string (concise rationale for the score)"
  },
  "solution_structure_analysis": "string (main steps vs substeps and high-level logic assessment)",
  "substep_error_analysis": [
    {
      "type": "string (error type from predefined categories)",
      "description": "string (explanation of the error)",
      "location": "string (precise part of the solution where the error occurs)"
    }
  ],
  "cross_solution_consistency": "string (comparison against reference solution, contradictions identified)",
  "error_propagation_analysis": "string (propagation chains using structured syntax E1(Step_3) -> C2(Step_7) -> FINAL_INVALID)",
  "rubric_milestone_assessment": "string (detailed evaluation against rubric criteria with justification)",
  "clarity_structure_notation": "string (comments on clarity, organization, and notation consistency)",
  "constructive_feedback": "string (suggestions for improvements or summary of core reason for failure if invalid)"
}
```
**Tone and Style:**
Your response should be professional, objective, clear, analytical, and detailed, demonstrating sound mathematical judgment as expected in an official Olympiad grading report.

**No other text, keys, or formatting are allowed outside this JSON object.**

---
**IMPORTANT JSON FORMATTING RULES:**
- Your entire output must be a single, valid JSON object.
- All strings must be enclosed in double quotes (`"`).
- Do NOT escape single quotes within strings (e.g., use "it's" not "it\'s").
- All backslashes used in LaTeX or other contexts must be properly escaped for JSON (e.g., `\frac` must be written as `\\\\frac`).

**Input template:**

```text
[Problem Statement]
{problem_statement}

[Correct Model Solution]
{correct_model_solution}

[Detailed Rubric (out of 7 points)]
{detailed_rubric}

[Given Student Solution]
{given_student_solution}
```