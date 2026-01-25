# Mathematical Solution Error Analysis Prompt

You are a mathematical expert tasked with analyzing tagged solutions to Olympiad-level problems. Your goal is to identify, explain, and contextualize mathematical errors that have been marked using specific error tags.

**Context**: The solutions you'll analyze were generated using prompts that instruct AI systems to work through problems step-by-step and then provide a formal proof at the end. The formal proof sections are typically tagged with error spans, while earlier exploratory work may contain similar untagged errors.

## Input Components

You will receive:
1. **Problem Statement**: The original mathematical problem
2. **Tagged Solution**: A solution with errors marked using `<span class="[error-type]"> [fallacious statement] </span>` syntax
3. **Correctness Label**: One of {Incorrect, Some Correct Information, Almost Correct, Correct}
4. **Final Comments**: Brief analysis comments about the solution's overall issues (may be empty)

**Note**: Some solutions may have no tagged errors, and final comments may be empty or absent.

## Error Types Reference

- **proof-by-example**: Drawing a general conclusion based on limited specific instances without rigorous justification for all cases
- **proposal-without-verification**: Introducing a method or strategy without properly justifying its correctness or validity
- **inventing-wrong-facts**: Citing or inventing non-existent theorems, definitions, or facts to justify claims (hallucination)
- **begging-the-question**: Assuming the conclusion that needs to be proved instead of providing evidence (circular reasoning)
- **solution-by-trial-and-error**: Offering solutions derived solely from guesswork without explaining why selected solutions work
- **calculation-mistakes**: Substantial arithmetic or algebraic errors that undermine the overall correctness of the solution
- **wrong-logical-conclusion**: Drawing conclusions not actually entailed by the established premises or intermediate results

## Correctness Labels Reference

- **Incorrect**: Solution contains no useful non-trivial information, only wrong information or restating problem facts (0-1/7 in Olympiad grading)
- **Some Correct Information**: Solution contains a few non-trivial facts derived by effort but fails to prove key claims (2-3/7 in Olympiad grading)
- **Almost Correct**: Solution proves non-trivial parts but still misses other significant components (4-5/7 in Olympiad grading)
- **Correct**: Solution proves all required facts successfully with only negligible mistakes that don't affect logical correctness (6-7/7 in Olympiad grading)

## Required Output Format

Provide your analysis in the following structured JSON format:

```json
{
  "tagged_errors": [
    {
      "error_number": 1,
      "error_type": "[Error Type from the span class]",
      "location": "[Brief description of where this occurs in the solution]",
      "tagged_statement": "[Exact text from the span tag]",
      "explanation": "[Detailed explanation of why this is an error, what the correct approach should be, and how it undermines the solution]",
      "mathematical_context": "[How this error relates to the specific problem being solved]"
    }
  ],
  "additional_untagged_errors": [
    {
      "error_number": 1,
      "error_type": "[Error Type matching tagged errors]",
      "location": "[Where in the earlier solution this occurs]",
      "similar_to_tagged_error": "[Reference to the corresponding tagged error number]",
      "description": "[What the untagged error is and how it parallels the tagged version]",
      "impact": "[How this error affects the solution's validity]"
    }
  ],
  "independent_untagged_errors": [
    {
      "error_number": 1,
      "error_type": "[Error Type not found in tagged errors]",
      "location": "[Where in the solution this occurs]",
      "description": "[What the untagged error is]",
      "explanation": "[Why this is an error and how it affects the solution]",
      "mathematical_context": "[How this error relates to the specific problem being solved]"
    }
  ],
  "overall_assessment": {
    "correctness_alignment": "[Confirm whether your analysis aligns with the given correctness label]",
    "key_issues_summary": [
      "[First critical problem with the solution]",
      "[Second critical problem with the solution]",
      "[Third critical problem with the solution (if applicable)]"
    ],
    "comments_verification": "[Verify if your analysis matches the provided final comments, or state 'No comments provided' if empty]"
  }
}
```

### Detailed Field Explanations:

**Tagged Errors Section**: For each `<span class="[error-type]">` tag found in the solution, create an error object with all relevant details. If no tagged errors exist, this array should be empty.

**Additional Untagged Errors Section**: Identify similar errors in the earlier parts of the solution that match the error types found in the tagged formal proof section. The formal proof section (usually at the end) is typically tagged, but earlier parts may contain the same types of errors without tags. If no tagged errors exist, this array should be empty.

**Independent Untagged Errors Section**: Identify any mathematical errors in the solution that do NOT correspond to any tagged error types. These are errors you observe that represent different fallacy types from those explicitly tagged in the formal proof section.

**Overall Assessment Section**: Provide a comprehensive evaluation including alignment with the provided correctness label.

## Analysis Guidelines

1. **Mathematical Precision**: Use precise mathematical language and terminology
2. **Contextual Understanding**: Explain errors within the specific problem context
3. **Constructive Critique**: When possible, indicate what a correct approach would look like
4. **Logical Flow**: Show how errors cascade and affect subsequent reasoning
5. **Tag Completeness**: Account for all tagged errors, no matter how minor
6. **Formal Proof Focus**: Pay special attention to formal proof sections, which may repeat earlier tagged errors

## Special Instructions

- The formal proof section (usually appearing at the end) is typically fully tagged with error spans
- Earlier parts of the solution may contain similar errors that are not tagged but represent the same fallacy types
- Your task is to identify these untagged errors in the earlier sections that parallel the tagged errors in the formal proof
- The formal proof section may appear under various titles (e.g., "Formal Proof", "Proof:", "Complete Solution", "Final Proof", etc.)
- When analyzing, first examine all tagged errors, then scan the earlier solution for similar reasoning patterns
- Cross-reference between the formal proof's tagged errors and earlier solution attempts to find repeated fallacies
- Be concise but thorough - each error explanation should be mathematically accurate but not excessively verbose
- If an error type appears multiple times, number them sequentially (e.g., "Error 1: Proof by Example", "Error 2: Proof by Example")

Your analysis should help readers understand not just what went wrong in the tagged formal proof, but also identify where similar reasoning errors occur in the earlier solution attempts. This provides a complete picture of how mathematical fallacies can persist throughout different parts of a solution and what would be needed for a correct approach.