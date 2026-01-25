Is the following text claiming that the solution is complete?
==========================================================

{solution_text}

==========================================================

Response in exactly "yes" or "no". No other words.

### JSON Schema for Output ###

You MUST provide your response in a JSON object that strictly adheres to the following schema. Do NOT wrap the JSON in markdown backticks.

```json
{
    "title": "VerificationOutput",
    "description": "The structured output for a verification stage.",
    "type": "object",
    "properties": {
        "verdict": {
            "title": "Verdict",
            "description": "The verdict, 'yes' or 'no'.",
            "type": "string"
        }
    },
    "required": [
        "verdict"
    ]
}
```
