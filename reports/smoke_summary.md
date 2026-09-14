# BoolQ smoke summary — adaptation

This 10-item smoke run validates experimental integrity; it is not evidence for or against intrinsic self-correction.

## Results

- Completed paired items: 10
- Initial accuracy (all items): 50.0%
- Revised accuracy (all items): 40.0%
- Delta accuracy (`A1 - A0`): -10.0%
- Transitions: CC=4, CW=1, WC=0, WW=5
- Exact two-sided McNemar p-value: 1
- Paired bootstrap 95% CI for delta: [-30.0%, 0.0%]
- Valid-only initial accuracy: 50.0% (n=10)

## Limits

Adaptation reasons: different model family, GGUF Q8 quantization, Ollama runtime/chat template, and 10-item subset. Invalid and error outputs remain in all-item denominators. The raw journal stores exact messages, output, request/response metadata, and retries.
