# Data and security boundaries

PolicyCompass is a local learning tool. It does not execute input code, load serialized Python objects, make network calls, or deploy policies. The CLI reads a bounded UTF-8 JSON file and prints JSON. Shell redirection and callers control where output is saved.

Use synthetic or properly de-identified logs. Keep real logs in an ignored directory outside public repositories. Dataset IDs, segment names, model cells, and evaluation summaries can reveal business or user information. Hashes are identifiers, not anonymization or authentication. Review every staged file before publishing it.

Probability validity is not provenance. The tool cannot establish that logged propensities were the actual selection probabilities, that decisions were unconfounded, or that row partitions are independent. Do not use the output as sole evidence for consequential automated decisions.

Input limits reduce accidental resource use; this is not a hardened multi-tenant service or a defense against arbitrary hostile Python callers. Bootstrap work grows with eval rows times resamples. No credentials, cloud integrations, or provider adapters are included.

Report a reproducible issue through this repository's Issues tab using synthetic data. Do not attach credentials or private logs.
