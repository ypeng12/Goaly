# Emotion feature ablation

Paired training seeds; same profiles, rewards, masks, architecture and step budget. Only emotion input dimensions 20:25 are zeroed in the ablation.

Each run requests 25,000 steps. Source/data SHA256: `09c6d4738f568b47f2cd68809cdd94e912212da0849db2bd8b140a1213488f25`.

| Seed | Condition | Split | Case success | Appropriate handoff | Truncated | Turns | Return |
|---:|---|---|---:|---:|---:|---:|---:|
| 42 | full | train | 85.71% | 0.0% | 14.29% | 7.286 | 12.986 |
| 42 | full | val | 100.0% | 0.0% | 0.0% | 4.0 | 16.6 |
| 42 | full | test | 50.0% | 50.0% | 0.0% | 4.25 | 11.325 |
| 42 | no_emotion | train | 71.43% | 0.0% | 28.57% | 9.429 | 9.271 |
| 42 | no_emotion | val | 100.0% | 0.0% | 0.0% | 6.5 | 16.35 |
| 42 | no_emotion | test | 50.0% | 50.0% | 0.0% | 4.25 | 10.325 |
| 7 | full | train | 85.71% | 0.0% | 14.29% | 5.857 | 13.843 |
| 7 | full | val | 100.0% | 0.0% | 0.0% | 4.0 | 16.6 |
| 7 | full | test | 50.0% | 50.0% | 0.0% | 4.25 | 11.325 |
| 7 | no_emotion | train | 57.14% | 0.0% | 42.86% | 8.714 | 5.129 |
| 7 | no_emotion | val | 100.0% | 0.0% | 0.0% | 4.0 | 16.6 |
| 7 | no_emotion | test | 50.0% | 50.0% | 0.0% | 4.25 | 11.325 |

Small finite simulator and two seeds; no significance or deployment claim. These fresh runs are separate from the 75k delivery checkpoints.

Unsuccessful runs remain in this report; acceptance failures are outcomes, not grounds to discard a condition.
