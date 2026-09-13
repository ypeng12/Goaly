"""State featurizer converting AgentPolicyEnv observations into PyTorch tensors."""
from typing import Any, Dict, List, Union
import torch
import numpy as np

from ..harness.types import Phase, AGENT_ACTIONS
from ..harness.dialogue import EMOTIONS, dialogue_signals

PHASE_ORDER = [
    Phase.VERIFY_ID.value,
    Phase.RESOLVE_INTENT.value,
    Phase.PROCESS_CASE.value,
    Phase.POST_PROCESS.value,
    Phase.ESCALATED.value,
    Phase.CONCLUDED.value,
]

PII_FIELDS = ["name", "dob", "phone", "email", "id_last4"]
MEMORY_KEYS = ["case_type_hint", "status_hint", "date_hint", "topic_hint"]


class StateFeaturizer:
    """Extracts a fixed-size numerical feature vector from AgentPolicyEnv observations.
    
    Feature Layout (total 30 dimensions, schema version 2):
      [0..5]   (6) Phase one-hot
      [6..10]  (5) Verified PII fields multi-hot (name, dob, phone, email, id_last4)
      [11]     (1) Verified PII count / 5.0
      [12..15] (4) Cross-phase memory slots multi-hot (case_type, status, date, topic)
      [16]     (1) Memory slots count / 4.0
      [17]     (1) Data shield active (1.0 or 0.0)
      [18]     (1) Active case ID present (1.0 or 0.0)
      [19]     (1) Turn ratio (turn / max_turns)
      [20..24] (5) Observable emotion one-hot
      [25..27] (3) Privacy concern, refusal, human request
      [28..29] (2) Resolution attempts / 2, grounded answer delivered
    """

    FEATURE_DIM = 30
    VERSION = 2

    def __init__(self, max_turns: int = 15, use_emotion_features: bool = True):
        self.max_turns = max_turns
        self.use_emotion_features = use_emotion_features

    def featurize(self, obs: Dict[str, Any], turn: int = 0) -> np.ndarray:
        """Convert a single observation dict into a versioned float32 vector."""
        vec = np.zeros(self.FEATURE_DIM, dtype=np.float32)

        # 1. Phase one-hot (dim 6)
        phase_str = obs.get("phase", Phase.VERIFY_ID.value)
        if phase_str in PHASE_ORDER:
            vec[PHASE_ORDER.index(phase_str)] = 1.0

        # 2. Verified fields (dim 5 + 1)
        verified_fields = set(obs.get("verified_fields", []))
        for i, field in enumerate(PII_FIELDS):
            if field in verified_fields:
                vec[6 + i] = 1.0
        vec[11] = len(verified_fields & set(PII_FIELDS)) / 5.0

        # 3. Memory slots (dim 4 + 1)
        memory_slots = obs.get("memory_slots", {})
        for i, key in enumerate(MEMORY_KEYS):
            if memory_slots.get(key):
                vec[12 + i] = 1.0
        vec[16] = sum(1.0 for k in MEMORY_KEYS if memory_slots.get(k)) / 4.0

        # 4. Data shield active (dim 1)
        vec[17] = 1.0 if obs.get("data_shield_active", True) else 0.0

        # 5. Active case ID present (dim 1)
        vec[18] = float(bool(obs.get("active_case_id")))

        # 6. Turn progress (dim 1)
        vec[19] = min(1.0, float(turn) / max(1.0, float(self.max_turns)))
        signals = dialogue_signals(obs.get('caller_utterance', ''))
        emotion = obs.get('emotion', signals['emotion'])
        if self.use_emotion_features:
            vec[20 + EMOTIONS.index(emotion if emotion in EMOTIONS else 'neutral')] = 1.0
        vec[25] = float(obs.get('privacy_concern', signals['privacy_concern']))
        vec[26] = float(obs.get('refusal', signals['is_refusal']))
        vec[27] = float(obs.get('demands_human', signals['demands_human']))
        vec[28] = min(1.0, obs.get('resolution_attempts', 0) / 2.0)
        vec[29] = float(obs.get('grounded_answered', False))

        return vec

    def featurize_tensor(
        self,
        obs: Dict[str, Any],
        turn: int = 0,
        device: Union[str, torch.device] = "cpu",
    ) -> torch.Tensor:
        """Return 1D tensor [FEATURE_DIM] on specified device."""
        arr = self.featurize(obs, turn=turn)
        return torch.tensor(arr, dtype=torch.float32, device=device)

    def extract_mask_tensor(
        self,
        obs: Dict[str, Any],
        device: Union[str, torch.device] = "cpu",
    ) -> torch.Tensor:
        """Return 1D bool tensor [ACTION_SPACE_SIZE] of legal actions."""
        mask = obs.get("action_mask", [True] * len(AGENT_ACTIONS))
        return torch.tensor(mask, dtype=torch.bool, device=device)
