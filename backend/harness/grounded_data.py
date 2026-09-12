import json
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from .types import PolicyHolder, ClaimRecord, PIIFields, CrossPhaseMemory

# Resolve path to fixtures directory
FIXTURES_DIR = Path(__file__).resolve().parent.parent.parent / "fixtures"

class GroundedDataManager:
    def __init__(self, fixtures_dir: Optional[Path] = None):
        self.fixtures_dir = fixtures_dir or FIXTURES_DIR
        self.policyholders: List[PolicyHolder] = []
        self.claims: List[ClaimRecord] = []
        self.claim_schema: Dict[str, Any] = {}
        self.document_guideline: Dict[str, Any] = {}
        self.representatives: List[Dict[str, Any]] = []
        self.consent_scenarios: Dict[str, Any] = {}
        self._load_fixtures()

    def _load_fixtures(self):
        # 1. Policyholders
        ph_path = self.fixtures_dir / "policyholders.json"
        if ph_path.exists():
            with open(ph_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.policyholders = [PolicyHolder(**item) for item in data]

        # 2. Claims
        claims_path = self.fixtures_dir / "claims.json"
        if claims_path.exists():
            with open(claims_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.claims = [ClaimRecord(**item) for item in data]

        # 3. Claim schema
        schema_path = self.fixtures_dir / "claim_schema.json"
        if schema_path.exists():
            with open(schema_path, "r", encoding="utf-8") as f:
                self.claim_schema = json.load(f)

        # 4. Document guidelines
        guideline_path = self.fixtures_dir / "required_document_guideline.json"
        if guideline_path.exists():
            with open(guideline_path, "r", encoding="utf-8") as f:
                self.document_guideline = json.load(f)

        # 5. Representatives
        rep_path = self.fixtures_dir / "representatives.json"
        if rep_path.exists():
            with open(rep_path, "r", encoding="utf-8") as f:
                self.representatives = json.load(f)

        # 6. Consent scenarios
        consent_path = self.fixtures_dir / "consent_scenarios.json"
        if consent_path.exists():
            with open(consent_path, "r", encoding="utf-8") as f:
                self.consent_scenarios = json.load(f)

    @staticmethod
    def _clean_phone(phone: Optional[str]) -> str:
        if not phone:
            return ""
        return re.sub(r"\D", "", phone)

    @staticmethod
    def _clean_str(s: Optional[str]) -> str:
        if not s:
            return ""
        return s.strip().lower()

    def verify_identity(self, pii: PIIFields) -> Tuple[bool, Optional[PolicyHolder], List[str]]:
        """
        Verify if accumulated PII matches a known policyholder with at least 3 matching fields.
        Allowed verification fields: Full name, DOB, Phone, Email, SSN/National ID last 4 digits, Policy number.
        Returns: (is_verified, matched_policyholder, matched_field_names)
        """
        best_match: Optional[PolicyHolder] = None
        max_matched_fields: List[str] = []

        for ph in self.policyholders:
            matched_fields: List[str] = []

            # 1. Full name (support primary name and aliases)
            if pii.name:
                user_name = self._clean_str(pii.name)
                candidate_names = [self._clean_str(ph.name)] + [self._clean_str(a) for a in ph.name_aliases]
                if any(user_name in c or c in user_name for c in candidate_names):
                    matched_fields.append("name")

            # 2. Policy number
            if pii.policy_number:
                if self._clean_str(pii.policy_number) == self._clean_str(ph.policy_number):
                    matched_fields.append("policy_number")

            # 3. DOB (supports exact or normalized date)
            if pii.dob:
                if self._clean_str(pii.dob) == self._clean_str(ph.dob):
                    matched_fields.append("dob")

            # 4. SSN last 4 or ID last 4
            if pii.id_last4:
                if pii.id_last4.strip() == ph.id_last4.strip():
                    matched_fields.append("id_last4")

            # 5. Phone
            if pii.phone:
                clean_user_phone = self._clean_phone(pii.phone)
                candidate_phones = [self._clean_phone(ph.phone)] + [self._clean_phone(a) for a in ph.phone_aliases]
                if any(clean_user_phone[-10:] == c[-10:] for c in candidate_phones if len(c) >= 10):
                    matched_fields.append("phone")

            # 6. Email
            if pii.email:
                clean_user_email = self._clean_str(pii.email)
                candidate_emails = [self._clean_str(ph.email)] + [self._clean_str(a) for a in ph.email_aliases]
                if clean_user_email in candidate_emails:
                    matched_fields.append("email")

            if len(matched_fields) > len(max_matched_fields):
                max_matched_fields = matched_fields
                best_match = ph

        # Rule: At least 3 PII info fields verified
        is_verified = len(max_matched_fields) >= 3 and best_match is not None
        return is_verified, best_match, max_matched_fields

    def get_claims_for_party(self, party_id: str) -> List[ClaimRecord]:
        return [c for c in self.claims if c.party_id == party_id]

    def find_claim(self, party_id: str, memory: CrossPhaseMemory) -> Tuple[Optional[ClaimRecord], List[ClaimRecord]]:
        """
        Attempts to resolve the target claim using cross-phase memory hints.
        Returns: (uniquely_matched_claim, candidate_claims)
        """
        user_claims = self.get_claims_for_party(party_id)
        if not user_claims:
            return None, []

        candidates = list(user_claims)

        # Filter by case type if hint present
        if memory.case_type_hint:
            ct = memory.case_type_hint.lower()
            filtered = [c for c in candidates if c.case_type.lower() in ct or ct in c.case_type.lower()]
            if filtered:
                candidates = filtered

        # Filter by status if hint present
        if memory.status_hint:
            st = memory.status_hint.lower()
            filtered = [c for c in candidates if c.status.lower() == st]
            if filtered:
                candidates = filtered

        # Filter by date hint if present
        if memory.date_hint:
            dh = memory.date_hint.lower()
            filtered = []
            for c in candidates:
                # e.g. "January" -> month "01", or "2026-01-12"
                if "jan" in dh and "-01-" in c.created_at:
                    filtered.append(c)
                elif "feb" in dh and "-02-" in c.created_at:
                    filtered.append(c)
                elif "mar" in dh and "-03-" in c.created_at:
                    filtered.append(c)
                elif "nov" in dh and "-11-" in c.created_at:
                    filtered.append(c)
                elif dh in c.created_at:
                    filtered.append(c)
            if filtered:
                candidates = filtered

        if len(candidates) == 1:
            return candidates[0], candidates
        return None, candidates

    def get_claim_by_id(self, case_id: str) -> Optional[ClaimRecord]:
        for c in self.claims:
            if c.case_id == case_id:
                return c
        return None

    def get_document_guidance_for_claim(self, claim: ClaimRecord) -> Dict[str, Any]:
        """
        Retrieves specific document instructions and alternatives for the claim's missing docs.
        """
        doc_guidances = {}
        alt_guidances = {}

        all_doc_g = self.document_guideline.get("document_guidance", {})
        all_alt_g = self.document_guideline.get("document_alternative_guidance", {})

        for doc in claim.documents_needed:
            # find matching key
            matched_key = None
            for k in all_doc_g:
                if doc.lower() in k.lower() or k.lower() in doc.lower():
                    matched_key = k
                    break
            if matched_key:
                doc_guidances[doc] = all_doc_g[matched_key].get("en", "")
                alt_guidances[doc] = all_alt_g.get(matched_key, {}).get("en", all_alt_g.get("default", {}).get("en", ""))
            else:
                doc_guidances[doc] = self.document_guideline.get("default_guidance", {}).get("en", "")
                alt_guidances[doc] = all_alt_g.get("default", {}).get("en", "")

        return {
            "default_guidance": self.document_guideline.get("default_guidance", {}).get("en", ""),
            "case_type_guidance": self.document_guideline.get("case_type_guidance", {}).get(claim.case_type, {}).get("en", ""),
            "document_guidance": doc_guidances,
            "document_alternative_guidance": alt_guidances,
            "claim_followup_settings": self.document_guideline.get("claim_followup_settings", {}),
            "followup_qa": self.document_guideline.get("claim_followup_guidance", [])
        }

# Global singleton
grounded_data = GroundedDataManager()
