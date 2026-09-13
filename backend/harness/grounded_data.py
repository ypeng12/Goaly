import json
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from .types import PolicyHolder, ClaimRecord, PIIFields, CrossPhaseMemory, Representative

# Resolve path to fixtures directory
FIXTURES_DIR = Path(__file__).resolve().parent.parent.parent / "fixtures"

class GroundedDataManager:
    def __init__(self, fixtures_dir: Optional[Path] = None):
        self.fixtures_dir = fixtures_dir or FIXTURES_DIR
        self.policyholders: List[PolicyHolder] = []
        self.claims: List[ClaimRecord] = []
        self.claim_schema: Dict[str, Any] = {}
        self.document_guideline: Dict[str, Any] = {}
        self.representatives: List[Representative] = []
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
                data = json.load(f)
                self.representatives = [Representative(**item) for item in data]

        # 6. Consent scenarios
        consent_path = self.fixtures_dir / "consent_scenarios.json"
        if consent_path.exists():
            with open(consent_path, "r", encoding="utf-8") as f:
                self.consent_scenarios = json.load(f)

    def find_representative(self, rep_name: str, policyholder_name: Optional[str] = None) -> Optional[Representative]:
        """A roster entry is an authorization hint, never identity or consent proof."""
        clean_rep = self._clean_str(rep_name)
        if not clean_rep:
            return None
        clean_ph = self._clean_str(policyholder_name)
        candidates = [r for r in self.representatives
                      if self._clean_str(r.rep_name) == clean_rep
                      and (not clean_ph or self._clean_str(r.buyer_name) == clean_ph)]
        return candidates[0] if len(candidates) == 1 else None

    def simulate_consent_check(self, scenario_name: str = "default", attempt_index: int = 0) -> str:
        """Fixture viewer only. No workflow gate trusts this simulated result."""
        scenario = self.consent_scenarios.get(scenario_name, {})
        sequence = scenario.get("status_sequence", ["pending"])
        if not sequence:
            return "pending"
        return sequence[max(0, min(attempt_index, len(sequence) - 1))]

    def get_document_alternative(self, doc_name: str) -> str:
        all_alt_g = self.document_guideline.get("document_alternative_guidance", {})
        for k, v in all_alt_g.items():
            if doc_name.lower() in k.lower() or k.lower() in doc_name.lower():
                return v.get("en", "")
        return all_alt_g.get("default", {}).get("en", "")

    @staticmethod
    def _clean_phone(phone: Optional[str]) -> str:
        digits = re.sub(r"\D", "", phone or "")
        # This demo contains North American numbers; normalize the local form,
        # but never authenticate by a matching suffix of an arbitrary number.
        return "1" + digits if len(digits) == 10 else digits

    @staticmethod
    def _clean_str(s: Optional[str]) -> str:
        return " ".join((s or "").split()).casefold()

    def verify_identity(self, pii: PIIFields) -> Tuple[bool, Optional[PolicyHolder], List[str]]:
        """Require three distinct allowed PII fields and no contradictory input.

        A policy number only narrows the lookup. Last-four digits count only
        when the fixture AND submitted ID type identify an SSN. Aliases must
        match a complete configured value. Multiple eligible people fail closed.
        """
        eligible = []
        partial = []
        for ph in self.policyholders:
            if pii.policy_number and self._clean_str(pii.policy_number) != self._clean_str(ph.policy_number):
                continue
            checks = {}
            if pii.name:
                checks["name"] = self._clean_str(pii.name) in {
                    self._clean_str(n) for n in [ph.name, *ph.name_aliases]
                }
            if pii.dob:
                checks["dob"] = pii.dob.strip() == ph.dob
            if pii.phone:
                checks["phone"] = self._clean_phone(pii.phone) in {
                    self._clean_phone(n) for n in [ph.phone, *ph.phone_aliases]
                }
            if pii.email:
                checks["email"] = self._clean_str(pii.email) in {
                    self._clean_str(e) for e in [ph.email, *ph.email_aliases]
                }
            if pii.id_last4:
                supplied_type = pii.id_type or "ssn_last4"
                # An explicitly supplied non-SSN identifier cannot satisfy this SOP.
                if supplied_type != ph.id_type or not re.fullmatch(r"\d{4}", pii.id_last4) or pii.id_last4 != ph.id_last4:
                    continue
                if supplied_type == "ssn_last4":
                    checks["id_last4"] = True
            matched = [field for field, matches in checks.items() if matches]
            if not all(checks.values()):
                continue
            partial.append((ph, matched))
            if len(matched) >= 3:
                eligible.append((ph, matched))
        if len(eligible) == 1:
            ph, matched = eligible[0]
            return True, ph, matched
        if len(eligible) > 1:
            return False, None, []
        if len(partial) == 1:
            ph, matched = partial[0]
            return False, ph, matched
        return False, None, []

    def get_claims_for_party(self, party_id: str) -> List[ClaimRecord]:
        return [c for c in self.claims if c.party_id == party_id]

    @staticmethod
    def _date_matches(created_at: str, hint: str) -> bool:
        """Intersect every supplied calendar component, including full dates."""
        import calendar
        from datetime import date
        try:
            created = date.fromisoformat(created_at)
        except ValueError:
            return False
        text = hint.strip().casefold()
        components = 0
        dates = re.findall(r"\b((?:19|20)\d{2})[-/]([01]?\d)(?:[-/]([0-3]?\d))?\b", text)
        if dates:
            for year, month, day in dates:
                components += 1
                if created.year != int(year) or created.month != int(month) or (day and created.day != int(day)):
                    return False
        for month in range(1, 13):
            names = {calendar.month_name[month].casefold(), calendar.month_abbr[month].casefold()}
            if any(re.search(r"\b" + re.escape(name) + r"\b", text) for name in names):
                components += 1
                if created.month != month:
                    return False
        for year in re.findall(r"\b(?:19|20)\d{2}\b", text):
            components += 1
            if created.year != int(year):
                return False
        return components > 0

    def find_claim(self, party_id: str, memory: CrossPhaseMemory) -> Tuple[Optional[ClaimRecord], List[ClaimRecord]]:
        """Resolve only within the owner and by strict intersection of all hints.

        Empty filters remain empty. An explicit foreign case, wrong year, or
        contradictory hint can never fall back to a convenient owned claim.
        """
        candidates = list(self.get_claims_for_party(party_id))
        hint_fields = ("case_id_hint", "case_type_hint", "status_hint", "date_hint")
        has_hint = any(getattr(memory, field, None) or memory.hint_history.get(field) for field in hint_fields)
        if not has_hint and not memory.topic_hint:
            return None, candidates
        for field in hint_fields:
            values = list(memory.hint_history.get(field, []))
            current = getattr(memory, field, None)
            if current and current not in values:
                values.append(current)
            for value in values:
                if field == "date_hint":
                    candidates = [c for c in candidates if self._date_matches(c.created_at, value)]
                else:
                    claim_field = {"case_id_hint": "case_id", "case_type_hint": "case_type", "status_hint": "status"}[field]
                    candidates = [c for c in candidates if self._clean_str(getattr(c, claim_field)) == self._clean_str(value)]
        return (candidates[0] if len(candidates) == 1 else None), candidates

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
