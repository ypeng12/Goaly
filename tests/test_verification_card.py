"""Tests for direct Security Verification Card form submission and deterministic LLM-bypass gate."""
import pytest
from fastapi.testclient import TestClient
from backend.app import app, sessions
from backend.harness.state_machine import SOPStateMachine
from backend.harness.types import Phase


def test_state_machine_verify_card_data_success():
    sm = SOPStateMachine("test_session_card_1")
    assert sm.state.phase == Phase.VERIFY_ID
    assert not sm.state.identity_verified

    # Submit 3 matching fields on card (Margaret Chen)
    res = sm.verify_card_data({
        "name": "Margaret Chen",
        "dob": "1985-03-15",
        "phone": "(650) 521-2836",
        "email": "margaret@email.com",
        "id_last4": "4472"
    })

    assert sm.state.identity_verified is True
    assert sm.state.phase in {Phase.RESOLVE_INTENT, Phase.PROCESS_CASE}
    assert "Thank you, Margaret Chen" in res["agent_reply"]
    assert sm.get_verified_policyholder() is not None
    assert sm.get_verified_policyholder().name == "Margaret Chen"


def test_state_machine_verify_card_data_partial_failure():
    sm = SOPStateMachine("test_session_card_2")
    assert sm.state.phase == Phase.VERIFY_ID

    # Submit only 1 field
    res = sm.verify_card_data({
        "name": "Margaret Chen",
        "dob": "",
        "phone": "",
        "email": "",
        "id_last4": ""
    })

    assert sm.state.identity_verified is False
    assert sm.state.phase == Phase.VERIFY_ID
    assert "We need 2 more matching field(s)" in res["agent_reply"]


def test_api_verify_card_endpoint():
    client = TestClient(app)

    # 1. Reset session
    reset_resp = client.post("/api/reset", json={})
    assert reset_resp.status_code == 200
    sid = reset_resp.json()["session_id"]

    # 2. Verify card via API
    card_resp = client.post("/api/verify-card", json={
        "session_id": sid,
        "name": "Margaret Chen",
        "dob": "03/15/1985",
        "phone": "(650) 521-2836",
        "email": "margaret@email.com",
        "id_last4": "4472"
    })


    assert card_resp.status_code == 200
    data = card_resp.json()
    assert data["sop_state"]["identity_verified"] is True
    assert data["current_phase"] in ["RESOLVE_INTENT", "PROCESS_CASE"]
    assert "Margaret Chen" in data["reply"]


def test_national_id_policyholder_verification():
    sm = SOPStateMachine("test_session_matian")
    res = sm.verify_card_data({
        "name": "Ma Tian",
        "dob": "1964-09-10",
        "phone": "(650) 208-8799",
        "email": "matian@example.com",
        "id_last4": "6688",
        "id_type": "national_id_last4"
    })
    assert sm.state.identity_verified is True
    assert sm.get_verified_policyholder().name == "Ma Tian"

