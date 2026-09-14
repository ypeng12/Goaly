"""Offline checks of the live acceptance rubric; these make no model calls."""
from eval.live_acceptance import provider_checks, expected_response, scenarios


def response(**updates):
    return {'current_phase': 'PROCESS_CASE', 'engine_mode': 'live',
            'policy_decision': {'grounded_answered': True, 'response_generation': 'live_fact_plan'},
            'model_activity': {'interpretation_requests': 1, 'interpretation_succeeded': 1,
                               'planning_requests': 1, 'planning_succeeded': 1}, **updates}


def test_actual_validated_calls_and_plan_meet_the_live_rubric():
    assert all(provider_checks(response(), 'PROCESS_CASE').values())


def test_live_mode_label_without_request_evidence_is_not_a_live_pass():
    checks = provider_checks(response(model_activity=None), 'PROCESS_CASE')
    assert not checks['provider_activity_valid']
    assert not checks['actual_interpretation_or_terminal_no_call']
    assert not checks['grounded_live_planning']


def test_live_interpretation_does_not_count_as_live_response_planning():
    data = response(policy_decision={'grounded_answered': True, 'response_generation': 'deterministic_grounded'},
                    model_activity={'interpretation_requests': 1, 'interpretation_succeeded': 1,
                                    'planning_requests': 0, 'planning_succeeded': 0})
    assert provider_checks(data, 'PROCESS_CASE')['actual_interpretation_or_terminal_no_call']
    assert not provider_checks(data, 'PROCESS_CASE')['grounded_live_planning']


def test_forced_terminal_replay_requires_zero_requests_despite_live_mode_label():
    data = response(current_phase='CONCLUDED', policy_decision={'source': 'sop', 'forced': True},
                    model_activity={'interpretation_requests': 0, 'interpretation_succeeded': 0,
                                    'planning_requests': 0, 'planning_succeeded': 0})
    assert all(provider_checks(data, 'CONCLUDED').values())
    data['model_activity']['interpretation_requests'] = 1
    assert not provider_checks(data, 'CONCLUDED')['actual_interpretation_or_terminal_no_call']


def test_claiming_a_successful_plan_with_no_planning_request_fails():
    data = response(model_activity={'interpretation_requests': 1, 'interpretation_succeeded': 1,
                                    'planning_requests': 0, 'planning_succeeded': 1})
    assert not provider_checks(data, 'PROCESS_CASE')['provider_activity_valid']


def test_fallback_is_reported_even_if_the_mode_label_is_live():
    assert not provider_checks(response(fallback_reason='Using deterministic fallback.'), 'PROCESS_CASE')['no_provider_fallback']


def test_contextual_suite_checks_referent_specific_answer_and_excludes_other_document():
    cases = scenarios()
    expectation = cases['model_referent_paraphrase'][2][2]
    assert expected_response('Ask for a visit summary or a re-sent office note.', expectation)
    assert not expected_response('Ask the hospital or lab for a replacement copy, or a visit summary and office note.', expectation)
    assert cases['contextual_document_chain'][2][0] == 'What about the second one?'
    assert cases['emotional_context_recovery'][2][0] == 'This is frustrating. What is the first one?'
