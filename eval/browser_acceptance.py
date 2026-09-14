"""Browser acceptance for customer conversation, trained action selection and RL Lab.

Install Playwright and Chromium, then start the app. Customer sessions explicitly
use offline language; no paid model calls. Saved records use synthetic identities.
"""
import argparse
import json
import math
import re
from datetime import date
from pathlib import Path

from playwright.sync_api import expect, sync_playwright


CONTROLLERS = {'rule': 'Rule baseline', 'ppo42': 'Trained PPO · 42', 'ppo7': 'Trained PPO · 7'}


def check_decision(data, source=None, terminal=False):
    """Validate server evidence without inventing checkpoint hashes or scores."""
    decision = data['policy_decision']
    if terminal:
        assert decision['source'] == 'sop'
        assert decision['selected_action'] is None
        assert decision['checkpoint_sha256'] is None
        assert decision['allowed_actions'] == []
        assert decision['forced'] is True
        assert all(value == 0 for value in decision['probabilities'].values())
        return decision
    if source:
        assert decision['source'] == source, decision
    probabilities = decision['probabilities']
    allowed = decision['allowed_actions']
    assert decision['selected_action'] in allowed
    assert all(math.isfinite(value) and 0 <= value <= 1 for value in probabilities.values())
    assert abs(sum(probabilities.values()) - 1) < 1e-5
    assert all(probabilities[action] == 0 for action in probabilities if action not in allowed)
    if decision['source'].startswith('ppo'):
        assert re.fullmatch('[0-9a-f]{64}', decision['checkpoint_sha256'])
        assert decision['selected_action'] == max(probabilities, key=probabilities.get)
    elif decision['source'] == 'rule':
        assert decision['checkpoint_sha256'] is None
    return decision


def run(url, output_dir, require_customer_report=False):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    evidence = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        for width, height in [(1440, 1000), (390, 844)]:
            page = browser.new_page(viewport={'width': width, 'height': height})
            errors, customer_turns, controller_records = [], [], {}
            page.on('pageerror', lambda e: errors.append(str(e)))

            def no_overflow():
                assert page.evaluate('() => document.documentElement.scrollWidth <= innerWidth')

            def composer_in_view():
                assert page.locator('#user-input').evaluate('''e => {
                    const input = e.getBoundingClientRect();
                    const panel = e.closest('.conversation-panel').getBoundingClientRect();
                    return input.top >= 0 && input.bottom <= innerHeight &&
                        input.bottom <= panel.bottom && input.top >= panel.top;
                }'''), f'The composer must be visible on first load at {width}×{height}'

            def ready():
                expect(page.locator('#btn-config')).to_be_enabled()

            def configure(controller):
                ready()
                page.locator('#btn-config').click()
                page.locator('#cfg-controller').select_option(controller)
                page.locator('#cfg-engine-mode').select_option('mock')
                with page.expect_response(lambda r: '/api/config' in r.url and r.request.method == 'POST') as response:
                    page.locator('#config-save').click()
                saved = response.value.json()
                assert saved['controller'] == controller and saved['use_mock_only']
                expect(page.locator('#config-modal')).to_be_hidden()
                expect(page.locator('#controller-label')).to_contain_text(CONTROLLERS[controller])
                expect(page.locator('#language-label')).to_have_text('Language: offline')

            def send(message):
                expect(page.locator('#user-input')).to_be_enabled()
                page.locator('#user-input').fill(message)
                with page.expect_response(lambda r: '/api/chat' in r.url) as response:
                    page.locator('#btn-send').click()
                assert response.value.status == 200
                data = response.value.json()
                expect(page.locator('#chat-window .assistant').last).to_contain_text(data['reply'].split('\n')[0].strip())
                page.wait_for_function('''() => {
                    const chat = document.getElementById('chat-window');
                    return chat.scrollHeight - chat.scrollTop - chat.clientHeight < 2;
                }''')
                expect(page.locator('#policy-decision')).to_be_visible()
                customer_turns.append({'message': message, **data})
                return data

            def new_call():
                with page.expect_response(lambda r: '/api/reset' in r.url):
                    page.locator('#btn-reset').click()
                ready()
                expect(page.locator('#user-input')).to_be_enabled()

            def show_suggestions():
                suggestions = page.locator('#suggested-topics')
                if suggestions.get_attribute('open') is None:
                    page.locator('#suggested-topics > summary').click()

            page.goto(url, wait_until='networkidle')
            ready()
            no_overflow()
            composer_in_view()
            assert page.locator('.inspector-panel').is_hidden()
            assert page.locator('#topic-choices button').count() == 4
            assert page.locator('#suggested-topics').get_attribute('open') is None
            page.screenshot(path=str(out / f'welcome-{width}.png'), full_page=True)
            configure('ppo42')

            # Free-text, cross-phase memory and linked follow-ups, not button scripts.
            data = send("My name is Margaret Chen, policy POL-9921. I'm calling about my denied healthcare claim from January.")
            assert data['current_phase'] == 'VERIFY_ID' and data['active_case'] is None
            assert data['sop_state']['cross_phase_memory']['date_hint'] == 'January'
            assert data['sop_state']['verified_fields'] == ['name']
            check_decision(data, 'ppo42')
            data = send('DOB 1985-03-15, SSN last four 4472.')
            assert data['active_case']['case_id'] == 'CL-2048'
            claim = data['active_case']
            data = send('What documents do I need?')
            assert all(document.lower() in data['reply'].lower() for document in claim['documents_needed'])
            data = send('What is the second one?')
            assert data['current_phase'] == 'PROCESS_CASE'
            assert claim['documents_needed'][1].lower() in data['reply'].lower()
            assert 'assessment' in data['reply'].lower()
            data = send("I can't get it.")
            assert data['current_phase'] == 'PROCESS_CASE'
            assert any(term in data['reply'].lower() for term in ['visit summary', 'discharge', 're-sent office note'])
            data = send('Can I send photos?')
            assert data['current_phase'] == 'PROCESS_CASE'
            assert re.search(r'photo|scan|cop(?:y|ies)|readab|legib', data['reply'], re.I), data['reply']
            data = send('What is the deadline?')
            assert data['current_phase'] == 'PROCESS_CASE'
            assert claim['appeal_deadline'] in data['reply']
            if date.fromisoformat(claim['appeal_deadline']) < date.today():
                assert 'passed' in data['reply'].lower() and 'cannot promise' in data['reply'].lower()
            check_decision(data, 'ppo42')
            if page.locator('#policy-decision').get_attribute('open') is None:
                page.locator('#policy-decision > summary').click()
            expect(page.locator('#policy-decision-boundary')).to_contain_text('SOP required')
            no_overflow()
            page.screenshot(path=str(out / f'customer-followups-{width}.png'), full_page=True)
            data = send('That answers my question.')
            assert data['current_phase'] == 'POST_PROCESS'
            expect(page.locator('#btn-decline-email')).to_be_visible()
            with page.expect_response(lambda r: '/api/chat' in r.url) as response:
                page.locator('#btn-decline-email').click()
            data = response.value.json()
            assert data['current_phase'] == 'CONCLUDED' and data['sop_state']['outbox_count'] == 0
            check_decision(data, terminal=True)
            customer_turns.append({'message': response.value.request.post_data_json['message'], **data})
            expect(page.locator('#policy-decision-title')).to_contain_text('Required SOP step')
            assert 'Trained PPO' not in page.locator('#policy-decision-title').inner_text()
            expect(page.locator('#user-input')).to_be_disabled()

            # Switch all three actual customer controllers and inspect genuine
            # distributions at a state with multiple choices, not a forced action.
            for controller in CONTROLLERS:
                new_call()
                configure(controller)
                data = send('This is frustrating. I need help with my claim.')
                assert data['current_phase'] == 'VERIFY_ID' and data['active_case'] is None
                decision = check_decision(data, controller)
                assert len(decision['allowed_actions']) > 1 and not decision['forced']
                if controller != 'rule':
                    assert sum(value > 0 for value in decision['probabilities'].values()) > 1
                controller_records[controller] = decision
                if page.locator('#policy-decision').get_attribute('open') is None:
                    page.locator('#policy-decision > summary').click()
                expect(page.locator('#policy-decision-title')).to_contain_text(CONTROLLERS[controller])
                assert page.locator('.decision-option').count() == len(decision['allowed_actions'])
                expect(page.locator('.decision-option.selected')).to_have_count(1)
                rendered_record = json.loads(page.locator('#policy-decision-record').text_content())
                assert rendered_record == decision
                no_overflow()
                page.screenshot(path=str(out / f'controller-{controller}-{width}.png'), full_page=True)

            # Single-field verification remains locked and typo recovery is explicit.
            new_call()
            page.locator('#btn-open-verification').click()
            page.locator('#card-name').fill('Ma Tian')
            with page.expect_response(lambda r: '/api/verify-card' in r.url) as response:
                page.locator('#btn-card-submit').click()
            data = response.value.json()
            assert data['current_phase'] == 'VERIFY_ID' and data['active_case'] is None
            assert data['sop_state']['verified_fields'] == ['name']
            page.locator('#verification-close').click()
            expect(page.locator('#verification-progress')).to_contain_text('1 of 3')
            expect(page.locator('#verification-progress')).to_contain_text('locked')
            page.screenshot(path=str(out / f'name-only-locked-{width}.png'), full_page=True)
            new_call()
            show_suggestions()
            with page.expect_response(lambda r: '/api/chat' in r.url) as response:
                page.get_by_role('button', name='Not sure where to start', exact=True).click()
            assert 'request for your insurer to pay' in response.value.json()['reply']
            assert response.value.json()['current_phase'] == 'VERIFY_ID'
            data = send('My name is Margret Chen. My name is Margaret Chen. Why was my healthcare cliam deneid in January?')
            assert data['active_case'] is None and data['claim_choices'] == []
            page.locator('#btn-open-verification').click()
            expect(page.locator('#card-name')).to_be_empty()
            page.locator('#btn-card-fill').click()
            expect(page.locator('#verification-modal')).to_be_visible()
            page.locator('#card-email').fill('mistyped@')
            page.locator('#btn-card-submit').click()
            assert page.locator('#error-email').inner_text()
            assert page.locator('#card-name').input_value() == 'Margaret Chen'
            page.locator('#card-email').fill('margaret@email.com')
            page.locator('#card-phone').fill('+1 (650) 521-2836')
            page.locator('#card-phone').blur()
            assert page.locator('#card-phone').input_value() == '+1 (650) 521-2836'
            assert page.locator('#card-dob').get_attribute('type') == 'date'
            page.screenshot(path=str(out / f'verification-{width}.png'), full_page=True)
            with page.expect_response(lambda r: '/api/verify-card' in r.url) as response:
                page.locator('#btn-card-submit').click()
            data = response.value.json()
            assert data['current_phase'] == 'PROCESS_CASE' and data['active_case']['case_id'] == 'CL-2048'
            expect(page.locator('#verification-modal')).to_be_hidden()
            assert page.locator('#btn-open-verification').is_hidden()
            assert page.locator('#security-card-el').count() == 0
            no_overflow()
            show_suggestions()
            with page.expect_response(lambda r: '/api/chat' in r.url) as response:
                page.get_by_role('button', name='How to send documents', exact=True).click()
            assert response.value.json()['current_phase'] == 'PROCESS_CASE'
            page.screenshot(path=str(out / f'guided-chat-{width}.png'), full_page=True)

            # Simulator metrics stay separate from the actual customer transcript.
            with page.expect_response(lambda r: '/api/lab/evidence' in r.url) as saved_evidence:
                page.goto(url.rstrip('/') + '/lab', wait_until='networkidle')
            customer_report = saved_evidence.value.json().get('customer_policy')
            if require_customer_report:
                assert customer_report and customer_report['environment_version'] == 4, 'A saved v4 customer report is required for this acceptance run.'
            if customer_report:
                expect(page.locator('#customer-evidence-results')).to_be_visible()
                expect(page.locator('#customer-result-policy')).to_have_value('customer_ppo42')
                report_policy = customer_report['policies']['customer_ppo42']
                expect(page.locator('#customer-evidence-status')).to_contain_text('not a blind test')
                expect(page.locator('#customer-safety')).to_have_text(str(report_policy['metrics']['safety_violations']))
                expect(page.locator('#customer-repetition')).to_have_text(str(report_policy['metrics']['unnecessary_gate_explanations']))
                assert 'Mean policy decisions' in page.locator('.customer-comparison').text_content()
                assert 'Unresolved handoffs' in page.locator('.customer-comparison').text_content()
                expected_completion = report_policy['metrics']['case_completion_rate'] * 100
                displayed_completion = float(page.locator('#customer-completion').inner_text().rstrip('%').replace(',', ''))
                assert abs(displayed_completion - expected_completion) < 0.06
                for controller in ('ppo42', 'ppo7'):
                    assert controller_records[controller]['checkpoint_sha256'] == customer_report['policies'][f'customer_{controller}']['checkpoint']['sha256'], 'Displayed customer evaluation must correspond to the checkpoint used in customer chat.'
                    assert controller_records[controller]['checkpoint']['environment_version'] == 4
                with page.expect_download() as customer_download:
                    page.locator('#customer-report-download').click()
                customer_report_path = out / f'customer-report-{width}.json'
                customer_download.value.save_as(str(customer_report_path))
                downloaded_report = json.loads(customer_report_path.read_text())
                assert downloaded_report['policies']['customer_ppo42']['checkpoint']['sha256'] == report_policy['checkpoint']['sha256']
            else:
                expect(page.locator('#customer-evidence-results')).to_be_hidden()
                expect(page.locator('#customer-evidence-status')).to_contain_text('No compatible v4')
            expect(page.locator('#lab-open-chat')).to_have_attribute('href', '/?controller=ppo42')
            expect(page.locator('#lab-profile option').first).to_be_attached()
            with page.expect_response(lambda r: '/api/lab/run' in r.url, timeout=90000) as response:
                page.locator('#lab-run').click()
            assert response.value.status == 200
            data = response.value.json()
            expect(page.locator('#lab-results')).to_be_visible()
            expect(page.locator('#lab-summary tr')).to_have_count(4)
            assert len(data['runs']) == 4
            assert page.locator('#lab-takeaway').inner_text()
            assert 'not the chance' in page.locator('#lab-results').inner_text()
            assert page.locator('.lab-action-row.masked').count() > 0
            no_overflow()
            page.locator('#lab-turn').evaluate('(e) => {e.value=e.max;e.dispatchEvent(new Event("input",{bubbles:true}));}')
            expect(page.locator('#lab-phase')).to_contain_text('→')
            page.locator('#lab-turn').evaluate('(e) => {e.value=0;e.dispatchEvent(new Event("input",{bubbles:true}));}')
            page.screenshot(path=str(out / f'rl-lab-{width}.png'), full_page=True)
            with page.expect_download() as download:
                page.locator('#lab-download').click()
            download.value.save_as(str(out / f'lab-run-{width}.json'))
            assert not errors, errors
            (out / f'customer-runtime-{width}.json').write_text(json.dumps({
                'controllers': controller_records, 'customer_turns': customer_turns,
            }, indent=2))
            evidence.append({
                'viewport': width, 'customer_finish': 'passed', 'consent_skip': 'passed',
                'multi_turn_reference_and_alternatives': 'passed', 'photos_and_deadline': 'passed',
                'three_actual_controllers': 'passed', 'checkpoint_and_distribution_evidence': 'passed',
                'sop_terminal_attribution': 'passed', 'beginner_choices': 'passed',
                'form_error_recovery': 'passed', 'remembered_typo_hint': 'passed',
                'native_date_and_phone_preservation': 'passed', 'plain_language_lab': 'passed',
                'customer_report_matches_live_checkpoints': 'passed' if customer_report else 'not generated',
                'name_only_remains_locked': 'passed', 'composer_visible_on_first_load': 'passed',
                'rl_comparison': 'passed', 'replay': 'passed', 'download': 'passed', 'page_errors': errors,
            })
            page.close()
        # An additional narrow viewport covers first-load geometry without
        # repeating policy experiments that are already exercised above.
        page = browser.new_page(viewport={'width': 320, 'height': 700})
        page.goto(url, wait_until='networkidle')
        expect(page.locator('#user-input')).to_be_enabled()
        assert page.locator('#user-input').evaluate('e => e.getBoundingClientRect().bottom <= innerHeight')
        assert page.evaluate('() => document.documentElement.scrollWidth <= innerWidth')
        page.screenshot(path=str(out / 'welcome-320.png'), full_page=True)
        evidence.append({'viewport': 320, 'composer_visible_on_first_load': 'passed', 'horizontal_overflow': False})
        browser.close()
    (out / 'browser_acceptance.json').write_text(json.dumps({'passed': True, 'checks': evidence}, indent=2))
    print(json.dumps(evidence))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:8080')
    parser.add_argument('--output-dir', default='artifacts/browser_acceptance')
    parser.add_argument('--require-customer-report', action='store_true', help='Require the saved v4 report and verify its hashes match actual customer decisions.')
    args = parser.parse_args()
    run(args.url, args.output_dir, args.require_customer_report)
