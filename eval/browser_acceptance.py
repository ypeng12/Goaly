"""Browser acceptance for the customer flow and the actual RL Lab.

Install the optional Playwright package and Chromium, then start the app first.
No paid model calls: the customer session is explicitly set to offline mode.
"""
import argparse
import json
from pathlib import Path
from playwright.sync_api import sync_playwright


def run(url, output_dir):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    evidence = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        for width, height in [(1440, 1000), (390, 844)]:
            page = browser.new_page(viewport={'width':width,'height':height})
            errors = []
            page.on('pageerror', lambda e: errors.append(str(e)))
            page.goto(url,wait_until='networkidle')
            page.wait_for_function("document.getElementById('engine-label').textContent !== 'Connecting'")
            page.locator('#btn-config').click()
            page.locator('#cfg-engine-mode').select_option('mock')
            with page.expect_response(lambda r:'/api/config' in r.url and r.request.method=='POST'):
                page.locator('#config-save').click()
            page.locator('#config-modal').wait_for(state='hidden')
            assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
            assert page.locator('.inspector-panel').is_hidden()
            assert page.locator('#topic-choices button').count() == 4
            # The input can exist in the DOM yet be clipped by a fixed-height
            # flex panel. Assert actual geometry, not just element visibility.
            assert page.locator('#user-input').evaluate('''e => {
                const input = e.getBoundingClientRect();
                const panel = e.closest('.conversation-panel').getBoundingClientRect();
                return input.bottom <= panel.bottom && input.top >= panel.top;
            }''')
            page.screenshot(path=str(out/f'welcome-{width}.png'), full_page=True)

            def send(message):
                page.locator('#user-input').fill(message)
                with page.expect_response(lambda r:'/api/chat' in r.url) as response:
                    page.locator('#btn-send').click()
                data = response.value.json()
                page.wait_for_timeout(100)
                return data

            data = send("My name is Margaret Chen, policy POL-9921. I'm calling about my denied healthcare claim from January.")
            assert data['current_phase']=='VERIFY_ID' and data['active_case'] is None
            assert data['sop_state']['cross_phase_memory']['date_hint']=='January'
            data = send('DOB 1985-03-15, SSN last four 4472.')
            assert data['active_case']['case_id']=='CL-2048'
            data = send('That answers my question.')
            assert data['current_phase']=='POST_PROCESS'
            page.locator('#btn-decline-email').wait_for(state='visible')
            page.screenshot(path=str(out/f'customer-{width}.png'),full_page=True)
            with page.expect_response(lambda r:'/api/chat' in r.url) as response:
                page.locator('#btn-decline-email').click()
            data=response.value.json()
            assert data['current_phase']=='CONCLUDED' and data['sop_state']['outbox_count']==0

            # A new caller can get beginner help, correct a typo with the form,
            # and keep their original claim hint without bypassing the gate.
            with page.expect_response(lambda r:'/api/reset' in r.url):
                page.locator('#btn-reset').click()
            page.locator('#btn-open-verification').wait_for(state='visible')
            page.locator('#btn-open-verification').click()
            page.locator('#card-name').fill('Ma Tian')
            with page.expect_response(lambda r:'/api/verify-card' in r.url) as response:
                page.locator('#btn-card-submit').click()
            data = response.value.json()
            assert data['current_phase'] == 'VERIFY_ID' and data['active_case'] is None
            assert data['sop_state']['verified_fields'] == ['name']
            page.locator('#verification-close').click()
            assert '1 of 3' in page.locator('#verification-progress').inner_text()
            assert 'locked' in page.locator('#verification-progress').inner_text()
            page.screenshot(path=str(out/f'name-only-locked-{width}.png'), full_page=True)
            with page.expect_response(lambda r:'/api/reset' in r.url):
                page.locator('#btn-reset').click()
            with page.expect_response(lambda r:'/api/chat' in r.url) as response:
                page.get_by_role('button', name='Not sure where to start', exact=True).click()
            assert 'request for your insurer to pay' in response.value.json()['reply']
            assert response.value.json()['current_phase'] == 'VERIFY_ID'
            data = send('My name is Margret Chen. My name is Margaret Chen. Why was my healthcare cliam deneid in January?')
            assert data['active_case'] is None and data['claim_choices'] == []
            page.locator('#btn-open-verification').click()
            assert page.locator('#card-name').input_value() == ''
            page.locator('#btn-card-fill').click()
            assert page.locator('#verification-modal').is_visible()
            page.locator('#card-email').fill('mistyped@')
            page.locator('#btn-card-submit').click()
            assert page.locator('#error-email').inner_text()
            assert page.locator('#card-name').input_value() == 'Margaret Chen'
            page.locator('#card-email').fill('margaret@email.com')
            page.locator('#card-phone').fill('+1 (650) 521-2836')
            page.locator('#card-phone').blur()
            assert page.locator('#card-phone').input_value() == '+1 (650) 521-2836'
            assert page.locator('#card-dob').get_attribute('type') == 'date'
            page.screenshot(path=str(out/f'verification-{width}.png'), full_page=True)
            with page.expect_response(lambda r:'/api/verify-card' in r.url) as response:
                page.locator('#btn-card-submit').click()
            data = response.value.json()
            assert data['current_phase'] == 'PROCESS_CASE' and data['active_case']['case_id'] == 'CL-2048'
            page.locator('#verification-modal').wait_for(state='hidden')
            assert page.locator('#btn-open-verification').is_hidden()
            assert page.locator('#security-card-el').count() == 0
            assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
            with page.expect_response(lambda r:'/api/chat' in r.url) as response:
                page.get_by_role('button', name='How to send documents', exact=True).click()
            assert response.value.json()['current_phase'] == 'PROCESS_CASE'
            page.screenshot(path=str(out/f'guided-chat-{width}.png'), full_page=True)

            page.goto(url.rstrip('/')+'/lab',wait_until='networkidle')
            page.locator('#lab-profile option').first.wait_for(state='attached')
            with page.expect_response(lambda r:'/api/lab/run' in r.url,timeout=90000) as response:
                page.locator('#lab-run').click()
            assert response.value.status==200
            data=response.value.json()
            page.locator('#lab-results').wait_for(state='visible')
            assert page.locator('#lab-summary tr').count()==4
            assert len(data['runs'])==4
            assert page.locator('#lab-takeaway').inner_text()
            assert 'not the chance' in page.locator('#lab-results').inner_text()
            assert page.locator('.lab-action-row.masked').count()>0
            assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
            page.locator('#lab-turn').evaluate('(e)=>{e.value=e.max;e.dispatchEvent(new Event("input",{bubbles:true}));}')
            assert '→' in page.locator('#lab-phase').inner_text()
            page.locator('#lab-turn').evaluate('(e)=>{e.value=0;e.dispatchEvent(new Event("input",{bubbles:true}));}')
            page.screenshot(path=str(out/f'rl-lab-{width}.png'),full_page=True)
            with page.expect_download() as download:
                page.locator('#lab-download').click()
            download.value.save_as(str(out/f'lab-run-{width}.json'))
            assert not errors, errors
            evidence.append({'viewport':width,'customer_finish':'passed','consent_skip':'passed',
                             'beginner_choices':'passed','form_error_recovery':'passed','remembered_typo_hint':'passed',
                             'native_date_and_phone_preservation':'passed','plain_language_lab':'passed',
                             'name_only_remains_locked':'passed','composer_not_clipped':'passed',
                             'rl_comparison':'passed','replay':'passed','download':'passed','page_errors':errors})
            page.close()
        browser.close()
    (out/'browser_acceptance.json').write_text(json.dumps({'passed':True,'checks':evidence},indent=2))
    print(json.dumps(evidence))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url',default='http://127.0.0.1:8080')
    parser.add_argument('--output-dir',default='artifacts/browser_acceptance')
    args=parser.parse_args()
    run(args.url,args.output_dir)
