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

            page.goto(url.rstrip('/')+'/lab',wait_until='networkidle')
            page.locator('#lab-profile option').first.wait_for(state='attached')
            with page.expect_response(lambda r:'/api/lab/run' in r.url,timeout=90000) as response:
                page.locator('#lab-run').click()
            assert response.value.status==200
            data=response.value.json()
            page.locator('#lab-results').wait_for(state='visible')
            assert page.locator('#lab-summary tr').count()==4
            assert len(data['runs'])==4
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
