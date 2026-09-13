"""Optional real-browser check: start ./run.sh, then python tests/browser_smoke.py.
Requires: pip install playwright && python -m playwright install chromium
"""
from pathlib import Path
from playwright.sync_api import sync_playwright, expect

ROOT = Path(__file__).resolve().parents[1]
(ROOT / 'artifacts').mkdir(exist_ok=True)
with sync_playwright() as playwright:
    chrome = Path('/Applications/Google Chrome.app/Contents/MacOS/Google Chrome')
    browser = playwright.chromium.launch(headless=True, **({'executable_path': str(chrome)} if chrome.exists() else {}))
    page = browser.new_page(viewport={'width': 1440, 'height': 1080})
    errors = []
    page.on('pageerror', lambda error: errors.append(str(error)))
    page.goto('http://127.0.0.1:8080', wait_until='networkidle')
    page.locator('#scenario-margaret').click()
    expect(page.locator('#call-status')).to_have_text('Working through your claim')
    expect(page.locator('#engine-label')).to_have_text('Offline demo')
    assert '3 / 3 matched' in page.locator('#verify-status-badge').inner_text()
    assert 'CL-2048' in page.locator('#claim-details-container').inner_text()
    page.locator('#user-input').fill('How do I submit the missing documents?')
    page.locator('#btn-send').click()
    expect(page.locator('#btn-send')).to_be_enabled()
    assert 'portal' in page.locator('#chat-window').inner_text()
    page.locator('#btn-wrap-up').click()
    expect(page.locator('#call-status')).to_have_text('Your follow-up choice')
    page.screenshot(path=str(ROOT / 'artifacts/demo-desktop.png'), full_page=True)
    page.locator('#time-travel-slider').fill('0')
    assert page.locator('#user-input').is_disabled()
    assert 'unavailable' in page.locator('#claim-details-container').inner_text()
    page.locator('#btn-revert-live').click()
    page.locator('#btn-accept-email').click()
    expect(page.locator('#call-status')).to_have_text('Call complete')
    assert 'simulated' in page.locator('#chat-window').inner_text()
    page.locator('#btn-reset').click()
    expect(page.locator('#btn-send')).to_be_enabled()
    page.locator('#scenario-margaret').click()
    expect(page.locator('#call-status')).to_have_text('Working through your claim')
    page.locator('#btn-wrap-up').click()
    expect(page.locator('#call-status')).to_have_text('Your follow-up choice')
    page.locator('#btn-decline-email').click()
    expect(page.locator('#call-status')).to_have_text('Call complete')
    assert 'skip the email' in page.locator('#chat-window').inner_text()
    page.locator('#btn-reset').click()
    expect(page.locator('#btn-send')).to_be_enabled()
    page.locator('#btn-config').click()
    assert page.locator('#config-modal').is_visible()
    page.locator('#cfg-engine-mode').select_option('llm')
    page.locator('#cfg-api-key').fill('test-ui-secret')
    page.locator('#modal-cancel').click()
    assert page.locator('#cfg-api-key').input_value() == ''
    assert page.evaluate('localStorage.length + sessionStorage.length') == 0
    page.set_viewport_size({'width': 390, 'height': 844})
    page.screenshot(path=str(ROOT / 'artifacts/demo-mobile.png'), full_page=True)
    assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth')
    assert not errors, errors
    print('Browser checks passed: send/skip flows, replay, settings cleanup, mobile width; no JS errors.')
    browser.close()
