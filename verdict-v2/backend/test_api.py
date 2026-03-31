import json
import mimetypes
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / 'backend'
SAMPLE = ROOT / 'sample.csv'
BASE = 'http://127.0.0.1:8765'


def wait_for_server(timeout=20):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f'{BASE}/', timeout=2) as response:
                if response.status == 200:
                    return
        except Exception:
            time.sleep(0.5)
    raise RuntimeError('Server did not start in time')


def request_json(path, payload):
    req = urllib.request.Request(
        f'{BASE}{path}',
        data=json.dumps(payload).encode('utf-8'),
        headers={'Content-Type': 'application/json'},
        method='POST',
    )
    with urllib.request.urlopen(req, timeout=20) as response:
        return json.loads(response.read().decode('utf-8'))


def multipart_body(parts):
    boundary = '----VERDICTBOUNDARY123456'
    body = bytearray()
    for part in parts:
        body.extend(f'--{boundary}\r\n'.encode())
        disposition = f'Content-Disposition: form-data; name="{part["name"]}"'
        if part.get('filename'):
            disposition += f'; filename="{part["filename"]}"'
        body.extend(f'{disposition}\r\n'.encode())
        if part.get('content_type'):
            body.extend(f'Content-Type: {part["content_type"]}\r\n'.encode())
        body.extend(b'\r\n')
        value = part['value']
        if isinstance(value, str):
            value = value.encode('utf-8')
        body.extend(value)
        body.extend(b'\r\n')
    body.extend(f'--{boundary}--\r\n'.encode())
    return boundary, bytes(body)


def post_multipart(path, parts, expect_json=True):
    boundary, body = multipart_body(parts)
    req = urllib.request.Request(
        f'{BASE}{path}',
        data=body,
        headers={'Content-Type': f'multipart/form-data; boundary={boundary}'},
        method='POST',
    )
    with urllib.request.urlopen(req, timeout=20) as response:
        payload = response.read()
        if expect_json:
            return json.loads(payload.decode('utf-8'))
        return response.getheader('Content-Type', ''), payload


def main():
    process = subprocess.Popen(
        [sys.executable, '-m', 'uvicorn', 'verdict_api:app', '--host', '127.0.0.1', '--port', '8765'],
        cwd=BACKEND,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        wait_for_server()
        sample_bytes = SAMPLE.read_bytes()
        upload = post_multipart('/api/upload', [{
            'name': 'file',
            'filename': SAMPLE.name,
            'content_type': mimetypes.guess_type(SAMPLE.name)[0] or 'text/csv',
            'value': sample_bytes,
        }])
        dataset_id = upload['dataset_id']
        outcome_col = upload['detected_outcome']
        sensitive_col = upload['detected_sensitive'][0]

        scan = request_json('/api/scan', {
            'dataset_id': dataset_id,
            'outcome_col': outcome_col,
            'sensitive_col': sensitive_col,
        })

        if len(upload['detected_sensitive']) >= 2:
            request_json('/api/scan-intersectional', {
                'dataset_id': dataset_id,
                'outcome_col': outcome_col,
                'sensitive_cols': upload['detected_sensitive'],
            })

        request_json('/api/attack', {
            'dataset_id': dataset_id,
            'outcome_col': outcome_col,
            'sensitive_col': sensitive_col,
            'original_val': scan['group_stats'][0]['group'],
            'new_val': scan['group_stats'][-1]['group'],
        })
        request_json('/api/fix', {
            'dataset_id': dataset_id,
            'outcome_col': outcome_col,
            'sensitive_col': sensitive_col,
            'original_val': scan['group_stats'][0]['group'],
        })
        request_json('/api/debias', {
            'dataset_id': dataset_id,
            'outcome_col': outcome_col,
            'sensitive_col': sensitive_col,
        })
        preview_row = dict(upload['preview'][0])
        preview_row.pop(outcome_col, None)
        request_json('/api/predict-fair', {
            'dataset_id': dataset_id,
            'outcome_col': outcome_col,
            'sensitive_col': sensitive_col,
            'record': preview_row,
        })
        post_multipart('/api/compare-models', [
            {'name': 'file1', 'filename': 'model_a.csv', 'content_type': 'text/csv', 'value': sample_bytes},
            {'name': 'file2', 'filename': 'model_b.csv', 'content_type': 'text/csv', 'value': sample_bytes},
        ])
        verdict = request_json('/api/verdict', {
            'dataset_id': dataset_id,
            'is_fair': scan['severity'] == 'LOW',
            'severity': scan['severity'],
            'disparity': scan['disparity_ratio'],
            'feature': sensitive_col,
        })
        pdf_req = urllib.request.Request(
            f'{BASE}/api/export-pdf',
            data=json.dumps({
                'decision_id': verdict['decision_id'],
                'verdict': verdict['verdict'],
                'integrity_hash': verdict['integrity_hash'],
                'report_lines': verdict['detailed_report'],
                'compliance': verdict['compliance'],
            }).encode('utf-8'),
            headers={'Content-Type': 'application/json'},
            method='POST',
        )
        with urllib.request.urlopen(pdf_req, timeout=20) as response:
            if 'application/pdf' not in response.getheader('Content-Type', ''):
                raise RuntimeError(f'Unexpected PDF content type: {response.getheader("Content-Type", "")}')

        print('All VERDICT API endpoints passed against sample.csv')
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()


if __name__ == '__main__':
    main()
