import React, { useRef, useState } from 'react';

const API = 'http://localhost:8000/api';

function ArcGauge({ valueStr, label }) {
  const value = valueStr === 'HIGH' ? 86 : valueStr === 'MEDIUM' ? 52 : 18;
  const color = valueStr === 'HIGH' ? '#FF3B3B' : valueStr === 'MEDIUM' ? '#F5A623' : '#00C896';
  const radius = 75;
  const cx = 90;
  const cy = 85;
  const angle = Math.PI + (value / 100) * Math.PI;
  const x = cx + radius * Math.cos(angle);
  const y = cy + radius * Math.sin(angle);

  return (
    <svg width="180" height="95" viewBox="0 0 180 95">
      <path d="M 15 85 A 75 75 0 0 1 165 85" fill="none" stroke="#2A2A2A" strokeWidth="10" strokeLinecap="round" />
      <path d={`M 15 85 A 75 75 0 0 1 ${x.toFixed(1)} ${y.toFixed(1)}`} fill="none" stroke={color} strokeWidth="10" strokeLinecap="round" />
      <text x="90" y="78" textAnchor="middle" fill={color} fontSize="28" fontFamily="Playfair Display, serif" fontWeight="900">{valueStr}</text>
      <text x="90" y="92" textAnchor="middle" fill="#666" fontSize="9" fontFamily="IBM Plex Mono, monospace" letterSpacing="3">{label}</text>
    </svg>
  );
}

function StatusTag({ value }) {
  const cls = value === 'HIGH' || String(value).includes('NON') || String(value).includes('RISK')
    ? 'tag-red'
    : value === 'MEDIUM'
      ? 'tag-amber'
      : 'tag-green';
  return <span className={`tag ${cls}`}>{value}</span>;
}

function AppShell() {
  const [step, setStep] = useState(1);
  const [datasetId, setDatasetId] = useState('');
  const [info, setInfo] = useState(null);
  const [scan, setScan] = useState(null);
  const [intersections, setIntersections] = useState(null);
  const [attack, setAttack] = useState(null);
  const [fix, setFix] = useState(null);
  const [debias, setDebias] = useState(null);
  const [shieldInputs, setShieldInputs] = useState({});
  const [shieldResult, setShieldResult] = useState(null);
  const [verdict, setVerdict] = useState(null);
  const [compareFiles, setCompareFiles] = useState({ fileA: null, fileB: null });
  const [compareResult, setCompareResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [loadingText, setLoadingText] = useState('');
  const [outcomeCol, setOutcomeCol] = useState('');
  const [sensitiveCol, setSensitiveCol] = useState('');
  const [faceOffRevealed, setFaceOffRevealed] = useState(false);
  const [userChoice, setUserChoice] = useState('');
  const [showModal, setShowModal] = useState(false);
  const fileRef = useRef(null);

  const withLoader = async (text, fn) => {
    setLoading(true);
    setLoadingText(text);
    try {
      return await fn();
    } finally {
      setLoading(false);
    }
  };

  const postJSON = async (endpoint, body) => {
    const response = await fetch(`${API}${endpoint}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || 'Request failed');
    }
    return data;
  };

  const handleUpload = async (event) => {
    const file = event.target.files?.[0];
    if (!file) return;

    await withLoader('Parsing dataset variables', async () => {
      const form = new FormData();
      form.append('file', file);
      const response = await fetch(`${API}/upload`, { method: 'POST', body: form });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail || 'Upload failed');
      }
      setDatasetId(data.dataset_id);
      setInfo(data);
      setOutcomeCol(data.detected_outcome || data.columns[0] || '');
      setSensitiveCol(data.detected_sensitive?.[0] || data.columns[1] || data.columns[0] || '');
      setScan(null);
      setIntersections(null);
      setAttack(null);
      setFix(null);
      setDebias(null);
      setShieldResult(null);
      setVerdict(null);
      setStep(1);
    }).catch((error) => window.alert(error.message));
  };

  const runScan = async () => {
    await withLoader('Analysing disparity metrics', async () => {
      const scanData = await postJSON('/scan', { dataset_id: datasetId, outcome_col: outcomeCol, sensitive_col: sensitiveCol });
      setScan(scanData);
      if ((info?.detected_sensitive || []).length >= 2) {
        const intersectionData = await postJSON('/scan-intersectional', {
          dataset_id: datasetId,
          outcome_col: outcomeCol,
          sensitive_cols: info.detected_sensitive,
        });
        setIntersections(intersectionData);
      } else {
        setIntersections(null);
      }
      setStep(2);
    }).catch((error) => window.alert(error.message));
  };

  const runAttack = async () => {
    await withLoader('Generating adversarial profiles', async () => {
      const original = scan?.group_stats?.[0]?.group || '';
      const modified = scan?.group_stats?.[1]?.group || scan?.group_stats?.[0]?.group || '';
      const data = await postJSON('/attack', {
        dataset_id: datasetId,
        outcome_col: outcomeCol,
        sensitive_col: sensitiveCol,
        original_val: original,
        new_val: modified,
      });
      setAttack(data);
      setUserChoice('');
      setFaceOffRevealed(false);
      setStep(3);
    }).catch((error) => window.alert(error.message));
  };

  const runFix = async () => {
    await withLoader('Generating fair assessment', async () => {
      const original = scan?.group_stats?.[0]?.group || '';
      const data = await postJSON('/fix', {
        dataset_id: datasetId,
        outcome_col: outcomeCol,
        sensitive_col: sensitiveCol,
        original_val: original,
      });
      setFix(data);
      setStep(4);
    }).catch((error) => window.alert(error.message));
  };

  const runDebias = async () => {
    await withLoader('Running debiasing engine', async () => {
      const data = await postJSON('/debias', { dataset_id: datasetId, outcome_col: outcomeCol, sensitive_col: sensitiveCol });
      setDebias(data);
    }).catch((error) => window.alert(error.message));
  };

  const runShield = async () => {
    await withLoader('Running counterfactual analysis', async () => {
      const data = await postJSON('/predict-fair', {
        dataset_id: datasetId,
        outcome_col: outcomeCol,
        sensitive_col: sensitiveCol,
        record: shieldInputs,
      });
      setShieldResult(data);
    }).catch((error) => window.alert(error.message));
  };

  const runVerdict = async () => {
    await withLoader('Generating comprehensive audit report', async () => {
      const data = await postJSON('/verdict', {
        dataset_id: datasetId,
        is_fair: (scan?.severity || 'HIGH') === 'LOW',
        severity: scan?.severity || 'HIGH',
        disparity: scan?.disparity_ratio || 'UNKNOWN',
        feature: scan?.affected_feature || sensitiveCol,
      });
      setVerdict(data);
    }).catch((error) => window.alert(error.message));
  };

  const runComparison = async () => {
    if (!compareFiles.fileA || !compareFiles.fileB) {
      window.alert('Choose two CSV files first.');
      return;
    }

    await withLoader('Comparing model fairness profiles', async () => {
      const form = new FormData();
      form.append('file1', compareFiles.fileA);
      form.append('file2', compareFiles.fileB);
      const response = await fetch(`${API}/compare-models`, { method: 'POST', body: form });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail || 'Comparison failed');
      }
      setCompareResult(data);
    }).catch((error) => window.alert(error.message));
  };

  const downloadPdf = async () => {
    if (!verdict) return;
    await withLoader('Generating PDF certificate', async () => {
      const response = await fetch(`${API}/export-pdf`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          decision_id: verdict.decision_id,
          verdict: verdict.verdict,
          integrity_hash: verdict.integrity_hash,
          report_lines: verdict.detailed_report,
          compliance: verdict.compliance,
        }),
      });
      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement('a');
      anchor.href = url;
      anchor.download = `VERDICT_Certificate_${verdict.decision_id}.pdf`;
      anchor.click();
      URL.revokeObjectURL(url);
    }).catch((error) => window.alert(error.message));
  };

  const resetSession = () => {
    setStep(1);
    setScan(null);
    setIntersections(null);
    setAttack(null);
    setFix(null);
    setDebias(null);
    setShieldResult(null);
    setVerdict(null);
    setUserChoice('');
    setFaceOffRevealed(false);
    setShowModal(false);
  };

  return (
    <>
      <nav className="nav">
        <span className="wordmark">VERD<span>I</span>CT</span>
        <div className="nav-steps">
          <button className={`step-btn ${step === 1 ? 'active' : ''} ${info ? 'done' : ''}`} onClick={() => setStep(1)}>01 Ingestion</button>
          <button className={`step-btn ${step === 2 ? 'active' : ''} ${scan ? 'done' : ''}`} onClick={() => info && setStep(2)}>02 The Mirror</button>
          <button className={`step-btn ${step === 3 ? 'active' : ''} ${attack ? 'done' : ''}`} onClick={() => attack && setStep(3)}>03 The Face-Off</button>
          <button className={`step-btn ${step === 4 ? 'active' : ''} ${fix ? 'done' : ''}`} onClick={() => fix && setStep(4)}>04 The Fix</button>
          <button className={`step-btn ${step === 5 ? 'active' : ''} ${shieldResult || compareResult ? 'done' : ''}`} onClick={() => fix && setStep(5)}>05 Live Shield</button>
          <button className={`step-btn ${step === 6 ? 'active' : ''}`} onClick={() => (fix || verdict) && setStep(6)}>06 Verdict</button>
        </div>
      </nav>

      {loading && (
        <div className="screen visible fade-up">
          <div className="loading">{loadingText} <span className="dots"><span>.</span><span>.</span><span>.</span></span></div>
        </div>
      )}

      {step === 1 && !loading && (
        <div className="screen fade-up">
          <p className="section-label">Screen 01 - Data Ingestion</p>
          <h1 className="display-title">What&apos;s hiding<br />in the data?</h1>
          <p className="subtitle">Upload a decision dataset, preview the first ten rows, and confirm which attributes should anchor the fairness scan.</p>

          <div className="card">
            <div className="screen-header-row">
              <span className="section-label" style={{ marginBottom: 0 }}>Target System</span>
              <StatusTag value={info ? 'Dataset Loaded' : 'Pending Data'} />
            </div>
            <p className="muted-copy">{info ? `${info.filename} loaded with ${info.row_count} records.` : 'Awaiting CSV payload for automated vulnerability scanning.'}</p>
          </div>

          <div className="file-upload-wrapper" onClick={() => fileRef.current?.click()}>[ Click to select CSV payload ]</div>
          <input ref={fileRef} className="file-input" type="file" accept=".csv" onChange={handleUpload} />

          {info && (
            <>
              <div className="screen-grid" style={{ marginTop: '2rem' }}>
                <div className="card">
                  <label className="shield-label">Target Outcome Column</label>
                  <select value={outcomeCol} onChange={(event) => setOutcomeCol(event.target.value)}>
                    {info.columns.map((column) => <option key={column} value={column}>{column}</option>)}
                  </select>
                </div>
                <div className="card">
                  <label className="shield-label">Sensitive Attribute</label>
                  <select value={sensitiveCol} onChange={(event) => setSensitiveCol(event.target.value)}>
                    {info.columns.map((column) => <option key={column} value={column}>{column}</option>)}
                  </select>
                </div>
              </div>

              <div style={{ marginTop: '1.5rem' }}>
                <p className="section-label">Data Preview - First {info.preview.length} Rows</p>
                <div className="table-wrapper">
                  <table className="data-table">
                    <thead>
                      <tr>{info.columns.map((column) => <th key={column}>{column}</th>)}</tr>
                    </thead>
                    <tbody>
                      {info.preview.map((row, index) => (
                        <tr key={index}>
                          {info.columns.map((column) => <td key={column}>{String(row[column] ?? '')}</td>)}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>

              <button className="btn-primary" onClick={runScan}>Initiate Vulnerability Scan -&gt;</button>
            </>
          )}
        </div>
      )}

      {step === 2 && !loading && scan && (
        <div className="screen fade-up">
          <p className="section-label">Screen 02 - The Mirror</p>
          <h1 className="display-title">System Bias Profile</h1>
          <p className="subtitle">Single-axis disparity, proxy leakage, significance testing, and compound intersection checks are all mapped together here.</p>

          <div className="card hero-metrics">
            <ArcGauge valueStr={scan.severity} label="SYSTEM RISK" />
            <div className="metric-column">
              <div className="metric-chip">
                <span className="section-label" style={{ marginBottom: '.4rem' }}>Tracked Outcome</span>
                <strong>{scan.tracked_outcome}</strong>
              </div>
              <div className="metric-chip">
                <span className="section-label" style={{ marginBottom: '.4rem' }}>Disparity Ratio</span>
                <strong>{scan.disparity_ratio}</strong>
              </div>
              <div className="metric-chip">
                <span className="section-label" style={{ marginBottom: '.4rem' }}>Significance</span>
                <StatusTag value={scan.statistical_significance ? 'SIGNIFICANT' : 'LOW SIGNAL'} />
              </div>
            </div>
          </div>

          <div className="screen-grid">
            <div className="card">
              <p className="section-label">Bias Fingerprint</p>
              {scan.bias_fingerprint.map((item) => (
                <div key={item.feature} className="stat-row">
                  <span>{item.feature}</span>
                  <StatusTag value={item.influence} />
                </div>
              ))}
            </div>
            <div className="card">
              <p className="section-label">Proxy Correlation Matrix</p>
              {scan.proxy_correlations.length > 0 ? scan.proxy_correlations.map((item) => (
                <div key={item.feature} className="stat-row">
                  <span>{item.feature}</span>
                  <span className="mono-small">{item.proxy_correlation} / {item.outcome_correlation}</span>
                </div>
              )) : <p className="muted-copy">No proxy features crossed the alert threshold.</p>}
            </div>
          </div>

          <div className="finding-bar">
            <p className="section-label">Key Finding</p>
            <p>{scan.key_finding}</p>
          </div>

          <div className="card">
            <p className="section-label">Group Breakdown</p>
            <div className="stats-table">
              {scan.group_stats.map((group) => (
                <div key={group.group} className="stat-row">
                  <span>{group.group}</span>
                  <span>{group.rate}% on {group.tracked_outcome}</span>
                </div>
              ))}
            </div>
          </div>

          {intersections?.intersections?.length > 0 && (
            <div style={{ marginTop: '1.5rem' }}>
              <p className="section-label" style={{ marginBottom: '1rem' }}>Intersectionality Analysis</p>
              <div className="intersection-grid">
                {intersections.intersections.map((item) => (
                  <div key={item.axes} className="intersection-card">
                    <div className="screen-header-row">
                      <strong>{item.axes}</strong>
                      <StatusTag value={item.severity} />
                    </div>
                    <p className="muted-copy">{item.disparity} disparity on {item.tracked_outcome}</p>
                    {item.all_groups.slice(0, 4).map((group) => (
                      <div key={group.intersection} className="stat-row">
                        <span>{group.intersection}</span>
                        <span>{group.rate}%</span>
                      </div>
                    ))}
                  </div>
                ))}
              </div>
            </div>
          )}

          <div style={{ marginTop: '2rem' }}>
            <button className="btn-primary" onClick={runAttack}>Show Me a Real Case -&gt;</button>
          </div>
        </div>
      )}

      {step === 3 && !loading && attack && (
        <div className="screen fade-up">
          <p className="section-label">Screen 03 - The Face-Off</p>
          <h1 className="display-title">{faceOffRevealed ? 'The system lied.' : 'Which profile is higher risk?'}</h1>
          <p className="subtitle">{faceOffRevealed ? 'The original model split these people by demographic patterning.' : 'Choose the profile you believe the model should score as higher risk before the outcome is revealed.'}</p>

          <div className="profile-grid">
            {['original', 'modified'].map((side, index) => {
              const record = attack[side];
              const label = index === 0 ? 'A' : 'B';
              return (
                <div key={side} className={`profile-card ${faceOffRevealed ? (index === 0 ? 'high' : 'low') : ''}`}>
                  <span className="profile-letter">{label}</span>
                  <div style={{ marginBottom: '1rem' }}>
                    <h2 style={{ fontFamily: "'Playfair Display', serif", fontSize: '1.4rem' }}>Profile {label}</h2>
                    <span className="tag tag-amber">{sensitiveCol}: {record[sensitiveCol]}</span>
                  </div>
                  {Object.entries(record).filter(([key]) => key !== outcomeCol && !key.toLowerCase().includes('unnamed')).slice(0, 5).map(([key, value]) => (
                    <div key={key} className="profile-field">
                      <p className="pf-label">{key}</p>
                      <p className="pf-val">{String(value)}</p>
                    </div>
                  ))}
                  {faceOffRevealed && (
                    <div style={{ marginTop: '1rem', paddingTop: '1rem', borderTop: '1px solid var(--border)' }}>
                      <span className={`score-badge ${index === 0 ? 'score-high' : 'score-low'}`}>{record[outcomeCol]}</span>
                    </div>
                  )}
                </div>
              );
            })}
          </div>

          {!faceOffRevealed ? (
            <div className="choice-btns">
              <button className={`choice-btn ${userChoice === 'A' ? 'selected' : ''}`} onClick={() => { setUserChoice('A'); setShowModal(true); }}>Profile A is Higher Risk</button>
              <button className={`choice-btn ${userChoice === 'B' ? 'selected' : ''}`} onClick={() => { setUserChoice('B'); setShowModal(true); }}>Profile B is Higher Risk</button>
            </div>
          ) : (
            <div className="finding-bar">
              <p className="section-label">Exploit Assessment</p>
              <p>{userChoice === 'A' ? attack.review_a : attack.review_b}</p>
            </div>
          )}

          {faceOffRevealed && <button className="btn-primary" onClick={runFix}>See the Fair Decision -&gt;</button>}
        </div>
      )}

      {step === 4 && !loading && fix && (
        <div className="screen fade-up">
          <p className="section-label">Screen 04 - The Fix</p>
          <h1 className="display-title">What objective assessment<br />actually looks like.</h1>
          <p className="subtitle">The Decision Firewall removes protected signals and their nearest proxies, then reruns the assessment using only legitimate evidence.</p>

          <div className="compare-grid">
            <div className="compare-col biased">
              <StatusTag value="Biased System" />
              <p className="big-num" style={{ color: 'var(--red)', marginTop: '1rem' }}>{fix.biased_score}</p>
              <p className="section-label" style={{ marginTop: '1.5rem' }}>Removed Signals</p>
              {fix.factors_removed.map((item) => <div key={item} className="factor-row"><span style={{ color: 'var(--red)' }}>x</span><span>{item}</span></div>)}
            </div>
            <div className="compare-col fair">
              <StatusTag value="Audited Decision" />
              <p className="big-num" style={{ color: 'var(--green)', marginTop: '1rem' }}>{fix.fair_score}</p>
              <p className="section-label" style={{ marginTop: '1.5rem' }}>Retained Signals</p>
              {fix.factors_used.map((item) => <div key={item} className="factor-row"><span style={{ color: 'var(--green)' }}>+</span><span>{item}</span></div>)}
            </div>
          </div>

          <div className="card">
            <p className="section-label">What Changed</p>
            <p className="muted-copy">{fix.what_changed}</p>
          </div>

          {!debias ? (
            <div style={{ display: 'flex', gap: '1rem', flexWrap: 'wrap' }}>
              <button className="btn-primary" onClick={runDebias}>Run Debiasing Engine -&gt;</button>
              <button className="btn-secondary" onClick={() => setStep(5)}>Skip to Live Shield -&gt;</button>
            </div>
          ) : (
            <div className="card debias-card">
              <div className="summary-grid">
                <div><p className="big-num" style={{ color: 'var(--amber)' }}>{debias.records_affected}</p><p className="muted-copy">Records changed</p></div>
                <div><p className="big-num" style={{ color: 'var(--green)' }}>{debias.disparity_after}x</p><p className="muted-copy">Post-fix disparity</p></div>
                <div><p className="big-num">{debias.counterfactual_flips}</p><p className="muted-copy">Counterfactual flips</p></div>
              </div>
              <p className="section-label">Proxy Features Removed</p>
              <div className="factor-list">
                {debias.proxy_features_removed.map((item) => <span key={item} className="tag tag-amber">{item}</span>)}
              </div>
              <p className="muted-copy" style={{ marginTop: '1rem' }}>{debias.summary}</p>
              <div className="compare-grid" style={{ marginTop: '1.5rem' }}>
                <div className="compare-col biased">
                  <p className="section-label">Before</p>
                  {debias.before.map((item) => <div key={item.group} className="stat-row"><span>{item.group}</span><span>{item.rate}%</span></div>)}
                </div>
                <div className="compare-col fair">
                  <p className="section-label">After</p>
                  {debias.after.map((item) => <div key={item.group} className="stat-row"><span>{item.group}</span><span>{item.rate}%</span></div>)}
                </div>
              </div>
              <button className="btn-primary" onClick={() => setStep(5)}>Proceed to Live Shield -&gt;</button>
            </div>
          )}
        </div>
      )}

      {step === 5 && !loading && (
        <div className="screen fade-up">
          <p className="section-label">Screen 05 - Live Shield</p>
          <h1 className="display-title">Real-time<br />decision firewall.</h1>
          <p className="subtitle">Test a live record through the fair middleware, then compare two model outputs side by side before you seal the verdict.</p>

          <div className="card">
            <p className="section-label">Live Predict-Fair API</p>
            <div className="shield-form">
              {info?.columns.filter((column) => column !== outcomeCol && !column.toLowerCase().includes('unnamed')).slice(0, 8).map((column) => (
                <div key={column}>
                  <p className="shield-label">{column}</p>
                  <input className="shield-input" value={shieldInputs[column] || ''} onChange={(event) => setShieldInputs((prev) => ({ ...prev, [column]: event.target.value }))} />
                </div>
              ))}
            </div>
            <button className="btn-primary" onClick={runShield}>Run Counterfactual Analysis -&gt;</button>
          </div>

          {shieldResult && (
            <div className="card">
              <div className="compare-grid">
                <div className="compare-col biased">
                  <p className="section-label">Biased Prediction</p>
                  <p className="big-num" style={{ color: 'var(--red)' }}>{shieldResult.biased_prediction}</p>
                </div>
                <div className="compare-col fair">
                  <p className="section-label">Fair Prediction</p>
                  <p className="big-num" style={{ color: 'var(--green)' }}>{shieldResult.fair_prediction}</p>
                </div>
              </div>
              <p className="muted-copy">{shieldResult.explanation}</p>
              <div className="counterfactual-grid">
                {Object.entries(shieldResult.counterfactual_outcome || {}).map(([group, outcome]) => (
                  <div key={group} className="counterfactual-card">
                    <span className="section-label" style={{ marginBottom: '.25rem' }}>{group}</span>
                    <strong>{String(outcome)}</strong>
                  </div>
                ))}
              </div>
            </div>
          )}

          <div className="card" style={{ marginTop: '2rem' }}>
            <p className="section-label">Multi-Model Comparison</p>
            <p className="muted-copy">Upload two model-output CSVs and compare which one produces the lower demographic disparity.</p>
            <div className="screen-grid" style={{ marginTop: '1rem' }}>
              <div>
                <p className="shield-label">Model A CSV</p>
                <input type="file" accept=".csv" onChange={(event) => setCompareFiles((prev) => ({ ...prev, fileA: event.target.files?.[0] || null }))} />
              </div>
              <div>
                <p className="shield-label">Model B CSV</p>
                <input type="file" accept=".csv" onChange={(event) => setCompareFiles((prev) => ({ ...prev, fileB: event.target.files?.[0] || null }))} />
              </div>
            </div>
            <button className="btn-secondary" style={{ marginTop: '1rem' }} onClick={runComparison}>Compare Models -&gt;</button>

            {compareResult && (
              <div className="comparison-results">
                <div className="compare-grid">
                  {['model_a', 'model_b'].map((key) => (
                    <div key={key} className={`compare-card ${key === 'model_a' ? 'left' : 'right'}`}>
                      <p className="section-label">{compareResult[key].label}</p>
                      <p className="big-num" style={{ color: compareResult[key].risk_level === 'LOW' ? 'var(--green)' : compareResult[key].risk_level === 'MEDIUM' ? 'var(--amber)' : 'var(--red)' }}>{compareResult[key].disparity_ratio}x</p>
                      <StatusTag value={compareResult[key].risk_level} />
                      {compareResult[key].group_breakdown.slice(0, 3).map((group) => (
                        <div key={group.group} className="stat-row">
                          <span>{group.group}</span>
                          <span>{group.rate}%</span>
                        </div>
                      ))}
                    </div>
                  ))}
                </div>
                <div className="finding-bar green">
                  <p className="section-label">Recommendation</p>
                  <p>{compareResult.recommendation}</p>
                </div>
              </div>
            )}
          </div>

          <button className="btn-primary" onClick={() => setStep(6)}>Proceed to Final Verdict -&gt;</button>
        </div>
      )}

      {step === 6 && !loading && (
        <div className="screen fade-up">
          <p className="section-label">Screen 06 - Verdict</p>
          <h1 className="display-title">Audit Sealed.</h1>
          <p className="subtitle">The final certificate packages the fairness call, regulatory mapping, integrity hash, and recent audit history.</p>

          {!verdict ? (
            <button className="btn-primary" onClick={runVerdict}>Generate Comprehensive Audit Report -&gt;</button>
          ) : (
            <>
              <div className={`verdict-box ${verdict.verdict === 'FAIR' ? 'FAIR' : 'NOT_FAIR'}`}>
                <div style={{ fontSize: '1rem', letterSpacing: '2px', marginBottom: '1rem' }}>FINAL DETERMINATION</div>
                <div className="big-num" style={{ color: verdict.verdict === 'FAIR' ? 'var(--green)' : 'var(--red)' }}>{verdict.verdict}</div>
                <div style={{ marginTop: '.75rem' }}><StatusTag value={verdict.verdict} /></div>
              </div>

              <div className="hash-block">
                <p className="section-label">Integrity Hash</p>
                <code>{verdict.integrity_hash}</code>
              </div>

              <div className="compliance-grid">
                {verdict.compliance.map((item) => (
                  <div key={item.regulation} className={`compliance-card ${item.status.includes('COMPLIANT') && !item.status.includes('NON') ? 'pass' : 'fail'}`}>
                    <p style={{ fontSize: '.75rem', fontWeight: 600, marginBottom: '.4rem' }}>{item.regulation}</p>
                    <StatusTag value={item.status} />
                    <p className="muted-copy" style={{ marginTop: '.8rem' }}>{item.detail}</p>
                  </div>
                ))}
              </div>

              <div className="card">
                <p className="section-label">Detailed Audit Log</p>
                <div className="log-block">
                  {verdict.detailed_report.map((line, index) => <div key={index}>{line}</div>)}
                </div>
              </div>

              <div className="card">
                <p className="section-label">Recent Audit Trail</p>
                {verdict.audit_history.map((item) => (
                  <div key={item.decision_id} className="stat-row">
                    <span>{item.decision_id}</span>
                    <span>{item.verdict} / {item.severity}</span>
                  </div>
                ))}
              </div>

              <div style={{ display: 'flex', gap: '1rem', flexWrap: 'wrap' }}>
                <button className="btn-primary pdf-btn" onClick={downloadPdf}>Download Audit Certificate</button>
                <button className="btn-secondary" onClick={resetSession}>Start New Session</button>
              </div>
            </>
          )}
        </div>
      )}

      {showModal && (
        <div className="modal-backdrop">
          <div className="modal-card fade-up">
            <p className="section-label">Decision Interceptor</p>
            <h3 style={{ fontFamily: "'Playfair Display', serif", fontSize: '1.8rem', marginBottom: '.8rem' }}>Bias Risk Detected</h3>
            <p className="muted-copy" style={{ marginBottom: '1.5rem' }}>This choice aligns with a historically skewed decision pattern. Continue with the original outcome or re-evaluate the case through the fair model.</p>
            <div style={{ display: 'flex', gap: '.75rem', flexDirection: 'column' }}>
              <button className="btn-secondary" onClick={() => { setShowModal(false); setFaceOffRevealed(true); }}>Proceed with Original Decision</button>
              <button className="btn-red" onClick={() => { setShowModal(false); setFaceOffRevealed(true); runFix(); }}>Re-evaluate with Fair Model</button>
            </div>
          </div>
        </div>
      )}

      <footer>
        <span>VERDICT v2 · AI Decision Security Platform</span>
        <span>Decision Firewall Engine</span>
      </footer>
    </>
  );
}

export default AppShell;
