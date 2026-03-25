import { useState, useEffect, useRef, useCallback } from 'react';
import './index.css';

async function apiFetch(path, opts = {}) {
  const res = await fetch(path, opts);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

export default function App() {
  const [gameState, setGameState] = useState(null);
  const [logs, setLogs] = useState(['> Global Strategy Link Online...']);
  const [showSettings, setShowSettings] = useState(false);
  const [settingsForm, setSettingsForm] = useState({ mode: 'PvE' });
  const [targetSelectMode, setTargetSelectMode] = useState(null);
  const [isResolving, setIsResolving] = useState(false);
  const [hoverNation, setHoverNation] = useState(null);

  const logEndRef = useRef(null);

  const addLogs = useCallback((newLogs) => {
    if (!newLogs || newLogs.length === 0) return;
    setLogs(prev => [...prev.slice(-200), ...newLogs.map(l => `> ${l}`)]);
  }, []);

  const addLog = useCallback((msg) => {
    setLogs(prev => [...prev.slice(-200), `> ${msg}`]);
  }, []);

  useEffect(() => {
    logEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [logs]);

  const fetchState = useCallback(async () => {
    try {
      const data = await apiFetch('/api/state');
      setGameState(data);
      return data;
    } catch {
      return null;
    }
  }, []);

  useEffect(() => { fetchState(); }, [fetchState]);

  const queueCommand = useCallback(async (cmdStr) => {
    try {
      const data = await apiFetch('/api/command', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ command: `QUEUE ${cmdStr}` })
      });
      if (!data.success) addLog(data.message);
      setTargetSelectMode(null);
      await fetchState();
    } catch (e) { addLog(`Error: ${e}`); }
  }, [addLog, fetchState]);

  const cancelLast = useCallback(async () => {
    try {
      await apiFetch('/api/command', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ command: 'CANCEL_LAST' })
      });
      await fetchState();
    } catch (e) { addLog(`Error: ${e}`); }
  }, [fetchState, addLog]);

  const submitTurn = useCallback(async () => {
    setIsResolving(true);
    addLog("Submitting turn to server...");
    try {
      const data = await apiFetch('/api/command', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ command: 'SUBMIT_TURN' })
      });
      if (data.logs) addLogs(data.logs);
      await fetchState();
    } catch (e) {
      addLog(`Error resolving turn: ${e}`);
    } finally {
      setIsResolving(false);
    }
  }, [addLogs, addLog, fetchState]);

  useEffect(() => {
    if (!gameState || gameState.winner !== null) return;
    // Only auto-play if all 4 factions are AI (EvE mode)
    const isEve = gameState.ai_players?.length === 4;
    if (isEve && !isResolving) {
      const timer = setTimeout(() => {
        submitTurn();
      }, 5000);
      return () => clearTimeout(timer);
    }
  }, [gameState?.turn, gameState?.winner, gameState?.ai_players, isResolving, submitTurn]);

  const applySettingsAndReset = useCallback(async () => {
    setShowSettings(false);
    try {
      const data = await apiFetch('/api/reset', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ mode: settingsForm.mode })
      });
      addLog(data.message);
      setTargetSelectMode(null);
      await fetchState();
    } catch (e) { addLog(`Reset error: ${e}`); }
  }, [addLog, fetchState, settingsForm]);

  if (!gameState) return <div className="loading-screen">Connecting to Strategy Link...</div>;

  const { turn, current_player, ai_players, winner, nations, diplomacy } = gameState;
  const isHumanTurn = !ai_players?.includes(current_player) && !winner;
  const myNation = nations[current_player];

  const handleTargetClick = (targetId) => {
    if (targetSelectMode) {
      queueCommand(`${targetSelectMode} ${targetId}`);
    }
  };

  const ACHIEVEMENTS_MAP = {
    fertile_lands: "Fertile Lands (+10% Gold)",
    trade_routes: "Trade Routes",
    mercantilism: "Mercantilism",
    treasury_reserve: "Treasury Reserve",
    economic_hegemony: "Economic Hegemony",
    levies: "Levies (+10% MP recovery)",
    standing_army: "Standing Army",
    war_college: "War College",
    total_mobilization: "Total Mobilization",
    marshal_legacy: "Marshal Legacy",
    workshops: "Workshops (+15% Prod)",
    factory_system: "Factory System",
    assembly_line: "Assembly Line",
    industrial_heartland: "Industrial Heartland",
    industrial_revolution: "Industrial Revolution (+Sci)",
    literacy: "Literacy",
    university_system: "University System",
    scientific_method: "Scientific Method (-15% Tech Cost)",
    enlightenment: "Enlightenment (+Sci per MP)",
    technological_supremacy: "Tech Supremacy",
    common_law: "Common Law",
    representation: "Representation",
    diplomatic_corps: "Diplomatic Corps",
    national_identity: "National Identity (+20% MP)",
    golden_age: "Golden Age (+25% Gold)"
  };

  return (
    <div className="dashboard">
      <div className="top-bar">
        <div className="turn-info">
          <h2>TURN {turn}</h2>
          <div className={`active-player-tag bg-${myNation.color.replace('#', '')}`}>
            {myNation.name}
          </div>
        </div>
        <div style={{ display: 'flex', gap: 10 }}>
          {(gameState?.ai_players?.length === 4) && <span style={{ color: '#94a3b8', display: 'flex', alignItems: 'center' }}>Auto-Play Active (5s/turn)</span>}
          <button className="settings-btn" onClick={() => setShowSettings(true)}>⚙ Game Settings</button>
        </div>
      </div>

      {showSettings && (
        <div className="modal-backdrop">
          <div className="modal">
            <h2>Game Setup</h2>
            <select value={settingsForm.mode} onChange={e => setSettingsForm(s => ({ ...s, mode: e.target.value }))}>
              <option value="PvE">1 Player vs CPU</option>
              <option value="PvP">Local Multiplayer (PvP)</option>
              <option value="EvE">CPU Auto-Play</option>
            </select>
            <div className="modal-actions">
              <button className="primary" onClick={applySettingsAndReset}>Restart Simulation</button>
              <button onClick={() => setShowSettings(false)}>Cancel</button>
            </div>
          </div>
        </div>
      )}

      <div className="main-stage">
        <div className="nations-list">
          {Object.values(nations).map(n => {
            const isMe = n.id === current_player;
            const myRelations = isMe ? 'SELF' : diplomacy[current_player]?.[n.id];
            const canTarget = targetSelectMode && !isMe && !n.is_defeated;

            const earned_achievements = Object.entries(n.achievements)
              .filter(([k, v]) => v === true && ACHIEVEMENTS_MAP[k])
              .map(([k]) => k);

            return (
              <div
                key={n.id}
                className={`nation-card ${isMe ? 'active-nation' : ''} ${n.is_defeated ? 'defeated' : ''} ${canTarget ? 'targetable' : ''}`}
                style={{ borderTop: `4px solid ${n.color}` }}
                onClick={() => canTarget ? handleTargetClick(n.id) : null}
              >
                <div className="nation-header">
                  <h3>{n.name} <span style={{fontSize: 10, color: '#94a3b8', fontWeight: 'normal'}}>({n.personality})</span></h3>
                  {!isMe && <span className={`diplomacy-badge dip-${myRelations.replace(/\s/g, '-').toLowerCase()}`}>{myRelations}</span>}
                  {n.is_defeated && <span className="diplomacy-badge dip-dead">DESTROYED</span>}
                </div>
                <div className="nation-stats">
                  <div title="Gold">💰 {n.gold} <small>(+{n.gold_yield})</small></div>
                  <div title="Manpower">🪖 {n.manpower} <small>(+{n.manpower_yield})</small></div>
                  <div title="Production">🏭 {n.production} <small>(+{n.production_yield})</small></div>
                  <div title="Military">⚔️ {n.military}</div>
                  {n.war_exhaustion > 0 && (
                    <div title="War Exhaustion" style={{color: '#ef4444'}}>💀 {n.war_exhaustion} <small>(-{n.war_exhaustion * 5}% Manpower)</small></div>
                  )}
                  <div title="Science">🔬 {n.science_yield}/t</div>
                  <div title="Civics">📜 {n.civic_yield}/t</div>
                </div>
                
                {(n.active_trade_agreements?.length > 0 || n.active_research_pacts?.length > 0) && (
                  <div style={{fontSize: 11, color: '#38bdf8', marginBottom: 8, fontWeight: 600}}>
                    🤝 {n.active_trade_agreements.length} Trade | 🔬 {n.active_research_pacts.length} Research Pacts
                  </div>
                )}

                {earned_achievements.length > 0 && (
                  <div className="nation-achievements">
                    <strong>🏆 Achievements:</strong>
                    <div className="ach-list">
                      {earned_achievements.map(k => <span key={k} className="ach-badge">✓ {ACHIEVEMENTS_MAP[k]}</span>)}
                    </div>
                  </div>
                )}
              </div>
            );
          })}
        </div>

        <div className="control-center">
          {winner !== null ? (
            <div className="victory-banner">
              <h1>{nations[winner].name} HAS ACHIEVED TOTAL VICTORY</h1>
              <button onClick={() => setShowSettings(true)}>Play Again</button>
            </div>
          ) : (gameState?.ai_players?.length === 4) ? (
            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', height: '100%', justifyContent: 'center' }}>
              <h2 style={{fontFamily: 'Cinzel, serif', color: '#fff', marginBottom: 40}}>Global Strategy Network</h2>
              <svg width="600" height="400" style={{ overflow: 'visible' }}>
                {Object.values(nations).map((n1, i, arr) => {
                  const getPos = (idx) => ({ x: 300 + 180 * Math.cos((idx / arr.length) * Math.PI * 2), y: 200 + 180 * Math.sin((idx / arr.length) * Math.PI * 2) });
                  const p1 = getPos(i);
                  return arr.map((n2, j) => {
                    if (n1.id >= n2.id) return null;
                    const p2 = getPos(j);
                    const relations = diplomacy[n1.id]?.[n2.id];
                    
                    let stroke = 'rgba(255,255,255,0.05)';
                    let dash = '0';
                    let width = 1;
                    
                    if (relations === 'WAR') { stroke = '#ef4444'; width = 4; }
                    else if (relations === 'ALLIED') { stroke = '#10b981'; width = 4; }
                    else if (n1.active_trade_agreements?.includes(n2.id)) { stroke = '#f59e0b'; dash = '5,5'; width = 2; }
                    else if (n1.active_research_pacts?.includes(n2.id)) { stroke = '#3b82f6'; dash = '2,4'; width = 2; }
                    
                    if (hoverNation !== null && hoverNation !== n1.id && hoverNation !== n2.id) return null;
                    
                    return (
                      <line key={`${n1.id}-${n2.id}`} x1={p1.x} y1={p1.y} x2={p2.x} y2={p2.y} stroke={stroke} strokeWidth={width} strokeDasharray={dash} />
                    );
                  })
                })}
                
                {Object.values(nations).map((n, i, arr) => {
                  const p = { x: 300 + 180 * Math.cos((i / arr.length) * Math.PI * 2), y: 200 + 180 * Math.sin((i / arr.length) * Math.PI * 2) };
                  const isHovered = hoverNation === n.id;
                  return (
                    <g key={n.id} transform={`translate(${p.x}, ${p.y})`} 
                       onMouseEnter={() => setHoverNation(n.id)}
                       onMouseLeave={() => setHoverNation(null)}
                       style={{cursor: 'pointer'}}>
                      <circle r={n.is_defeated ? "20" : "45"} fill={n.is_defeated ? '#333' : n.color} stroke={isHovered ? '#fff' : '#000'} strokeWidth="4" />
                      {!n.is_defeated && <text y="5" textAnchor="middle" fill="#fff" fontSize="14" fontWeight="bold">{n.name}</text>}
                      {!n.is_defeated && <text y="22" textAnchor="middle" fill="rgba(255,255,255,0.8)" fontSize="10">{n.personality}</text>}
                    </g>
                  );
                })}
              </svg>
              <div style={{marginTop: 60, display: 'flex', gap: 20, fontSize: 12, color: '#94a3b8'}}>
                <span style={{color: '#10b981'}}>━ Alliance</span>
                <span style={{color: '#ef4444'}}>━ War</span>
                <span style={{color: '#f59e0b'}}>╍ Trade Pact</span>
                <span style={{color: '#3b82f6'}}>╍ Research Pact</span>
              </div>
            </div>
          ) : (
            <div className="action-dashboard">
              <div className="action-header">
                <h2>Command Dashboard - {myNation.name}</h2>
                <div className="ap-tracker">
                  ACTION POINTS REMAINING: <span className="ap-count">{myNation.action_points}</span>
                </div>
              </div>

              <div className="queue-panel">
                <h3>Queued Actions pending resolution:</h3>
                {myNation.queued_actions.length === 0 ? <p className="empty-queue">No actions queued.</p> : (
                  <ol className="action-queue">
                    {myNation.queued_actions.map((act, i) => <li key={i}>{act}</li>)}
                  </ol>
                )}
                {myNation.queued_actions.length > 0 && (
                  <button className="cancel-btn" onClick={cancelLast}>Undo Last Action</button>
                )}
              </div>

              {targetSelectMode && (
                <div className="targeting-banner">
                  <p>Select a target nation for: <strong>{targetSelectMode}</strong></p>
                  <button onClick={() => setTargetSelectMode(null)}>Cancel</button>
                </div>
              )}

              <div className={`action-grid ${(!isHumanTurn || myNation.action_points <= 0 || targetSelectMode) ? 'disabled' : ''}`}>
                <div className="action-category">
                  <h3>Economy & Industry</h3>
                  <button onClick={() => queueCommand('HARVEST GOLD')}>Harvest Wealth</button>
                  <button onClick={() => queueCommand('HARVEST MANPOWER')}>Draft Conscripts</button>
                  <button onClick={() => queueCommand('HARVEST PRODUCTION')}>Mobilize Industry</button>
                </div>

                <div className="action-category">
                  <h3>Development (Science & Culture)</h3>
                  <div className="current-progress">
                    Tech: {myNation.current_tech || 'None'} {myNation.current_tech && `(${myNation.tech_progress} pts)`}<br />
                    Civic: {myNation.current_civic || 'None'} {myNation.current_civic && `(${myNation.civic_progress} pts)`}
                  </div>
                  <div style={{ display: 'flex', gap: '4px' }}>
                    <button onClick={() => queueCommand('HARVEST SCIENCE')} style={{ flex: 1 }}>Fund Acads (+Sci)</button>
                    <button onClick={() => queueCommand('HARVEST CIVICS')} style={{ flex: 1 }}>Fund Culture (+Civ)</button>
                  </div>
                  <hr style={{ borderColor: '#333', margin: '8px 0' }} />
                  <button onClick={() => queueCommand('RESEARCH Iron Working')}>Research Iron Working</button>
                  <button onClick={() => queueCommand('RESEARCH Gunpowder')}>Research Gunpowder</button>
                  <button onClick={() => queueCommand('RESEARCH Industrialization')}>Research Industrialization</button>
                  <button onClick={() => queueCommand('PURSUE_CIVIC Code of Laws')}>Civic: Code of Laws</button>
                </div>

                <div className="action-category">
                  <h3>Diplomacy & Military</h3>
                  <button onClick={() => setTargetSelectMode('PROPOSE_ALLIANCE')}>Propose Alliance</button>
                  <button onClick={() => setTargetSelectMode('ACCEPT_ALLIANCE')}>Accept Alliance</button>
                  <button onClick={() => setTargetSelectMode('DECLARE_WAR')} className="btn-warn">Declare War</button>
                  <button onClick={() => setTargetSelectMode('MILITARY_STRIKE')} className="btn-danger">Launch Military Strike</button>
                  <small style={{ display: 'block', marginTop: 10, color: '#94a3b8' }}>Strike consumes 100 🪖 and 50 🏭.</small>
                </div>
              </div>

              {isHumanTurn && (
                <button
                  className="end-turn-btn"
                  disabled={isResolving}
                  onClick={submitTurn}
                  style={{ background: myNation.action_points > 0 ? '#b45309' : '#16a34a' }}
                >
                  {isResolving ? "RESOLVING WORLD ACTIONS..." : (myNation.action_points > 0 ? "SKIP REMAINING ACTIONS & SUBMIT TURN" : "SUBMIT TURN")}
                </button>
              )}
            </div>
          )}
        </div>

        <div className="log-panel">
          <h3>Simulation Event Log</h3>
          <div className="log-container">
            <div className="log-scroll">
              {logs.map((L, i) => <div key={i}>{L}</div>)}
              <div ref={logEndRef} />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

