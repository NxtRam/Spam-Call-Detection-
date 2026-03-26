document.addEventListener('DOMContentLoaded', () => {
    const socket = io();

    // Elements
    const screenCall = document.getElementById('callScreen');
    const screenDash = document.getElementById('dashboardScreen');
    const screenBlock = document.getElementById('blockOverlay');
    
    const btnAccept = document.getElementById('btnAccept');
    const btnDecline = document.getElementById('btnDecline');
    const btnEndCall = document.getElementById('btnEndCall');
    const btnReturn = document.getElementById('btnReturn');
    
    const transcriptBox = document.getElementById('transcriptBox');
    const threatProgress = document.getElementById('threatProgress');
    const threatPct = document.getElementById('threatPct');
    const riskBadge = document.getElementById('riskBadge');
    
    const totalFrags = document.getElementById('totalFrags');
    const spamFrags = document.getElementById('spamFrags');
    
    const callTimer = document.getElementById('callTimer');
    const blockReason = document.getElementById('blockReason');
    const reportFile = document.getElementById('reportFile');

    let timerInterval = null;
    let seconds = 0;
    
    // Canvas visualizer
    const canvas = document.getElementById('visualizer');
    const ctx = canvas.getContext('2d');
    canvas.width = window.innerWidth;
    canvas.height = 100;

    let currentEnergy = 0;

    // --- UI state management ---
    
    btnAccept.addEventListener('click', () => {
        screenCall.classList.add('hidden');
        screenDash.classList.remove('hidden');
        
        // Start Socket
        socket.emit('start_call');
        startTimer();
    });

    btnDecline.addEventListener('click', () => {
        alert("Call declined.");
    });

    btnEndCall.addEventListener('click', () => {
        socket.emit('end_call');
        resetHome();
    });

    btnReturn.addEventListener('click', () => {
        resetHome();
    });

    function startTimer() {
        seconds = 0;
        callTimer.innerText = "00:00";
        if (timerInterval) clearInterval(timerInterval);
        timerInterval = setInterval(() => {
            seconds++;
            const m = Math.floor(seconds / 60).toString().padStart(2, '0');
            const s = (seconds % 60).toString().padStart(2, '0');
            callTimer.innerText = `${m}:${s}`;
        }, 1000);
    }

    function resetHome() {
        clearInterval(timerInterval);
        screenDash.classList.add('hidden');
        screenBlock.classList.add('hidden');
        screenCall.classList.remove('hidden');
        transcriptBox.innerHTML = '';
        updateThreatLevel(0);
        updateRiskBadge('⚪ SAFE', 0);
        totalFrags.innerText = '0';
        spamFrags.innerText = '0';
        
        ctx.clearRect(0, 0, canvas.width, canvas.height); // clear audio
    }

    // --- Socket Event Listeners ---
    
    let partialEl = null;

    socket.on('transcript_partial', (data) => {
        if (!partialEl) {
            partialEl = document.createElement('div');
            partialEl.className = 't-partial';
            transcriptBox.appendChild(partialEl);
        }
        partialEl.innerText = data.text + '...';
        transcriptBox.scrollTop = transcriptBox.scrollHeight;
    });

    socket.on('transcript_final', (data) => {
        if (partialEl) {
            partialEl.remove();
            partialEl = null;
        }
        
        const el = document.createElement('div');
        
        // Formatting based on confidence
        if (data.label === 'Spam') {
            if (data.confidence >= 0.7) {
                el.className = 't-spam-high';
                el.innerHTML = `⚠️ [${data.risk_level}] ${data.text}`;
            } else {
                el.className = 't-spam-low';
                el.innerHTML = `⚠️ [${data.risk_level}] ${data.text}`;
            }
        } else {
            el.className = 't-ham';
            el.innerText = data.text;
        }

        if (data.is_robo) {
            el.innerHTML += `<span class="t-robo">ROBOCALL</span>`;
        }

        transcriptBox.appendChild(el);
        transcriptBox.scrollTop = transcriptBox.scrollHeight;
    });

    socket.on('call_stats', (data) => {
        totalFrags.innerText = data.total_fragments;
        spamFrags.innerText = data.spam_fragments;
        
        updateThreatLevel(data.spam_percentage);
        
        let riskText = '⚪ SAFE';
        if (data.spam_percentage >= 70) riskText = '⛔ CRITICAL';
        else if (data.spam_percentage >= 40) riskText = '🟠 HIGH';
        else if (data.spam_percentage >= 15) riskText = '🟡 MEDIUM';
        
        updateRiskBadge(riskText, data.spam_percentage);
    });

    socket.on('scam_alert', (data) => {
        // We could play a beep or flash the screen
    });

    socket.on('auto_block', (data) => {
        clearInterval(timerInterval);
        blockReason.innerText = data.reason;
        screenDash.classList.add('hidden');
        screenBlock.classList.remove('hidden');
    });

    socket.on('call_dropped', (data) => {
        reportFile.innerText = "Evidence saved: " + data.report;
    });

    socket.on('audio_level', (data) => {
        // Map RMS energy (0 - 500) to visualizer scale
        currentEnergy = Math.min(data.energy, 300) / 300;
    });

    // --- Helper Functions ---
    
    function updateThreatLevel(pct) {
        threatPct.innerText = pct.toFixed(0) + '%';
        const angle = (pct / 100) * 360;
        
        let color = '#00ff88'; // green
        if (pct >= 40) color = '#ffcc00'; // yellow
        if (pct >= 70) color = '#ff3366'; // red
        
        threatProgress.style.background = `conic-gradient(${color} ${angle}deg, #222 ${angle}deg)`;
        threatPct.style.color = color;
    }

    function updateRiskBadge(text, pct) {
        riskBadge.innerText = text;
        
        let bg = 'rgba(0, 255, 136, 0.1)';
        let border = '#00ff88';
        let box = 'rgba(0, 255, 136, 0.2)';
        
        if (pct >= 40) { bg = 'rgba(255, 204, 0, 0.1)'; border = '#ffcc00'; box = 'rgba(255, 204, 0, 0.2)'; }
        if (pct >= 70) { bg = 'rgba(255, 51, 102, 0.1)'; border = '#ff3366'; box = 'rgba(255, 51, 102, 0.2)'; }

        riskBadge.style.background = bg;
        riskBadge.style.borderColor = border;
        riskBadge.style.color = border;
        riskBadge.style.boxShadow = `0 0 15px ${box}`;
    }

    // --- Visualizer Animation Loop ---
    function animateVisualizer() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        // Draw some bars centered in the canvas depending on energy
        const numBars = 40;
        const barWidth = canvas.width / numBars;
        
        ctx.fillStyle = 'rgba(0, 240, 255, 0.5)';
        
        for (let i = 0; i < numBars; i++) {
            // make a sine wave that peaks in the middle and depends on energy
            const distFromCenter = Math.abs(i - numBars/2) / (numBars/2);
            const peakHeight = (1 - distFromCenter) * (currentEnergy * 100);
            
            // add some random flutter
            const h = peakHeight * (0.8 + Math.random() * 0.4);
            
            ctx.fillRect(i * barWidth + 2, canvas.height - h, barWidth - 4, h);
        }
        
        // Slowly decay energy
        currentEnergy *= 0.9;
        
        requestAnimationFrame(animateVisualizer);
    }
    
    animateVisualizer();

    // Resize canvas
    window.addEventListener('resize', () => {
        canvas.width = window.innerWidth;
    });
});
