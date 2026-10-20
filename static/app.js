/**
 * pattupetti - Modern Cute Music Intelligence Frontend
 * Features:
 *  - Waveform Canvas with Dual-Tone Bars (Dark Charcoal Played / Silver Unplayed)
 *  - Iconic Pentagonal Marker Needle Scrubber with Drag & Click Scrubbing
 *  - Live Mic Audio Capture with Real-Time Audio Visualizer Animation
 *  - File Dropzone & 1-Click FMA Benchmark Preset Chips
 *  - Genre Intelligence Telemetry & Smooth Animated Probability Bars
 */

// --------------------------------------------------------------------------
// 1. DOM Elements
// --------------------------------------------------------------------------

// Waveform & Scrubber Needle
const waveformTrack = document.getElementById('waveformTrack');
const waveformCanvas = document.getElementById('waveformCanvas');
const canvasCtx = waveformCanvas.getContext('2d');
const scrubberNeedle = document.getElementById('scrubberNeedle');
const needleHandle = document.getElementById('needleHandle');

const capsulePlayBtn = document.getElementById('capsulePlayBtn');
const playBtnIcon = document.getElementById('playBtnIcon');
const capsuleTrackTitle = document.getElementById('capsuleTrackTitle');
const capsuleTime = document.getElementById('capsuleTime');
const audioElement = document.getElementById('audioElement');

// Tabs & Navigation
const tabs = document.querySelectorAll('.tab');
const tabViews = document.querySelectorAll('.tab-view');

// Live Voice / Mic
const micBtn = document.getElementById('micBtn');
const micEmoji = document.getElementById('micEmoji');
const micHeading = document.getElementById('micHeading');
const micTip = document.getElementById('micTip');
const liveRecBadge = document.getElementById('liveRecBadge');
const recTimer = document.getElementById('recTimer');

// File Upload
const dropZone = document.getElementById('dropZone');
const audioFileInput = document.getElementById('audioFileInput');

// Preset Sample Chips
const sampleChipsContainer = document.getElementById('sampleChipsContainer');

// Processing Overlay
const processingCurtain = document.getElementById('processingCurtain');

// Genre Intelligence & Telemetry
const heroGenreName = document.getElementById('heroGenreName');
const confidencePill = document.getElementById('confidencePill');
const confidenceText = document.getElementById('confidenceText');
const valTempo = document.getElementById('valTempo');
const valCentroid = document.getElementById('valCentroid');
const valEnergy = document.getElementById('valEnergy');
const valZcr = document.getElementById('valZcr');
const probabilityBarsList = document.getElementById('probabilityBarsList');

// --------------------------------------------------------------------------
// 2. State & Audio WebAPI Engine
// --------------------------------------------------------------------------

const GENRE_EMOJIS = {
    'Pop': '💖',
    'Rock': '🎸',
    'Hip-Hop': '🎤',
    'Electronic': '🎹',
    'Instrumental': '🎷',
    'Folk': '🎻',
    'Experimental': '🥁',
    'International': '🌍'
};

const GENRE_COLORS = {
    'Pop': '#ff6384',
    'Rock': '#ff5252',
    'Hip-Hop': '#6366f1',
    'Electronic': '#8b5cf6',
    'Instrumental': '#eab308',
    'Folk': '#10b981',
    'Experimental': '#ec4899',
    'International': '#06b6d4'
};

let playbackAudioCtx = null;
let playbackAnalyser = null;
let playbackSourceNode = null;

let micAudioCtx = null;
let micAnalyser = null;
let micStream = null;
let micSourceNode = null;
let mediaRecorder = null;
let recordedChunks = [];

let isRecording = false;
let recordTimerInterval = null;
let recordSeconds = 0;

let currentProgress = 0.48; // Default initial needle scrubber position (~48% as in reference image)
let isDraggingScrubber = false;
let isAudioLoaded = false;
let sampleTracksList = [];

// Base waveform bar heights (68 bars representing natural song dynamics like in the reference)
const NUM_BARS = 72;
const baseWaveformHeights = [];
// Generate organic music waveform curve (peaks, valleys, rhythmic accents)
for (let i = 0; i < NUM_BARS; i++) {
    const norm = i / NUM_BARS;
    // Harmonic envelope with variation
    let h = Math.sin(norm * Math.PI) * 0.65;
    h += Math.sin(norm * Math.PI * 4.2) * 0.22;
    h += Math.cos(norm * Math.PI * 7.5) * 0.15;
    h += (Math.sin(i * 1.33) * 0.5 + 0.5) * 0.25;
    h = Math.max(0.12, Math.min(1.0, h));
    baseWaveformHeights.push(h);
}

// --------------------------------------------------------------------------
// 3. Waveform Rendering Engine & Dual-Tone Scrubber
// --------------------------------------------------------------------------

function ensurePlaybackContext() {
    if (!playbackAudioCtx) {
        const AudioContext = window.AudioContext || window.webkitAudioContext;
        playbackAudioCtx = new AudioContext();
        playbackAnalyser = playbackAudioCtx.createAnalyser();
        playbackAnalyser.fftSize = 256;
        playbackAnalyser.smoothingTimeConstant = 0.78;

        playbackSourceNode = playbackAudioCtx.createMediaElementSource(audioElement);
        playbackSourceNode.connect(playbackAnalyser);
        playbackAnalyser.connect(playbackAudioCtx.destination);
    }
    if (playbackAudioCtx.state === 'suspended') {
        playbackAudioCtx.resume();
    }
}

let animPhase = 0;

function drawWaveform() {
    requestAnimationFrame(drawWaveform);

    const width = waveformCanvas.width;
    const height = waveformCanvas.height;
    canvasCtx.clearRect(0, 0, width, height);

    // Audio frequency data when playing or recording
    let freqData = null;
    const isPlaying = !audioElement.paused && audioElement.currentTime > 0;

    if (isPlaying && playbackAnalyser) {
        freqData = new Uint8Array(playbackAnalyser.frequencyBinCount);
        playbackAnalyser.getByteFrequencyData(freqData);
    } else if (isRecording && micAnalyser) {
        freqData = new Uint8Array(micAnalyser.frequencyBinCount);
        micAnalyser.getByteFrequencyData(freqData);
    }

    // Geometry calculations
    const paddingX = 14;
    const availableWidth = width - (paddingX * 2);
    const barSpacing = availableWidth / NUM_BARS;
    const barWidth = Math.max(2.5, barSpacing * 0.42);
    const centerY = height / 2;
    const maxBarHalfHeight = (height - 14) / 2;

    const scrubberX = paddingX + (currentProgress * availableWidth);

    // Dynamic wave idle float
    animPhase += 0.035;

    for (let i = 0; i < NUM_BARS; i++) {
        const barX = paddingX + (i * barSpacing) + (barSpacing / 2);
        let barHeightRatio = baseWaveformHeights[i];

        if (freqData) {
            const freqIdx = Math.floor((i / NUM_BARS) * (freqData.length * 0.75));
            const binVal = freqData[freqIdx] || 0;
            const boost = (binVal / 255.0) * 0.75;
            barHeightRatio = Math.max(0.12, Math.min(1.0, barHeightRatio * 0.5 + boost));
        } else if (isPlaying) {
            barHeightRatio += Math.sin(animPhase + i * 0.3) * 0.12;
            barHeightRatio = Math.max(0.12, Math.min(1.0, barHeightRatio));
        }

        const barHalfHeight = Math.max(3, barHeightRatio * maxBarHalfHeight);

        // Color coding matching the user's reference image:
        // Left of needle scrubber = Solid Charcoal (#1c2024)
        // Right of needle scrubber = Soft Silvery-Lavender (#d2d7e3)
        if (barX <= scrubberX) {
            canvasCtx.fillStyle = '#1c2024';
        } else {
            canvasCtx.fillStyle = '#d2d7e3';
        }

        // Draw vertical pill bar with rounded caps
        const topY = centerY - barHalfHeight;
        const barTotalHeight = barHalfHeight * 2;
        const radius = barWidth / 2;

        canvasCtx.beginPath();
        if (canvasCtx.roundRect) {
            canvasCtx.roundRect(barX - radius, topY, barWidth, barTotalHeight, radius);
        } else {
            canvasCtx.rect(barX - radius, topY, barWidth, barTotalHeight);
        }
        canvasCtx.fill();
    }
}

// Start visualizer loop immediately
drawWaveform();

// --------------------------------------------------------------------------
// 4. Interactive Scrubber Needle Positioning & Scrubbing
// --------------------------------------------------------------------------

function updateNeedleUI(progressPct) {
    currentProgress = Math.max(0.0, Math.min(1.0, progressPct));
    // Position scrubber needle horizontally
    // Need needle center to align with currentProgress
    const trackWidth = waveformTrack.clientWidth;
    const pixelX = currentProgress * trackWidth;
    scrubberNeedle.style.left = `calc(${currentProgress * 100}% - 11px)`;
}

// Initialize needle at default ~48% matching reference photo
updateNeedleUI(0.48);

// Time and progress tracking for audio playback
audioElement.addEventListener('timeupdate', () => {
    if (!isDraggingScrubber && audioElement.duration) {
        const progress = audioElement.currentTime / audioElement.duration;
        updateNeedleUI(progress);
        const curSec = Math.floor(audioElement.currentTime);
        const totalSec = Math.floor(audioElement.duration);
        capsuleTime.textContent = `${curSec}s`;
    }
});

audioElement.addEventListener('ended', () => {
    setPlaybackState(false);
    updateNeedleUI(0);
    capsuleTime.textContent = '0s';
});

// Scrubbing via click or drag across the waveform track
function seekToEvent(e) {
    const rect = waveformTrack.getBoundingClientRect();
    const clickX = e.clientX - rect.left;
    const ratio = Math.max(0, Math.min(1, clickX / rect.width));
    updateNeedleUI(ratio);

    if (audioElement.duration) {
        audioElement.currentTime = ratio * audioElement.duration;
        capsuleTime.textContent = `${Math.floor(audioElement.currentTime)}s`;
    } else {
        capsuleTime.textContent = `${Math.round(ratio * 28)}s`;
    }
}

waveformTrack.addEventListener('click', (e) => {
    seekToEvent(e);
});

scrubberNeedle.addEventListener('mousedown', (e) => {
    isDraggingScrubber = true;
    e.stopPropagation();
});

window.addEventListener('mousemove', (e) => {
    if (isDraggingScrubber) {
        seekToEvent(e);
    }
});

window.addEventListener('mouseup', () => {
    if (isDraggingScrubber) {
        isDraggingScrubber = false;
    }
});

// Touch support for mobile/tablets
scrubberNeedle.addEventListener('touchstart', (e) => {
    isDraggingScrubber = true;
}, { passive: true });

window.addEventListener('touchmove', (e) => {
    if (isDraggingScrubber && e.touches[0]) {
        const touch = e.touches[0];
        const rect = waveformTrack.getBoundingClientRect();
        const touchX = touch.clientX - rect.left;
        const ratio = Math.max(0, Math.min(1, touchX / rect.width));
        updateNeedleUI(ratio);
        if (audioElement.duration) {
            audioElement.currentTime = ratio * audioElement.duration;
            capsuleTime.textContent = `${Math.floor(audioElement.currentTime)}s`;
        }
    }
}, { passive: true });

window.addEventListener('touchend', () => {
    isDraggingScrubber = false;
});

// --------------------------------------------------------------------------
// 5. Play / Pause Capsule Button
// --------------------------------------------------------------------------

function setPlaybackState(isPlaying) {
    if (isPlaying) {
        // Switch icon to reference image's `▌▌` minimalist pause bars
        playBtnIcon.textContent = '▌▌';
        playBtnIcon.style.fontSize = '0.95rem';
        capsulePlayBtn.title = 'Pause Audio';
    } else {
        playBtnIcon.textContent = '▶';
        playBtnIcon.style.fontSize = '1.15rem';
        capsulePlayBtn.title = 'Play Audio';
    }
}

capsulePlayBtn.addEventListener('click', () => {
    if (!audioElement.src) {
        // If no audio loaded yet, load the first sample track seamlessly
        if (sampleTracksList.length > 0) {
            selectSampleTrack(0);
            return;
        }
    }

    ensurePlaybackContext();
    if (audioElement.paused) {
        audioElement.play().then(() => {
            setPlaybackState(true);
        }).catch(err => {
            console.warn('Playback resume needed:', err);
        });
    } else {
        audioElement.pause();
        setPlaybackState(false);
    }
});

// --------------------------------------------------------------------------
// 6. Cute Segmented Tabs Switcher
// --------------------------------------------------------------------------

tabs.forEach(tab => {
    tab.addEventListener('click', () => {
        tabs.forEach(t => t.classList.remove('active'));
        tabViews.forEach(v => v.classList.remove('active'));

        tab.classList.add('active');
        const tabKey = tab.dataset.tab;
        const view = document.getElementById('view' + tabKey.charAt(0).toUpperCase() + tabKey.slice(1));
        if (view) view.classList.add('active');
    });
});

// --------------------------------------------------------------------------
// 7. Live Singing & Microphone Streaming Capture (Lossless PCM WAV Engine)
// --------------------------------------------------------------------------

let pcmChunks = [];
let micScriptNode = null;
let recordingSampleRate = 22050;

micBtn.addEventListener('click', async () => {
    if (!isRecording) {
        // ── START LIVE MICROPHONE RECORDING ──
        try {
            micStream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    echoCancellation: false,
                    noiseSuppression: false,
                    autoGainControl: true
                }
            });

            // Set up AudioContext for pure PCM capture & real-time visualizer
            const AudioContext = window.AudioContext || window.webkitAudioContext;
            micAudioCtx = new AudioContext();
            if (micAudioCtx.state === 'suspended') {
                await micAudioCtx.resume();
            }
            recordingSampleRate = micAudioCtx.sampleRate || 44100;

            micAnalyser = micAudioCtx.createAnalyser();
            micAnalyser.fftSize = 256;
            micAnalyser.smoothingTimeConstant = 0.72;

            micSourceNode = micAudioCtx.createMediaStreamSource(micStream);
            micSourceNode.connect(micAnalyser);

            // ScriptProcessorNode to reliably capture pure PCM Float32 audio samples
            pcmChunks = [];
            const bufferSize = 4096;
            micScriptNode = micAudioCtx.createScriptProcessor(bufferSize, 1, 1);
            micScriptNode.onaudioprocess = (e) => {
                if (!isRecording) return;
                const channelData = e.inputBuffer.getChannelData(0);
                // Clone samples
                pcmChunks.push(new Float32Array(channelData));
            };

            // Connect script processor to stream and a dummy silent destination to keep it alive
            micSourceNode.connect(micScriptNode);
            micScriptNode.connect(micAudioCtx.destination);

            // Update UI State
            isRecording = true;
            micBtn.classList.add('recording');
            micEmoji.textContent = '■';
            micHeading.textContent = 'Recording...';
            micTip.textContent = 'Sing or hum. Click again to stop and analyze.';
            liveRecBadge.style.display = 'flex';

            recordSeconds = 0;
            recTimer.textContent = '00:00';
            recordTimerInterval = setInterval(() => {
                recordSeconds++;
                const mins = String(Math.floor(recordSeconds / 60)).padStart(2, '0');
                const secs = String(recordSeconds % 60).padStart(2, '0');
                recTimer.textContent = `${mins}:${secs}`;
                capsuleTime.textContent = `${recordSeconds}s`;
                // Move needle during recording
                updateNeedleUI(Math.min(1.0, recordSeconds / 30));
            }, 1000);

        } catch (err) {
            console.error('Microphone access error:', err);
            if (err.name === 'NotAllowedError') {
                showPredictionError('Microphone permission was denied. Please allow microphone access in your browser.');
            } else {
                showPredictionError('Could not start microphone: ' + err.message);
            }
        }
    } else {
        // ── STOP LIVE MICROPHONE RECORDING ──
        isRecording = false;
        clearInterval(recordTimerInterval);

        micBtn.classList.remove('recording');
        micEmoji.textContent = '●';
        micHeading.textContent = 'Analyzing...';
        liveRecBadge.style.display = 'none';

        // Disconnect and release audio nodes
        if (micScriptNode) {
            try { micScriptNode.disconnect(); } catch (e) {}
            micScriptNode = null;
        }
        if (micSourceNode) {
            try { micSourceNode.disconnect(); } catch (e) {}
            micSourceNode = null;
        }
        if (micStream) {
            micStream.getTracks().forEach(t => t.stop());
            micStream = null;
        }
        if (micAudioCtx) {
            try { micAudioCtx.close(); } catch (e) {}
            micAudioCtx = null;
        }

        // Process collected PCM samples
        if (!pcmChunks || pcmChunks.length === 0 || recordSeconds < 1) {
            resetMicState();
            showPredictionError('Recording was too short. Please sing or speak for at least 2-3 seconds.');
            return;
        }

        // Merge all Float32Array chunks
        let totalSamples = 0;
        for (let i = 0; i < pcmChunks.length; i++) {
            totalSamples += pcmChunks[i].length;
        }
        const mergedSamples = new Float32Array(totalSamples);
        let offset = 0;
        let maxAmp = 0;
        for (let i = 0; i < pcmChunks.length; i++) {
            mergedSamples.set(pcmChunks[i], offset);
            offset += pcmChunks[i].length;
        }

        // Check audio amplitude level
        for (let i = 0; i < mergedSamples.length; i++) {
            const abs = Math.abs(mergedSamples[i]);
            if (abs > maxAmp) maxAmp = abs;
        }

        console.log(`[Mic PCM] Total samples: ${totalSamples}, Duration: ${(totalSamples / recordingSampleRate).toFixed(2)}s, Peak Amp: ${maxAmp}`);

        if (maxAmp < 0.002) {
            resetMicState();
            showPredictionError('Microphone signal was completely silent. Please ensure your microphone is unmuted and singing aloud.');
            return;
        }

        // Auto-normalize if volume was soft
        if (maxAmp < 0.4) {
            const gain = Math.min(0.65 / maxAmp, 8.0);
            for (let i = 0; i < mergedSamples.length; i++) {
                mergedSamples[i] *= gain;
            }
        }

        // Encode to standard 16-bit PCM WAV
        const wavBlob = encodeWAV(mergedSamples, recordingSampleRate);
        console.log(`[Mic WAV] Generated WAV blob size: ${wavBlob.size} bytes`);

        // Set as current audio stream in capsule
        const audioUrl = URL.createObjectURL(wavBlob);
        audioElement.src = audioUrl;
        capsuleTrackTitle.textContent = `Live Recording (${recordSeconds}s)`;
        capsuleTime.textContent = `${recordSeconds}s`;
        updateNeedleUI(0);
        setPlaybackState(false);

        // Send to prediction API
        showCurtain(true);
        const formData = new FormData();
        formData.append('audio', wavBlob, 'live_singing.wav');
        await triggerPredictionAPI(formData);

        resetMicState();
    }
});

/**
 * Standard 16-bit PCM WAV File Encoder
 */
function encodeWAV(samples, sampleRate) {
    const buffer = new ArrayBuffer(44 + samples.length * 2);
    const view = new DataView(buffer);

    // RIFF identifier
    writeString(view, 0, 'RIFF');
    // File length
    view.setUint32(4, 36 + samples.length * 2, true);
    // RIFF type
    writeString(view, 8, 'WAVE');
    // format chunk identifier
    writeString(view, 12, 'fmt ');
    // format chunk length
    view.setUint32(16, 16, true);
    // sample format (1 = raw PCM)
    view.setUint16(20, 1, true);
    // channel count (1 = mono)
    view.setUint16(22, 1, true);
    // sample rate
    view.setUint32(24, sampleRate, true);
    // byte rate (sampleRate * 1 channel * 2 bytes)
    view.setUint32(28, sampleRate * 2, true);
    // block align (1 channel * 2 bytes)
    view.setUint16(32, 2, true);
    // bits per sample (16 bits)
    view.setUint16(34, 16, true);
    // data chunk identifier
    writeString(view, 36, 'data');
    // data chunk length
    view.setUint32(40, samples.length * 2, true);

    // Write 16-bit PCM samples
    let index = 44;
    for (let i = 0; i < samples.length; i++, index += 2) {
        const s = Math.max(-1, Math.min(1, samples[i]));
        view.setInt16(index, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
    }

    return new Blob([view], { type: 'audio/wav' });
}

function writeString(view, offset, string) {
    for (let i = 0; i < string.length; i++) {
        view.setUint8(offset + i, string.charCodeAt(i));
    }
}

function resetMicState() {
    micHeading.textContent = 'Ready';
    micTip.textContent = 'Tap to start recording';
    micEmoji.textContent = '●';
    micBtn.classList.remove('recording');
    liveRecBadge.style.display = 'none';
}

// --------------------------------------------------------------------------
// 8. File Dropzone & Upload Handler
// --------------------------------------------------------------------------

dropZone.addEventListener('click', () => audioFileInput.click());

dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('dragover');
});

dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('dragover');
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    if (e.dataTransfer.files.length > 0) {
        handleAudioUpload(e.dataTransfer.files[0]);
    }
});

audioFileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleAudioUpload(e.target.files[0]);
    }
});

function handleAudioUpload(file) {
    const rawName = file.name;
    capsuleTrackTitle.textContent = rawName;
    updateNeedleUI(0);

    const blobUrl = URL.createObjectURL(file);
    audioElement.src = blobUrl;
    audioElement.onloadedmetadata = () => {
        capsuleTime.textContent = `${Math.round(audioElement.duration)}s`;
    };
    setPlaybackState(false);

    const formData = new FormData();
    formData.append('audio', file);
    triggerPredictionAPI(formData);
}

// --------------------------------------------------------------------------
// 9. Benchmark Preset Chips Handler
// --------------------------------------------------------------------------

async function initPresetSamples() {
    try {
        const res = await fetch('/api/samples');
        const data = await res.json();
        if (data.status === 'success') {
            sampleTracksList = data.samples;
            renderPresetChips();
        }
    } catch (e) {
        console.error('Failed to load sample tracks:', e);
    }
}

function renderPresetChips() {
    sampleChipsContainer.innerHTML = '';
    sampleTracksList.forEach((item, index) => {
        const chip = document.createElement('div');
        chip.className = 'bench-chip-item';
        chip.innerHTML = `
            <span class="chip-emoji">${GENRE_EMOJIS[item.genre] || '🎵'}</span>
            <div class="chip-details">
                <span class="chip-genre-tag">${item.genre}</span>
                <span class="chip-name" title="${item.title}">${item.title}</span>
            </div>
        `;
        chip.addEventListener('click', () => selectSampleTrack(index));
        sampleChipsContainer.appendChild(chip);
    });
}

function selectSampleTrack(index) {
    const track = sampleTracksList[index];
    if (!track) return;

    capsuleTrackTitle.textContent = `${track.title}`;
    updateNeedleUI(0);

    audioElement.src = `/api/sample-audio/${track.id}`;
    audioElement.onloadedmetadata = () => {
        capsuleTime.textContent = `${Math.round(audioElement.duration || 30)}s`;
    };

    ensurePlaybackContext();
    audioElement.play().then(() => {
        setPlaybackState(true);
    }).catch(e => console.log('Autoplay policy caught:', e));

    fetchSamplePrediction(track.id);
}

async function fetchSamplePrediction(trackId) {
    showCurtain(true);
    try {
        const res = await fetch(`/api/predict-sample/${trackId}`);
        const data = await res.json();
        if (data.status === 'success') {
            renderResults(data);
        } else {
            alert('Prediction error: ' + data.message);
        }
    } catch (err) {
        alert('Server connection error: ' + err.message);
    } finally {
        showCurtain(false);
    }
}

function showPredictionError(msg) {
    heroGenreName.textContent = 'Error';
    confidenceText.textContent = msg;
    confidencePill.style.display = 'flex';
    showCurtain(false);
}

async function triggerPredictionAPI(formData) {
    showCurtain(true);
    try {
        const res = await fetch('/api/predict', {
            method: 'POST',
            body: formData
        });
        const data = await res.json();
        if (data.status === 'success') {
            renderResults(data);
        } else {
            showPredictionError(data.message || 'Could not classify audio');
        }
    } catch (err) {
        showPredictionError('Server connection failed: ' + err.message);
    } finally {
        showCurtain(false);
    }
}

function renderResults(data) {
    const genre = data.predicted_genre;
    const topProb = data.probabilities[genre] || 0;

    // 1. Genre result
    heroGenreName.textContent = genre;
    confidenceText.textContent = `${topProb.toFixed(1)}%`;
    confidencePill.style.display = 'flex';

    // 2. Extracted Acoustic Descriptors (Telemetry)
    if (data.metrics) {
        valTempo.textContent = `${data.metrics.tempo || '--'} BPM`;
        valCentroid.textContent = `${data.metrics.brightness_hz ? Math.round(data.metrics.brightness_hz) : '--'} Hz`;
        valEnergy.textContent = `${data.metrics.energy_rms ? data.metrics.energy_rms.toFixed(3) : '--'}`;
        valZcr.textContent = `${data.metrics.zcr ? data.metrics.zcr.toFixed(3) : '--'}`;
    }

    // 3. Cute Probability Distribution Bars
    probabilityBarsList.innerHTML = '';
    const sortedProbs = Object.entries(data.probabilities).sort((a, b) => b[1] - a[1]);

    sortedProbs.forEach(([g, prob], rank) => {
        const isTop = rank === 0;
        const row = document.createElement('div');
        row.className = 'prob-row';
        row.innerHTML = `
            <div class="prob-meta">
                <span class="prob-genre-name">${g}</span>
                <span class="prob-percent">${prob.toFixed(1)}%</span>
            </div>
            <div class="prob-track">
                <div class="prob-fill ${isTop ? 'top-rank' : ''}" style="width: 0%"></div>
            </div>
        `;
        probabilityBarsList.appendChild(row);

        // Animate width smoothly
        setTimeout(() => {
            const fill = row.querySelector('.prob-fill');
            if (fill) fill.style.width = `${prob}%`;
        }, 40);
    });
}

function showCurtain(isVisible) {
    processingCurtain.style.display = isVisible ? 'flex' : 'none';
}

// --------------------------------------------------------------------------
// 11. Initial Startup Initialization
// --------------------------------------------------------------------------

initPresetSamples();

// Responsive canvas sizing
function resizeCanvas() {
    if (waveformTrack) {
        waveformCanvas.width = waveformTrack.clientWidth * window.devicePixelRatio || 900;
        waveformCanvas.height = waveformTrack.clientHeight * window.devicePixelRatio || 76;
    }
}
window.addEventListener('resize', resizeCanvas);
resizeCanvas();
