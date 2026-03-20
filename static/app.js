const socket = io();

// UI Elements
const lobbyScreen = document.getElementById('lobby-screen');
const gameScreen = document.getElementById('game-screen');
const gameOverScreen = document.getElementById('game-over-screen');

socket.on('connect', () => { console.log('Connected'); });
socket.on('disconnect', () => { console.log('Disconnected'); });

let myIdx = 0;
let aiIdx = 1;

// Initialization
let stats = JSON.parse(localStorage.getItem('coupStats')) || {wins: 0, losses: 0};
let savedName = localStorage.getItem('coupPlayerName') || 'Human';
document.getElementById('player-name').value = savedName;

function updateStatsDisplay() {
    const msg = `Wins: ${stats.wins} | Losses: ${stats.losses}`;
    const statsDiv = document.getElementById('lobby-stats');
    if (statsDiv) statsDiv.innerText = msg;
    const topStats = document.getElementById('top-stats');
    if (topStats) topStats.innerText = msg;
}
updateStatsDisplay();

document.getElementById('clear-score-btn')?.addEventListener('click', () => {
    stats = {wins: 0, losses: 0};
    localStorage.setItem('coupStats', JSON.stringify(stats));
    updateStatsDisplay();
});

// Add logic for Play Again button
document.getElementById('rematch-btn').addEventListener('click', () => {
    gameOverScreen.classList.add('hidden');
    gameOverScreen.style.display = 'none';
    
    document.getElementById('my-cards').innerHTML = '';
    document.getElementById('ai-cards').innerHTML = '';
    document.getElementById('action-feed').innerHTML = '';
    document.getElementById('ai-reveal').innerHTML = '';

    const name = localStorage.getItem('coupPlayerName') || 'Human';
    socket.emit('start_game', { player_name: name });
});

document.getElementById('start-btn').addEventListener('click', () => {
    const name = document.getElementById('player-name').value || 'Human';
    localStorage.setItem('coupPlayerName', name);
    socket.emit('start_game', { player_name: name });
    lobbyScreen.classList.remove('active');
    gameScreen.classList.add('active');
});

socket.on('game_started', (data) => {
    myIdx = data.player_idx;
    aiIdx = data.ai_idx;
});

function getCardImageUrl(cardName) {
    if (!cardName) return 'images/card_back.png?v=2';
    return `images/${cardName.toLowerCase()}.png?v=2`;
}

function renderView(view) {
    // 1. My Area
    document.getElementById('my-name').innerText = view.name;
    document.getElementById('my-coins').innerText = view.my_coins;

    const myCards = document.getElementById('my-cards');
    myCards.innerHTML = '';
    
    // Helper to add click-to-zoom for any card
    const applyZoom = (div, imgUrl) => {
        div.onclick = function() {
            document.getElementById('modal-img').src = imgUrl;
            document.getElementById('card-modal').classList.remove('hidden');
        };
    };

    view.my_cards.forEach((cardName, idx) => {
        const div = document.createElement('div');
        div.className = 'card';
        const imgUrl = getCardImageUrl(cardName);
        div.style.backgroundImage = `url(${imgUrl})`;
        applyZoom(div, imgUrl);
        myCards.appendChild(div);
    });

    view.my_revealed.forEach((cardName) => {
        const div = document.createElement('div');
        div.className = 'card revealed';
        const imgUrl = getCardImageUrl(cardName);
        div.style.backgroundImage = `url(${imgUrl})`;
        applyZoom(div, imgUrl);
        myCards.appendChild(div);
    });

    // 2. Opponent Area
    const opp = view.opponents[0];
    if (opp) {
        document.getElementById('ai-name').innerText = opp.name;
        document.getElementById('ai-coins').innerText = opp.coins;
        document.getElementById('ai-influence').innerText = opp.influence_count;

        const aiCards = document.getElementById('ai-cards');
        aiCards.innerHTML = '';
        for (let i = 0; i < opp.influence_count; i++) {
            const div = document.createElement('div');
            div.className = 'card hidden';
            aiCards.appendChild(div);
        }
        opp.revealed.forEach((cardName) => {
            const div = document.createElement('div');
            div.className = 'card revealed';
            const imgUrl = getCardImageUrl(cardName);
            div.style.backgroundImage = `url(${imgUrl})`;
            
            // Add zoom to revealed AI cards
            div.onclick = function() {
                document.getElementById('modal-img').src = imgUrl;
                document.getElementById('card-modal').classList.remove('hidden');
            };
            
            aiCards.appendChild(div);
        });
    }

    // 3. Action Feed and Deck Info
    document.getElementById('deck-size').innerText = view.court_deck_size;

    const feed = document.getElementById('action-feed');
    feed.innerHTML = '';
    const reversedHistory = [...view.action_history].reverse();
    reversedHistory.forEach((h) => {
        const div = document.createElement('div');
        div.className = 'feed-item';
        // h.player is an index. we try to match it.
        const actor = h.player === view.player_id ? view.name : view.opponents[0].name;
        const target = h.target !== null ? (h.target === view.player_id ? view.name : view.opponents[0].name) : '';
        let msg = `${actor} did ${h.action}`;
        if (target) msg += ` on ${target}`;
        if (h.was_blocked) msg += ` (BLOCKED)`;
        if (h.was_challenged) msg += ` (CHALLENGED - ${h.challenge_won ? 'won' : 'lost'})`;
        div.innerText = msg;
        feed.appendChild(div);
    });
}

function showPrompt(title, optionsHTML) {
    document.getElementById('prompt-title').innerText = title;
    document.getElementById('prompt-options').innerHTML = optionsHTML;

    const container = document.getElementById('prompt-container');
    container.classList.remove('prompt-hidden');
    container.classList.add('prompt-active');
}

function hidePrompt() {
    const container = document.getElementById('prompt-container');
    container.classList.remove('prompt-active');
    container.classList.add('prompt-hidden');
}

window.submitPrompt = function (data) {
    socket.emit('player_action', data);
    hidePrompt();
};

const actionMap = {
    'choose_action': (payload) => {
        let html = '';
        payload.legal_actions.forEach((a, i) => {
            let label = a.action_type;
            if (a.target_idx !== null && label !== "Income" && label !== "Foreign Aid" && label !== "Tax" && label !== "Exchange") {
                label += ' (Target Opponent)';
            }
            html += `<button class="btn action-btn" onclick="submitPrompt({choice_index: ${i}})">${label}</button>`;
        });
        showPrompt("Choose an Action", html);
    },
    'choose_challenge': (payload) => {
        let html = `
            <button class="btn action-btn" onclick="submitPrompt({challenge: true});" style="background:var(--accent)">Challenge!</button>
            <button class="btn action-btn" onclick="submitPrompt({challenge: false});">Don't Challenge</button>
        `;
        showPrompt(`Challenge ${payload.claimed_card}?`, html);
    },
    'choose_counteraction': (payload) => {
        let html = `<button class="btn action-btn" onclick="submitPrompt({choice_index: -1});">Pass</button>`;
        payload.blocking_cards.forEach((c, i) => {
            html += `<button class="btn action-btn" onclick="submitPrompt({choice_index: ${i}})">Block with ${c}</button>`;
        });
        showPrompt(`Block ${payload.action_type}?`, html);
    },
    'choose_challenge_counter': (payload) => {
        let html = `
            <button class="btn action-btn" onclick="submitPrompt({challenge: true});" style="background:var(--accent)">Challenge Block!</button>
            <button class="btn action-btn" onclick="submitPrompt({challenge: false});">Pass</button>
        `;
        showPrompt(`Challenge Block (${payload.blocking_card})?`, html);
    },
    'choose_card_to_lose': (payload) => {
        let html = '';
        payload.view.my_cards.forEach((c, i) => {
            html += `<button class="btn action-btn" style="background:var(--accent)" onclick="submitPrompt({choice_index: ${i}})">Lose ${c}</button>`;
        });
        showPrompt("You must lose an influence!", html);
    },
    'choose_exchange_cards': (payload) => {
        window._exchangeSelection = [];
        window._exchangeTotal = payload.num_to_keep;
        let html = `<div id="exchange-grid" style="display:flex; justify-content:center; gap:10px; margin-bottom:15px;">`;
        payload.all_cards.forEach((c, i) => {
            html += `<div id="exc-${i}" class="card selectable" style="background-image:url(${getCardImageUrl(c)}); height:120px; width:85px;" onclick="toggleExchange(${i})"></div>`;
        });
        html += `</div><button id="confirm-exch-btn" class="btn primary-btn" disabled onclick="confirmExchange()">Confirm</button>`;
        showPrompt(`Keep ${payload.num_to_keep} cards`, html);
    }
};

window.toggleExchange = function (idx) {
    const el = document.getElementById('exc-' + idx);
    const selIdx = window._exchangeSelection.indexOf(idx);
    if (selIdx > -1) {
        window._exchangeSelection.splice(selIdx, 1);
        el.style.border = "2px solid rgba(255,255,255,0.1)";
        el.style.transform = "scale(1)";
    } else {
        if (window._exchangeSelection.length < window._exchangeTotal) {
            window._exchangeSelection.push(idx);
            el.style.border = "2px solid var(--primary)";
            el.style.transform = "scale(1.1)";
        }
    }
    document.getElementById('confirm-exch-btn').disabled = window._exchangeSelection.length !== window._exchangeTotal;
};

window.confirmExchange = function () {
    submitPrompt({ choice_indices: window._exchangeSelection });
};

socket.on('game_prompt', (payload) => {
    if (payload.view) renderView(payload.view);
    const handler = actionMap[payload.type];
    if (handler) {
        handler(payload);
    }
});

socket.on('game_over', (data) => {
    hidePrompt();
    gameScreen.classList.remove('active');
    gameOverScreen.classList.remove('hidden');
    gameOverScreen.style.display = 'flex'; // override hidden 

    let msg = "It's a Draw";
    if (data.winner !== 'Draw') {
        msg = `${data.winner} Wins!`;
        const myName = document.getElementById('player-name').value || 'Human';
        if (data.winner === myName) {
            stats.wins++;
        } else {
            stats.losses++;
        }
        localStorage.setItem('coupStats', JSON.stringify(stats));
        updateStatsDisplay();
    }
    document.getElementById('win-message').innerText = msg;

    if (data.ai_final_cards) {
        const rev = document.getElementById('ai-reveal');
        rev.innerHTML = '';
        data.ai_final_cards.forEach(c => {
            const div = document.createElement('div');
            div.className = 'card';
            const imgUrl = getCardImageUrl(c);
            div.style.backgroundImage = `url(${imgUrl})`;
            div.onclick = function() {
                document.getElementById('modal-img').src = imgUrl;
                document.getElementById('card-modal').classList.remove('hidden');
            };
            rev.appendChild(div);
        });
    }
});

socket.on('game_error', (data) => {
    alert("Error: " + data.error);
});
