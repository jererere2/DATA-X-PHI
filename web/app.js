document.addEventListener('DOMContentLoaded', () => {
    const API_BASE = 'http://127.0.0.1:3030';
    const historyLog = document.getElementById('historyLog');
    const userInput = document.getElementById('userInput');
    const sendBtn = document.getElementById('sendBtn');
    const clearBtn = document.getElementById('clearBtn');
    const refreshBtn = document.getElementById('refreshBtn');
    const thinkingIndicator = document.getElementById('thinkingIndicator');
    const statusText = document.getElementById('statusText');
    const chatContainer = document.getElementById('chatContainer');

    function scrollToBottom() {
        if(historyLog) {
            historyLog.scrollTop = historyLog.scrollHeight;
        }
    }

    async function appendMessageNode(role, text) {
        if(!historyLog) return;
        const msgDiv = document.createElement('div');
        msgDiv.classList.add('message', role);

        const badge = document.createElement('div');
        badge.classList.add('role-badge');
        badge.textContent = role === 'data' ? 'DATA' : 'AZ12';
        msgDiv.appendChild(badge);

        const content = document.createElement('div');
        content.classList.add('content');
        
        const lines = text.split('\n');
        for (const line of lines) {
            const p = document.createElement('p');
            p.textContent = line;
            content.appendChild(p);
        }
        msgDiv.appendChild(content);
        historyLog.appendChild(msgDiv);
        scrollToBottom();
        
        return new Promise(resolve => setTimeout(resolve, 50));
    }

    async function loadState() {
        if (statusText) statusText.textContent = 'Sincronizando estado...';
        try {
            const res = await fetch(`${API_BASE}/state`);
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            const data = await res.json();
            
            if(historyLog) historyLog.innerHTML = '';
            for (const msg of data.history) {
                await appendMessageNode(msg.role, msg.text);
            }
            if (statusText) statusText.textContent = 'Sincronizado con el núcleo';
        } catch (err) {
            console.error('Error cargando estado:', err);
            if (statusText) statusText.textContent = 'Error de conexión';
        }
    }

    async function transmitMessage() {
        if(!userInput) return;
        const text = userInput.value.trim();
        if (!text) return;

        userInput.value = '';
        userInput.disabled = true;
        if(sendBtn) sendBtn.disabled = true;

        await appendMessageNode('user', text);
        if (thinkingIndicator) thinkingIndicator.style.display = 'block';
        if (statusText) statusText.textContent = 'DATA procesando...';
        scrollToBottom();

        try {
            const res = await fetch(`${API_BASE}/chat`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message: text })
            });

            if (!res.ok) throw new Error(`HTTP error ${res.status}`);
            const data = await res.json();

            await appendMessageNode('data', data.reply);
            if (statusText) statusText.textContent = 'Sinapsis consolidada.';
        } catch (err) {
            console.error('Fallo en la transmisión:', err);
            await appendMessageNode('data', `[ALERTA DE SISTEMA] No se pudo conectar con el núcleo en ${API_BASE}. Asegúrate de que el servidor Rust esté ejecutándose.`);
            if (statusText) statusText.textContent = 'Error de comunicación.';
        } finally {
            if (thinkingIndicator) thinkingIndicator.style.display = 'none';
            userInput.disabled = false;
            if(sendBtn) sendBtn.disabled = false;
            userInput.focus();
            scrollToBottom();
        }
    }

    async function clearMemory() {
        if (!confirm('¿Confirmas reiniciar el historial?')) return;
        if (statusText) statusText.textContent = 'Purgando registros...';
        try {
            const res = await fetch(`${API_BASE}/clear`, { method: 'POST' });
            if (!res.ok) throw new Error(`HTTP error ${res.status}`);
            if(historyLog) historyLog.innerHTML = '';
            if (statusText) statusText.textContent = 'Memoria borrada.';
        } catch (err) {
            console.error('Error al limpiar:', err);
            if (statusText) statusText.textContent = 'Fallo al purgar.';
        }
    }

    if(sendBtn) sendBtn.addEventListener('click', transmitMessage);
    if(userInput) {
        userInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                transmitMessage();
            }
        });
    }

    if (clearBtn) clearBtn.addEventListener('click', clearMemory);
    if (refreshBtn) refreshBtn.addEventListener('click', loadState);

    // Initial load
    loadState();
});
