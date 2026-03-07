#!/bin/sh

# 1. Limpa apenas o Python e túneis antigos do Serveo (sem matar o SEU SSH)
# Procuramos por processos ssh que contenham 'serveo.net' no comando                                      pkill -9 python3
ps aux | grep 'serveo.net' | grep -v grep | awk '{print $1}' | xargs kill -9 2>/dev/null

echo "🚀 Iniciando o Gotapp Server..."
# 2. Inicia o Uvicorn em segundo plano
python3 -m uvicorn main:app --host 0.0.0.0 --port 8000 &
PYTHON_PID=$!

sleep 2

echo "🔗 Abrindo túnel no Serveo (redegotapp.serveo.net)..."
echo "⚠️  Se aparecer um link do 'console.serveo.net', clique e faça login!"

# 3. Inicia o SSH do túnel usando a chave RSA
# -i ~/.ssh/id_rsa envia sua identidade para o subdomínio fixo
ssh -i ~/.ssh/id_rsa -o ServerAliveInterval=60 -o StrictHostKeyChecking=accept-new -R gotapp:80:localhost:8000 serveo.net

# 4. Ao encerrar o túnel (Ctrl+C), limpa o Python
kill $PYTHON_PID
echo "🛑 Servidor encerrado."

