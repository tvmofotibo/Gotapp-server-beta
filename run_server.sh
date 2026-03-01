#!/bin/sh

echo "🚀 Iniciando o Gotapp Server..."
# Inicia o Uvicorn em segundo plano (&)
# Redireciona a saída para um log para não sujar a tela
python3 -m uvicorn main:app --host 0.0.0.0 --port 8000 > server.log 2>&1 &

# Guarda o ID do processo do python para encerrar depois se precisar
PYTHON_PID=$!

echo "🔗 Abrindo túnel no Serveo (redegotapp.serveo.net)..."
echo "Pressione CTRL+C para parar tudo."

# Inicia o SSH do Serveo
# -o ServerAliveInterval=60 mantém a conexão ativa
ssh -o ServerAliveInterval=60 -R redegotapp:80:localhost:8000 serveo.net

# Quando você fechar o SSH, ele mata o processo do Python também
kill $PYTHON_PID
echo "🛑 Servidor encerrado."

