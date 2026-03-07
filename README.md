# 🚀 Gotapp Server — Beta

![Status](https://img.shields.io/badge/Status-Em%20Desenvolvimento-yellow?style=for-the-badge&logo=statuspage)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)

Pequena rede social desenvolvida com tecnologias modernas para comunicação rápida e simples entre usuários. 

Este servidor é o "coração" do ecossistema Gotapp, sendo responsável pela API, autenticação segura, gerenciamento de usuários e toda a lógica principal da aplicação.

---

## 🛠️ Tecnologias Utilizadas

O projeto utiliza o que há de mais eficiente no ecossistema Python atual:

* **[Python](https://www.python.org/):** Linguagem base pela sua versatilidade e robustez.
* **[FastAPI](https://fastapi.tiangolo.com/):** Framework web de alta performance para construção de APIs.
* **[JWT (JSON Web Tokens)](https://jwt.io/):** Implementação de autenticação segura e escalável (Stateless).

---

## 📌 Funcionalidades Principais (Backend)

* ✅ **Autenticação:** Sistema de Login/Registro com criptografia de senhas.
* ✅ **Gestão de Usuários:** CRUD completo de perfis.
* ✅ **Comunicação:** Endpoints otimizados para troca de mensagens rápidas.
* ✅ Suporte a notificações e integração com WebSockets.
* 🚧 **Em breve:** 

---

## 🚀 Como Executar o Projeto (Desenvolvimento)

### Pré-requisitos
* Python 3.9+
* Pip (Gerenciador de pacotes)

### Instalação

1. Clone o repositório:
```bash
git clone [https://github.com/seu-usuario/gotapp-server.git](https://github.com/seu-usuario/gotapp-server.git)
```
 * Crie um ambiente virtual e ative-o:
<!-- end list -->
```bash
python -m venv venv
```
# No Windows:
```bash
venv\Scripts\activate
```
# No Linux/Mac:
``` bash
source venv/bin/activate
```
 * Instale as dependências:
<!-- end list -->
``` bash
pip install -r requirements.txt
```
 * Inicie o servidor:
<!-- end list -->
``` bash
uvicorn main:app --reload
```

🚧 Status do Projeto
Atualmente o Gotapp Server encontra-se em versão Beta. Novas funcionalidades de segurança e otimização de banco de dados estão sendo implementadas.
⭐ Desenvolvido para ser simples, rápido e eficiente.
