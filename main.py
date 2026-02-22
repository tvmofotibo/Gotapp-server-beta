import os
import shutil
import asyncio
import logging
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Dict
from fastapi import FastAPI, File, UploadFile, Depends, HTTPException, status, WebSocket, WebSocketDisconnect
from fastapi.security import OAuth2PasswordBearer
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy import create_engine, Column, Integer, String, DateTime, ForeignKey, Text, Boolean, desc
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
from uuid import uuid4

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === Configurações ===
SECRET_KEY = "09d25e094faa6ca2556c818166b7a9563b93f7099f6f0f4caa6cf63b88e8d3e7"  # Troque em produção!
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt_sha256"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="users/login")

# === Banco de Dados SQLite ===
SQLALCHEMY_DATABASE_URL = "sqlite:///./got_app.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# === Modelos ===
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    name = Column(String, nullable=False)
    hashed_password = Column(String, nullable=False)
    bio = Column(Text, default="")
    avatar_url = Column(String, default="")
    created_at = Column(DateTime, default=datetime.utcnow)

    sent_messages = relationship("Message", foreign_keys="Message.sender_id", back_populates="sender")
    received_messages = relationship("Message", foreign_keys="Message.recipient_id", back_populates="recipient")

class Message(Base):
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True, index=True)
    sender_id = Column(Integer, ForeignKey("users.id"))
    recipient_id = Column(Integer, ForeignKey("users.id"))
    # NOVO: content pode ser nulo para áudio
    content = Column(Text, nullable=True)
    # NOVO: tipo da mensagem e URL do arquivo
    message_type = Column(String, default='text')  # 'text' ou 'audio'
    file_url = Column(String, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    is_read = Column(Boolean, default=False)

    sender = relationship("User", foreign_keys=[sender_id], back_populates="sent_messages")
    recipient = relationship("User", foreign_keys=[recipient_id], back_populates="received_messages")

# Recria as tabelas (em desenvolvimento). Em produção, use migrações.
Base.metadata.drop_all(bind=engine)
Base.metadata.create_all(bind=engine)

# === Schemas ===
class UserCreate(BaseModel):
    email: EmailStr
    name: str
    password: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class UserOut(BaseModel):
    id: int
    email: EmailStr
    name: str
    bio: str = ""
    avatar_url: str = ""

class UserProfileUpdate(BaseModel):
    name: Optional[str] = None
    bio: Optional[str] = None

# NOVO: Schema para envio de mensagem com tipo e URL
class MessageSend(BaseModel):
    recipient_id: int
    content: Optional[str] = None
    message_type: str = 'text'  # 'text' ou 'audio'
    file_url: Optional[str] = None

# NOVO: Schema de saída com os campos adicionais
class MessageOut(BaseModel):
    id: int
    sender_id: int
    recipient_id: int
    content: Optional[str]
    message_type: str
    file_url: Optional[str]
    timestamp: datetime
    is_read: bool

class ConversationOut(BaseModel):
    user_id: int
    name: str
    avatar_url: str = ""
    last_message: Optional[str] = None
    last_message_time: Optional[datetime] = None
    unread_count: int = 0

class Token(BaseModel):
    access_token: str
    token_type: str

# === Dependências ===
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def authenticate_user(db: Session, email: str, password: str):
    user = db.query(User).filter(User.email == email).first()
    if not user or not verify_password(password, user.hashed_password):
        return False
    return user

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: int = payload.get("sub")
        if user_id is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    user = db.query(User).filter(User.id == user_id).first()
    if user is None:
        raise credentials_exception
    return user

# === WebSocket Connection Manager com suporte a chamadas ===
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[int, Dict] = {}  # user_id -> {"ws": WebSocket, "name": str}
        self.call_sessions: Dict[int, int] = {}  # user_id -> other_user_id (chamada ativa)

    async def connect(self, websocket: WebSocket, user_id: int, user_name: str):
        await websocket.accept()
        self.active_connections[user_id] = {"ws": websocket, "name": user_name}
        logger.info(f"Usuário {user_id} ({user_name}) conectado via WebSocket")

    def disconnect(self, user_id: int):
        if user_id in self.active_connections:
            del self.active_connections[user_id]
            # Se estava em chamada, notificar o outro lado e limpar
            if user_id in self.call_sessions:
                other = self.call_sessions.pop(user_id)
                if other in self.call_sessions:
                    del self.call_sessions[other]
                # Se o outro ainda estiver online, enviar sinal de chamada encerrada
                if other in self.active_connections:
                    asyncio.create_task(self.send_personal_message({"type": "call_end", "from": user_id}, other))
            logger.info(f"Usuário {user_id} desconectado")

    async def send_personal_message(self, message: dict, user_id: int):
        if user_id in self.active_connections:
            try:
                await self.active_connections[user_id]["ws"].send_json(message)
                logger.info(f"Mensagem enviada para usuário {user_id}")
            except Exception as e:
                logger.error(f"Erro ao enviar mensagem para {user_id}: {e}")

    def get_user_name(self, user_id: int) -> Optional[str]:
        if user_id in self.active_connections:
            return self.active_connections[user_id]["name"]
        return None

manager = ConnectionManager()

# === App FastAPI ===
app = FastAPI(title="Got App API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cria pastas necessárias
os.makedirs("uploads", exist_ok=True)
os.makedirs("uploads/audio", exist_ok=True)  # NOVO: pasta para áudios
os.makedirs("static", exist_ok=True)

app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")
app.mount("/static", StaticFiles(directory="static"), name="static")

# === WebSocket endpoint com signaling para chamadas ===
@app.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: int, token: str, db: Session = Depends(get_db)):
    # Validar token
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        token_user_id = payload.get("sub")
        if token_user_id is None or int(token_user_id) != user_id:
            logger.warning(f"Token user {token_user_id} não corresponde ao user_id {user_id}")
            await websocket.close(code=1008)
            return
    except (JWTError, ValueError) as e:
        logger.error(f"Erro na validação do token: {e}")
        await websocket.close(code=1008)
        return

    # Buscar nome do usuário no banco (para exibir nas chamadas)
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        await websocket.close(code=1008)
        return

    await manager.connect(websocket, user_id, user.name)
    try:
        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type")
            target = data.get("target")  # ID do destinatário

            if msg_type == "call_offer":
                # Verificar se o destinatário está online e não ocupado
                if target not in manager.active_connections:
                    await manager.send_personal_message({"type": "call_offline"}, user_id)
                    continue
                if target in manager.call_sessions:
                    await manager.send_personal_message({"type": "call_busy"}, user_id)
                    continue

                # Registrar chamada
                manager.call_sessions[user_id] = target
                manager.call_sessions[target] = user_id

                # Encaminhar oferta com nome do remetente
                await manager.send_personal_message({
                    "type": "call_offer",
                    "from": user_id,
                    "fromName": user.name,
                    "offer": data["offer"],
                    "hasVideo": data.get("hasVideo", False),
                    "hasAudio": data.get("hasAudio", True)
                }, target)

            elif msg_type == "call_answer":
                if target in manager.active_connections:
                    await manager.send_personal_message({
                        "type": "call_answer",
                        "from": user_id,
                        "answer": data["answer"]
                    }, target)

            elif msg_type == "ice_candidate":
                if target in manager.active_connections:
                    await manager.send_personal_message({
                        "type": "ice_candidate",
                        "from": user_id,
                        "candidate": data["candidate"]
                    }, target)

            elif msg_type == "call_end":
                # Limpar sessão e avisar o outro participante
                if user_id in manager.call_sessions:
                    other = manager.call_sessions.pop(user_id)
                    if other in manager.call_sessions:
                        del manager.call_sessions[other]
                    if other in manager.active_connections:
                        await manager.send_personal_message({"type": "call_end", "from": user_id}, other)

            elif msg_type == "call_reject":
                # O destinatário recusou a chamada
                if target in manager.active_connections:
                    await manager.send_personal_message({"type": "call_reject", "from": user_id}, target)
                # Limpar registro se existir (quem chamou)
                if user_id in manager.call_sessions:
                    other = manager.call_sessions.pop(user_id)
                    if other in manager.call_sessions:
                        del manager.call_sessions[other]

            elif msg_type == "call_accept":
                # O destinatário aceitou (já temos a chamada registrada)
                if target in manager.active_connections:
                    await manager.send_personal_message({
                        "type": "call_accept",
                        "from": user_id,
                        "hasVideo": data.get("hasVideo", False)
                    }, target)

            else:
                logger.warning(f"Tipo de mensagem desconhecido: {msg_type}")

    except WebSocketDisconnect:
        manager.disconnect(user_id)
    except Exception as e:
        logger.error(f"Erro no WebSocket: {e}")
        manager.disconnect(user_id)

# === Endpoints HTTP ===
@app.post("/users/register", response_model=UserOut)
def register(user: UserCreate, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.email == user.email).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    hashed_password = get_password_hash(user.password)
    new_user = User(email=user.email, name=user.name, hashed_password=hashed_password)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return new_user

@app.post("/users/login", response_model=Token)
def login(user_data: UserLogin, db: Session = Depends(get_db)):
    user = authenticate_user(db, user_data.email, user_data.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": str(user.id)}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/users/me", response_model=UserOut)
def read_users_me(current_user: User = Depends(get_current_user)):
    return current_user

@app.put("/users/me", response_model=UserOut)
def update_profile(profile: UserProfileUpdate, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if profile.name is not None:
        current_user.name = profile.name
    if profile.bio is not None:
        current_user.bio = profile.bio
    db.commit()
    db.refresh(current_user)
    return current_user

@app.post("/users/me/avatar", response_model=UserOut)
async def upload_avatar(file: UploadFile = File(...), current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    ext = os.path.splitext(file.filename)[1]
    filename = f"avatar_{current_user.id}_{uuid4().hex}{ext}"
    file_path = os.path.join("uploads", filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    avatar_url = f"/uploads/{filename}"
    current_user.avatar_url = avatar_url
    db.commit()
    db.refresh(current_user)
    return current_user

@app.get("/users/{user_id}", response_model=UserOut)
def get_user(user_id: int, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user

@app.get("/conversations/", response_model=List[ConversationOut])
def get_conversations(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    other_users = db.query(User).filter(User.id != current_user.id).all()
    conversations = []
    for other in other_users:
        last_msg = db.query(Message).filter(
            ((Message.sender_id == current_user.id) & (Message.recipient_id == other.id)) |
            ((Message.sender_id == other.id) & (Message.recipient_id == current_user.id))
        ).order_by(desc(Message.timestamp)).first()
        unread = db.query(Message).filter(
            Message.sender_id == other.id,
            Message.recipient_id == current_user.id,
            Message.is_read == False
        ).count()
        # NOVO: para exibir um ícone se for áudio
        last_msg_display = None
        if last_msg:
            if last_msg.message_type == 'audio':
                last_msg_display = '🎤 Áudio'
            else:
                last_msg_display = last_msg.content
        conversations.append({
            "user_id": other.id,
            "name": other.name,
            "avatar_url": other.avatar_url,
            "last_message": last_msg_display,
            "last_message_time": last_msg.timestamp if last_msg else None,
            "unread_count": unread
        })
    return conversations

@app.get("/messages/{user_id}", response_model=List[MessageOut])
def get_messages(user_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    other = db.query(User).filter(User.id == user_id).first()
    if not other:
        raise HTTPException(status_code=404, detail="User not found")
    messages = db.query(Message).filter(
        ((Message.sender_id == current_user.id) & (Message.recipient_id == user_id)) |
        ((Message.sender_id == user_id) & (Message.recipient_id == current_user.id))
    ).order_by(Message.timestamp).all()
    db.query(Message).filter(
        Message.sender_id == user_id,
        Message.recipient_id == current_user.id,
        Message.is_read == False
    ).update({"is_read": True})
    db.commit()
    return messages

# NOVO: endpoint para upload de áudio
@app.post("/messages/audio")
async def upload_audio(file: UploadFile = File(...), current_user: User = Depends(get_current_user)):
    # Gera nome único
    ext = os.path.splitext(file.filename)[1]
    if not ext:
        ext = '.webm'
    filename = f"audio_{current_user.id}_{uuid4().hex}{ext}"
    file_path = os.path.join("uploads", "audio", filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    file_url = f"/uploads/audio/{filename}"
    return {"file_url": file_url}

@app.post("/messages/", response_model=MessageOut)
async def send_message(message: MessageSend, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    recipient = db.query(User).filter(User.id == message.recipient_id).first()
    if not recipient:
        raise HTTPException(status_code=404, detail="Recipient not found")
    new_msg = Message(
        sender_id=current_user.id,
        recipient_id=message.recipient_id,
        content=message.content,
        message_type=message.message_type,
        file_url=message.file_url
    )
    db.add(new_msg)
    db.commit()
    db.refresh(new_msg)

    # Preparar dados para WebSocket
    msg_data = {
        "id": new_msg.id,
        "sender_id": new_msg.sender_id,
        "recipient_id": new_msg.recipient_id,
        "content": new_msg.content,
        "message_type": new_msg.message_type,
        "file_url": new_msg.file_url,
        "timestamp": new_msg.timestamp.isoformat(),
        "is_read": new_msg.is_read
    }

    # Enviar apenas para o destinatário
    asyncio.create_task(manager.send_personal_message(msg_data, message.recipient_id))

    return new_msg

@app.get("/")
def root():
    return FileResponse("static/index.html")

