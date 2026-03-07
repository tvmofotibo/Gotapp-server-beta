import os
import re
import json
import shutil
import asyncio
import logging
import bcrypt
import hashlib
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Dict
from fastapi import FastAPI, File, UploadFile, Depends, HTTPException, status, WebSocket, WebSocketDisconnect
from fastapi.security import OAuth2PasswordBearer
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
from jose import JWTError, jwt
from sqlalchemy import create_engine, Column, Integer, String, DateTime, ForeignKey, Text, Boolean, desc, UniqueConstraint
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
from uuid import uuid4

# ── Push Notifications ──────────────────────────────────────
# pip install pywebpush py-vapid
try:
    from pywebpush import webpush, WebPushException
    from py_vapid import Vapid
    PUSH_AVAILABLE = True
except ImportError:
    PUSH_AVAILABLE = False
    logging.warning("pywebpush não instalado. Notificações push desativadas.")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === Configurações ===
SECRET_KEY = "09d25e094faa6ca2556c818166b7a9563b93f7099f6f0f4caa6cf63b88e8d3e7"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# ── VAPID Keys ──────────────────────────────────────────────
VAPID_FILE = "vapid.json"

def load_or_create_vapid():
    import base64
    from cryptography.hazmat.primitives.serialization import (
        Encoding, PublicFormat, PrivateFormat, NoEncryption
    )

    if not PUSH_AVAILABLE:
        return None, None, None

    if os.path.exists(VAPID_FILE):
        with open(VAPID_FILE) as f:
            data = json.load(f)
        if data.get("private_key", "").startswith("-----"):
            logger.warning("[VAPID] Chave no formato PEM antigo — regenerando...")
            os.remove(VAPID_FILE)
        else:
            return data["private_key"], data["public_key"], data["claims"]

    vapid = Vapid()
    vapid.generate_keys()

    pub_raw = vapid.public_key.public_bytes(Encoding.X962, PublicFormat.UncompressedPoint)
    public_key = base64.urlsafe_b64encode(pub_raw).rstrip(b"=").decode()

    priv_numbers = vapid.private_key.private_numbers()
    priv_raw = priv_numbers.private_value.to_bytes(32, "big")
    private_key = base64.urlsafe_b64encode(priv_raw).rstrip(b"=").decode()

    claims = {"sub": "mailto:admin@gotapp.com"}
    with open(VAPID_FILE, "w") as f:
        json.dump({"private_key": private_key, "public_key": public_key, "claims": claims}, f)
    logger.info(f"[VAPID] Chaves geradas. Pública: {public_key}")
    return private_key, public_key, claims

VAPID_PRIVATE, VAPID_PUBLIC, VAPID_CLAIMS = load_or_create_vapid()

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
    username = Column(String, unique=True, index=True, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    name = Column(String, nullable=False)
    hashed_password = Column(String, nullable=False)
    bio = Column(Text, default="")
    avatar_url = Column(String, default="")
    created_at = Column(DateTime, default=datetime.utcnow)

    sent_messages = relationship("Message", foreign_keys="Message.sender_id", back_populates="sender")
    received_messages = relationship("Message", foreign_keys="Message.recipient_id", back_populates="recipient")
    posts = relationship("Post", back_populates="user", cascade="all, delete-orphan")
    reels = relationship("Reel", back_populates="user", cascade="all, delete-orphan")
    followers = relationship("Follow", foreign_keys="Follow.followed_id", back_populates="followed", cascade="all, delete-orphan")
    following = relationship("Follow", foreign_keys="Follow.follower_id", back_populates="follower", cascade="all, delete-orphan")
    push_subscriptions = relationship("PushSubscription", back_populates="user", cascade="all, delete-orphan")

class Message(Base):
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True, index=True)
    sender_id = Column(Integer, ForeignKey("users.id"))
    recipient_id = Column(Integer, ForeignKey("users.id"))
    content = Column(Text, nullable=True)
    message_type = Column(String, default='text')
    file_url = Column(String, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    is_read = Column(Boolean, default=False)

    sender = relationship("User", foreign_keys=[sender_id], back_populates="sent_messages")
    recipient = relationship("User", foreign_keys=[recipient_id], back_populates="received_messages")

class Post(Base):
    __tablename__ = "posts"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    image_url = Column(String, nullable=True)
    text_content = Column(Text, nullable=True)
    caption = Column(Text, default="")
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="posts")

class Reel(Base):
    __tablename__ = "reels"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    youtube_id = Column(String, nullable=False)
    caption = Column(Text, default="")
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="reels")

class Follow(Base):
    __tablename__ = "follows"
    id = Column(Integer, primary_key=True, index=True)
    follower_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    followed_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    follower = relationship("User", foreign_keys=[follower_id], back_populates="following")
    followed = relationship("User", foreign_keys=[followed_id], back_populates="followers")

    __table_args__ = (UniqueConstraint('follower_id', 'followed_id', name='unique_follow'),)

class PushSubscription(Base):
    __tablename__ = "push_subscriptions"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    endpoint = Column(Text, nullable=False, unique=True)
    p256dh = Column(Text, nullable=False)
    auth = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="push_subscriptions")

Base.metadata.create_all(bind=engine)

# === Funções de hash ===
def get_password_hash(password: str) -> str:
    password_sha256 = hashlib.sha256(password.encode()).digest()
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password_sha256, salt)
    return hashed.decode()

def verify_password(plain_password: str, hashed_password: str) -> bool:
    password_sha256 = hashlib.sha256(plain_password.encode()).digest()
    return bcrypt.checkpw(password_sha256, hashed_password.encode())

def extract_youtube_id(url: str) -> Optional[str]:
    patterns = [
        r'youtube\.com/watch\?v=([a-zA-Z0-9_-]{11})',
        r'youtu\.be/([a-zA-Z0-9_-]{11})',
        r'youtube\.com/shorts/([a-zA-Z0-9_-]{11})',
        r'youtube\.com/embed/([a-zA-Z0-9_-]{11})',
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

# === Schemas ===
class UserCreate(BaseModel):
    username: str
    email: EmailStr
    name: str
    password: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class UserOut(BaseModel):
    id: int
    username: str
    email: EmailStr
    name: str
    bio: str = ""
    avatar_url: str = ""
    followers_count: int = 0
    following_count: int = 0
    posts_count: int = 0

class UserProfileUpdate(BaseModel):
    name: Optional[str] = None
    bio: Optional[str] = None

class MessageSend(BaseModel):
    recipient_id: int
    content: Optional[str] = None
    message_type: str = 'text'
    file_url: Optional[str] = None

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
    i_follow_them: bool = False
    they_follow_me: bool = False
    is_mutual: bool = False
    remaining_messages: Optional[int] = None

class Token(BaseModel):
    access_token: str
    token_type: str

class TextPostCreate(BaseModel):
    text: str
    caption: Optional[str] = None

class PostOut(BaseModel):
    id: int
    user_id: int
    image_url: Optional[str]
    text_content: Optional[str]
    caption: Optional[str]
    created_at: datetime
    post_type: str = "image"
    user_name: Optional[str] = None
    user_avatar: Optional[str] = None

class ReelCreate(BaseModel):
    youtube_url: str
    caption: Optional[str] = None

class ReelOut(BaseModel):
    id: int
    user_id: int
    youtube_id: str
    caption: Optional[str]
    created_at: datetime
    user_name: Optional[str] = None
    user_avatar: Optional[str] = None

class PushSubscriptionCreate(BaseModel):
    endpoint: str
    p256dh: str
    auth: str

# === Dependências ===
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def authenticate_user(db: Session, email: str, password: str):
    user = db.query(User).filter(User.email == email).first()
    if not user or not verify_password(password, user.hashed_password):
        return False
    return user

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                                           detail="Could not validate credentials",
                                           headers={"WWW-Authenticate": "Bearer"})
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

# ── Helper de envio push ─────────────────────────────────────
async def send_push_to_user(user_id: int, title: str, body: str, db: Session = None,
                            data: dict = None, sender_avatar: str = None):
    if not PUSH_AVAILABLE or not VAPID_PRIVATE:
        return

    own_db = SessionLocal()
    try:
        subs = own_db.query(PushSubscription).filter(PushSubscription.user_id == user_id).all()
        logger.info(f"[Push] Tentando enviar para user {user_id}: {len(subs)} subscription(s) | titulo='{title}'")
        if not subs:
            logger.warning(f"[Push] user {user_id} não tem subscriptions no banco — push ignorado")
            return

        base_url = os.environ.get("APP_BASE_URL", "").rstrip("/")

        def abs_url(path):
            if not path:
                return f"{base_url}/static/icon.png" if base_url else "/static/icon.png"
            if path.startswith("http"):
                return path
            return f"{base_url}{path}" if base_url else path

        payload = json.dumps({
            "title": title,
            "body": body,
            "icon": abs_url("/static/icon.png"),
            "badge": abs_url(sender_avatar),
            "data": data or {}
        })

        stale_ids = []
        for sub in subs:
            try:
                webpush(
                    subscription_info={"endpoint": sub.endpoint, "keys": {"p256dh": sub.p256dh, "auth": sub.auth}},
                    data=payload,
                    vapid_private_key=VAPID_PRIVATE,
                    vapid_claims=VAPID_CLAIMS
                )
            except WebPushException as e:
                if e.response and e.response.status_code in (404, 410):
                    stale_ids.append(sub.id)
                else:
                    logger.warning(f"[Push] user {user_id}: {e}")
            except Exception as e:
                logger.warning(f"[Push] erro inesperado: {e}")

        if stale_ids:
            own_db.query(PushSubscription).filter(
                PushSubscription.id.in_(stale_ids)
            ).delete(synchronize_session=False)
            own_db.commit()
            logger.info(f"[Push] {len(stale_ids)} subscription(s) expirada(s) removida(s) do user {user_id}")
    finally:
        own_db.close()

# === WebSocket Connection Manager ===
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[int, Dict] = {}
        self.call_sessions: Dict[int, int] = {}
        # Chamadas pendentes para usuários offline: {target_id: {offer_data}}
        self.pending_calls: Dict[int, Dict] = {}

    async def connect(self, websocket: WebSocket, user_id: int, user_name: str):
        await websocket.accept()
        # Substitui conexão antiga silenciosamente (reconexão do cliente)
        self.active_connections[user_id] = {"ws": websocket, "name": user_name}
        # Entrega chamada pendente (caller ligou enquanto estava offline)
        if user_id in self.pending_calls:
            pending = self.pending_calls.pop(user_id)
            # Só entrega se o caller ainda está conectado
            if self.is_connected(pending["from"]):
                self.call_sessions[user_id] = pending["from"]
                self.call_sessions[pending["from"]] = user_id
                asyncio.create_task(self.send_personal_message(pending["offer_msg"], user_id))
                asyncio.create_task(self.send_personal_message(
                    {"type": "call_connecting", "target": user_id}, pending["from"]
                ))
            else:
                # Caller desistiu enquanto esperava
                asyncio.create_task(self.send_personal_message(
                    {"type": "call_missed_offline"}, user_id
                ))

    def disconnect(self, user_id: int):
        if user_id not in self.active_connections:
            return
        del self.active_connections[user_id]
        if user_id in self.call_sessions:
            other = self.call_sessions.pop(user_id)
            if other in self.call_sessions:
                del self.call_sessions[other]
            if other in self.active_connections:
                asyncio.create_task(
                    self.send_personal_message({"type": "call_end", "from": user_id}, other)
                )

    async def send_personal_message(self, message: dict, user_id: int):
        from starlette.websockets import WebSocketState
        entry = self.active_connections.get(user_id)
        if not entry:
            return
        ws: WebSocket = entry["ws"]
        try:
            if ws.client_state == WebSocketState.CONNECTED:
                await ws.send_json(message)
            else:
                # Conexão já fechou — limpa o registro para não acumular
                self.active_connections.pop(user_id, None)
        except Exception as e:
            logger.warning(f"[WS] send falhou user {user_id}, removendo: {e}")
            self.active_connections.pop(user_id, None)

    def get_user_name(self, user_id: int) -> Optional[str]:
        return self.active_connections.get(user_id, {}).get("name")

    def is_connected(self, user_id: int) -> bool:
        """Retorna True apenas se o WS existe E está realmente aberto."""
        from starlette.websockets import WebSocketState
        entry = self.active_connections.get(user_id)
        if not entry:
            return False
        try:
            return entry["ws"].client_state == WebSocketState.CONNECTED
        except Exception:
            return False

manager = ConnectionManager()

# === App FastAPI ===
app = FastAPI(title="Got App API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

@app.middleware("http")
async def capture_base_url(request, call_next):
    """Captura a URL base do primeiro request para usar nas notificações push."""
    if not os.environ.get("APP_BASE_URL"):
        scheme = request.headers.get("x-forwarded-proto", request.url.scheme)
        host   = request.headers.get("x-forwarded-host", request.headers.get("host", request.url.netloc))
        os.environ["APP_BASE_URL"] = f"{scheme}://{host}"
        logger.info(f"[Push] APP_BASE_URL detectada: {os.environ['APP_BASE_URL']}")
    return await call_next(request)

os.makedirs("uploads", exist_ok=True)
os.makedirs("uploads/audio", exist_ok=True)
os.makedirs("uploads/posts", exist_ok=True)
os.makedirs("static", exist_ok=True)

app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")
app.mount("/static", StaticFiles(directory="static"), name="static")

# ── Push Endpoints ────────────────────────────────────────────

@app.get("/push/vapid-public-key")
def get_vapid_public_key():
    if not PUSH_AVAILABLE or not VAPID_PUBLIC:
        raise HTTPException(status_code=503, detail="Push notifications não disponíveis")
    return {"public_key": VAPID_PUBLIC}

@app.post("/push/subscribe")
async def subscribe_push(sub: PushSubscriptionCreate,
                         current_user: User = Depends(get_current_user),
                         db: Session = Depends(get_db)):
    existing = db.query(PushSubscription).filter(PushSubscription.endpoint == sub.endpoint).first()
    if existing:
        existing.p256dh = sub.p256dh
        existing.auth = sub.auth
        existing.user_id = current_user.id
    else:
        db.add(PushSubscription(user_id=current_user.id, endpoint=sub.endpoint,
                                 p256dh=sub.p256dh, auth=sub.auth))
    db.commit()
    return {"status": "subscribed"}

@app.delete("/push/unsubscribe")
async def unsubscribe_push(sub: PushSubscriptionCreate,
                           current_user: User = Depends(get_current_user),
                           db: Session = Depends(get_db)):
    db.query(PushSubscription).filter(
        PushSubscription.user_id == current_user.id,
        PushSubscription.endpoint == sub.endpoint
    ).delete()
    db.commit()
    return {"status": "unsubscribed"}

@app.post("/push/refresh")
async def refresh_push(sub: PushSubscriptionCreate,
                       current_user: User = Depends(get_current_user),
                       db: Session = Depends(get_db)):
    db.query(PushSubscription).filter(
        PushSubscription.user_id == current_user.id
    ).delete()
    db.add(PushSubscription(
        user_id=current_user.id,
        endpoint=sub.endpoint,
        p256dh=sub.p256dh,
        auth=sub.auth
    ))
    db.commit()
    return {"status": "refreshed"}

# === WebSocket ===
@app.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: int, token: str, db: Session = Depends(get_db)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        token_user_id = payload.get("sub")
        if token_user_id is None or int(token_user_id) != user_id:
            await websocket.close(code=1008); return
    except (JWTError, ValueError):
        await websocket.close(code=1008); return

    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        await websocket.close(code=1008); return

    await manager.connect(websocket, user_id, user.name)
    try:
        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type")
            target = data.get("target")

            if msg_type == "call_offer":
                if target in manager.call_sessions:
                    await manager.send_personal_message({"type": "call_busy"}, user_id)
                    continue
                if not manager.is_connected(target):
                    # Destinatário offline — guarda chamada pendente e notifica via push
                    manager.pending_calls[target] = {
                        "from": user_id,
                        "offer_msg": {
                            "type": "call_offer", "from": user_id, "fromName": user.name,
                            "offer": data["offer"], "hasVideo": data.get("hasVideo", False),
                            "hasAudio": data.get("hasAudio", True)
                        }
                    }
                    await send_push_to_user(target, f"📞 Chamada de {user.name}",
                                            "Toque para atender", db,
                                            {"type": "call", "from_id": user_id},
                                            sender_avatar=user.avatar_url or None)
                    # Avisa o caller que está aguardando o destinatário abrir o app
                    await manager.send_personal_message(
                        {"type": "call_waiting", "target": target}, user_id
                    )
                    continue
                manager.call_sessions[user_id] = target
                manager.call_sessions[target] = user_id
                await manager.send_personal_message({
                    "type": "call_offer", "from": user_id, "fromName": user.name,
                    "offer": data["offer"], "hasVideo": data.get("hasVideo", False), "hasAudio": data.get("hasAudio", True)
                }, target)
                await send_push_to_user(target, f"📞 Chamada de {user.name}", "Toque para atender", db,
                                        {"type": "call", "from_id": user_id},
                                        sender_avatar=user.avatar_url or None)

            elif msg_type == "call_answer":
                if target in manager.active_connections:
                    await manager.send_personal_message({"type": "call_answer", "from": user_id, "answer": data["answer"]}, target)

            elif msg_type == "ice_candidate":
                if target in manager.active_connections:
                    await manager.send_personal_message({"type": "ice_candidate", "from": user_id, "candidate": data["candidate"]}, target)

            elif msg_type == "call_end":
                # Cancela chamada pendente se o caller desistir antes do destinatário abrir
                for tid, pending in list(manager.pending_calls.items()):
                    if pending["from"] == user_id:
                        del manager.pending_calls[tid]
                        await send_push_to_user(tid, f"📵 Chamada perdida de {user.name}",
                                                "Você perdeu uma chamada",
                                                db, {"type": "missed_call", "from_id": user_id},
                                                sender_avatar=user.avatar_url or None)
                        break
                if user_id in manager.call_sessions:
                    other = manager.call_sessions.pop(user_id)
                    if other in manager.call_sessions: del manager.call_sessions[other]
                    if manager.is_connected(other):
                        await manager.send_personal_message({"type": "call_end", "from": user_id}, other)
                    else:
                        await send_push_to_user(other, f"📵 Chamada perdida de {user.name}",
                                                "Você perdeu uma chamada",
                                                db, {"type": "missed_call", "from_id": user_id},
                                                sender_avatar=user.avatar_url or None)

            elif msg_type == "call_reject":
                if target in manager.active_connections:
                    await manager.send_personal_message({"type": "call_reject", "from": user_id}, target)
                if user_id in manager.call_sessions:
                    other = manager.call_sessions.pop(user_id)
                    if other in manager.call_sessions: del manager.call_sessions[other]

            elif msg_type == "call_accept":
                if target in manager.active_connections:
                    await manager.send_personal_message({"type": "call_accept", "from": user_id, "hasVideo": data.get("hasVideo", False)}, target)

    except WebSocketDisconnect:
        manager.disconnect(user_id)
    except Exception as e:
        logger.error(f"Erro no WebSocket: {e}")
        manager.disconnect(user_id)

# === Usuários ===
@app.post("/users/register", response_model=UserOut)
def register(user: UserCreate, db: Session = Depends(get_db)):
    if db.query(User).filter(User.username == user.username).first():
        raise HTTPException(status_code=400, detail="Username already taken")
    if db.query(User).filter(User.email == user.email).first():
        raise HTTPException(status_code=400, detail="Email already registered")
    if ' ' in user.username:
        raise HTTPException(status_code=400, detail="Username cannot contain spaces")
    new_user = User(username=user.username, email=user.email, name=user.name,
                    hashed_password=get_password_hash(user.password))
    db.add(new_user); db.commit(); db.refresh(new_user)
    return UserOut(id=new_user.id, username=new_user.username, email=new_user.email, name=new_user.name,
                   bio=new_user.bio, avatar_url=new_user.avatar_url)

@app.post("/users/login", response_model=Token)
def login(user_data: UserLogin, db: Session = Depends(get_db)):
    user = authenticate_user(db, user_data.email, user_data.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    access_token = create_access_token(data={"sub": str(user.id)},
                                        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/users/me", response_model=UserOut)
def read_users_me(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    return UserOut(id=current_user.id, username=current_user.username, email=current_user.email, name=current_user.name,
                   bio=current_user.bio, avatar_url=current_user.avatar_url,
                   followers_count=db.query(Follow).filter(Follow.followed_id == current_user.id).count(),
                   following_count=db.query(Follow).filter(Follow.follower_id == current_user.id).count(),
                   posts_count=db.query(Post).filter(Post.user_id == current_user.id).count())

@app.put("/users/me", response_model=UserOut)
def update_profile(profile: UserProfileUpdate, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if profile.name is not None: current_user.name = profile.name
    if profile.bio is not None: current_user.bio = profile.bio
    db.commit(); db.refresh(current_user)
    return UserOut(id=current_user.id, username=current_user.username, email=current_user.email, name=current_user.name,
                   bio=current_user.bio, avatar_url=current_user.avatar_url,
                   followers_count=db.query(Follow).filter(Follow.followed_id == current_user.id).count(),
                   following_count=db.query(Follow).filter(Follow.follower_id == current_user.id).count(),
                   posts_count=db.query(Post).filter(Post.user_id == current_user.id).count())

@app.post("/users/me/avatar", response_model=UserOut)
async def upload_avatar(file: UploadFile = File(...), current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    ext = os.path.splitext(file.filename)[1]
    filename = f"avatar_{current_user.id}_{uuid4().hex}{ext}"
    with open(os.path.join("uploads", filename), "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    current_user.avatar_url = f"/uploads/{filename}"
    db.commit(); db.refresh(current_user)
    return UserOut(id=current_user.id, username=current_user.username, email=current_user.email, name=current_user.name,
                   bio=current_user.bio, avatar_url=current_user.avatar_url,
                   followers_count=db.query(Follow).filter(Follow.followed_id == current_user.id).count(),
                   following_count=db.query(Follow).filter(Follow.follower_id == current_user.id).count(),
                   posts_count=db.query(Post).filter(Post.user_id == current_user.id).count())

@app.get("/users/search", response_model=List[UserOut])
def search_users(q: str = "", current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    query = db.query(User).filter(User.id != current_user.id)
    if q:
        query = query.filter((User.username.ilike(f"%{q}%")) | (User.name.ilike(f"%{q}%")))
    return [UserOut(id=u.id, username=u.username, email=u.email, name=u.name, bio=u.bio, avatar_url=u.avatar_url,
                    followers_count=db.query(Follow).filter(Follow.followed_id == u.id).count(),
                    following_count=db.query(Follow).filter(Follow.follower_id == u.id).count(),
                    posts_count=db.query(Post).filter(Post.user_id == u.id).count())
            for u in query.limit(20).all()]

@app.get("/users/{user_id}", response_model=UserOut)
def get_user(user_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == user_id).first()
    if not user: raise HTTPException(status_code=404, detail="User not found")
    return UserOut(id=user.id, username=user.username, email=user.email, name=user.name, bio=user.bio,
                   avatar_url=user.avatar_url,
                   followers_count=db.query(Follow).filter(Follow.followed_id == user_id).count(),
                   following_count=db.query(Follow).filter(Follow.follower_id == user_id).count(),
                   posts_count=db.query(Post).filter(Post.user_id == user_id).count())

# === Relacionamentos ===
@app.get("/users/{user_id}/followers", response_model=List[UserOut])
def get_followers(user_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if not db.query(User).filter(User.id == user_id).first():
        raise HTTPException(status_code=404, detail="User not found")
    return [UserOut(id=f.follower.id, username=f.follower.username, email=f.follower.email, name=f.follower.name,
                    bio=f.follower.bio, avatar_url=f.follower.avatar_url,
                    followers_count=db.query(Follow).filter(Follow.followed_id == f.follower.id).count(),
                    following_count=db.query(Follow).filter(Follow.follower_id == f.follower.id).count(),
                    posts_count=db.query(Post).filter(Post.user_id == f.follower.id).count())
            for f in db.query(Follow).filter(Follow.followed_id == user_id).all()]

@app.get("/users/{user_id}/following", response_model=List[UserOut])
def get_following(user_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if not db.query(User).filter(User.id == user_id).first():
        raise HTTPException(status_code=404, detail="User not found")
    return [UserOut(id=f.followed.id, username=f.followed.username, email=f.followed.email, name=f.followed.name,
                    bio=f.followed.bio, avatar_url=f.followed.avatar_url,
                    followers_count=db.query(Follow).filter(Follow.followed_id == f.followed.id).count(),
                    following_count=db.query(Follow).filter(Follow.follower_id == f.followed.id).count(),
                    posts_count=db.query(Post).filter(Post.user_id == f.followed.id).count())
            for f in db.query(Follow).filter(Follow.follower_id == user_id).all()]

@app.post("/follow/{user_id}")
async def follow_user(user_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if user_id == current_user.id:
        raise HTTPException(status_code=400, detail="You cannot follow yourself")
    target = db.query(User).filter(User.id == user_id).first()
    if not target: raise HTTPException(status_code=404, detail="User not found")
    if db.query(Follow).filter(Follow.follower_id == current_user.id, Follow.followed_id == user_id).first():
        raise HTTPException(status_code=400, detail="Already following")
    db.add(Follow(follower_id=current_user.id, followed_id=user_id))
    db.commit()
    await manager.send_personal_message({"type": "followed", "follower_id": current_user.id, "follower_name": current_user.name}, user_id)
    await manager.send_personal_message({"type": "follow_update", "target_id": user_id, "follower_id": current_user.id, "action": "followed"}, current_user.id)
    await send_push_to_user(user_id, "✨ Novo seguidor",
                            f"{current_user.name} começou a te seguir!", db,
                            {"type": "follow", "follower_id": current_user.id},
                            sender_avatar=current_user.avatar_url or None)
    return {"status": "followed"}

@app.delete("/follow/{user_id}")
async def unfollow_user(user_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    follow = db.query(Follow).filter(Follow.follower_id == current_user.id, Follow.followed_id == user_id).first()
    if not follow: raise HTTPException(status_code=404, detail="Not following")
    db.delete(follow); db.commit()
    await manager.send_personal_message({"type": "unfollowed", "follower_id": current_user.id, "follower_name": current_user.name}, user_id)
    await manager.send_personal_message({"type": "follow_update", "target_id": user_id, "follower_id": current_user.id, "action": "unfollowed"}, current_user.id)
    return {"status": "unfollowed"}

@app.get("/follow/status/{user_id}")
def follow_status(user_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    return {"following": db.query(Follow).filter(Follow.follower_id == current_user.id, Follow.followed_id == user_id).first() is not None}

# === Mensagens ===
@app.get("/conversations/", response_model=List[ConversationOut])
def get_conversations(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    conversations = []
    for other in db.query(User).filter(User.id != current_user.id).all():
        i_follow_them = db.query(Follow).filter(Follow.follower_id == current_user.id, Follow.followed_id == other.id).first() is not None
        they_follow_me = db.query(Follow).filter(Follow.follower_id == other.id, Follow.followed_id == current_user.id).first() is not None
        remaining = None
        if not they_follow_me:
            msg_count = db.query(Message).filter(Message.sender_id == current_user.id, Message.recipient_id == other.id).count()
            remaining = max(0, 3 - msg_count)
        last_msg = db.query(Message).filter(
            ((Message.sender_id == current_user.id) & (Message.recipient_id == other.id)) |
            ((Message.sender_id == other.id) & (Message.recipient_id == current_user.id))
        ).order_by(desc(Message.timestamp)).first()
        unread = db.query(Message).filter(Message.sender_id == other.id, Message.recipient_id == current_user.id, Message.is_read == False).count()
        last_msg_display = ('🎤 Áudio' if last_msg.message_type == 'audio' else last_msg.content) if last_msg else None
        if i_follow_them or they_follow_me:
            conversations.append({
                "user_id": other.id, "name": other.name, "avatar_url": other.avatar_url,
                "last_message": last_msg_display, "last_message_time": last_msg.timestamp if last_msg else None,
                "unread_count": unread, "i_follow_them": i_follow_them, "they_follow_me": they_follow_me,
                "is_mutual": i_follow_them and they_follow_me, "remaining_messages": remaining
            })
    return conversations

@app.get("/messages/{user_id}", response_model=List[MessageOut])
def get_messages(user_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if not db.query(User).filter(User.id == user_id).first():
        raise HTTPException(status_code=404, detail="User not found")
    messages = db.query(Message).filter(
        ((Message.sender_id == current_user.id) & (Message.recipient_id == user_id)) |
        ((Message.sender_id == user_id) & (Message.recipient_id == current_user.id))
    ).order_by(Message.timestamp).all()
    db.query(Message).filter(Message.sender_id == user_id, Message.recipient_id == current_user.id,
                              Message.is_read == False).update({"is_read": True})
    db.commit()
    return messages

@app.post("/messages/audio")
async def upload_audio(file: UploadFile = File(...), current_user: User = Depends(get_current_user)):
    ext = os.path.splitext(file.filename)[1] or '.webm'
    filename = f"audio_{current_user.id}_{uuid4().hex}{ext}"
    with open(os.path.join("uploads", "audio", filename), "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return {"file_url": f"/uploads/audio/{filename}"}

@app.post("/messages/", response_model=MessageOut)
async def send_message(message: MessageSend, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    recipient = db.query(User).filter(User.id == message.recipient_id).first()
    if not recipient: raise HTTPException(status_code=404, detail="Recipient not found")

    recipient_follows_sender = db.query(Follow).filter(
        Follow.follower_id == recipient.id, Follow.followed_id == current_user.id
    ).first() is not None

    if not recipient_follows_sender:
        if db.query(Message).filter(Message.sender_id == current_user.id, Message.recipient_id == recipient.id).count() >= 3:
            raise HTTPException(status_code=403, detail="Você só pode enviar 3 mensagens para este usuário até ele te seguir.")

    new_msg = Message(sender_id=current_user.id, recipient_id=message.recipient_id,
                      content=message.content, message_type=message.message_type, file_url=message.file_url)
    db.add(new_msg); db.commit(); db.refresh(new_msg)

    msg_data = {
        "id": new_msg.id, "sender_id": new_msg.sender_id, "recipient_id": new_msg.recipient_id,
        "content": new_msg.content, "message_type": new_msg.message_type, "file_url": new_msg.file_url,
        "timestamp": new_msg.timestamp.isoformat(), "is_read": new_msg.is_read
    }
    if not recipient_follows_sender:
        msg_data["from_non_follower"] = True

    await manager.send_personal_message(msg_data, message.recipient_id)

    # Envia push sempre — o browser suprime automaticamente se a aba estiver em foco.
    # Checar só manager.active_connections não é suficiente: o WS pode estar
    # registrado mas a conexão já fechou (app em background ou aba sem foco).
    if not manager.is_connected(message.recipient_id):
        preview = "🎤 Áudio" if message.message_type == "audio" else (message.content or "")
        await send_push_to_user(
            message.recipient_id,
            f"💬 {current_user.name}",
            (preview[:80] if preview else "Nova mensagem"),
            db,
            {"type": "message", "sender_id": current_user.id},
            sender_avatar=current_user.avatar_url or None
        )

    return new_msg

# === Posts ===
@app.post("/posts", response_model=PostOut)
async def create_post(caption: str = "", file: UploadFile = File(...),
                      current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    ext = os.path.splitext(file.filename)[1]
    filename = f"post_{current_user.id}_{uuid4().hex}{ext}"
    with open(os.path.join("uploads", "posts", filename), "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    post = Post(user_id=current_user.id, image_url=f"/uploads/posts/{filename}", caption=caption)
    db.add(post); db.commit(); db.refresh(post)
    asyncio.create_task(_notify_followers_new_post(current_user, post, db))
    return PostOut(id=post.id, user_id=post.user_id, image_url=post.image_url, text_content=post.text_content,
                   caption=post.caption, created_at=post.created_at, post_type="image",
                   user_name=current_user.name, user_avatar=current_user.avatar_url)

@app.post("/posts/text", response_model=PostOut)
async def create_text_post(body: TextPostCreate, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if not body.text or not body.text.strip():
        raise HTTPException(status_code=400, detail="O texto não pode estar vazio.")
    post = Post(user_id=current_user.id, text_content=body.text.strip(), caption=body.caption or "")
    db.add(post); db.commit(); db.refresh(post)
    asyncio.create_task(_notify_followers_new_post(current_user, post, db))
    return PostOut(id=post.id, user_id=post.user_id, image_url=post.image_url, text_content=post.text_content,
                   caption=post.caption, created_at=post.created_at, post_type="text",
                   user_name=current_user.name, user_avatar=current_user.avatar_url)

async def _notify_followers_new_post(author: User, post: Post, db: Session):
    followers = db.query(Follow).filter(Follow.followed_id == author.id).all()
    preview = (post.text_content or post.caption or "Novo post")[:60]
    for f in followers:
        await send_push_to_user(f.follower_id, f"📸 {author.name} publicou", preview, db,
                                {"type": "post", "post_id": post.id, "author_id": author.id},
                                sender_avatar=author.avatar_url or None)

@app.get("/users/{user_id}/posts", response_model=List[PostOut])
def get_user_posts(user_id: int, db: Session = Depends(get_db)):
    posts = db.query(Post).filter(Post.user_id == user_id).order_by(desc(Post.created_at)).all()
    return [PostOut(id=p.id, user_id=p.user_id, image_url=p.image_url, text_content=p.text_content,
                    caption=p.caption, created_at=p.created_at,
                    post_type="text" if p.text_content and not p.image_url else "image",
                    user_name=db.query(User).filter(User.id == p.user_id).first().name if db.query(User).filter(User.id == p.user_id).first() else "",
                    user_avatar=db.query(User).filter(User.id == p.user_id).first().avatar_url if db.query(User).filter(User.id == p.user_id).first() else "")
            for p in posts]

@app.get("/feed", response_model=List[PostOut])
def get_feed(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    following_ids = [f.followed_id for f in db.query(Follow).filter(Follow.follower_id == current_user.id).all()]
    followed_posts = db.query(Post).filter(Post.user_id.in_(following_ids)).order_by(desc(Post.created_at)).limit(50).all() if following_ids else []
    excluded_ids = following_ids + [current_user.id]
    other_posts = db.query(Post).filter(~Post.user_id.in_(excluded_ids)).order_by(desc(Post.created_at)).limit(50).all()
    result = []
    for p in (followed_posts + other_posts)[:50]:
        user = db.query(User).filter(User.id == p.user_id).first()
        result.append(PostOut(id=p.id, user_id=p.user_id, image_url=p.image_url, text_content=p.text_content,
                               caption=p.caption, created_at=p.created_at,
                               post_type="text" if p.text_content and not p.image_url else "image",
                               user_name=user.name if user else "", user_avatar=user.avatar_url if user else ""))
    return result

# === Reels ===
@app.post("/reels", response_model=ReelOut)
def create_reel(body: ReelCreate, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    yt_id = extract_youtube_id(body.youtube_url)
    if not yt_id:
        raise HTTPException(status_code=400, detail="URL do YouTube inválida.")
    reel = Reel(user_id=current_user.id, youtube_id=yt_id, caption=body.caption or "")
    db.add(reel); db.commit(); db.refresh(reel)
    return ReelOut(id=reel.id, user_id=reel.user_id, youtube_id=reel.youtube_id, caption=reel.caption,
                   created_at=reel.created_at, user_name=current_user.name, user_avatar=current_user.avatar_url)

@app.get("/reels", response_model=List[ReelOut])
def get_reels(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    following_ids = [f.followed_id for f in db.query(Follow).filter(Follow.follower_id == current_user.id).all()]
    followed_reels = db.query(Reel).filter(Reel.user_id.in_(following_ids)).order_by(desc(Reel.created_at)).limit(50).all() if following_ids else []
    excluded_ids = following_ids + [current_user.id]
    other_reels = db.query(Reel).filter(~Reel.user_id.in_(excluded_ids)).order_by(desc(Reel.created_at)).limit(50).all()
    result = []
    for r in (followed_reels + other_reels)[:50]:
        user = db.query(User).filter(User.id == r.user_id).first()
        result.append(ReelOut(id=r.id, user_id=r.user_id, youtube_id=r.youtube_id, caption=r.caption,
                               created_at=r.created_at, user_name=user.name if user else "", user_avatar=user.avatar_url if user else ""))
    return result

# ── Rotas dos Service Workers (precisam estar na raiz "/") ───
# O header Service-Worker-Allowed="/" permite que o SW registrado
# em /sw.js ou /sw-pwa.js controle todas as páginas do app.

@app.get("/sw.js")
def serve_sw_push():
    """SW original de notificações push — servido na raiz."""
    return FileResponse(
        "static/sw.js",
        media_type="application/javascript",
        headers={"Service-Worker-Allowed": "/"}
    )

@app.get("/sw-pwa.js")
def serve_sw_pwa():
    """SW de cache/offline PWA — servido na raiz para ter scope "/"."""
    return FileResponse(
        "static/sw-pwa.js",
        media_type="application/javascript",
        headers={"Service-Worker-Allowed": "/"}
    )


@app.get("/.well-known/assetlinks.json")
def assetlinks():
    """
    Necessário para TWA (Trusted Web Activity) no Android / Play Store.
    Retorna lista vazia — evita o 404 e pode ser preenchido com os dados
    reais do app Android quando necessário.
    """
    return []

@app.get("/")
def root():
    return FileResponse("static/index.html")

