// sw.js — Service Worker UNIFICADO da got app
// Push Notifications + Cache PWA num único arquivo.
// sw-pwa.js foi removido — não registre mais ele no index.html.

const CACHE_SHELL   = 'gotapp-shell-v1';
const CACHE_DYNAMIC = 'gotapp-dynamic-v1';
const APP_SHELL     = ['/', '/static/manifest.json'];

// ── Install ──────────────────────────────────────────────────
self.addEventListener('install', event => {
  event.waitUntil(
    caches.open(CACHE_SHELL)
      .then(cache => cache.addAll(APP_SHELL))
      .then(() => self.skipWaiting())
  );
});

// ── Activate: limpa caches antigos ───────────────────────────
self.addEventListener('activate', event => {
  const KEEP = [CACHE_SHELL, CACHE_DYNAMIC];
  event.waitUntil(
    caches.keys()
      .then(keys => Promise.all(
        keys.filter(k => !KEEP.includes(k)).map(k => caches.delete(k))
      ))
      .then(() => self.clients.claim())
  );
});

// ── Fetch: estratégia por tipo de rota ───────────────────────
self.addEventListener('fetch', event => {
  const { request } = event;
  const url = new URL(request.url);

  if (!url.href.startsWith(self.location.origin)) return;
  if (request.method !== 'GET') return;

  // API e WebSocket — Network-only, nunca cacheia
  const API_PREFIXES = [
    '/users', '/messages', '/conversations', '/follow',
    '/feed', '/posts', '/reels', '/push', '/ws'
  ];
  if (API_PREFIXES.some(p => url.pathname.startsWith(p))) {
    event.respondWith(
      fetch(request).catch(() => new Response(
        JSON.stringify({ detail: 'Sem conexão com o servidor.' }),
        { status: 503, headers: { 'Content-Type': 'application/json' } }
      ))
    );
    return;
  }

  // Uploads (avatares, fotos, áudios) — Cache-first
  if (url.pathname.startsWith('/uploads/')) {
    event.respondWith(
      caches.match(request).then(cached => {
        if (cached) return cached;
        return fetch(request).then(res => {
          if (res.ok) caches.open(CACHE_DYNAMIC).then(c => c.put(request, res.clone()));
          return res;
        });
      })
    );
    return;
  }

  // Shell e static — Cache-first com fallback offline
  event.respondWith(
    caches.match(request).then(cached => {
      if (cached) return cached;
      return fetch(request).then(res => {
        if (res.ok) caches.open(CACHE_SHELL).then(c => c.put(request, res.clone()));
        return res;
      }).catch(() => {
        if (request.mode === 'navigate') return caches.match('/');
      });
    })
  );
});

// ── Push: receber notificação ─────────────────────────────────
self.addEventListener('push', event => {
  if (!event.data) return;

  let payload;
  try { payload = event.data.json(); }
  catch { payload = { title: 'got app', body: event.data.text() }; }

  const title = payload.title || 'got app';
  const type  = (payload.data && payload.data.type) || 'general';

  const options = {
    body:     payload.body  || '',
    icon:     payload.icon  || '/static/icon.png',
    badge:    payload.badge || '/static/icon.png',
    data:     payload.data  || {},
    tag:      type,
    renotify: true,
    vibrate:  [200, 100, 200],
    actions:  _actionsFor(type)
  };

  event.waitUntil(self.registration.showNotification(title, options));
});

// ── Notification click ────────────────────────────────────────
self.addEventListener('notificationclick', event => {
  event.notification.close();

  const data   = event.notification.data || {};
  const action = event.action;

  event.waitUntil(
    self.clients.matchAll({ type: 'window', includeUncontrolled: true }).then(clients => {
      const existing = clients.find(c => c.url.startsWith(self.location.origin));
      if (existing) {
        existing.focus();
        existing.postMessage({ type: 'notification_click', data, action });
      } else {
        self.clients.openWindow('/').then(win => {
          if (win) win.postMessage({ type: 'notification_click', data, action });
        });
      }
    })
  );
});

// ── Push subscription change (rotação de chaves) ──────────────
self.addEventListener('pushsubscriptionchange', event => {
  event.waitUntil(
    self.registration.pushManager.subscribe(event.oldSubscription.options)
      .then(sub => {
        const j = sub.toJSON();
        return fetch('/push/refresh', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ endpoint: j.endpoint, p256dh: j.keys.p256dh, auth: j.keys.auth })
        });
      })
  );
});

// ── Ações rápidas por tipo ────────────────────────────────────
function _actionsFor(type) {
  if (type === 'message')     return [{ action: 'open', title: '💬 Abrir chat' }];
  if (type === 'call')        return [{ action: 'open', title: '📞 Atender' }];
  if (type === 'missed_call') return [{ action: 'open', title: '🔁 Retornar' }];
  if (type === 'follow')      return [{ action: 'open', title: '👤 Ver perfil' }];
  if (type === 'post')        return [{ action: 'open', title: '📸 Ver post' }];
  return [];
}

