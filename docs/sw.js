// Minimal Service Worker so Android Chrome installs TradingPro as a full WebAPK
// with a full-bleed maskable adaptive icon (no white circle border).
const CACHE = "tradingpro-v2";

self.addEventListener("install", (event) => {
  self.skipWaiting();
});

self.addEventListener("activate", (event) => {
  event.waitUntil(self.clients.claim());
});

self.addEventListener("fetch", (event) => {
  // Always prefer network so state.json signals are always fresh
  event.respondWith(
    fetch(event.request).catch(() => caches.match(event.request))
  );
});
