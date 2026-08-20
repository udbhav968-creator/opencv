// static/sw.js
// ICC Real DRS Hawk-Eye 3D Progressive Web App Service Worker

const CACHE_NAME = 'drs-hawkeye-v1';
const ASSETS_TO_CACHE = [
  '/',
  '/analytics',
  '/records',
  '/admin',
  '/favicon.ico',
  '/manifest.json'
];

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => {
      console.log('[DRS PWA] Caching offline app shell assets');
      return cache.addAll(ASSETS_TO_CACHE);
    })
  );
});

self.addEventListener('fetch', (event) => {
  event.respondWith(
    caches.match(event.request).then((response) => {
      return response || fetch(event.request);
    })
  );
});
