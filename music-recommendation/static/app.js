const state = {
  token: sessionStorage.getItem("soundscout_token") || "",
  user: null,
  currentTrack: null,
  queue: [],
  queueIndex: -1,
  liked: new Set(),
  playlists: [],
  catalog: [],
  repeat: false,
  shuffle: false,
};

const $ = (selector) => document.querySelector(selector);
const audio = $("#audioElement");

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (char) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[char]));
}

function toast(message) {
  const element = $("#toast");
  element.textContent = message;
  element.classList.add("show");
  window.clearTimeout(toast.timer);
  toast.timer = window.setTimeout(() => element.classList.remove("show"), 2600);
}

async function api(path, options = {}) {
  const headers = new Headers(options.headers || {});
  if (state.token) headers.set("Authorization", "Bearer " + state.token);
  if (options.body && !(options.body instanceof FormData)) {
    headers.set("Content-Type", "application/json");
  }
  const response = await fetch(path, { ...options, headers });
  const contentType = response.headers.get("content-type") || "";
  const result = contentType.includes("application/json")
    ? await response.json()
    : await response.text();
  if (!response.ok) {
    if (response.status === 401 && state.token) logout(false);
    const message = typeof result === "object" && result ? result.detail : result;
    throw new Error(message || "The request could not be completed.");
  }
  return result;
}

function showAuth() {
  $("#authView").classList.remove("hidden");
  $("#appView").classList.add("hidden");
}

function showApp() {
  $("#authView").classList.add("hidden");
  $("#appView").classList.remove("hidden");
}

function logout(showMessage = true) {
  state.token = "";
  state.user = null;
  state.currentTrack = null;
  state.queue = [];
  state.queueIndex = -1;
  sessionStorage.removeItem("soundscout_token");
  audio.pause();
  audio.removeAttribute("src");
  showAuth();
  if (showMessage) toast("You have signed out.");
}

async function login(username, password) {
  const result = await api("/api/auth/login", {
    method: "POST",
    body: JSON.stringify({ username, password }),
  });
  state.token = result.access_token;
  state.user = result.user;
  sessionStorage.setItem("soundscout_token", state.token);
  await enterApp();
}

async function register(username, email, password) {
  const result = await api("/api/auth/register", {
    method: "POST",
    body: JSON.stringify({ username, email, password }),
  });
  state.token = result.access_token;
  state.user = result.user;
  sessionStorage.setItem("soundscout_token", state.token);
  await enterApp();
  toast("Your account is ready. Welcome in.");
}

async function enterApp() {
  state.user = await api("/api/auth/me");
  showApp();
  $("#sideUsername").textContent = state.user.username;
  $("#userAvatar").textContent = state.user.username.slice(0, 1).toUpperCase();
  $("#welcomeHeading").textContent = "Made for " + state.user.username;
  state.catalog = await api("/api/tracks?limit=250");
  await loadLiked();
  await Promise.all([loadHome(), loadPlaylists(), loadStats()]);
}

function genreClass(genre) {
  return "genre-" + String(genre || "other").toLowerCase().replace(/[^a-z0-9]+/g, "-");
}

function cardMarkup(track) {
  const liked = state.liked.has(track.track_id);
  const symbol = String(track.title || "S").trim().slice(0, 1).toUpperCase();
  const reason = track.reason ? '<p class="track-reason">' + escapeHtml(track.reason) + "</p>" : "";
  return '<article class="track-card ' + genreClass(track.genre) + '">' +
    '<div class="track-art"><span class="art-symbol">' + escapeHtml(symbol) + '</span>' +
    '<button class="quick-play" data-play-track="' + escapeHtml(track.track_id) + '" aria-label="Play ' + escapeHtml(track.title) + '">▶</button></div>' +
    '<div class="track-info"><div class="track-copy"><strong>' + escapeHtml(track.title) + '</strong><span>' +
    escapeHtml(track.artist) + " · " + escapeHtml(track.genre) + "</span></div></div>" + reason +
    '<div class="card-actions"><button class="icon-button ' + (liked ? "like-active" : "") +
    '" data-like-track="' + escapeHtml(track.track_id) + '" title="' + (liked ? "Unlike" : "Like") +
    ' song">' + (liked ? "♥" : "♡") + '</button><button class="icon-button" data-add-track="' +
    escapeHtml(track.track_id) + '" title="Add to playlist">＋</button><span class="muted" style="font-size:8px">' +
    Number(track.tempo || 0).toFixed(0) + " BPM</span></div></article>";
}

function renderTracks(target, tracks, emptyMessage = "Nothing to show yet.") {
  const container = typeof target === "string" ? $(target) : target;
  if (!tracks || tracks.length === 0) {
    container.innerHTML = '<p class="empty-copy">' + escapeHtml(emptyMessage) + "</p>";
    return;
  }
  container.innerHTML = tracks.map(cardMarkup).join("");
}

async function loadHome() {
  const strategy = $("#modelSelect").value;
  $("#recommendationSubtitle").textContent = strategy === "hybrid"
    ? "A blend of song characteristics, listener patterns and popularity."
    : "Showing the " + strategy + " recommendation model.";
  const results = await Promise.all([
    api("/api/recommendations/" + strategy + "?limit=8"),
    api("/api/recommendations/popularity?limit=8"),
    api("/api/genres"),
  ]);
  renderTracks("#recommendationsGrid", results[0], "Play a few tracks to personalise your recommendations.");
  renderTracks("#trendingGrid", results[1], "The collection is waiting for its first listens.");
  $("#genreChips").innerHTML = results[2].map((genre) =>
    '<button class="genre-chip" data-genre="' + escapeHtml(genre) + '">' + escapeHtml(genre) + "</button>"
  ).join("");
}

async function loadStats() {
  const health = await api("/api/health");
  $("#statTracks").textContent = health.tracks;
  $("#statEvents").textContent = health.interactions;
  $("#statSvd").textContent = health.recommendation_engine.svd_available
    ? health.recommendation_engine.svd_components + " factors"
    : "warming up";
}

async function loadPlaylists() {
  state.playlists = await api("/api/playlists");
  $("#playlistList").innerHTML = state.playlists.length
    ? state.playlists.map((playlist) =>
      '<button class="playlist-link" data-open-playlist="' + escapeHtml(playlist.playlist_id) + '">♫ &nbsp;' +
      escapeHtml(playlist.title) + ' <span class="muted">(' + Number(playlist.track_count) + ")</span></button>"
    ).join("")
    : '<p class="side-empty">Create a playlist to start collecting tracks.</p>';
  $("#libraryPlaylists").innerHTML = state.playlists.length
    ? state.playlists.map((playlist) =>
      '<article class="playlist-card"><div class="playlist-cover">♫</div><h3>' +
      escapeHtml(playlist.title) + "</h3><p>" + Number(playlist.track_count) +
      " tracks · " + escapeHtml(playlist.description || "Your collection") +
      '</p><button class="button button-outline" data-open-playlist="' +
      escapeHtml(playlist.playlist_id) + '">Open playlist</button></article>'
    ).join("")
    : '<p class="empty-copy">No playlists yet. Use the + beside Your playlists to create one.</p>';
}

async function loadLiked() {
  const tracks = await api("/api/library/liked");
  state.liked = new Set(tracks.map((track) => track.track_id));
  renderTracks("#likedGrid", tracks, "Songs you like will appear here.");
}

async function loadHistory() {
  const history = await api("/api/history?limit=40");
  renderTracks("#historyGrid", history, "Your listening history will appear here.");
}

async function openPlaylist(id) {
  const result = await api("/api/playlists/" + encodeURIComponent(id));
  $("#playlistHeading").textContent = result.playlist.title;
  $("#playlistDescription").textContent = result.playlist.description || (result.tracks.length + " tracks");
  renderTracks("#playlistTracks", result.tracks, "This playlist is empty. Add a track using the plus button.");
  switchView("playlist");
}

async function showGenre(genre) {
  const tracks = await api("/api/tracks?genre=" + encodeURIComponent(genre) + "&limit=100");
  $("#searchInput").value = "";
  $("#searchCount").textContent = tracks.length + " tracks in " + genre;
  renderTracks("#searchGrid", tracks, "No tracks found in this genre.");
  switchView("search");
}

function switchView(view) {
  document.querySelectorAll(".page-view").forEach((element) => element.classList.add("hidden"));
  const target = $("#" + view + "View");
  if (target) target.classList.remove("hidden");
  document.querySelectorAll(".nav-item").forEach((button) => {
    button.classList.toggle("active", button.dataset.view === view);
  });
  if (view === "library") loadPlaylists().catch(showError);
  if (view === "liked") loadLiked().catch(showError);
  if (view === "history") loadHistory().catch(showError);
}

function showError(error) {
  toast(error.message || "Something went wrong.");
}

async function recordEvent(trackId, eventType, duration = 0) {
  if (!state.token || !trackId) return;
  try {
    await api("/api/interactions", {
      method: "POST",
      body: JSON.stringify({
        track_id: trackId,
        event_type: eventType,
        duration_seconds: Math.max(0, Math.floor(duration)),
      }),
    });
  } catch (error) {
    console.warn("Could not save listening event:", error.message);
  }
}

async function playTrack(track, queue = null) {
  if (!track) return;
  if (!track.audio_url) {
    toast("No audio file is available for this track. Add an authorised recording to enable playback.");
    return;
  }
  if (state.currentTrack && state.currentTrack.track_id !== track.track_id && !audio.ended) {
    await recordEvent(state.currentTrack.track_id, "skip", audio.currentTime);
  }
  if (queue) {
    state.queue = queue;
    state.queueIndex = queue.findIndex((item) => item.track_id === track.track_id);
  } else if (!state.queue.some((item) => item.track_id === track.track_id)) {
    state.queue = [track];
    state.queueIndex = 0;
  } else {
    state.queueIndex = state.queue.findIndex((item) => item.track_id === track.track_id);
  }
  state.currentTrack = track;
  $("#playerTitle").textContent = track.title;
  $("#playerArtist").textContent = track.artist;
  $("#playerArtwork").textContent = String(track.title || "S").slice(0, 1).toUpperCase();
  $("#playerLike").disabled = false;
  updatePlayerLike();
  audio.src = track.audio_url;
  audio.load();
  try {
    await audio.play();
    $("#playPauseButton").textContent = "Ⅱ";
    await recordEvent(track.track_id, "play", 0);
    const similar = await api("/api/recommendations/similar/" + encodeURIComponent(track.track_id) + "?limit=4");
    renderTracks("#similarGrid", similar, "No similar tracks found.");
    $("#similarSubtitle").textContent = "Musical neighbours of " + track.title;
    loadStats().catch(() => {});
  } catch (error) {
    toast("Playback was blocked or the audio file is missing.");
    console.error(error);
  }
}

function updatePlayerLike() {
  const liked = state.currentTrack && state.liked.has(state.currentTrack.track_id);
  $("#playerLike").textContent = liked ? "♥" : "♡";
  $("#playerLike").classList.toggle("like-active", Boolean(liked));
}

async function toggleLike(trackId) {
  const wasLiked = state.liked.has(trackId);
  await recordEvent(trackId, wasLiked ? "unlike" : "like");
  if (wasLiked) state.liked.delete(trackId);
  else state.liked.add(trackId);
  updatePlayerLike();
  document.querySelectorAll('[data-like-track="' + CSS.escape(trackId) + '"]').forEach((button) => {
    button.textContent = wasLiked ? "♡" : "♥";
    button.classList.toggle("like-active", !wasLiked);
  });
  toast(wasLiked ? "Removed from liked songs." : "Added to liked songs.");
  loadStats().catch(() => {});
}

async function addTrackToPlaylist(trackId) {
  if (!state.playlists.length) {
    toast("Create a playlist first using the + button in Your playlists.");
    $("#playlistForm").classList.remove("hidden");
    $("#playlistTitle").focus();
    return;
  }
  let playlist = state.playlists[0];
  if (state.playlists.length > 1) {
    const name = window.prompt("Enter the exact playlist name:\n" + state.playlists.map((item) => item.title).join("\n"));
    if (name === null) return;
    playlist = state.playlists.find((item) => item.title.toLowerCase() === name.trim().toLowerCase());
    if (!playlist) {
      toast("Playlist name not found.");
      return;
    }
  }
  try {
    await api("/api/playlists/" + encodeURIComponent(playlist.playlist_id) + "/tracks", {
      method: "POST",
      body: JSON.stringify({ track_id: trackId }),
    });
    toast("Added to " + playlist.title + ".");
    await loadPlaylists();
    loadStats().catch(() => {});
  } catch (error) {
    showError(error);
  }
}

async function searchTracks(query) {
  const tracks = await api("/api/tracks?q=" + encodeURIComponent(query) + "&limit=100");
  $("#searchCount").textContent = tracks.length + " result" + (tracks.length === 1 ? "" : "s");
  renderTracks("#searchGrid", tracks, "No matching tracks. Try an artist, album or genre.");
}

document.addEventListener("click", async (event) => {
  const playButton = event.target.closest("[data-play-track]");
  if (playButton) {
    const trackId = playButton.dataset.playTrack;
    const found = state.catalog.find((item) => item.track_id === trackId);
    if (found) {
      const cards = Array.from(playButton.closest(".track-grid").querySelectorAll(".track-card"));
      const queue = cards.map((card) => {
        const button = card.querySelector("[data-play-track]");
        return button ? state.catalog.find((item) => item.track_id === button.dataset.playTrack) : null;
      }).filter(Boolean);
      await playTrack(found, queue.length ? queue : [found]);
    }
    return;
  }

  const likeButton = event.target.closest("[data-like-track]");
  if (likeButton) {
    await toggleLike(likeButton.dataset.likeTrack).catch(showError);
    return;
  }

  const addButton = event.target.closest("[data-add-track]");
  if (addButton) {
    await addTrackToPlaylist(addButton.dataset.addTrack);
    return;
  }

  const genreButton = event.target.closest("[data-genre]");
  if (genreButton) {
    await showGenre(genreButton.dataset.genre).catch(showError);
    return;
  }

  const playlistButton = event.target.closest("[data-open-playlist]");
  if (playlistButton) {
    await openPlaylist(playlistButton.dataset.openPlaylist).catch(showError);
    return;
  }

  const navButton = event.target.closest("[data-view]");
  if (navButton) switchView(navButton.dataset.view);
});

$("#loginForm").addEventListener("submit", async (event) => {
  event.preventDefault();
  try { await login($("#loginUsername").value.trim(), $("#loginPassword").value); }
  catch (error) { showError(error); }
});

$("#demoLogin").addEventListener("click", async () => {
  $("#loginUsername").value = "demo";
  $("#loginPassword").value = "music12345";
  try { await login("demo", "music12345"); } catch (error) { showError(error); }
});

$("#registerForm").addEventListener("submit", async (event) => {
  event.preventDefault();
  try {
    await register(
      $("#registerUsername").value.trim(),
      $("#registerEmail").value.trim(),
      $("#registerPassword").value
    );
  } catch (error) { showError(error); }
});

$("#logoutButton").addEventListener("click", () => logout());
$("#modelSelect").addEventListener("change", () => loadHome().catch(showError));
$("#newPlaylistToggle").addEventListener("click", () => {
  $("#playlistForm").classList.toggle("hidden");
  $("#playlistTitle").focus();
});

$("#playlistForm").addEventListener("submit", async (event) => {
  event.preventDefault();
  try {
    await api("/api/playlists", {
      method: "POST",
      body: JSON.stringify({ title: $("#playlistTitle").value.trim(), description: "" }),
    });
    $("#playlistTitle").value = "";
    $("#playlistForm").classList.add("hidden");
    await loadPlaylists();
    toast("Playlist created.");
  } catch (error) { showError(error); }
});

let searchTimer;
$("#searchInput").addEventListener("input", () => {
  switchView("search");
  window.clearTimeout(searchTimer);
  searchTimer = window.setTimeout(() => searchTracks($("#searchInput").value.trim()).catch(showError), 180);
});
$("#clearSearch").addEventListener("click", () => {
  $("#searchInput").value = "";
  searchTracks("").catch(showError);
});

$("#playRecommendations").addEventListener("click", async () => {
  try {
    const items = await api("/api/recommendations/" + $("#modelSelect").value + "?limit=20");
    if (items.length) await playTrack(items[0], items);
    else toast("No recommendations yet. Try exploring the collection.");
  } catch (error) { showError(error); }
});

$("#playerLike").addEventListener("click", () => {
  if (state.currentTrack) toggleLike(state.currentTrack.track_id).catch(showError);
});

$("#playPauseButton").addEventListener("click", async () => {
  if (!state.currentTrack) {
    $("#playRecommendations").click();
    return;
  }
  if (audio.paused) {
    try { await audio.play(); $("#playPauseButton").textContent = "Ⅱ"; }
    catch { toast("Could not start playback."); }
  } else {
    audio.pause();
    $("#playPauseButton").textContent = "▶";
  }
});

$("#nextButton").addEventListener("click", () => moveQueue(1));
$("#previousButton").addEventListener("click", () => {
  if (audio.currentTime > 3) {
    audio.currentTime = 0;
    return;
  }
  moveQueue(-1);
});

$("#shuffleButton").addEventListener("click", () => {
  state.shuffle = !state.shuffle;
  $("#shuffleButton").classList.toggle("like-active", state.shuffle);
  toast(state.shuffle ? "Shuffle is on." : "Shuffle is off.");
});

$("#repeatButton").addEventListener("click", () => {
  state.repeat = !state.repeat;
  audio.loop = state.repeat;
  $("#repeatButton").classList.toggle("like-active", state.repeat);
  toast(state.repeat ? "Repeat is on." : "Repeat is off.");
});

$("#queueButton").addEventListener("click", () => {
  toast(state.queue.length ? state.queue.length + " tracks in the current queue." : "Your queue is empty.");
});
$("#volumeInput").addEventListener("input", (event) => { audio.volume = Number(event.target.value); });
audio.volume = Number($("#volumeInput").value);

$("#progressInput").addEventListener("input", (event) => {
  if (Number.isFinite(audio.duration) && audio.duration > 0) {
    audio.currentTime = Number(event.target.value) / 100 * audio.duration;
  }
});
audio.addEventListener("timeupdate", () => {
  const duration = Number.isFinite(audio.duration) ? audio.duration : 0;
  $("#progressInput").value = duration ? audio.currentTime / duration * 100 : 0;
  $("#currentTime").textContent = formatTime(audio.currentTime);
  $("#totalTime").textContent = formatTime(duration);
});
audio.addEventListener("play", () => { $("#playPauseButton").textContent = "Ⅱ"; });
audio.addEventListener("pause", () => { $("#playPauseButton").textContent = "▶"; });
audio.addEventListener("ended", async () => {
  if (state.currentTrack) await recordEvent(state.currentTrack.track_id, "complete", audio.duration || 0);
  if (!state.repeat) await moveQueue(1, true);
  loadStats().catch(() => {});
});
audio.addEventListener("error", () => toast("This audio file could not be played."));

function formatTime(seconds) {
  if (!Number.isFinite(seconds) || seconds < 0) return "0:00";
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return mins + ":" + String(secs).padStart(2, "0");
}

async function moveQueue(direction, ended = false) {
  if (!state.queue.length) return;
  let next = state.queueIndex + direction;
  if (state.shuffle && direction > 0 && state.queue.length > 1) {
    next = Math.floor(Math.random() * state.queue.length);
  }
  if (next < 0 || next >= state.queue.length) {
    if (!ended) toast("You've reached the end of this queue.");
    return;
  }
  state.queueIndex = next;
  await playTrack(state.queue[next], state.queue);
}

$("#backButton").addEventListener("click", () => window.history.back());
$("#forwardButton").addEventListener("click", () => window.history.forward());
document.addEventListener("keydown", (event) => {
  if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "k") {
    event.preventDefault();
    $("#searchInput").focus();
  }
});

(async function initialise() {
  if (!state.token) {
    showAuth();
    return;
  }
  try { await enterApp(); }
  catch { logout(false); }
})();