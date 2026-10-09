// Helpers shared by index.html and history.html.

const $ = (id) => document.getElementById(id);

// Builds an element with its text set through textContent, never innerHTML: text is shown
// as text, so nothing in a response can be run as HTML.
function el(tag, className = "", text = "") {
  const node = document.createElement(tag);
  if (className) node.className = className;
  node.textContent = text;
  return node;
}

// Seconds as m:ss.
function clock(seconds) {
  const s = Math.floor(seconds);
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}

// A colour token from style.css, for Chart.js, which needs real colour values.
function css(token) {
  return getComputedStyle(document.documentElement).getPropertyValue(token).trim();
}

// "9 Oct, 14:53" from a stored ISO time, in the viewer's own time zone.
function when(iso) {
  return new Date(iso).toLocaleString(undefined, {
    day: "numeric", month: "short", hour: "2-digit", minute: "2-digit",
  });
}

// The profile chosen on the landing page (profiles.html), remembered by this browser. It's a
// name, not a login: it decides whose history uploads go to and which history is shown.
// Storage can be blocked (private windows), so every access is allowed to fail.
function currentProfile() {
  try { return localStorage.getItem("profile"); } catch { return null; }
}

function chooseProfile(name) {
  try { localStorage.setItem("profile", name); return true; } catch { return false; }
}

// Pages that work on a profile send a visitor without one to the landing page, which sends
// them back here once they've picked.
function requireProfile() {
  const name = currentProfile();
  if (!name) {
    const here = location.pathname + location.search;
    location.replace(`profiles.html?next=${encodeURIComponent(here)}`);
  }
  return name;
}

// A circle with the name's first letter, in a colour that is always the same for that name.
const AVATAR_COLOURS = ["#2a78d6", "#0f8a7a", "#7a5cd6", "#d0682f", "#c8457a", "#3b8a3b"];

function avatar(name, size = "") {
  const sum = [...name].reduce((total, char) => total + char.codePointAt(0), 0);
  const circle = el("span", `avatar ${size}`.trim(), [...name][0].toUpperCase());
  circle.style.background = AVATAR_COLOURS[sum % AVATAR_COLOURS.length];
  circle.setAttribute("aria-hidden", "true");
  return circle;
}

// The header's "who am I" chip: the avatar, the name, and a way to switch.
function showProfileChip(name) {
  const chip = $("me");
  if (!chip || !name) return;
  const change = el("a", "", "Switch");
  change.href = "profiles.html";
  chip.replaceChildren(avatar(name), el("strong", "", name), change);
}
