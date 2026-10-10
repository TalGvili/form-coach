// Helpers shared by index.html, history.html and profiles.html.

const $ = (id) => document.getElementById(id);

// The static demo (GitHub Pages) is these same pages, built with <html data-demo> by
// scripts/build_site.py. There is no server there: the pages read the bundled demo/*.json
// instead, and uploading and profiles are switched off.
const DEMO = document.documentElement.hasAttribute("data-demo");
const DEMO_PROFILE = "Demo";

// Google sign-in is on when the server's config.js gave a client id. Otherwise the app is in
// local mode, with profiles that are names, not accounts.
const SIGN_IN = !DEMO && Boolean(window.FORM_COACH?.googleClientId);

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

// Who the page works for, as { name }, or null after sending the visitor to the landing page
// (which sends them back here). With sign-in on, the server says: the login cookie is HttpOnly,
// so no script, ours included, can read it. In local mode, the profile this browser remembers.
async function identify() {
  if (DEMO) return { name: DEMO_PROFILE };
  if (SIGN_IN) {
    const response = await fetch("/me");
    if (response.ok) return await response.json();
  } else {
    const name = currentProfile();
    if (name) return { name };
  }
  const here = location.pathname + location.search;
  location.replace(`profiles.html?next=${encodeURIComponent(here)}`);
  return null;
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

// The header's "who am I" chip: the avatar, the name, and Switch (local mode) or Sign out.
function showProfileChip(name) {
  const chip = $("me");
  if (!chip || !name) return;
  chip.replaceChildren(avatar(name), el("strong", "", name));
  if (DEMO) return; // nothing to switch to
  const change = el("a", "", SIGN_IN ? "Sign out" : "Switch");
  change.href = "profiles.html";
  if (SIGN_IN) {
    change.addEventListener("click", async (event) => {
      event.preventDefault();
      await fetch("/auth/logout", { method: "POST" }); // ends the login on the server
      location.href = "profiles.html";
    });
  }
  chip.append(change);
}

// The demo's banner: what this page shows, and where the real thing is.
function showDemoBanner(what) {
  if (!DEMO) return;
  const banner = $("demo-banner");
  const repo = el("a", "", "run the app yourself");
  repo.href = "https://github.com/TalGvili/form-coach";
  banner.append(
    el("strong", "", "Demo. "),
    `${what} To analyse your own video, `,
    repo,
    ".",
  );
  banner.hidden = false;
}
