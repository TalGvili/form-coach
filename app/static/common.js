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
