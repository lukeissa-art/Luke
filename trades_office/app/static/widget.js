/* Website chat button for trades shops.
   Usage: <script src="https://YOUR-APP/widget.js" data-shop="WIDGET_KEY" async></script>
   Optional: data-label="Book a visit"  data-color="#0f766e"  data-position="left" */
(function () {
  if (window.__tdChatLoaded) return;
  window.__tdChatLoaded = true;

  var script = document.currentScript || (function () {
    var all = document.querySelectorAll("script[data-shop]");
    return all[all.length - 1];
  })();
  if (!script || !script.getAttribute("data-shop")) return;

  var key = script.getAttribute("data-shop");
  var origin = new URL(script.src, location.href).origin;
  var color = script.getAttribute("data-color") || "#111c2e";
  var label = script.getAttribute("data-label") || "Chat with us";
  var side = script.getAttribute("data-position") === "left" ? "left" : "right";

  function start() {
    var btn = document.createElement("button");
    btn.type = "button";
    btn.setAttribute("aria-expanded", "false");
    btn.setAttribute("aria-label", label);
    btn.textContent = label;
    btn.style.cssText = [
      "position:fixed", side + ":16px", "bottom:16px", "z-index:2147483000",
      "background:" + color, "color:#fff", "border:0", "border-radius:999px",
      "padding:14px 20px", "font:600 16px/1 system-ui,-apple-system,'Segoe UI',sans-serif",
      "box-shadow:0 6px 20px rgba(0,0,0,.25)", "cursor:pointer", "max-width:calc(100vw - 32px)"
    ].join(";");

    var panel = document.createElement("div");
    panel.style.cssText = [
      "position:fixed", side + ":16px", "bottom:80px", "z-index:2147483001",
      "width:min(380px, calc(100vw - 32px))", "height:min(600px, calc(100vh - 110px))",
      "border-radius:14px", "overflow:hidden", "box-shadow:0 12px 40px rgba(0,0,0,.3)",
      "background:#fff", "display:none"
    ].join(";");
    var frame = null;

    function open() {
      if (!frame) {
        frame = document.createElement("iframe");
        frame.src = origin + "/chat/" + encodeURIComponent(key) + "?embed=1&color=" + encodeURIComponent(color);
        frame.title = "Chat";
        frame.setAttribute("allow", "clipboard-write");
        frame.style.cssText = "border:0;width:100%;height:100%;display:block";
        panel.appendChild(frame);
      }
      panel.style.display = "block";
      btn.setAttribute("aria-expanded", "true");
    }
    function close() {
      panel.style.display = "none";
      btn.setAttribute("aria-expanded", "false");
      btn.focus();
    }
    btn.addEventListener("click", function () { panel.style.display === "none" ? open() : close(); });
    window.addEventListener("message", function (e) {
      if (e.origin === origin && e.data && e.data.type === "tradedesk-chat-close") close();
    });
    document.addEventListener("keydown", function (e) {
      if (e.key === "Escape" && panel.style.display !== "none") close();
    });

    document.body.appendChild(panel);
    document.body.appendChild(btn);
  }

  if (document.body) start();
  else document.addEventListener("DOMContentLoaded", start);
})();
