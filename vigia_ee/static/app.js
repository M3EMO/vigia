/* Panel de Vigía: mejoras progresivas. Todo funciona sin este archivo (AT-E07).
   Reglas: el texto del modelo entra solo con textContent o nodos de texto; sin eval ni handlers inline. */
(function () {
  "use strict";

  var THEME_KEY = "vigia-theme";
  var root = document.documentElement;
  root.classList.add("js");

  function readTheme() {
    try { return window.localStorage.getItem(THEME_KEY); } catch (e) { return null; }
  }
  function writeTheme(value) {
    try {
      if (value) { window.localStorage.setItem(THEME_KEY, value); } else { window.localStorage.removeItem(THEME_KEY); }
    } catch (e) { /* almacenamiento bloqueado: el tema vale solo para esta página */ }
  }
  function applyTheme(value) {
    if (value === "light" || value === "dark") { root.setAttribute("data-theme", value); }
    else { root.removeAttribute("data-theme"); }
  }
  applyTheme(readTheme());

  var LEVELS = [[0.8, "high", "alto"], [0.5, "medium", "medio"], [0, "low", "bajo"]];
  function levelFor(score) {
    for (var i = 0; i < LEVELS.length; i++) { if (score >= LEVELS[i][0]) { return LEVELS[i]; } }
    return LEVELS[LEVELS.length - 1];
  }
  function fmt(x, digits) {
    if (typeof x !== "number" || !isFinite(x)) { return "s/d"; }
    return x.toFixed(digits).replace(".", ",");
  }
  function typing(el) {
    if (!el) { return false; }
    var tag = el.tagName;
    return tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT" || el.isContentEditable;
  }
  function el(tag, cls, text) {
    var node = document.createElement(tag);
    if (cls) { node.className = cls; }
    if (text !== undefined && text !== null) { node.textContent = text; }
    return node;
  }

  /* --- tema y navegación ------------------------------------------------------------------- */

  function initTheme() {
    var btn = document.querySelector("[data-theme-toggle]");
    if (!btn) { return; }
    var order = [null, "light", "dark"];
    var names = { "null": "sistema", "light": "claro", "dark": "oscuro" };
    function label() { btn.textContent = "Tema: " + names[String(readThemeSafe())]; }
    function readThemeSafe() { var t = root.getAttribute("data-theme"); return t === "light" || t === "dark" ? t : null; }
    btn.hidden = false;
    label();
    btn.addEventListener("click", function () {
      var next = order[(order.indexOf(readThemeSafe()) + 1) % order.length];
      applyTheme(next);
      writeTheme(next);
      label();
    });
  }

  function initNav() {
    var btn = document.querySelector("[data-nav-toggle]");
    var nav = document.getElementById("vg-nav");
    if (!btn || !nav) { return; }
    btn.addEventListener("click", function () {
      var open = nav.classList.toggle("is-open");
      btn.setAttribute("aria-expanded", open ? "true" : "false");
      if (open) { var first = nav.querySelector("a"); if (first) { first.focus(); } }
    });
  }

  function initCopy() {
    document.querySelectorAll("[data-copy]").forEach(function (btn) {
      if (!navigator.clipboard) { return; }
      btn.hidden = false;
      btn.addEventListener("click", function () {
        var target = document.getElementById(btn.getAttribute("data-copy"));
        var status = btn.parentNode.querySelector("[data-copy-status]");
        navigator.clipboard.writeText(target ? target.textContent : "").then(function () {
          if (status) { status.textContent = "Copiado al portapapeles."; }
        }, function () {
          if (status) { status.textContent = "No se pudo copiar; seleccioná el texto y copialo a mano."; }
        });
      });
    });
  }

  function openDialog(id) {
    var dialog = document.getElementById(id);
    if (dialog && typeof dialog.showModal === "function" && !dialog.open) { dialog.showModal(); }
  }

  function initDialogs() {
    document.querySelectorAll("[data-open-dialog]").forEach(function (btn) {
      var dialog = document.getElementById(btn.getAttribute("data-open-dialog"));
      if (!dialog || typeof dialog.showModal !== "function") { return; }
      btn.hidden = false;
      btn.addEventListener("click", function () { openDialog(dialog.id); });
    });
  }

  /* --- detalle: fragmentos, señales y umbral de vista ----------------------------------------- */

  function initDetail() {
    var container = document.querySelector("body[data-page='detail'] [data-marked-text]");
    var marks = container ? Array.prototype.slice.call(container.querySelectorAll("mark[data-fragment]")) : [];

    function showSignals(n, focus) {
      var items = document.querySelectorAll("[data-signals]");
      if (!items.length) { return; }
      items.forEach(function (item) { item.classList.toggle("is-current", item.getAttribute("data-signals") === n); });
      marks.forEach(function (m) { m.classList.toggle("is-current", m.getAttribute("data-fragment") === n); });
      if (focus) {
        var heading = document.querySelector("#sig-" + n + " h3 a");
        if (heading) { heading.focus(); }
      }
    }
    function move(step) {
      if (!marks.length) { return; }
      var idx = marks.indexOf(document.activeElement);
      var next = idx === -1 ? (step > 0 ? 0 : marks.length - 1) : (idx + step + marks.length) % marks.length;
      marks[next].focus();
      showSignals(marks[next].getAttribute("data-fragment"), false);
    }
    marks.forEach(function (m) {
      m.addEventListener("click", function () { showSignals(m.getAttribute("data-fragment"), false); });
      m.addEventListener("focus", function () { showSignals(m.getAttribute("data-fragment"), false); });
    });
    if (document.body.getAttribute("data-page") === "detail") {
      document.addEventListener("keydown", function (ev) {
        if (ev.ctrlKey || ev.metaKey || ev.altKey || typing(ev.target) || document.querySelector("dialog[open]")) { return; }
        if (ev.key === "j") { ev.preventDefault(); move(1); }
        else if (ev.key === "k") { ev.preventDefault(); move(-1); }
        else if (ev.key === "Enter" && marks.indexOf(document.activeElement) !== -1) {
          ev.preventDefault();
          showSignals(document.activeElement.getAttribute("data-fragment"), true);
        } else if (ev.key === "?") { ev.preventDefault(); openDialog("shortcuts"); }
      });
    }

    var form = document.querySelector("[data-threshold-form]");
    if (form) {
      var range = form.querySelector("[data-threshold-range]");
      var number = form.querySelector("input[name='umbral']");
      if (range && number) {
        range.hidden = false;
        // WCAG 3.2.2: con el teclado, cada flecha dispara 'change'. Enviar ahí recarga la página en
        // cada paso y el foco cae al body. Solo se envía al soltar el puntero o con Enter.
        var pointer = false;
        var send = function () { if (form.requestSubmit) { form.requestSubmit(); } else { form.submit(); } };
        range.addEventListener("input", function () { number.value = Number(range.value).toFixed(2); });
        range.addEventListener("pointerdown", function () { pointer = true; });
        range.addEventListener("change", function () {
          if (!pointer) { return; }
          pointer = false;
          send();
        });
        range.addEventListener("keydown", function (ev) {
          pointer = false;
          if (ev.key === "Enter") { ev.preventDefault(); send(); }
        });
        number.addEventListener("input", function () {
          var v = parseFloat(number.value.replace(",", "."));
          if (isFinite(v) && v >= 0 && v <= 1) { range.value = String(v); }
        });
      }
    }
  }

  /* --- cola de revisión: 1/2/3, Ctrl+Enter, flechas ------------------------------------------- */

  function initReview() {
    if (document.body.getAttribute("data-page") !== "review-item") { return; }
    var form = document.querySelector("[data-review-form]");
    document.addEventListener("keydown", function (ev) {
      if (document.querySelector("dialog[open]")) { return; }
      if (form && (ev.ctrlKey || ev.metaKey) && ev.key === "Enter") {
        var next = form.querySelector("[data-submit-next]");
        if (next) { ev.preventDefault(); next.click(); }
        return;
      }
      if (ev.ctrlKey || ev.metaKey || ev.altKey || typing(ev.target) && ev.target.type !== "radio") { return; }
      if (form && (ev.key === "1" || ev.key === "2" || ev.key === "3")) {
        var radio = form.querySelector("input[data-key='" + ev.key + "']");
        if (radio) { ev.preventDefault(); radio.checked = true; radio.focus(); }
      } else if (ev.key === "ArrowLeft" || ev.key === "ArrowRight") {
        if (ev.target && ev.target.type === "radio") { return; }
        var link = document.querySelector(ev.key === "ArrowLeft" ? "[data-nav-prev]" : "[data-nav-next]");
        if (link) { ev.preventDefault(); window.location.assign(link.href); }
      } else if (ev.key === "?") { ev.preventDefault(); openDialog("shortcuts"); }
    });
  }

  /* --- playground con streaming SSE ----------------------------------------------------------- */

  function renderFinal(target, text, spans, prefix) {
    var chars = Array.from(text);
    var box = el("div", "vg-response");
    box.setAttribute("data-marked-text", "");
    var descs = el("div");
    descs.hidden = true;
    var pos = 0;
    var sorted = (spans || []).slice().sort(function (a, b) { return a.start - b.start; });
    sorted.forEach(function (sp, i) {
      var start = Math.max(pos, Math.min(sp.start, chars.length));
      var end = Math.min(sp.end, chars.length);
      if (end <= start) { return; }
      if (start > pos) { box.appendChild(document.createTextNode(chars.slice(pos, start).join(""))); }
      var lvl = levelFor(sp.score);
      var n = i + 1;
      var mark = el("mark", "vg-mark vg-lvl-" + lvl[1], chars.slice(start, end).join(""));
      mark.id = prefix + "-" + n;
      mark.tabIndex = 0;
      mark.setAttribute("aria-describedby", prefix + "-" + n + "-desc");
      box.appendChild(mark);
      var desc = el("span", null, "Fragmento dudoso " + n + ", nivel " + lvl[2] + ", P(error) " + fmt(sp.score, 2) + ".");
      desc.id = prefix + "-" + n + "-desc";
      descs.appendChild(desc);
      pos = end;
    });
    if (pos < chars.length) { box.appendChild(document.createTextNode(chars.slice(pos).join(""))); }
    target.appendChild(box);
    target.appendChild(descs);
  }

  var ACTION_LABELS = { pass: "Pasa", annotate: "Anotar", human_review: "Revisión humana", rag_retry: "Re-consulta RAG", abstain: "Abstención" };
  // Mismas clases que la macro action_badge de _macros.html.
  var ACTION_BADGES = { pass: "vg-badge", annotate: "vg-badge vg-badge-info", human_review: "vg-badge vg-badge-warn", rag_retry: "vg-badge vg-badge-warn", abstain: "vg-badge vg-badge-danger" };

  function initPlayground() {
    var form = document.querySelector("[data-playground-form]");
    var output = document.querySelector("[data-playground-output]");
    if (!form || !output || !window.fetch || !window.TextDecoder) { return; }
    var status = output.querySelector("[data-playground-status]");
    var errorBox = output.querySelector("[data-playground-error]");
    var result = output.querySelector("[data-playground-result]");
    var submit = form.querySelector("[data-playground-submit]");
    var busy = false;

    function clearFieldErrors() {
      form.querySelectorAll("[data-js-error]").forEach(function (n) { n.remove(); });
      form.querySelectorAll("[aria-invalid='true']").forEach(function (n) { n.removeAttribute("aria-invalid"); n.removeAttribute("aria-describedby"); });
    }
    function showError(message) {
      errorBox.textContent = "";
      var alert = el("div", "vg-alert vg-alert-error");
      alert.setAttribute("role", "alert");
      alert.appendChild(el("p", null, message));
      var retry = el("button", "vg-btn vg-btn-secondary vg-btn-small", "Reintentar");
      retry.type = "button";
      retry.addEventListener("click", function () { form.requestSubmit ? form.requestSubmit() : start(); });
      var p = el("p");
      p.appendChild(retry);
      alert.appendChild(p);
      errorBox.appendChild(alert);
    }
    function finish() {
      busy = false;
      output.setAttribute("aria-busy", "false");
      if (submit) { submit.removeAttribute("aria-disabled"); }
    }

    function start() {
      if (busy) { return; }
      clearFieldErrors();
      var data = new FormData(form);
      var payload = {
        prompt: String(data.get("prompt") || ""),
        system: String(data.get("system") || ""),
        max_tokens: String(data.get("max_tokens") || ""),
        temperature: String(data.get("temperature") || "")
      };
      busy = true;
      output.setAttribute("aria-busy", "true");
      // aria-disabled y no disabled: un botón deshabilitado pierde el foco y Chrome lo manda a <body>.
      // `busy` ya bloquea el doble envío.
      if (submit) { submit.setAttribute("aria-disabled", "true"); }
      errorBox.textContent = "";
      result.textContent = "";
      status.textContent = "Generando";
      var live = el("div", "vg-response vg-stream-out");
      result.appendChild(live);
      var nTokens = 0;
      var nFlagged = 0;
      var done = null;
      fetch("/playground/stream", {
        method: "POST",
        credentials: "same-origin",
        headers: {
          "Content-Type": "application/json",
          "Accept": "text/event-stream",
          "X-CSRF-Token": String(data.get("csrf_token") || "")
        },
        body: JSON.stringify(payload)
      }).then(function (resp) {
        if (resp.status === 422) {
          return resp.json().then(function (body) {
            var fields = (body && body.fields) || {};
            Object.keys(fields).forEach(function (name) {
              var input = form.querySelector("[name='" + name + "']");
              if (!input) { return; }
              var msg = el("p", "vg-field-error", fields[name]);
              msg.id = "js-err-" + name;
              msg.setAttribute("data-js-error", "");
              input.setAttribute("aria-invalid", "true");
              input.setAttribute("aria-describedby", msg.id);
              input.insertAdjacentElement("afterend", msg);
            });
            throw new Error("Revisá los campos marcados.");
          });
        }
        if (resp.status === 401) { throw new Error("La sesión venció. Volvé a ingresar."); }
        if (resp.status === 403) { throw new Error("Tu rol no puede usar el playground o el token de seguridad venció; recargá la página."); }
        if (resp.status === 429) {
          return resp.json().catch(function () { return null; }).then(function (body) {
            throw new Error((body && body.message) || "Tu organización alcanzó el límite de generaciones por minuto.");
          });
        }
        if (!resp.ok || !resp.body) { throw new Error("El servidor respondió con un error (" + resp.status + ")."); }
        var reader = resp.body.getReader();
        var decoder = new TextDecoder("utf-8");
        var buffer = "";
        function handle(event) {
          if (event.type === "token") {
            nTokens += 1;
            if (event.flag) {
              nFlagged += 1;
              live.appendChild(el("span", "vg-token-flag", event.text));
            } else {
              live.appendChild(document.createTextNode(event.text));
            }
            status.textContent = "Generando: " + nTokens + " tokens";
          } else if (event.type === "done") {
            done = event;
          } else if (event.type === "error") {
            throw new Error(event.message || "La generación falló.");
          }
        }
        function pump() {
          return reader.read().then(function (chunk) {
            if (chunk.done) { return; }
            buffer += decoder.decode(chunk.value, { stream: true });
            var parts = buffer.split("\n\n");
            buffer = parts.pop();
            parts.forEach(function (part) {
              part.split("\n").forEach(function (line) {
                if (line.indexOf("data: ") === 0) { handle(JSON.parse(line.slice(6))); }
              });
            });
            return pump();
          });
        }
        return pump();
      }).then(function () {
        if (!done) { throw new Error("La generación terminó sin resultado."); }
        result.textContent = "";
        renderFinal(result, done.text || live.textContent, done.spans, "pg");
        var meta = el("dl", "vg-meta");
        var badge = el("span", ACTION_BADGES[done.action] || "vg-badge", ACTION_LABELS[done.action] || done.action);
        [["Acción", badge],
         ["P(error) respuesta", fmt(done.response_p_error, 3)],
         ["Calibrado", done.calibrated ? "sí" : "no: sin detector activo no hay resaltado"]].forEach(function (row) {
          meta.appendChild(el("dt", null, row[0]));
          var dd = el("dd");
          if (typeof row[1] === "string") { dd.textContent = row[1]; } else { dd.appendChild(row[1]); }
          meta.appendChild(dd);
        });
        result.appendChild(meta);
        if (done.trace_id) {
          var p = el("p");
          var a = el("a", null, "Abrir el detalle con señales");
          a.href = "/generations/" + encodeURIComponent(done.trace_id);
          p.appendChild(a);
          result.appendChild(p);
        }
        status.textContent = "Listo: " + nTokens + " tokens, " + nFlagged + " marcados.";
      }).catch(function (err) {
        status.textContent = "La generación no terminó.";
        showError(err && err.message ? err.message : "La generación falló.");
      }).then(finish);
    }

    form.addEventListener("submit", function (ev) {
      ev.preventDefault();
      start();
    });
  }

  /* --- estado del job de calibración ---------------------------------------------------------- */

  function initJob() {
    var box = document.querySelector("[data-job-status]");
    if (!box || box.getAttribute("aria-busy") !== "true" || !window.fetch) { return; }
    var url = box.getAttribute("data-job-status");
    var label = box.querySelector("[data-job-label]");
    var errorBox = box.querySelector("[data-job-error]");
    var resultBox = box.querySelector("[data-job-result]");
    var refresh = box.querySelector("[data-job-refresh]");
    var failures = 0;
    function poll() {
      fetch(url, { credentials: "same-origin", headers: { "Accept": "application/json" } })
        .then(function (resp) { if (!resp.ok) { throw new Error(String(resp.status)); } return resp.json(); })
        .then(function (job) {
          failures = 0;
          if (job.state === "done" || job.state === "failed") {
            box.setAttribute("aria-busy", "false");
            label.textContent = job.label;
            var title = box.getAttribute("data-title-" + job.state);
            if (title) { document.title = title; }
            if (refresh) { refresh.parentNode.hidden = true; }
            if (job.error) {
              var alert = el("div", "vg-alert vg-alert-error");
              alert.setAttribute("role", "alert");
              alert.appendChild(el("p", null, job.error));
              errorBox.textContent = "";
              errorBox.appendChild(alert);
            }
            if (job.detector_id) {
              var a = el("a", "vg-btn", "Ver el reporte del detector");
              a.href = "/calibration/detectors/" + encodeURIComponent(job.detector_id);
              resultBox.textContent = "";
              resultBox.appendChild(a);
            }
            return;
          }
          label.textContent = job.label + ": el entrenamiento corre en segundo plano.";
          window.setTimeout(poll, 3000);
        })
        .catch(function () {
          failures += 1;
          if (failures >= 3) {
            box.setAttribute("aria-busy", "false");
            label.textContent = "No se pudo consultar el estado. Usá «Actualizar estado».";
            return;
          }
          window.setTimeout(poll, 5000);
        });
    }
    window.setTimeout(poll, 2000);
  }

  function init() {
    initTheme();
    initNav();
    initCopy();
    initDialogs();
    initDetail();
    initReview();
    initPlayground();
    initJob();
  }

  if (document.readyState === "loading") { document.addEventListener("DOMContentLoaded", init); }
  else { init(); }
})();
