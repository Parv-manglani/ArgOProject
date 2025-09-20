/* FloatChat AI Frontend JavaScript */
class FloatChatApp {
  constructor() {
    // === API configuration (relative paths for same-origin) ===
    this.apiBase = "";
    this.api = {
      chat: "/api/chat",
      history: (sid) => `/api/chat/history/${encodeURIComponent(sid)}`,
      upload: "/api/data/upload",
      files: "/api/data/files",
      health: "/api/health",
      azure_ping: "/api/azure/ping",
      variables: (fid) => `/api/data/variables?file_id=${encodeURIComponent(fid)}`,
      preview: (fid, var_name) =>
        `/api/data/preview?file_id=${encodeURIComponent(fid)}&var=${encodeURIComponent(var_name)}`,
      plot: {
        profile: (fid) => `/api/plot/profile${fid ? `?file_id=${encodeURIComponent(fid)}` : ""}`,
        timeseries: (fid, var_name, depth) =>
          `/api/plot/timeseries?var=${encodeURIComponent(var_name)}&depth=${depth}${
            fid ? `&file_id=${encodeURIComponent(fid)}` : ""
          }`,
        map: (fid) => `/api/plot/map${fid ? `?file_id=${encodeURIComponent(fid)}` : ""}`,
        scatter: (fid, x_var, y_var) =>
          `/api/plot/scatter?x_var=${encodeURIComponent(x_var)}&y_var=${encodeURIComponent(y_var)}${
            fid ? `&file_id=${encodeURIComponent(fid)}` : ""
          }`,
      },
    };

    // If your /api/data/files endpoint returns global files, set this true to list them.
    this.USE_GLOBAL_FILE_LIST = false;

    this.currentSessionId = null;
    this.sessions = [];            // list of session IDs for the left sidebar
    this.uploadedFiles = [];       // { file_id, filename, ... } scoped to THIS session
    this.chatHistory = [];
    this.varsCache = new Map();    // per-file variable list cache

    // Viz intent params (filled by chat prompts or UI)
    this.vizParams = {
      type: "map",
      fileId: null,
      var: "TEMP",    // for timeseries/profile
      depth: 50,      // dbar (≈ m)
      xVar: "PSAL",   // for scatter
      yVar: "TEMP",
    };

    // DOM helper
    this.$ = (id) => document.getElementById(id);

    this.init();
  }

  // =================== Init / Events ===================
  init() {
    this.bindEvents();
    this.checkSystemHealth();

    if (this.USE_GLOBAL_FILE_LIST) this.loadUploadedFiles();

    const saved = localStorage.getItem("floatchat_session");
    if (saved) {
      this.currentSessionId = saved;
      this.ensureSessionItem(saved, true);
      this.loadHistory(saved);
    } else {
      this.createNewSession();
    }

    this.autoResizeTextarea();
  }

  bindEvents() {
    // Guarded addEventListener helper
    const on = (id, evt, fn) => {
      const el = this.$(id);
      if (el) el.addEventListener(evt, fn);
    };

    // Chat
    on("sendBtn", "click", () => this.sendMessage());
    const chatInput = this.$("chatInput");
    if (chatInput) {
      chatInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
          e.preventDefault();
          this.sendMessage();
        }
      });
    }

    // Sessions
    on("newChatBtn", "click", () => this.createNewSession());
    on("clearChatBtn", "click", () => this.clearCurrentChat());
    on("exportChatBtn", "click", () => this.exportChat());

    // Upload
    const uploadArea = this.$("uploadArea");
    const fileInput = this.$("fileInput");
    on("attachBtn", "click", () => fileInput && fileInput.click());

    if (uploadArea && fileInput) {
      uploadArea.addEventListener("click", () => fileInput.click());
      uploadArea.addEventListener("dragover", (e) => this.handleDragOver(e));
      uploadArea.addEventListener("dragleave", (e) => this.handleDragLeave(e));
      uploadArea.addEventListener("drop", (e) => this.handleFileDrop(e));
      fileInput.addEventListener("change", (e) => this.handleFileSelect(e));
    }

    // Navigation & Modals
    on("aboutBtn", "click", () => this.showModal("aboutModal"));
    on("dataBtn", "click", () => this.toggleVizPanel());
    // IMPORTANT: id is "closevizBtn" in HTML (lowercase 'v' after close)
    on("closevizBtn", "click", () => this.toggleVizPanel(false));

    // Close modals
    document.querySelectorAll(".modal-close").forEach((btn) => {
      btn.addEventListener("click", (e) => {
        const modal = e.target.closest(".modal");
        if (modal?.id) this.hideModal(modal.id);
      });
    });
    // Click outside + ESC
    document.querySelectorAll(".modal").forEach((modal) => {
      modal.addEventListener("click", (e) => {
        if (e.target === modal) this.hideModal(modal.id);
      });
    });
    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape") {
        document.querySelectorAll(".modal.active").forEach((m) => this.hideModal(m.id));
      }
    });

    // Visualization
    on("generateVizBtn", "click", () => this.generateVisualization());
  }

  // =================== Chat ===================
  async sendMessage() {
    const input = this.$("chatInput");
    if (!input) return;

    const message = input.value.trim();
    if (!message) return;

    // Try prompt-driven viz first
    if (this.detectVisualizationCommand(message)) {
      this.addMessage("user", message);
      input.value = "";
      return;
    }

    input.value = "";
    this.showTypingIndicator(true);
    this.addMessage("user", message);

    try {
      const res = await fetch(this.api.chat, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message,
          session_id: this.currentSessionId,
          context: this.getRecentContext(),
        }),
      });

      let data;
      try {
        data = await res.json();
      } catch {
        throw new Error(`Server returned ${res.status}`);
      }

      if (res.ok && data?.status === "success") {
        this.addMessage("assistant", data.response, data.provider);
        this.currentSessionId = data.session_id || this.currentSessionId;
        localStorage.setItem("floatchat_session", this.currentSessionId);
        this.ensureSessionItem(this.currentSessionId, true);
      } else {
        const msg = data?.response || data?.detail || "Sorry, I encountered an error processing your request.";
        this.addMessage("assistant", msg, "error");
        this.showToast("error", "Failed to get AI response");
      }
    } catch (err) {
      console.error("Chat error:", err);
      this.addMessage("assistant", "I'm having trouble connecting to the server. Please try again.", "error");
      this.showToast("error", "Connection error");
    } finally {
      this.showTypingIndicator(false);
    }
  }

  // Parse prompt → viz intent
  detectVisualizationCommand(message) {
    const m = message.toLowerCase();

    // Pick file from prompt if present; else default to last uploaded in THIS session
    const filenameMatch = m.match(/(\d{7,}_[a-z]+\.nc|\w+\.nc)/i);
    if (filenameMatch) {
      const fname = filenameMatch[1];
      const found = this.uploadedFiles.find((f) => (f.filename || "").toLowerCase() === fname.toLowerCase());
      if (found) this.vizParams.fileId = found.file_id;
    } else if (!this.vizParams.fileId && this.uploadedFiles.length) {
      this.vizParams.fileId = this.uploadedFiles[this.uploadedFiles.length - 1].file_id;
    }

    // Depth (dbar ≈ m) – only set when user mentions units, to avoid grabbing random numbers
    const depthMatch = m.match(/(\d+(?:\.\d+)?)\s*(m|meter|metre|dbar)\b/);
    if (depthMatch) this.vizParams.depth = Number(depthMatch[1]);

    // Variable aliases
    const normalizeVar = (w) => {
      if (!w) return null;
      if (/temp|temperature/.test(w)) return "TEMP";
      if (/sal|salin/.test(w)) return "PSAL";
      if (/pres|pressure|depth/.test(w)) return "PRES";
      if (/ox|doxy|oxygen/.test(w)) return "DOXY";
      return null;
    };

    // Intent types
    if (/(time\s*series|timeseries|trend|over time)/.test(m)) {
      this.vizParams.type = "timeseries";
      const want = m.match(/(temperature|temp|salinity|sal|pressure|pres|oxygen|doxy)/);
      this.vizParams.var = normalizeVar(want?.[1]) || "TEMP";
      const vt = this.$("vizType"); if (vt) vt.value = "timeseries";
      this.toggleVizPanel(true);
      this.generateVisualization();
      this.addMessage("assistant", `Opening time series at ${this.vizParams.depth} dbar...`, "system");
      return true;
    }

    if (/(profile|depth profile)/.test(m)) {
      this.vizParams.type = "depth"; // HTML select uses "depth"
      const vt = this.$("vizType"); if (vt) vt.value = "depth";
      this.toggleVizPanel(true);
      this.generateVisualization();
      this.addMessage("assistant", "Opening vertical profile...", "system");
      return true;
    }

    if (/(map|position|location)/.test(m)) {
      this.vizParams.type = "map";
      const vt = this.$("vizType"); if (vt) vt.value = "map";
      this.toggleVizPanel(true);
      this.generateVisualization();
      this.addMessage("assistant", "Opening map...", "system");
      return true;
    }

    if (/(t-?s|temperature *salinity|ts diagram|scatter)/.test(m)) {
      this.vizParams.type = "scatter";
      // If user wrote "salinity vs temperature" etc.
      const vs = m.split(/\bvs\b/);
      if (vs.length === 2) {
        this.vizParams.xVar = normalizeVar(vs[0]) || "PSAL";
        this.vizParams.yVar = normalizeVar(vs[1]) || "TEMP";
      } else {
        this.vizParams.xVar = "PSAL";
        this.vizParams.yVar = "TEMP";
      }
      const vt = this.$("vizType"); if (vt) vt.value = "scatter";
      this.toggleVizPanel(true);
      this.generateVisualization();
      this.addMessage("assistant", "Opening T–S diagram...", "system");
      return true;
    }

    return false;
  }

  autoOpenVisualization(type) {
    const vt = this.$("vizType");
    if (vt) vt.value = type;
    this.toggleVizPanel(true);
    this.generateVisualization();
    this.addMessage("assistant", `Opening ${type} visualization for you...`, "system");
  }

  addMessage(role, content, provider = null) {
    const messagesContainer = this.$("chatMessages");
    if (!messagesContainer) return;

    const messageDiv = document.createElement("div");
    messageDiv.className = `message ${role}`;

    const timestamp = new Date().toLocaleTimeString();
    messageDiv.innerHTML = `
      <div class="message-avatar">
        <i class="fas fa-${role === "user" ? "user" : "robot"}"></i>
      </div>
      <div class="message-content">
        <div class="message-text">${this.renderMessage(content)}</div>
        <div class="message-time">${timestamp} ${provider ? `• ${this.escapeHtml(provider)}` : ""}</div>
      </div>
    `;

    messagesContainer.appendChild(messageDiv);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;

    this.chatHistory.push({ role, content, timestamp, provider });
  }

  renderMessage(content) {
    const s = String(content ?? "");

    // 1) Escape HTML so user/AI cannot inject <a> or any HTML
    const escaped = this.escapeHtml(s);

    // 2) Neutralize naked URLs (http/https/www) to plain text (no <a>)
    const noUrls = escaped
      .replace(/\bhttps?:\/\/[^\s<)]+/gi, "[link removed]")
      .replace(/\bwww\.[^\s<)]+/gi, "[link removed]");

    // 3) Basic markdown (bold/italic/code) without creating links
    return noUrls
      .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
      .replace(/\*(.*?)\*/g, "<em>$1</em>")
      .replace(/`([^`]+)`/g, "<code>$1</code>")
      .replace(/\n/g, "<br>");
  }

  escapeHtml(s) {
    return s.replace(/[&<>"']/g, (m) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[m]));
  }

  getRecentContext() {
    return this.chatHistory.slice(-10).map((m) => ({ role: m.role, content: m.content }));
  }

  showTypingIndicator(show) {
    const el = this.$("typingIndicator");
    if (el) el.style.display = show ? "flex" : "none";
  }

  async loadHistory(sessionId) {
    const sid = sessionId || this.currentSessionId;
    if (!sid) return;

    try {
      const res = await fetch(this.api.history(sid));
      if (!res.ok) return;
      const data = await res.json();

      const messagesContainer = this.$("chatMessages");
      if (!messagesContainer) return;

      const welcome = messagesContainer.querySelector(".message.assistant");
      messagesContainer.innerHTML = "";
      if (welcome) messagesContainer.appendChild(welcome);

      this.chatHistory = [];
      (data.messages || []).forEach((m) => this.addMessage(m.role, m.content, m.provider));
    } catch (e) {
      console.error("Failed to load history:", e);
    }

    localStorage.setItem("floatchat_session", sid);
    this.setActiveSessionItem(sid);
  }

  // =================== Sessions ===================
  createNewSession() {
    this.currentSessionId = (crypto?.randomUUID && crypto.randomUUID()) || this.generateUUID();
    this.sessions.push(this.currentSessionId);
    this.ensureSessionItem(this.currentSessionId, true);

    // Reset per-session state
    this.chatHistory = [];
    this.uploadedFiles = [];
    this.varsCache.clear();
    this.vizParams = { type: "map", fileId: null, var: "TEMP", depth: 50, xVar: "PSAL", yVar: "TEMP" };

    // Clear UI: messages (keep welcome), file list, viz panel
    const messagesContainer = this.$("chatMessages");
    if (messagesContainer) {
      const welcome = messagesContainer.querySelector(".message.assistant");
      messagesContainer.innerHTML = "";
      if (welcome) messagesContainer.appendChild(welcome);
    }

    const fileList = this.$("fileList");
    if (fileList) fileList.innerHTML = "";

    const vizDisplay = this.$("vizDisplay");
    if (vizDisplay)
      vizDisplay.innerHTML = `
      <div class="viz-placeholder">
        <i class="fas fa-chart-bar"></i>
        <p>Upload NetCDF data files to enable visualizations</p>
      </div>`;

    localStorage.setItem("floatchat_session", this.currentSessionId);
    this.showToast("success", "New chat session created");
  }

  clearCurrentChat() {
    if (confirm("Are you sure you want to clear this chat?")) {
      this.createNewSession();
    }
  }

  ensureSessionItem(sessionId, active = false) {
    const sessionList = this.$("sessionList");
    if (!sessionList) return;

    let item = sessionList.querySelector(`[data-session-id="${sessionId}"]`);
    if (!item) {
      item = document.createElement("div");
      item.className = "session-item";
      item.dataset.sessionId = sessionId;
      item.tabIndex = 0;
      item.textContent = `Session ${sessionId.slice(0, 8)}…`;
      item.addEventListener("click", () => {
        this.currentSessionId = sessionId;
        this.loadHistory(sessionId);
      });
      item.addEventListener("keydown", (e) => {
        if (e.key === "Enter" || e.key === " ") {
          this.currentSessionId = sessionId;
          this.loadHistory(sessionId);
        }
      });
      sessionList.insertBefore(item, sessionList.firstChild);
    }
    if (active) this.setActiveSessionItem(sessionId);
  }

  setActiveSessionItem(sessionId) {
    const sessionList = this.$("sessionList");
    if (!sessionList) return;
    sessionList.querySelectorAll(".session-item").forEach((i) => i.classList.remove("active"));
    const item = sessionList.querySelector(`[data-session-id="${sessionId}"]`);
    if (item) item.classList.add("active");
  }

  exportChat() {
    const chatData = {
      session_id: this.currentSessionId,
      timestamp: new Date().toISOString(),
      messages: this.chatHistory,
      uploaded_files: this.uploadedFiles.map((f) => ({ filename: f.filename, file_id: f.file_id })),
    };

    const blob = new Blob([JSON.stringify(chatData, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `floatchat-${this.currentSessionId.slice(0, 8)}.json`;
    a.click();
    URL.revokeObjectURL(url);
    this.showToast("success", "Chat exported successfully");
  }

  // =================== Uploads ===================
  handleDragOver(e) {
    e.preventDefault(); e.stopPropagation();
    this.$("uploadArea")?.classList.add("dragover");
  }
  handleDragLeave(e) {
    e.preventDefault(); e.stopPropagation();
    this.$("uploadArea")?.classList.remove("dragover");
  }
  handleFileDrop(e) {
    e.preventDefault(); e.stopPropagation();
    this.$("uploadArea")?.classList.remove("dragover");
    const files = Array.from(e.dataTransfer?.files || []);
    this.processFiles(files);
  }
  handleFileSelect(e) {
    const files = Array.from(e.target.files || []);
    e.target.value = "";
    this.processFiles(files);
  }
  async processFiles(files) {
    for (const file of files) {
      if (this.isNetCDFFile(file)) {
        await this.uploadFile(file);
      } else {
        this.showToast("error", `${file.name} is not a valid NetCDF file`);
      }
    }
  }
  isNetCDFFile(file) {
    const valid = [".nc", ".netcdf", ".cdf"];
    return valid.some((ext) => file.name.toLowerCase().endsWith(ext));
  }

  async uploadFile(file) {
    this.showLoading(true, `Uploading ${file.name}...`);
    try {
      const form = new FormData();
      form.append("file", file);

      const res = await fetch(this.api.upload, { method: "POST", body: form });
      let data;
      try {
        data = await res.json();
      } catch {
        throw new Error(`Upload failed (${res.status})`);
      }
      if (!res.ok) throw new Error(data?.detail || "Upload failed");

      const normalized = {
        file_id: data.file_id,
        filename: data.filename,
        total_variables: data.total_variables,
        total_dimensions: data.total_dimensions,
        dimensions: data.dimensions,
        variables: data.variables,
        global_attributes: data.global_attributes,
      };

      // Store ONLY in this session
      this.uploadedFiles.push(normalized);
      this.addFileToList(normalized);

      // Make the uploaded file the active one for plotting
      this.vizParams.fileId = normalized.file_id;

      this.showToast("success", `${file.name} uploaded successfully`);

      // Reset cached vars for this file (fresh)
      this.varsCache.delete(normalized.file_id);

      const msg =
        `📁 **File Uploaded**: ${this.escapeHtml(data.filename)}\n\n` +
        `**Dimensions**: ${data.total_dimensions}\n` +
        `**Variables**: ${data.total_variables}\n\n` +
        `You can now ask me questions about this data or generate visualizations!`;
      this.addMessage("assistant", msg, "system");
    } catch (err) {
      console.error("Upload error:", err);
      this.showToast("error", `Failed to upload ${file.name}: ${err.message}`);
    } finally {
      this.showLoading(false);
    }
  }

  async loadUploadedFiles() {
    // Only used if USE_GLOBAL_FILE_LIST === true
    try {
      const res = await fetch(this.api.files);
      if (!res.ok) return;
      const data = await res.json();

      const normalized = (data.files || []).map((f) => ({
        file_id: f.id,
        filename: f.filename,
        total_variables: f.total_variables ?? 0,
        total_dimensions: f.total_dimensions ?? 0,
      }));

      const byId = new Map(this.uploadedFiles.map((x) => [x.file_id, x]));
      normalized.forEach((n) => {
        if (!byId.has(n.file_id)) {
          byId.set(n.file_id, n);
          this.addFileToListSilent(n);
        }
      });

      this.uploadedFiles = Array.from(byId.values());
    } catch (e) {
      console.error("Failed to load files:", e);
    }
  }

  addFileToList(fileData) { this.addFileToListSilent(fileData); }

  addFileToListSilent(fileData) {
    const fileList = this.$("fileList");
    if (!fileList) return;

    if (fileList.querySelector(`[data-file-id="${fileData.file_id}"]`)) return;

    const varsCount = Number.isFinite(fileData.total_variables) ? fileData.total_variables : 0;
    const dimsCount = Number.isFinite(fileData.total_dimensions) ? fileData.total_dimensions : 0;

    const fileItem = document.createElement("div");
    fileItem.className = "file-item";
    fileItem.dataset.fileId = fileData.file_id;
    fileItem.innerHTML = `
      <div class="file-name">${this.escapeHtml(fileData.filename)}</div>
      <div class="file-meta">
        <span>${varsCount} variables</span>
        <span>${dimsCount} dimensions</span>
      </div>
    `;
    fileItem.addEventListener("click", () => this.showFileDetails(fileData));
    fileList.prepend(fileItem);
  }

  async showFileDetails(fileData) {
    const modalBody = this.$("fileModalBody");
    if (!modalBody) return;

    let dimsHtml = "";
    let varsHtml = "";
    let totalDims = fileData.total_dimensions ?? 0;
    let totalVars = fileData.total_variables ?? 0;

    let { dimensions, variables } = fileData;

    // If not present (e.g., global list), try to fetch variables list
    if (!variables) {
      try {
        const res = await fetch(this.api.variables(fileData.file_id));
        if (res.ok) {
          const v = await res.json();
          variables = {};
          (v.variables || []).forEach((entry) => {
            variables[entry.name] = {
              type: entry.dtype,
              shape: entry.shape,
              dimensions: entry.dimensions,
              attributes: { units: entry.units, standard_name: entry.standard_name },
            };
          });
          totalVars = Object.keys(variables).length;
        }
      } catch (e) {
        console.warn("Failed to fetch variables for details:", e);
      }
    }

    if (dimensions) {
      dimsHtml = Object.entries(dimensions)
        .map(
          ([name, info]) =>
            `<div class="detail-item"><strong>${this.escapeHtml(name)}</strong>: ${info.size} ${
              info.unlimited ? "(unlimited)" : ""
            }</div>`
        )
        .join("");
      totalDims = Object.keys(dimensions).length;
    }

    if (variables) {
      const varEntries = Object.entries(variables);
      const varsShown = varEntries
        .slice(0, 15)
        .map(([name, info]) => {
          const shape = Array.isArray(info.shape) ? `[${info.shape.join(", ")}]` : "";
          const units = info.attributes?.units || info.attributes?.unit || "";
          return `<div class="detail-item">
          <strong>${this.escapeHtml(name)}</strong>: ${this.escapeHtml(String(info.type || ""))} ${shape}
          ${units ? `<br><em>Units: ${this.escapeHtml(units)}</em>` : ""}</div>`;
        });
      const extra =
        varEntries.length > 15
          ? `<div class="detail-item">... and ${varEntries.length - 15} more variables</div>`
          : "";
      varsHtml = `${varsShown.join("")}${extra}`;
      totalVars = varEntries.length || totalVars;
    }

    modalBody.innerHTML = `
      <div class="file-details">
        <h4>📁 ${this.escapeHtml(fileData.filename)}</h4>
        <div class="detail-section">
          <h5>Dimensions (${totalDims})</h5>
          <div class="detail-grid">${dimsHtml || "<div class='text-muted'>No dimensions available</div>"}</div>
        </div>
        <div class="detail-section">
          <h5>Variables (${totalVars})</h5>
          <div class="detail-grid">${varsHtml || "<div class='text-muted'>No variables metadata available</div>"}</div>
        </div>
      </div>`;
    this.showModal("fileModal");
  }

  // =================== Health ===================
  async checkSystemHealth() {
    try {
      const [healthRes, azureRes] = await Promise.all([fetch(this.api.health), fetch(this.api.azure_ping)]);
      const healthData = await healthRes.json();
      const azureData = await azureRes.json();

      this.updateSystemStatus({ ...healthData, azure_ping: !!azureData.ok });
    } catch (err) {
      console.error("Health check failed:", err);
      this.updateSystemStatus({ status: "unhealthy", ai_service: "unknown", database: "unknown", azure_ping: false });
    }
  }

  updateSystemStatus(h) {
    const status = this.$("systemStatus");
    if (!status) return;

    const healthy = h.status === "healthy";
    const aiHealthy = h.ai_service === "configured" && h.azure_ping;

    status.innerHTML = `
      <div class="status-item">
        <span class="status-dot ${healthy ? "healthy" : "error"}"></span>
        <span>Backend: ${healthy ? "Online" : "Offline"}</span>
      </div>
      <div class="status-item">
        <span class="status-dot ${aiHealthy ? "healthy" : "warning"}"></span>
        <span>AI Service: ${aiHealthy ? "Connected" : "Not Available"}</span>
      </div>
      <div class="status-item">
        <span class="status-dot ${h.database === "connected" ? "healthy" : "error"}"></span>
        <span>Database: ${this.escapeHtml(h.database || "Unknown")}</span>
      </div>`;
  }

  // ======== NetCDF → Backend variable name mapper (for /plot endpoints) ========
  mapVarForBackend(ncVar) {
    if (!ncVar) return null;
    const up = String(ncVar).toUpperCase();
    if (up.startsWith("TEMP")) return "temperature";
    if (up.startsWith("PSAL")) return "salinity";
    if (up.startsWith("PRES")) return "pressure";
    if (up.startsWith("DOXY")) return "oxygen";
    return null; // unsupported
  }

  // =================== Visualization ===================
  toggleVizPanel(show = null) {
    const panel = this.$("vizPanel");
    if (!panel) return;

    const visible = panel.style.display !== "none";
    if (show === null) show = !visible;
    panel.style.display = show ? "block" : "none";

    if (show && this.uploadedFiles.length === 0) {
      const vizDisplay = this.$("vizDisplay");
      if (vizDisplay) {
        vizDisplay.innerHTML = `
          <div class="viz-placeholder">
            <i class="fas fa-chart-bar"></i>
            <p>Upload NetCDF data files to enable visualizations</p>
          </div>`;
      }
    }
  }

  async generateVisualization() {
    const vt = this.$("vizType");
    const typeFromUI = vt ? vt.value : null;
    // Prefer prompt-driven viz params; fallback to UI selects
    const type = this.vizParams.type || typeFromUI || "map";
    const fileId = this.vizParams.fileId || null;
    const varName = this.vizParams.var || "TEMP";
    const depth = Number.isFinite(this.vizParams.depth) ? this.vizParams.depth : 50;
    const xVarHint = this.vizParams.xVar || "PSAL";
    const yVarHint = this.vizParams.yVar || "TEMP";

    const vizDisplay = this.$("vizDisplay");
    if (!vizDisplay) return;

    if (this.uploadedFiles.length === 0) {
      this.showToast("warning", "Please upload NetCDF files first");
      return;
    }

    if (!fileId) {
      this.showToast("warning", "Select a file first (or upload one).");
      return;
    }

    vizDisplay.innerHTML = `
      <div class="viz-placeholder">
        <i class="fas fa-chart-line"></i>
        <p>Generating ${this.escapeHtml(type)} visualization...</p>
      </div>`;

    try {
      // Discover variables available in selected file (prefer *_ADJUSTED)
      const fileVars = await this.fetchVariables(fileId);
      const tempVar = this.pickVar(fileVars, [varName + "_ADJUSTED", varName, "TEMP_ADJUSTED", "TEMP"]);

      let url;
      switch (type) {
        case "scatter": { // T–S diagram
          // Map NetCDF var names to backend columns
          const xCandidate = this.pickVar(fileVars, [xVarHint + "_ADJUSTED", xVarHint, "PSAL_ADJUSTED", "PSAL"]);
          const yCandidate = this.pickVar(fileVars, [yVarHint + "_ADJUSTED", yVarHint, "TEMP_ADJUSTED", "TEMP"]);
          const bx = this.mapVarForBackend(xCandidate) || "salinity";
          const by = this.mapVarForBackend(yCandidate) || "temperature";
          url = this.api.plot.scatter(fileId, bx, by);
          break;
        }
        case "timeseries": {
          const chosen = tempVar || varName;
          const backendVar = this.mapVarForBackend(chosen);
          if (!backendVar) {
            throw new Error(
              `Unsupported variable for backend: ${chosen}. Allowed: temperature, salinity, pressure, oxygen.`
            );
          }
          url = this.api.plot.timeseries(fileId, backendVar, depth);
          break;
        }
        case "depth":     // Depth Profile
        case "profile": { // (alias)
          url = this.api.plot.profile(fileId);
          break;
        }
        case "map":
        default:
          url = this.api.plot.map(fileId);
      }

      const res = await fetch(url);
      const fig = await res.json();

      if (!res.ok) throw new Error(fig?.detail || "Plot API returned an error.");
      if (!fig || !Array.isArray(fig.data) || fig.data.length === 0) throw new Error("No data available for this plot.");

      vizDisplay.innerHTML = "";

      if (typeof Plotly?.newPlot !== "function") {
        vizDisplay.innerHTML = `<div class="viz-placeholder"><p>Plotly is not loaded on this page.</p></div>`;
        return;
      }

      await Plotly.newPlot(vizDisplay, fig.data, fig.layout || {}, {
        responsive: true,
        displayModeBar: true,
        modeBarButtonsToRemove: ["pan2d", "lasso2d"],
        displaylogo: false,
      });

      const cap = type.charAt(0).toUpperCase() + type.slice(1);
      this.showToast("success", `${cap} visualization generated`);
    } catch (e) {
      console.error("Visualization error:", e);
      vizDisplay.innerHTML = `
        <div class="viz-placeholder">
          <i class="fas fa-exclamation-circle"></i>
          <p>Failed to generate visualization</p>
          <p class="text-muted">${this.escapeHtml(e.message)}</p>
        </div>`;
      this.showToast("error", "Failed to generate visualization");
    }
  }

  // Fetch and cache variables for a file
  async fetchVariables(fileId) {
    if (!fileId) {
      const last = this.uploadedFiles[this.uploadedFiles.length - 1];
      if (last) fileId = last.file_id;
      this.vizParams.fileId = fileId || null;
    }
    if (!fileId) return [];

    if (this.varsCache.has(fileId)) return this.varsCache.get(fileId);

    try {
      const res = await fetch(this.api.variables(fileId));
      if (!res.ok) throw new Error("variables endpoint failed");
      const data = await res.json();
      const names = (data.variables || []).map((v) => v.name);
      this.varsCache.set(fileId, names);
      return names;
    } catch (e) {
      console.warn("fetchVariables failed:", e);
      return [];
    }
  }

  // Prefer the first candidate that exists in `available` (case-insensitive)
  pickVar(available, candidates) {
    if (!Array.isArray(available) || !available.length) return null;
    const availSet = new Set(available.map((v) => v.toUpperCase()));
    for (const cand of candidates) {
      if (!cand) continue;
      const up = cand.toUpperCase();
      if (availSet.has(up)) {
        const exact = available.find((v) => v.toUpperCase() === up);
        return exact || cand;
      }
    }
    return null;
  }

  // =================== UI Helpers ===================
  showModal(id) {
    const modal = this.$(id);
    if (!modal) return;
    modal.classList.add("active");
    document.body.style.overflow = "hidden";
    modal.querySelector(".modal-close")?.focus();
  }
  hideModal(id) {
    const modal = this.$(id);
    if (!modal) return;
    modal.classList.remove("active");
    document.body.style.overflow = "";
  }
  showLoading(show, text = "Processing...") {
    const overlay = this.$("loadingOverlay");
    if (!overlay) return;
    const lt = this.$("loadingText");
    if (lt) lt.textContent = text;
    overlay.classList.toggle("active", show);
  }
  showToast(type, message) {
    const toast = this.$("toast");
    if (!toast) return;
    const icon = toast.querySelector(".toast-icon");
    const msg = toast.querySelector(".toast-message");
    const icons = {
      success: "fas fa-check-circle",
      error: "fas fa-exclamation-circle",
      warning: "fas fa-exclamation-triangle",
      info: "fas fa-info-circle",
    };
    if (icon) icon.className = `toast-icon ${icons[type] || icons.info}`;
    if (msg) msg.textContent = message;
    toast.className = `toast ${type}`;
    toast.classList.add("active");
    setTimeout(() => toast.classList.remove("active"), 4000);
  }
  autoResizeTextarea() {
    const ta = this.$("chatInput");
    if (!ta) return;
    const resize = function () {
      this.style.height = "auto";
      this.style.height = Math.min(this.scrollHeight, 120) + "px";
    };
    ta.addEventListener("input", resize);
    resize.call(ta);
  }
  generateUUID() {
    return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, function (c) {
      const r = (Math.random() * 16) | 0,
        v = c === "x" ? r : (r & 0x3) | 0x8;
      return v.toString(16);
    });
  }
}

// Initialize the app when DOM is loaded
document.addEventListener("DOMContentLoaded", () => {
  window.floatChatApp = new FloatChatApp();
});
