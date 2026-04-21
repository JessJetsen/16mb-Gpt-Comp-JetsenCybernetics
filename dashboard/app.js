import { SETTINGS, TARGET_PRESETS } from "./schema.js";

const PROFILES_KEY = "parameter-golf-dashboard-profiles";

const settingsRoot = document.querySelector("#settings-root");
const searchInput = document.querySelector("#search-input");
const changedOnlyToggle = document.querySelector("#changed-only-toggle");
const targetSelect = document.querySelector("#target-select");
const launcherInput = document.querySelector("#launcher-input");
const profileNameInput = document.querySelector("#profile-name");
const commandOutput = document.querySelector("#command-output");
const envOutput = document.querySelector("#env-output");
const profilesRoot = document.querySelector("#profiles-root");
const copyCommandBtn = document.querySelector("#copy-command-btn");
const copyEnvBtn = document.querySelector("#copy-env-btn");
const saveProfileBtn = document.querySelector("#save-profile-btn");
const exportProfilesBtn = document.querySelector("#export-profiles-btn");
const importProfilesBtn = document.querySelector("#import-profiles-btn");
const importProfilesInput = document.querySelector("#import-profiles-input");

const state = {};
let activeTarget = "torch";
let launcherEdited = false;

for (const group of SETTINGS) {
  for (const field of group.fields) {
    state[field.env] = {
      ...field,
      initialValue: field.value,
      defaultValue: field.value,
    };
  }
}

function boolToEnv(value) {
  return value ? "1" : "0";
}

function getCurrentValue(field) {
  return field.type === "boolean" ? boolToEnv(Boolean(field.value)) : String(field.value);
}

function valueChanged(field) {
  return getCurrentValue(field) !== getCurrentValue({ ...field, value: field.defaultValue ?? field.initialValue });
}

function quoteShell(value) {
  if (/^[A-Za-z0-9_./:@,+-]+$/.test(value)) {
    return value;
  }
  return `'${String(value).replace(/'/g, `'\\''`)}'`;
}

function fieldMatchesTarget(field) {
  const targets = field.targets || ["all"];
  return activeTarget === "all" || targets.includes("all") || targets.includes(activeTarget);
}

function currentLauncher() {
  return launcherInput.value.trim() || TARGET_PRESETS[activeTarget].launcher;
}

function renderSettings() {
  const filter = searchInput.value.trim().toLowerCase();
  const changedOnly = changedOnlyToggle.checked;
  settingsRoot.innerHTML = "";

  for (const group of SETTINGS) {
    const visibleFields = group.fields.filter((field) => {
      if (!fieldMatchesTarget(field)) {
        return false;
      }
      const current = state[field.env];
      const haystack = `${field.label} ${field.env} ${field.help || ""}`.toLowerCase();
      if (filter && !haystack.includes(filter)) {
        return false;
      }
      if (changedOnly && !valueChanged(current)) {
        return false;
      }
      return true;
    });
    if (!visibleFields.length) continue;

    const details = document.createElement("details");
    details.className = "group";
    details.open = true;

    const summary = document.createElement("summary");
    summary.innerHTML = `<span>${group.title}</span><span class="muted">${visibleFields.length} knobs</span>`;
    details.appendChild(summary);

    if (group.description) {
      const description = document.createElement("p");
      description.className = "group-description";
      description.textContent = group.description;
      details.appendChild(description);
    }

    const body = document.createElement("div");
    body.className = "group-body";

    for (const field of visibleFields) {
      const current = state[field.env];
      const row = document.createElement("label");
      row.className = "field-row";

      const meta = document.createElement("span");
      meta.className = "field-meta";
      meta.innerHTML = `
        <span class="field-title">${field.label}</span>
        <span class="field-env">${field.env}</span>
        <span class="field-help">${field.help || ""}</span>
      `;

      let input;
      if (field.type === "boolean") {
        input = document.createElement("select");
        input.innerHTML = `
          <option value="1">Enabled</option>
          <option value="0">Disabled</option>
        `;
        input.value = boolToEnv(Boolean(current.value));
        input.addEventListener("change", () => {
          current.value = input.value === "1";
          refreshOutputs();
        });
      } else {
        input = document.createElement("input");
        input.type = field.type === "number" ? "number" : "text";
        input.value = current.value;
        if (field.type === "number") {
          input.step = "any";
        }
        input.addEventListener("input", () => {
          current.value = field.type === "number" && input.value !== "" ? Number(input.value) : input.value;
          refreshOutputs();
        });
      }

      row.append(meta, input);
      body.appendChild(row);
    }

    details.append(summary, body);
    settingsRoot.appendChild(details);
  }
}

function getEnvEntries() {
  const entries = [];
  for (const group of SETTINGS) {
    for (const field of group.fields) {
      if (!fieldMatchesTarget(field)) continue;
      const current = state[field.env];
      if (!valueChanged(current)) continue;
      entries.push([field.env, getCurrentValue(current)]);
    }
  }
  return entries;
}

function captureCurrentValues() {
  const values = {};
  for (const [env, field] of Object.entries(state)) {
    if (valueChanged(field)) {
      values[env] = field.value;
    }
  }
  return values;
}

function refreshOutputs() {
  const envEntries = getEnvEntries();
  const envLines = envEntries.map(([env, value]) => `${env}=${quoteShell(value)}`);
  envOutput.textContent = envLines.length ? envLines.join("\n") : "# No changed env vars for this target.";

  const launcher = currentLauncher();
  commandOutput.textContent = envLines.length
    ? `${envLines.map((line) => `${line} \\`).join("\n")}\n${launcher}`
    : launcher;

  renderProfiles();
}

function loadProfiles() {
  try {
    return JSON.parse(localStorage.getItem(PROFILES_KEY) || "[]");
  } catch {
    return [];
  }
}

function saveProfiles(profiles) {
  localStorage.setItem(PROFILES_KEY, JSON.stringify(profiles));
}

function sanitizeProfiles(input) {
  if (!Array.isArray(input)) return [];
  return input
    .filter((profile) => profile && typeof profile.name === "string" && profile.name.trim())
    .map((profile) => ({
      name: profile.name.trim(),
      target: profile.target || "torch",
      launcher: profile.launcher || TARGET_PRESETS[profile.target || "torch"]?.launcher || "python train_gpt.py",
      values: typeof profile.values === "object" && profile.values ? profile.values : {},
    }));
}

function renderProfiles() {
  const profiles = loadProfiles();
  profilesRoot.innerHTML = "";
  if (!profiles.length) {
    profilesRoot.innerHTML = `<p class="muted">No saved profiles yet.</p>`;
    return;
  }

  for (const profile of profiles) {
    const row = document.createElement("div");
    row.className = "profile-row";
    row.innerHTML = `
      <div>
        <strong>${profile.name}</strong>
        <div class="muted">${profile.target || "torch"} · ${Object.keys(profile.values).length} changed knobs</div>
      </div>
    `;

    const actions = document.createElement("div");
    actions.className = "profile-actions";

    const loadBtn = document.createElement("button");
    loadBtn.className = "button";
    loadBtn.textContent = "Load";
    loadBtn.addEventListener("click", () => {
      for (const group of SETTINGS) {
        for (const field of group.fields) {
          state[field.env].value = profile.values[field.env] ?? field.value;
        }
      }
      activeTarget = profile.target || "torch";
      targetSelect.value = activeTarget;
      launcherInput.value = profile.launcher || TARGET_PRESETS[activeTarget].launcher;
      launcherEdited = true;
      profileNameInput.value = profile.name;
      renderSettings();
      refreshOutputs();
    });

    const deleteBtn = document.createElement("button");
    deleteBtn.className = "button";
    deleteBtn.textContent = "Delete";
    deleteBtn.addEventListener("click", () => {
      saveProfiles(loadProfiles().filter((item) => item.name !== profile.name));
      renderProfiles();
    });

    actions.append(loadBtn, deleteBtn);
    row.appendChild(actions);
    profilesRoot.appendChild(row);
  }
}

async function copyText(text, button) {
  await navigator.clipboard.writeText(text);
  const original = button.textContent;
  button.textContent = "Copied";
  setTimeout(() => {
    button.textContent = original;
  }, 900);
}

targetSelect.innerHTML = Object.entries(TARGET_PRESETS)
  .map(([value, preset]) => `<option value="${value}">${preset.label}</option>`)
  .join("");
targetSelect.value = activeTarget;
targetSelect.addEventListener("change", () => {
  activeTarget = targetSelect.value;
  if (!launcherEdited) {
    launcherInput.value = TARGET_PRESETS[activeTarget].launcher;
  }
  renderSettings();
  refreshOutputs();
});

copyCommandBtn.addEventListener("click", () => copyText(commandOutput.textContent, copyCommandBtn));
copyEnvBtn.addEventListener("click", () => copyText(envOutput.textContent, copyEnvBtn));

saveProfileBtn.addEventListener("click", () => {
  const name = profileNameInput.value.trim();
  if (!name) {
    profileNameInput.focus();
    return;
  }
  const profiles = loadProfiles().filter((profile) => profile.name !== name);
  profiles.unshift({
    name,
    target: activeTarget,
    launcher: currentLauncher(),
    values: captureCurrentValues(),
  });
  saveProfiles(profiles.slice(0, 30));
  renderProfiles();
});

exportProfilesBtn.addEventListener("click", () => {
  const profiles = loadProfiles();
  const blob = new Blob([JSON.stringify(profiles, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = "parameter-golf-profiles.json";
  link.click();
  URL.revokeObjectURL(url);
});

importProfilesBtn.addEventListener("click", () => {
  importProfilesInput.click();
});

importProfilesInput.addEventListener("change", async () => {
  const [file] = importProfilesInput.files || [];
  if (!file) return;
  try {
    const text = await file.text();
    const imported = sanitizeProfiles(JSON.parse(text));
    const existing = loadProfiles().filter((profile) => !imported.some((item) => item.name === profile.name));
    saveProfiles([...imported, ...existing].slice(0, 50));
    renderProfiles();
  } catch (error) {
    console.error(error);
    window.alert("Could not import profiles JSON.");
  } finally {
    importProfilesInput.value = "";
  }
});

searchInput.addEventListener("input", renderSettings);
changedOnlyToggle.addEventListener("change", renderSettings);
launcherInput.addEventListener("input", () => {
  launcherEdited = true;
  refreshOutputs();
});

launcherInput.value = TARGET_PRESETS[activeTarget].launcher;
renderSettings();
refreshOutputs();
