//! The Exports window: the retake service's exports (cut MP4 with size / frame rate / codec / quality / audio options, audio only, subtitles of the kept text, cut list,
//! transcript) written into the project folder under `data/exports/`, never over an earlier export. The progress comes from the service's `export_render` stage events
//! (the ffmpeg percent); an ffmpeg error is shown exactly as ffmpeg wrote it. User presets live in the user's own config folder, not in the project.
//!
//! The form fields are plain window state; what the user does with them (open, export, cancel, save or delete a preset) goes through [`ExportAction`].

use egui::{Align2, RichText};
use serde_json::{Value, json};
use simplecut_core::{AnalysisJob, ExportOptionsInfo, ExportRequest, JobState, StageLog, UserCall};

use crate::app::{Action, SimpleCutApp};
use crate::theme;

#[derive(Debug, Clone, PartialEq)]
pub enum ExportAction {
    Open,
    Close,
    Start,
    Cancel,
    SavePreset,
    DeletePreset(String),
    /// Choose the file for the whole-timeline export.
    BrowseTimeline,
    /// Choose the destination folder for the service's exports.
    BrowseFolder,
    UseProjectFolder,
    /// The file already exists: go ahead and replace it (the whole-timeline export only; the service never overwrites).
    Overwrite,
    KeepExisting,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Res {
    Source,
    Height(String),
    Canvas(String),
    Custom,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Fps {
    Source,
    Fixed(u32),
    Custom,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum Quality {
    Preset,
    Crf,
    Bitrate,
}

/// Everything the user can set for one export.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Form {
    pub kind: String,
    pub format: String,
    pub preset: String,
    pub res: Res,
    pub custom_w: u32,
    pub custom_h: u32,
    pub lock_aspect: bool,
    pub fit: String,
    pub fps: Fps,
    pub custom_fps: f64,
    pub codec: String,
    pub quality: Quality,
    pub q_preset: String,
    pub crf: u32,
    pub bitrate: u32,
    pub speed: String,
    pub hardware: bool,
    pub audio_bitrate: u32,
    pub loudness: bool,
    pub fade_ms: f64,
}

impl Default for Form {
    fn default() -> Self {
        Self {
            kind: "video".into(),
            format: String::new(),
            preset: String::new(),
            res: Res::Source,
            custom_w: 1280,
            custom_h: 720,
            lock_aspect: true,
            fit: "letterbox".into(),
            fps: Fps::Source,
            custom_fps: 30.0,
            codec: "h264".into(),
            quality: Quality::Preset,
            q_preset: "standard".into(),
            crf: 23,
            bitrate: 8000,
            speed: "medium".into(),
            hardware: false,
            audio_bitrate: 192,
            loudness: false,
            fade_ms: 15.0,
        }
    }
}

impl Form {
    /// Fill the form from a preset's settings (as the service stores them). Anything the preset does not say keeps its default.
    pub(crate) fn from_preset(name: &str, settings: &Value, info: &ExportOptionsInfo) -> Self {
        let mut f = Self { preset: name.to_string(), fade_ms: info.fade_default, ..Self::default() };
        let res = &settings["resolution"];
        match res["mode"].as_str() {
            Some("preset") => {
                let p = res["preset"].as_str().unwrap_or_default().to_string();
                f.res = if info.canvases.contains(&p) { Res::Canvas(p) } else { Res::Height(p) };
                if let Some(fit) = res["fit"].as_str() {
                    f.fit = fit.to_string();
                }
            }
            Some("custom") => {
                f.res = Res::Custom;
                f.custom_w = res["width"].as_u64().and_then(|n| u32::try_from(n).ok()).unwrap_or(f.custom_w);
                f.custom_h = res["height"].as_u64().and_then(|n| u32::try_from(n).ok()).unwrap_or(f.custom_h);
                f.lock_aspect = res["lock_aspect"].as_bool().unwrap_or(false);
            }
            _ => {}
        }
        match &settings["fps"] {
            Value::Number(n) => {
                let v = n.as_f64().unwrap_or(30.0);
                f.fps = if info.fps_choices.iter().any(|c| (f64::from(*c) - v).abs() < 1e-9) { Fps::Fixed(v as u32) } else { Fps::Custom };
                f.custom_fps = v;
            }
            _ => f.fps = Fps::Source,
        }
        let v = &settings["video"];
        if let Some(c) = v["codec"].as_str() {
            f.codec = c.to_string();
        }
        if let Some(s) = v["speed"].as_str() {
            f.speed = s.to_string();
        }
        f.hardware = v["hardware"].as_bool().unwrap_or(false);
        let q = &v["quality"];
        if let Some(b) = q["bitrate_kbps"].as_u64() {
            f.quality = Quality::Bitrate;
            f.bitrate = u32::try_from(b).unwrap_or(f.bitrate);
        } else if let Some(c) = q["crf"].as_u64() {
            f.quality = Quality::Crf;
            f.crf = u32::try_from(c).unwrap_or(f.crf);
        } else if let Some(p) = q["preset"].as_str() {
            f.q_preset = p.to_string();
        }
        let a = &settings["audio"];
        f.audio_bitrate = a["bitrate_kbps"].as_u64().and_then(|n| u32::try_from(n).ok()).unwrap_or(f.audio_bitrate);
        f.loudness = a["normalize_loudness"].as_bool().unwrap_or(false);
        f.fade_ms = a["fade_ms"].as_f64().unwrap_or(f.fade_ms);
        f
    }

    /// The preset-shaped JSON of this form (also what a saved user preset holds).
    pub(crate) fn settings(&self) -> Value {
        let resolution = match &self.res {
            Res::Source => json!({"mode": "source"}),
            Res::Height(h) => json!({"mode": "preset", "preset": h}),
            Res::Canvas(c) => json!({"mode": "preset", "preset": c, "fit": self.fit}),
            Res::Custom if self.lock_aspect => json!({"mode": "custom", "width": self.custom_w, "height": self.custom_h, "lock_aspect": true}),
            Res::Custom => json!({"mode": "custom", "width": self.custom_w, "height": self.custom_h, "lock_aspect": false, "fit": self.fit}),
        };
        let fps = match &self.fps {
            Fps::Source => json!("source"),
            Fps::Fixed(n) => json!(n),
            Fps::Custom => json!(self.custom_fps),
        };
        let quality = match self.quality {
            Quality::Preset => json!({"preset": self.q_preset}),
            Quality::Crf => json!({"crf": self.crf}),
            Quality::Bitrate => json!({"bitrate_kbps": self.bitrate}),
        };
        json!({
            "resolution": resolution,
            "fps": fps,
            "video": {"codec": self.codec, "quality": quality, "speed": self.speed, "hardware": self.hardware},
            "audio": {"bitrate_kbps": self.audio_bitrate, "normalize_loudness": self.loudness, "fade_ms": self.fade_ms},
        })
    }

    /// The request body for the service.
    pub(crate) fn body(&self) -> Value {
        let mut b = if self.kind == "video" || self.kind == "audio" { self.settings() } else { json!({}) };
        if let Some(map) = b.as_object_mut() {
            map.insert("kind".into(), json!(self.kind));
            if self.kind != "video" {
                map.insert("format".into(), json!(self.format));
            }
            if self.kind == "video" && !self.preset.is_empty() {
                map.insert("preset".into(), json!(self.preset));
            }
        }
        b
    }

    /// Problems the window can see before asking the service (the service checks again).
    pub(crate) fn problem(&self, info: &ExportOptionsInfo) -> Option<String> {
        if self.kind != "video" {
            return None;
        }
        if self.res == Res::Custom {
            for (name, v) in [("Width", self.custom_w), ("Height", self.custom_h)] {
                if v % 2 != 0 {
                    return Some(format!("{name} must be an even number (H.264/H.265 need even sizes); {v} is odd."));
                }
                if v < info.dim_min || v > info.dim_max {
                    return Some(format!("{name} must be between {} and {} pixels.", info.dim_min, info.dim_max));
                }
            }
        }
        if self.fps == Fps::Custom && (self.custom_fps < info.fps_range.0 || self.custom_fps > info.fps_range.1) {
            return Some(format!("Frame rate must be between {} and {}.", info.fps_range.0, info.fps_range.1));
        }
        None
    }
}

#[derive(Default)]
pub(crate) struct ExportUi {
    pub open: bool,
    pub info: Option<ExportOptionsInfo>,
    pub form: Form,
    pub preset_name: String,
    pub note: Option<String>,
    pub error: Option<String>,
    pub result: Option<Vec<String>>,
    pub log: StageLog,
    /// The whole-timeline export (the editor's own): where the file goes, whether to use the graphics card, and the "replace it?" question.
    pub timeline_path: std::path::PathBuf,
    pub timeline_hw: bool,
    pub overwrite_confirm: bool,
    /// Where the service's exports go (None: the project's data/exports).
    pub dest_folder: Option<std::path::PathBuf>,
    /// The user picked a kind: the window no longer chooses for them when the service's options arrive.
    pub kind_chosen: bool,
    job: Option<Box<dyn AnalysisJob>>,
    calls: Vec<(bool, Box<dyn AnalysisJob>)>,
}

impl ExportUi {
    pub(crate) fn running(&self) -> bool {
        self.job.is_some()
    }
}

fn kind_label(kind: &str) -> &'static str {
    match kind {
        "timeline" => "Video — whole timeline (MP4)",
        "video" => "Video (MP4)",
        "audio" => "Audio only",
        "subtitles" => "Subtitles of the kept text",
        "cutlist" => "Cut list",
        "transcript" => "Transcript",
        _ => "?",
    }
}

fn res_label(r: &Res) -> String {
    match r {
        Res::Source => "Match source".into(),
        Res::Height(h) => match h.as_str() {
            "2160p" => "4K (2160p)".into(),
            other => other.to_string(),
        },
        Res::Canvas(c) => match c.as_str() {
            "vertical_1080x1920" => "Vertical 1080×1920".into(),
            "square_1080x1080" => "Square 1080×1080".into(),
            other => other.to_string(),
        },
        Res::Custom => "Custom size…".into(),
    }
}

fn fps_label(f: &Fps) -> String {
    match f {
        Fps::Source => "Match source (variable rate kept)".into(),
        Fps::Fixed(n) => format!("{n} fps"),
        Fps::Custom => "Custom…".into(),
    }
}

impl SimpleCutApp {
    pub(crate) fn perform_export(&mut self, action: ExportAction) {
        match action {
            ExportAction::Open => {
                self.export_ui.open = true;
                self.export_ui.note = None;
                self.export_ui.error = None;
                self.export_ui.result = None;
                self.export_ui.overwrite_confirm = false;
                let name = format!("{}.mp4", self.editor.project_name());
                self.export_ui.timeline_path = match self.editor.project_path() {
                    Some(p) => std::path::PathBuf::from(p).with_file_name(name),
                    None => std::path::PathBuf::from(name),
                };
                if !self.export_ui.kind_chosen {
                    self.export_ui.form.kind = "timeline".into();
                }
                if self.service.is_some() && self.export_ui.info.is_none() {
                    self.user_call(false, UserCall::ExportOptions);
                } else {
                    self.pick_default_kind();
                }
            }
            ExportAction::BrowseTimeline => {
                let suggested = self.export_ui.timeline_path.file_name().and_then(|n| n.to_str()).unwrap_or("output.mp4").to_string();
                if let Some(picked) = self.dialogs.pick_export_path(&suggested) {
                    self.export_ui.timeline_path = picked;
                    self.export_ui.overwrite_confirm = false;
                }
            }
            ExportAction::BrowseFolder => {
                let start = self.export_ui.dest_folder.clone();
                if let Some(picked) = self.dialogs.pick_export_folder(start.as_deref()) {
                    self.export_ui.dest_folder = Some(picked);
                }
            }
            ExportAction::UseProjectFolder => self.export_ui.dest_folder = None,
            ExportAction::KeepExisting => self.export_ui.overwrite_confirm = false,
            ExportAction::Overwrite => self.start_timeline_export(true),
            ExportAction::Close => self.export_ui.open = false,
            ExportAction::Cancel => {
                if let Some(job) = &mut self.export_ui.job {
                    job.cancel();
                }
            }
            ExportAction::Start if self.export_ui.form.kind == "timeline" || self.export_ui.info.is_none() => self.start_timeline_export(false),
            ExportAction::Start => self.start_export_job(),
            ExportAction::SavePreset => {
                let name = self.export_ui.preset_name.trim().to_string();
                if name.is_empty() {
                    self.export_ui.note = Some("Give the preset a name first.".into());
                    return;
                }
                let preset = self.export_ui.form.settings();
                self.user_call(true, UserCall::SavePreset { name: name.clone(), preset });
                self.export_ui.form.preset = name;
            }
            ExportAction::DeletePreset(name) => {
                self.user_call(true, UserCall::DeletePreset { name });
                self.export_ui.form.preset.clear();
            }
        }
    }

    /// `refresh`: ask for the options again when the call finishes (a preset was saved or deleted).
    fn user_call(&mut self, refresh: bool, call: UserCall) {
        match self.service.as_ref().and_then(|s| s.user_call(call)) {
            Some(job) => self.export_ui.calls.push((refresh, job)),
            None => self.export_ui.note = Some("The retake service cannot do that.".into()),
        }
    }

    /// The whole-timeline export: the editor's own render of everything on the timeline. A file that is already there is only replaced after the user said so.
    fn start_timeline_export(&mut self, confirmed: bool) {
        let path = self.export_ui.timeline_path.clone();
        if path.exists() && !confirmed {
            self.export_ui.overwrite_confirm = true;
            return;
        }
        self.export_ui.open = false;
        self.export_ui.overwrite_confirm = false;
        let use_hardware = self.export_ui.timeline_hw;
        self.perform(Action::StartExport { path, use_hardware });
    }

    /// With the project saved, the service's export (with its options) is the natural default; otherwise the whole timeline. Only until the user chooses.
    fn pick_default_kind(&mut self) {
        let have_folder = self.editor.project_path().is_some_and(|p| simplecut_core::project_folder::folder_of(std::path::Path::new(p)).is_some());
        if !self.export_ui.kind_chosen {
            self.export_ui.form.kind = if self.export_ui.info.is_some() && have_folder { "video".into() } else { "timeline".into() };
        }
    }

    fn start_export_job(&mut self) {
        let ui = &mut self.export_ui;
        ui.error = None;
        ui.result = None;
        ui.note = None;
        let Some(info) = ui.info.clone() else { return };
        if let Some(p) = ui.form.problem(&info) {
            ui.error = Some(p);
            return;
        }
        let request = match self.editor.analysis_requests() {
            Ok(r) => r.into_iter().next(),
            Err(e) => {
                self.export_ui.error = Some(e.to_string());
                return;
            }
        };
        let Some(source) = request else {
            self.export_ui.error = Some("Add a video to the timeline first.".into());
            return;
        };
        let mut body = self.export_ui.form.body();
        if let (Some(dest), Some(map)) = (&self.export_ui.dest_folder, body.as_object_mut()) {
            map.insert("destination".into(), json!(dest.to_string_lossy()));
        }
        match self.service.as_ref().and_then(|s| s.export(ExportRequest { item: source.item, path: source.path, body })) {
            Some(job) => {
                self.export_ui.log = StageLog::new();
                self.export_ui.job = Some(job);
            }
            None => self.export_ui.error = Some("The retake service cannot export.".into()),
        }
    }

    /// Called once per frame: follow the export job (its events and its end) and the preset / options calls.
    pub(crate) fn pump_exports(&mut self) {
        if let Some(mut job) = self.export_ui.job.take() {
            for e in job.events() {
                self.export_ui.log.push(e);
            }
            match job.poll() {
                JobState::Running { .. } => self.export_ui.job = Some(job),
                JobState::Done(text) => {
                    self.export_ui.log.finish();
                    self.export_ui.result = Some(result_lines(&text));
                }
                JobState::Failed(why) => {
                    for e in job.events() {
                        self.export_ui.log.push(e);
                    }
                    self.export_ui.log.finish();
                    // the service's sentence, then the exact message of the stage that failed (ffmpeg's own words for a render error)
                    let exact = self.export_ui.log.error().map(|(label, msg)| format!("\n{label}: {msg}")).unwrap_or_default();
                    self.export_ui.error = Some(format!("{why}{exact}"));
                }
                JobState::Cancelled => {
                    self.export_ui.log.finish();
                    self.export_ui.note = Some("Export cancelled.".into());
                }
            }
        }
        if self.export_ui.calls.is_empty() {
            return;
        }
        let mut again = false;
        let mut keep = Vec::new();
        for (refresh, mut job) in std::mem::take(&mut self.export_ui.calls) {
            match job.poll() {
                JobState::Running { .. } => keep.push((refresh, job)),
                JobState::Done(text) => {
                    let v: Value = serde_json::from_str(&text).unwrap_or(Value::Null);
                    let status = v["status"].as_u64().unwrap_or(0);
                    if (200..300).contains(&status) {
                        if let Some(info) = ExportOptionsInfo::from_json(&v["body"]) {
                            self.apply_options(info);
                        }
                        again |= refresh;
                    } else {
                        self.export_ui.note = Some(v["body"]["error"].as_str().map_or_else(|| "The retake service refused that.".to_string(), str::to_string));
                    }
                }
                JobState::Failed(why) => self.export_ui.note = Some(why),
                JobState::Cancelled => {}
            }
        }
        self.export_ui.calls = keep;
        if again {
            self.user_call(false, UserCall::ExportOptions);
        }
    }

    fn apply_options(&mut self, info: ExportOptionsInfo) {
        if self.export_ui.info.is_none() {
            let first = info.presets.iter().find(|p| p.name == "YouTube 1080p").or_else(|| info.presets.first());
            let kind = self.export_ui.form.kind.clone();
            self.export_ui.form = first.map_or_else(Form::default, |p| Form::from_preset(&p.name, &p.settings, &info));
            if self.export_ui.kind_chosen {
                self.export_ui.form.kind = kind;
            }
        }
        self.export_ui.info = Some(info);
        self.pick_default_kind();
    }
}

/// The lines shown for a finished export: where it is, and what it really is (ffprobe), including whether the frame rate is constant or variable.
fn result_lines(text: &str) -> Vec<String> {
    let v: Value = serde_json::from_str(text).unwrap_or(Value::Null);
    let mut lines = vec![format!("Saved: {}", v["relative"].as_str().unwrap_or("data/exports/…"))];
    let verify = &v["verify"];
    if verify.is_object() {
        let mut parts = Vec::new();
        if let (Some(w), Some(h)) = (verify["width"].as_u64(), verify["height"].as_u64()) {
            parts.push(format!("{w}×{h}"));
        }
        if let Some(f) = verify["fps"].as_f64() {
            parts.push(format!("{f:.3} fps"));
        }
        if let Some(m) = verify["frame_rate_mode"].as_str() {
            parts.push(format!("{m} ({})", if m == "VFR" { "variable frame rate" } else { "constant frame rate" }));
        }
        if let Some(d) = verify["duration_s"].as_f64() {
            parts.push(format!("{d:.1} s"));
        }
        if let Some(b) = verify["bytes"].as_u64() {
            parts.push(format!("{:.1} MB", b as f64 / 1e6));
        }
        lines.push(format!("Output: {}", parts.join(" · ")));
    }
    for n in v["notes"].as_array().into_iter().flatten().filter_map(Value::as_str) {
        lines.push(n.chars().take(300).collect());
    }
    lines
}

// ── the window ─────────────────────────────────────────────────────────────

/// A labelled drop-down: the label on the left (a fixed width, so the boxes line up), the box on the right. Returns the item the user picked, if any.
fn combo<T: Clone + PartialEq>(ui: &mut egui::Ui, id: &str, label: &str, current: &T, shown: String, items: Vec<(String, T)>) -> Option<T> {
    let mut picked = None;
    ui.horizontal(|ui| {
        ui.allocate_ui_with_layout(egui::vec2(150.0, 20.0), egui::Layout::left_to_right(egui::Align::Center), |ui| {
            ui.add(egui::Label::new(RichText::new(label).color(theme::tokens().text_dim)).truncate());
        });
        egui::ComboBox::from_id_salt(id).selected_text(shown).width(260.0).show_ui(ui, |ui| {
            for (text, value) in items {
                if crate::retake_options::menu_row(ui, &value == current, &text).clicked() {
                    picked = Some(value);
                }
            }
        });
    });
    picked
}

/// The one Export dialog. With the retake service connected it starts with "What to export": the whole timeline (the editor's own render, with music and volume), or the
/// retake cut and the other files the service makes. Each kind has its destination and the Export button is in the dialog.
pub(crate) fn window(app: &mut SimpleCutApp, ctx: &egui::Context) {
    if !app.export_ui.open {
        return;
    }
    let t = theme::tokens();
    let have_folder = app.editor.project_path().is_some_and(|p| simplecut_core::project_folder::folder_of(std::path::Path::new(p)).is_some());
    let mut queue: Vec<ExportAction> = Vec::new();
    let running = app.export_ui.running();
    let info = app.export_ui.info.clone();
    let service = app.service.is_some();
    let mut form = app.export_ui.form.clone();
    let mut preset_name = app.export_ui.preset_name.clone();
    let mut timeline_hw = app.export_ui.timeline_hw;
    let (note, error, result) = (app.export_ui.note.clone(), app.export_ui.error.clone(), app.export_ui.result.clone());
    let (timeline_path, overwrite) = (app.export_ui.timeline_path.clone(), app.export_ui.overwrite_confirm);
    let dest_folder = app.export_ui.dest_folder.clone();
    let log = app.export_ui.log.clone();
    let mut chosen_kind = false;

    egui::Window::new("Export options").collapsible(false).resizable(false).anchor(Align2::CENTER_CENTER, [0.0, 0.0]).show(ctx, |ui| {
        ui.set_min_width(500.0);
        // ── what to export (only when the service offers more than the timeline)
        if let Some(info) = &info {
            ui.add_enabled_ui(!running, |ui| {
                let mut kinds: Vec<(String, String)> = vec![("Video — whole timeline (MP4)".into(), "timeline".into())];
                kinds.extend(["video", "audio", "subtitles", "cutlist", "transcript"].iter().map(|k| (kind_label(k).to_string(), (*k).to_string())));
                if let Some(k) = combo(ui, "kind", "What to export", &form.kind.clone(), kind_label(&form.kind).to_string(), kinds) {
                    chosen_kind = true;
                    form.kind = k;
                    form.format = match form.kind.as_str() {
                        "audio" => info.audio_formats.first().cloned().unwrap_or_default(),
                        other => info.text_formats.iter().find(|(n, _)| n == other).and_then(|(_, f)| f.first().cloned()).unwrap_or_default(),
                    };
                }
            });
            ui.add_space(4.0);
        } else if service {
            ui.label(RichText::new("Asking the retake service what else it can export…").color(t.text_dim).small());
        }

        if form.kind == "timeline" || info.is_none() {
            // ── the whole timeline: the editor's own export
            ui.label(RichText::new("Preset: YouTube 1080p Full HD (1920×1080, H.264 + AAC)").color(t.text_dim));
            ui.add_space(8.0);
            ui.horizontal(|ui| {
                ui.label("Destination:");
                let shown = timeline_path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_else(|| timeline_path.to_string_lossy().into_owned());
                ui.label(RichText::new(shown).color(t.text));
                if ui.button("Browse...").clicked() {
                    queue.push(ExportAction::BrowseTimeline);
                }
            });
            if let Some(dir) = timeline_path.parent().filter(|d| !d.as_os_str().is_empty()) {
                ui.label(RichText::new(format!("in {}", dir.display())).color(t.text_dim).small());
            }
            ui.add_space(4.0);
            ui.checkbox(&mut timeline_hw, "Use graphics card if available");
            ui.add_space(8.0);
            if overwrite {
                ui.label(RichText::new("A file with this name already exists. Overwrite?").color(t.danger));
                ui.add_space(6.0);
                ui.horizontal(|ui| {
                    if ui.button("Overwrite").clicked() {
                        queue.push(ExportAction::Overwrite);
                    }
                    if ui.button("Cancel").clicked() {
                        queue.push(ExportAction::KeepExisting);
                    }
                });
            } else {
                ui.horizontal(|ui| {
                    if ui.button("Start export").clicked() {
                        queue.push(ExportAction::Start);
                    }
                    if ui.button("Cancel").clicked() {
                        queue.push(ExportAction::Close);
                    }
                });
            }
        } else if let Some(info) = &info {
            // ── the retake service's exports
            ui.add(egui::Label::new(RichText::new("The cut comes from the project's saved cut list; without one the whole video is exported. Names are never reused: an earlier export is never overwritten.").color(t.text_dim).small()).wrap());
            if !have_folder {
                ui.label(RichText::new("Save the project first: these exports use its folder.").color(t.danger));
            }
            ui.add_space(6.0);
            ui.add_enabled_ui(!running, |ui| {
                if form.kind != "video" {
                    let formats: Vec<String> = if form.kind == "audio" { info.audio_formats.clone() } else { info.text_formats.iter().find(|(n, _)| *n == form.kind).map(|(_, f)| f.clone()).unwrap_or_default() };
                    let items = formats.iter().map(|f| (f.to_uppercase(), f.clone())).collect();
                    if let Some(f) = combo(ui, "format", "Format", &form.format.clone(), form.format.to_uppercase(), items) {
                        form.format = f;
                    }
                }
                if form.kind == "video" {
                    // ── preset
                    let items: Vec<(String, String)> = info.presets.iter().map(|p| (if p.builtin { p.name.clone() } else { format!("{} (mine)", p.name) }, p.name.clone())).collect();
                    let shown = if form.preset.is_empty() { "Custom".to_string() } else { form.preset.clone() };
                    if let Some(name) = combo(ui, "preset", "Preset", &form.preset.clone(), shown, items)
                        && let Some(p) = info.preset(&name)
                    {
                        let kind = form.kind.clone();
                        form = Form::from_preset(&p.name, &p.settings, info);
                        form.kind = kind;
                    }
                    if let Some(p) = info.preset(&form.preset) {
                        ui.label(RichText::new(&p.description).color(t.text_dim).small());
                    }
                    ui.add_space(4.0);
                    // ── size
                    let mut res_items: Vec<(String, Res)> = vec![(res_label(&Res::Source), Res::Source)];
                    res_items.extend(info.heights.iter().map(|h| (res_label(&Res::Height(h.clone())), Res::Height(h.clone()))));
                    res_items.extend(info.canvases.iter().map(|c| (res_label(&Res::Canvas(c.clone())), Res::Canvas(c.clone()))));
                    res_items.push((res_label(&Res::Custom), Res::Custom));
                    if let Some(r) = combo(ui, "res", "Size", &form.res.clone(), res_label(&form.res), res_items) {
                        form.res = r;
                    }
                    if form.res == Res::Custom {
                        ui.horizontal(|ui| {
                            ui.label("Width");
                            ui.add(egui::DragValue::new(&mut form.custom_w).range(info.dim_min..=info.dim_max).speed(2.0));
                            ui.label("Height");
                            ui.add(egui::DragValue::new(&mut form.custom_h).range(info.dim_min..=info.dim_max).speed(2.0));
                            ui.checkbox(&mut form.lock_aspect, "Lock aspect ratio");
                        });
                    }
                    if matches!(form.res, Res::Canvas(_)) || (form.res == Res::Custom && !form.lock_aspect) {
                        let fits: Vec<(String, String)> = info.fits.iter().map(|f| (f.clone(), f.clone())).collect();
                        if let Some(f) = combo(ui, "fit", "Picture fit", &form.fit.clone(), form.fit.clone(), fits) {
                            form.fit = f;
                        }
                    }
                    // ── frame rate
                    let mut fps_items = vec![(fps_label(&Fps::Source), Fps::Source)];
                    fps_items.extend(info.fps_choices.iter().map(|n| (fps_label(&Fps::Fixed(*n)), Fps::Fixed(*n))));
                    fps_items.push((fps_label(&Fps::Custom), Fps::Custom));
                    if let Some(f) = combo(ui, "fps", "Frame rate", &form.fps.clone(), fps_label(&form.fps), fps_items) {
                        if let Fps::Fixed(n) = f {
                            form.custom_fps = f64::from(n);
                        }
                        form.fps = f;
                    }
                    if form.fps == Fps::Custom {
                        ui.horizontal(|ui| {
                            ui.label("Frames per second");
                            ui.add(egui::DragValue::new(&mut form.custom_fps).range(info.fps_range.0..=info.fps_range.1).speed(0.1));
                        });
                    }
                    // ── codec and quality
                    let codecs: Vec<(String, String)> = info.codecs.iter().map(|c| (if c == "h264" { "H.264".to_string() } else { "H.265 (HEVC)".to_string() }, c.clone())).collect();
                    if let Some(c) = combo(ui, "codec", "Codec", &form.codec.clone(), if form.codec == "h264" { "H.264".to_string() } else { "H.265 (HEVC)".to_string() }, codecs) {
                        form.codec = c;
                    }
                    ui.horizontal(|ui| {
                        ui.label("Quality");
                        ui.radio_value(&mut form.quality, Quality::Preset, "Preset");
                        ui.radio_value(&mut form.quality, Quality::Crf, "CRF");
                        ui.radio_value(&mut form.quality, Quality::Bitrate, "Bitrate");
                    });
                    ui.horizontal(|ui| match form.quality {
                        Quality::Preset => {
                            for p in ["draft", "standard", "high"] {
                                ui.selectable_value(&mut form.q_preset, p.to_string(), p);
                            }
                        }
                        Quality::Crf => {
                            ui.add(egui::DragValue::new(&mut form.crf).range(info.crf_range.0..=info.crf_range.1));
                            ui.label(RichText::new("lower = better quality, bigger file").color(t.text_dim).small());
                        }
                        Quality::Bitrate => {
                            ui.add(egui::DragValue::new(&mut form.bitrate).range(info.bitrate_range.0..=info.bitrate_range.1).suffix(" kbps"));
                        }
                    });
                    let speeds: Vec<(String, String)> = info.speeds.iter().map(|s| (s.clone(), s.clone())).collect();
                    if let Some(s) = combo(ui, "speed", "Encoder speed", &form.speed.clone(), form.speed.clone(), speeds) {
                        form.speed = s;
                    }
                    if info.hardware.contains(&form.codec) {
                        ui.checkbox(&mut form.hardware, "Use the NVIDIA GPU encoder (NVENC)");
                    } else {
                        form.hardware = false;
                    }
                }
                if form.kind == "video" || form.kind == "audio" {
                    // ── audio
                    let rates: Vec<(String, u32)> = info.audio_bitrates.iter().map(|b| (format!("{b} kbps"), *b)).collect();
                    if let Some(b) = combo(ui, "abr", "Audio bitrate", &form.audio_bitrate, format!("{} kbps", form.audio_bitrate), rates) {
                        form.audio_bitrate = b;
                    }
                    ui.horizontal(|ui| {
                        ui.add(egui::DragValue::new(&mut form.fade_ms).range(0.0..=info.fade_max).suffix(" ms"));
                        ui.label(RichText::new("short audio crossfade at every cut (0 = off)").color(t.text_dim));
                    });
                    if form.kind == "video" {
                        ui.checkbox(&mut form.loudness, "Normalise loudness (off by default)");
                    }
                }
                if form.kind == "video" {
                    // ── presets of your own
                    ui.add_space(6.0);
                    ui.horizontal(|ui| {
                        ui.add(egui::TextEdit::singleline(&mut preset_name).hint_text("Name for a new preset").desired_width(180.0));
                        if ui.button("Save as preset").on_hover_text("Kept in your own settings folder, not in the project").clicked() {
                            queue.push(ExportAction::SavePreset);
                        }
                        if info.preset(&form.preset).is_some_and(|p| !p.builtin) && ui.button("Delete this preset").clicked() {
                            queue.push(ExportAction::DeletePreset(form.preset.clone()));
                        }
                    });
                }
                // ── where the file goes
                ui.add_space(6.0);
                ui.horizontal(|ui| {
                    ui.label("Destination folder:");
                    let shown = dest_folder.as_ref().map_or_else(|| "the project's data/exports folder".to_string(), |d| d.display().to_string());
                    ui.add(egui::Label::new(RichText::new(shown).color(t.text)).truncate());
                    if ui.button("Browse folder…").clicked() {
                        queue.push(ExportAction::BrowseFolder);
                    }
                    if dest_folder.is_some() && ui.small_button("Use the project folder").clicked() {
                        queue.push(ExportAction::UseProjectFolder);
                    }
                });
            });
            if let Some(p) = form.problem(info) {
                ui.label(RichText::new(p).color(t.danger));
            }
            ui.add_space(8.0);
            if running {
                let frac = log.state("export_render");
                let pct = match frac {
                    simplecut_core::StageState::Running { pct, .. } => pct.unwrap_or(0.0),
                    _ => 0.0,
                };
                ui.add(egui::ProgressBar::new((pct / 100.0).clamp(0.0, 1.0)).text(format!("Rendering {pct:.0}%")).desired_width(460.0));
                if ui.button("Cancel export").clicked() {
                    queue.push(ExportAction::Cancel);
                }
            } else {
                ui.horizontal(|ui| {
                    if ui.add_enabled(have_folder && form.problem(info).is_none(), egui::Button::new("Start export")).clicked() {
                        queue.push(ExportAction::Start);
                    }
                    if ui.button("Cancel").clicked() {
                        queue.push(ExportAction::Close);
                    }
                });
            }
        }
        if let Some(lines) = &result {
            ui.add_space(8.0);
            for (i, l) in lines.iter().enumerate() {
                ui.add(egui::Label::new(RichText::new(l.as_str()).color(if i == 0 { t.text } else { t.text_dim })).wrap());
            }
        }
        if let Some(e) = &error {
            ui.add_space(8.0);
            ui.add(egui::Label::new(RichText::new(e.as_str()).color(t.danger)).wrap());
        }
        if let Some(n) = &note {
            ui.add_space(4.0);
            ui.add(egui::Label::new(RichText::new(n.as_str()).color(t.text_dim)).wrap());
        }
    });
    app.export_ui.form = form;
    app.export_ui.preset_name = preset_name;
    app.export_ui.timeline_hw = timeline_hw;
    app.export_ui.kind_chosen |= chosen_kind;
    for a in queue {
        app.queue(Action::Exports(a));
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;

    /// The real reply of the Python service (captured from `GET /v1/export/options`).
    fn info() -> ExportOptionsInfo {
        ExportOptionsInfo::from_json(&serde_json::from_str::<Value>(include_str!("../tests/fixtures/export_options.json")).unwrap()).unwrap()
    }

    #[test]
    fn every_preset_fills_the_form_and_the_form_gives_the_same_settings_back() {
        let info = info();
        for p in &info.presets {
            let form = Form::from_preset(&p.name, &p.settings, &info);
            let back = form.settings();
            for key in ["resolution", "fps", "video", "audio"] {
                let (a, b) = (&p.settings[key], &back[key]);
                // the form always spells every field out; the preset may leave some to the defaults, so compare what the preset does say
                for (k, v) in a.as_object().into_iter().flatten() {
                    if v.is_object() {
                        for (k2, v2) in v.as_object().unwrap() {
                            assert_eq!(&b[k][k2], v2, "{} / {key}.{k}.{k2}", p.name);
                        }
                    } else {
                        assert_eq!(&b[k], v, "{} / {key}.{k}", p.name);
                    }
                }
                if !a.is_object() {
                    assert_eq!(a, b, "{} / {key}", p.name);
                }
            }
        }
    }

    #[test]
    fn a_custom_size_and_frame_rate_are_checked_before_anything_is_sent() {
        let info = info();
        let mut f = Form { res: Res::Custom, custom_w: 1280, custom_h: 720, ..Form::default() };
        assert!(f.problem(&info).is_none());
        f.custom_w = 1281;
        assert!(f.problem(&info).unwrap().contains("Width must be an even number"));
        f.custom_w = 1280;
        f.custom_h = 9000;
        assert!(f.problem(&info).unwrap().contains("Height must be between 16 and 7680"));
        f.custom_h = 8;
        assert!(f.problem(&info).unwrap().contains("Height must be between"));
        f = Form { fps: Fps::Custom, custom_fps: 0.5, ..Form::default() };
        assert!(f.problem(&info).unwrap().contains("Frame rate must be between 1 and 240"));
        f.custom_fps = 23.976;
        assert!(f.problem(&info).is_none());
        assert!(
            Form { kind: "subtitles".into(), res: Res::Custom, custom_w: 7, ..Form::default() }.problem(&info).is_none(),
            "video options do not apply to a text export"
        );
    }

    #[test]
    fn the_request_body_matches_the_kind() {
        let info = info();
        let video = Form::from_preset("YouTube 1080p", &info.preset("YouTube 1080p").unwrap().settings, &info).body();
        assert_eq!((video["kind"].as_str(), video["preset"].as_str()), (Some("video"), Some("YouTube 1080p")));
        assert_eq!(video["resolution"], json!({"mode": "preset", "preset": "1080p"}));
        assert_eq!(video["video"]["quality"], json!({"preset": "high"}));
        assert_eq!(video["audio"]["normalize_loudness"], false, "loudness normalisation is off unless asked for");
        let custom = Form { res: Res::Custom, custom_w: 1000, custom_h: 1000, lock_aspect: true, ..Form::default() }.settings();
        assert_eq!(custom["resolution"], json!({"mode": "custom", "width": 1000, "height": 1000, "lock_aspect": true}));
        let free = Form { res: Res::Custom, lock_aspect: false, fit: "crop".into(), ..Form::default() }.settings();
        assert_eq!(free["resolution"]["fit"], "crop");
        let bitrate = Form { quality: Quality::Bitrate, bitrate: 4000, ..Form::default() }.settings();
        assert_eq!(bitrate["video"]["quality"], json!({"bitrate_kbps": 4000}));
        let text = Form { kind: "subtitles".into(), format: "srt".into(), ..Form::default() }.body();
        assert_eq!(text, json!({"kind": "subtitles", "format": "srt"}));
        let audio = Form { kind: "audio".into(), format: "mp3".into(), ..Form::default() }.body();
        assert_eq!((audio["kind"].as_str(), audio["format"].as_str()), (Some("audio"), Some("mp3")));
        assert!(audio.get("preset").is_none() && audio["audio"].is_object());
    }

    #[test]
    fn the_finished_output_says_whether_the_frame_rate_is_constant_or_variable() {
        let vfr = result_lines(&json!({"relative": "data/exports/a.mp4", "verify": {"width": 640, "height": 360, "fps": 15.0, "frame_rate_mode": "VFR", "duration_s": 5.0}, "notes": ["n"]}).to_string());
        assert_eq!(vfr[0], "Saved: data/exports/a.mp4");
        assert_eq!(vfr[1], "Output: 640×360 · 15.000 fps · VFR (variable frame rate) · 5.0 s");
        assert_eq!(vfr[2], "n");
        assert_eq!(result_lines("not json"), vec!["Saved: data/exports/…".to_string()], "an unreadable result still says something");
    }
}
