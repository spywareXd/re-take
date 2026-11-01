//! Retake options: the transcription model, the silence gap that splits sentences, and what goes with them: the project folder the settings live in, results that went
//! stale when a setting changed (and the confirmation before anything is replaced), the warning when the source video is not the one the project was made with, and the
//! import of data from the old per-machine location.
//!
//! The settings are stored by the service in the project folder (`X/data/settings.json`); the editor shows them, sends them with an analysis, and asks the service
//! (`ProjectCall`) when something has to change. A project without a folder yet keeps the choice in memory and sends it with the next analysis.

use std::path::Path;

use egui::{Align2, RichText};
use retake_editor_core::{AnalysisJob, AnalysisSettings, JobState, ProjectCall, WorkCall, WorkKind, project_folder};
use serde_json::Value;

use crate::app::{Action, RetakeEditorApp};
use crate::theme;

#[derive(Debug, Clone, PartialEq)]
pub enum OptionsAction {
    Open,
    Close,
    /// Choose a transcription model (by the name the service listed).
    SetModel(String),
    /// The slider moved: remember the value and refresh the sentence count.
    SetSilenceGap(f64),
    /// The slider was let go: store the value in the project.
    CommitSilenceGap,
    /// Re-transcribe now with the newly chosen model (the old transcript is kept by the service as a numbered copy).
    ConfirmRetranscribe,
    /// Go back to the model the transcript was made with.
    KeepOldTranscript,
    /// Re-split the saved transcript at the chosen gap without re-transcribing (asks what will be replaced first).
    ResplitNow,
    ConfirmResplit,
    CancelAsk,
    /// Copy the old per-machine data of this video into the project (the originals stay).
    ImportLegacy,
    /// Ask before setting the transcription / the keeper groups aside.
    AskReset(WorkKind),
    ConfirmReset,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Call {
    Status,
    Settings,
    Preview,
    ResplitCheck,
    Resplit,
    Import,
    /// The automatic copy of the old data into a project that was just saved.
    Adopt,
    /// Setting work aside / bringing it back (Ctrl+Z, Ctrl+Y, the Reset buttons).
    Work,
}

/// What the service said about the project (`GET /v1/project/status`).
#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct ProjectStatus {
    pub transcript_model: Option<String>,
    pub saved_model: Option<String>,
    pub saved_gap: Option<f64>,
    pub stale: Vec<String>,
    /// What the user set aside on purpose (Ctrl+Z, Reset): not "out of date".
    pub reset: Vec<String>,
    pub has_transcript: bool,
    pub has_units: bool,
    /// `ok`, `missing`, `changed` or `none`.
    pub source: String,
    pub source_message: String,
    pub legacy: Vec<String>,
}

impl ProjectStatus {
    fn from_json(v: &Value) -> Self {
        let text = |x: &Value| x.as_str().map(|s| s.chars().take(300).collect::<String>());
        let list = |x: &Value| x.as_array().map(|a| a.iter().filter_map(text).take(40).collect()).unwrap_or_default();
        Self {
            transcript_model: text(&v["transcript_model"]),
            saved_model: text(&v["settings"]["transcription_model"]),
            saved_gap: v["settings"]["silence_gap_s"].as_f64().filter(|g| g.is_finite()),
            stale: list(&v["stale"]),
            reset: list(&v["reset"]),
            has_transcript: v["has"]["transcript"].as_bool().unwrap_or(false),
            has_units: v["has"]["units"].as_bool().unwrap_or(false),
            source: text(&v["source"]["status"]).unwrap_or_else(|| "none".into()),
            source_message: text(&v["source"]["message"]).unwrap_or_default(),
            legacy: list(&v["legacy"]),
        }
    }
}

/// A question the user has to answer before anything is replaced.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Ask {
    Retranscribe { from: String, to: String },
    ResplitOffer { text: String },
    ResplitConfirm { gap: f64, will_replace: Vec<String> },
    Reset { what: WorkKind },
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Preview {
    pub gap: f64,
    pub units: u64,
    pub current: u64,
}

#[derive(Default)]
pub(crate) struct OptionsUi {
    pub open: bool,
    /// The chosen model / gap (None: the project's saved value, else the service's default).
    pub model: Option<String>,
    pub silence_gap_s: Option<f64>,
    /// The user agreed to replace an existing transcript with the next analysis.
    pub replace_next: bool,
    pub status: Option<ProjectStatus>,
    pub preview: Option<Preview>,
    pub ask: Option<Ask>,
    pub note: Option<String>,
    source_warned: bool,
    preview_wanted: Option<f64>,
    calls: Vec<(Call, Box<dyn AnalysisJob>)>,
}

/// One row of a list: a fixed-size slot, with the highlight painted by hand. Hovering can only change how the row is painted, never where it or the rows below it sit
/// (egui's own hovered button grows its layout a little, and the whole list jittered under the mouse). `name` is what a screen reader hears (default: the text).
pub(crate) fn menu_row_named(ui: &mut egui::Ui, selected: bool, text: &str, name: Option<&str>) -> egui::Response {
    let (rect, response) = ui.allocate_exact_size(egui::vec2(ui.available_width(), ui.spacing().interact_size.y), egui::Sense::click());
    response.widget_info(|| egui::WidgetInfo::selected(egui::WidgetType::Button, ui.is_enabled(), selected, name.unwrap_or(text)));
    if ui.is_rect_visible(rect) {
        let visuals = ui.style().interact_selectable(&response, selected);
        if selected || response.hovered() || response.has_focus() {
            ui.painter().rect_filled(rect, visuals.corner_radius, visuals.weak_bg_fill);
        }
        let font = egui::TextStyle::Button.resolve(ui.style());
        let at = egui::pos2(rect.left() + ui.spacing().button_padding.x, rect.center().y);
        ui.painter().text(at, egui::Align2::LEFT_CENTER, text, font, visuals.text_color());
    }
    response
}

pub(crate) fn menu_row(ui: &mut egui::Ui, selected: bool, text: &str) -> egui::Response {
    menu_row_named(ui, selected, text, None)
}

/// What a stored result is called for the user, each name once, in order.
fn friendly(names: &[String]) -> String {
    let mut out: Vec<&str> = Vec::new();
    for n in names {
        let label = match n.as_str() {
            "units" | "units_meta" => "sentences",
            "keepers" => "keeper groups",
            "cutlist" => "cut list",
            n if n.starts_with("pairs__") || n.starts_with("scores__") => "pair scores",
            other => other,
        };
        if !out.contains(&label) {
            out.push(label);
        }
    }
    out.join(", ")
}

impl RetakeEditorApp {
    /// The settings that go with the next analysis.
    pub(crate) fn options_settings(&self) -> AnalysisSettings {
        AnalysisSettings { model: self.options_ui.model.clone(), silence_gap_s: self.options_ui.silence_gap_s, replace_existing: self.options_ui.replace_next }
    }

    fn project_folder_path(&self) -> Option<std::path::PathBuf> {
        self.editor.project_path().and_then(|p| project_folder::folder_of(Path::new(p)))
    }

    /// The video on the timeline (the one the analysis is about), on disk.
    pub(crate) fn source_video_path(&self) -> Option<String> {
        self.editor.analysis_requests().ok().and_then(|r| r.into_iter().next()).map(|r| r.path)
    }

    /// The Reset buttons (after the question): set the transcription or the keeper groups aside. Nothing is deleted and the answers stay.
    fn reset_work(&mut self, what: WorkKind) {
        self.work(WorkCall::Reset { what, path: String::new() });
        match what {
            WorkKind::Transcription => {
                self.steps = Default::default();
                self.perform(Action::CloseReview);
                self.options_ui.note = Some("Transcription reset.".into());
            }
            WorkKind::KeeperGroups => {
                self.steps.done.retain(|s| !matches!(s.kind, crate::retake_steps::StepKind::Keepers { .. }));
                self.clear_ai_choices();
                self.options_ui.note = Some("Keeper groups reset.".into());
            }
        }
    }

    /// A project was just saved into its own folder: bring the transcript, sentences and pair scores that were made before the save (kept by the service in its old
    /// per-machine cache) into the project, so the work already done follows it. They are copied; the originals stay.
    pub(crate) fn adopt_existing_work(&mut self) {
        if self.project_folder_path().is_none() || self.service.is_none() {
            return;
        }
        if let Some(path) = self.source_video_path() {
            self.make_call(Call::Adopt, ProjectCall::ImportLegacy { path });
        }
    }

    pub(crate) fn make_work_call(&mut self, call: WorkCall) {
        match self.service.as_ref().and_then(|s| s.work_call(call)) {
            Some(job) => self.options_ui.calls.push((Call::Work, job)),
            None => self.options_ui.note = Some("The retake service cannot set work aside.".into()),
        }
    }

    fn make_call(&mut self, kind: Call, call: ProjectCall) {
        let job = self.service.as_ref().and_then(|s| s.project_call(call));
        match job {
            Some(job) => self.options_ui.calls.push((kind, job)),
            None => self.options_ui.note = Some("The retake service cannot do that.".into()),
        }
    }

    /// The project (file) changed: tell the service which folder it lives in, point the editor's own logs into it, and ask for the project's status (settings,
    /// stale results, whether the source video is still the one it was made with).
    pub(crate) fn sync_project_folder(&mut self) {
        let folder = self.project_folder_path();
        if let Some(service) = &self.service {
            service.set_project(folder.clone());
        }
        let (default_labels, default_spans) = (self.retakes.default_label_log.clone(), self.retakes.default_span_log.clone());
        self.retakes.label_log = folder.as_ref().map(|f| f.join("data").join("labels_editor.jsonl")).or(default_labels);
        self.retakes.span_log = folder.as_ref().map(|f| f.join("data").join("span_edits.jsonl")).or(default_spans);
        let ui = &mut self.options_ui;
        ui.status = None;
        ui.preview = None;
        ui.ask = None;
        ui.note = None;
        ui.model = None;
        ui.silence_gap_s = None;
        ui.replace_next = false;
        ui.source_warned = false;
        ui.calls.clear();
        if folder.is_some() && self.service.is_some() {
            self.make_call(Call::Status, ProjectCall::Status);
        }
    }

    pub(crate) fn perform_options(&mut self, action: OptionsAction) {
        let have_folder = self.project_folder_path().is_some();
        let silence = self.service.as_ref().and_then(|s| s.options().silence_gap);
        match action {
            OptionsAction::Open => {
                self.options_ui.open = true;
                if have_folder {
                    self.make_call(Call::Status, ProjectCall::Status);
                }
            }
            OptionsAction::Close => self.options_ui.open = false,
            OptionsAction::SetModel(name) => {
                self.options_ui.model = Some(name.clone());
                self.options_ui.ask = None;
                if have_folder {
                    self.make_call(Call::Settings, ProjectCall::SetSettings { model: Some(name), silence_gap_s: None });
                }
            }
            OptionsAction::SetSilenceGap(g) => {
                let g = silence.map_or(g, |s| s.clamp(g));
                self.options_ui.silence_gap_s = Some(g);
                if have_folder && self.options_ui.status.as_ref().is_some_and(|s| s.has_transcript) {
                    if self.options_ui.calls.iter().any(|(k, _)| *k == Call::Preview) {
                        self.options_ui.preview_wanted = Some(g); // one preview at a time; the latest value is asked for when it returns
                    } else {
                        self.make_call(Call::Preview, ProjectCall::UnitsPreview { silence_gap_s: g });
                    }
                }
            }
            OptionsAction::CommitSilenceGap => {
                if let (true, Some(g)) = (have_folder, self.options_ui.silence_gap_s) {
                    self.make_call(Call::Settings, ProjectCall::SetSettings { model: None, silence_gap_s: Some(g) });
                }
            }
            OptionsAction::ConfirmRetranscribe => {
                self.options_ui.ask = None;
                self.options_ui.replace_next = true;
                self.perform(Action::StartRetakeAnalysis);
            }
            OptionsAction::KeepOldTranscript => {
                let old = self.options_ui.status.as_ref().and_then(|s| s.transcript_model.clone());
                self.options_ui.ask = None;
                if let Some(old) = old {
                    self.perform_options(OptionsAction::SetModel(old));
                }
            }
            OptionsAction::ResplitNow => {
                let gap = self.options_ui.silence_gap_s.or(silence.map(|s| s.default)).unwrap_or(0.7);
                self.options_ui.ask = None;
                self.make_call(Call::ResplitCheck, ProjectCall::Resplit { silence_gap_s: gap, confirm: false });
            }
            OptionsAction::ConfirmResplit => {
                if let Some(Ask::ResplitConfirm { gap, .. }) = self.options_ui.ask.take() {
                    self.make_call(Call::Resplit, ProjectCall::Resplit { silence_gap_s: gap, confirm: true });
                }
            }
            OptionsAction::CancelAsk => self.options_ui.ask = None,
            OptionsAction::AskReset(what) => self.options_ui.ask = Some(Ask::Reset { what }),
            OptionsAction::ConfirmReset => {
                if let Some(Ask::Reset { what }) = self.options_ui.ask.take() {
                    self.reset_work(what);
                }
            }
            OptionsAction::ImportLegacy => match self.source_video_path() {
                Some(path) => self.make_call(Call::Import, ProjectCall::ImportLegacy { path }),
                None => self.options_ui.note = Some("Add a video to the timeline first: the old data is matched to the video.".into()),
            },
        }
    }

    /// Called once per frame: follow the calls to the service and apply their answers.
    pub(crate) fn pump_options(&mut self) {
        if self.options_ui.calls.is_empty() {
            return;
        }
        let mut finished = Vec::new();
        let mut keep = Vec::new();
        for (kind, mut job) in std::mem::take(&mut self.options_ui.calls) {
            match job.poll() {
                JobState::Running { .. } => keep.push((kind, job)),
                done => finished.push((kind, done)),
            }
        }
        self.options_ui.calls = keep;
        for (kind, end) in finished {
            match end {
                JobState::Done(text) => {
                    let v: Value = serde_json::from_str(&text).unwrap_or(Value::Null);
                    self.call_done(kind, v["status"].as_u64().unwrap_or(0), &v["body"]);
                }
                JobState::Failed(why) => self.options_ui.note = Some(why),
                _ => {}
            }
        }
    }

    fn call_done(&mut self, kind: Call, status: u64, body: &Value) {
        let error = || body["error"].as_str().map_or_else(|| "The retake service refused that.".to_string(), |e| e.chars().take(400).collect());
        match (kind, status) {
            (Call::Status, 200) => {
                let st = ProjectStatus::from_json(body);
                if self.options_ui.model.is_none() {
                    self.options_ui.model = st.saved_model.clone();
                }
                if self.options_ui.silence_gap_s.is_none() {
                    self.options_ui.silence_gap_s = st.saved_gap;
                }
                if matches!(st.source.as_str(), "missing" | "changed") && !self.options_ui.source_warned {
                    self.options_ui.source_warned = true;
                    let text = st.source_message.clone();
                    self.fail_text(&text);
                }
                self.options_ui.status = Some(st);
            }
            (Call::Settings, 200) => {
                let ask = body["ask"].as_str().map(str::to_string);
                let newly_stale: Vec<&str> = body["newly_stale"].as_array().map(|a| a.iter().filter_map(Value::as_str).collect()).unwrap_or_default();
                if let Some(text) = ask {
                    let from = self.options_ui.status.as_ref().and_then(|s| s.transcript_model.clone()).unwrap_or_default();
                    self.options_ui.ask = if newly_stale.contains(&"transcript") {
                        Some(Ask::Retranscribe { from, to: self.options_ui.model.clone().unwrap_or_default() })
                    } else {
                        Some(Ask::ResplitOffer { text: text.chars().take(400).collect() })
                    };
                }
                self.make_call(Call::Status, ProjectCall::Status);
            }
            (Call::Preview, 200) => {
                let gap = body["silence_gap_s"].as_f64().unwrap_or(0.0);
                self.options_ui.preview =
                    Some(Preview { gap, units: body["units"].as_u64().unwrap_or(0), current: body["current_units"].as_u64().unwrap_or(0) });
                if let Some(next) = self.options_ui.preview_wanted.take()
                    && (next - gap).abs() > 1e-9
                {
                    self.make_call(Call::Preview, ProjectCall::UnitsPreview { silence_gap_s: next });
                }
            }
            (Call::ResplitCheck, 409) if body["code"] == "resplit_needs_confirmation" => {
                let will_replace =
                    body["will_replace"].as_array().map(|a| a.iter().filter_map(Value::as_str).map(str::to_string).collect()).unwrap_or_default();
                self.options_ui.ask = Some(Ask::ResplitConfirm { gap: body["silence_gap_s"].as_f64().unwrap_or(0.7), will_replace });
            }
            (Call::ResplitCheck | Call::Resplit, 200) => {
                let n = body["units"].as_u64().unwrap_or(0);
                self.options_ui.note = Some(format!("Re-split into {n} sentences."));
                self.options_ui.preview = None;
                self.make_call(Call::Status, ProjectCall::Status);
            }
            (Call::Adopt, 200) => {
                let done: Vec<String> = body["imported"]
                    .as_array()
                    .map(|a| a.iter().filter_map(Value::as_str).filter(|n| *n != "units_meta").map(str::to_string).collect())
                    .unwrap_or_default();
                if !done.is_empty() {
                    self.options_ui.note = Some(format!("Copied into the project: {}.", friendly(&done)));
                    self.make_call(Call::Status, ProjectCall::Status);
                }
            }
            (Call::Work, 200) => {
                if self.project_folder_path().is_some() {
                    self.make_call(Call::Status, ProjectCall::Status);
                }
            }
            (Call::Adopt, _) => {} // nothing to bring along (no earlier analysis, or an older service): not worth a message
            (Call::Import, 200) => {
                let done: Vec<String> = body["imported"]
                    .as_array()
                    .map(|a| a.iter().filter_map(Value::as_str).filter(|n| *n != "units_meta").map(str::to_string).collect())
                    .unwrap_or_default();
                self.options_ui.note = Some(if done.is_empty() {
                    "Nothing to copy: the project already has this data.".into()
                } else {
                    format!("Copied into the project: {}.", friendly(&done))
                });
                self.make_call(Call::Status, ProjectCall::Status);
            }
            (_, _) => self.options_ui.note = Some(error()),
        }
    }
}

// ── the window ─────────────────────────────────────────────────────────────

fn model_line(m: &retake_editor_core::AsrModel) -> String {
    let mut parts = vec![m.name.clone() + if m.default { " (default)" } else { "" }];
    parts.push(if m.downloaded { "downloaded".to_string() } else { m.download_gb.map_or("not downloaded".to_string(), |g| format!("download ≈{g:.1} GB")) });
    if let Some(v) = m.vram_gb {
        parts.push(format!("≈{v:.1} GB GPU"));
    }
    if let Some(s) = m.relative_speed {
        parts.push(format!("{s:.1}× speed"));
    }
    parts.join("  ·  ")
}

pub(crate) fn window(app: &mut RetakeEditorApp, ctx: &egui::Context) {
    if !app.options_ui.open {
        return;
    }
    let t = theme::tokens();
    let amber = theme::GROUP_COLORS[0];
    let options = app.service.as_ref().map(|s| s.options()).unwrap_or_default();
    let busy = app.is_analyzing();
    let working = app.pipeline_running();
    let mut queue: Vec<OptionsAction> = Vec::new();
    let (chosen_model, chosen_gap) = (app.options_ui.model.clone(), app.options_ui.silence_gap_s);
    let status = app.options_ui.status.clone();
    let (preview, ask, note) = (app.options_ui.preview.clone(), app.options_ui.ask.clone(), app.options_ui.note.clone());

    egui::Window::new("Retake options").collapsible(false).resizable(false).anchor(Align2::CENTER_CENTER, [0.0, 0.0]).show(ctx, |ui| {
        ui.set_min_width(460.0);
        if let Some(st) = &status
            && matches!(st.source.as_str(), "missing" | "changed")
        {
            ui.label(RichText::new(&st.source_message).color(t.danger));
            ui.add_space(6.0);
        }

        // ── the transcription model
        ui.label(RichText::new("Transcription model").strong());
        if let Some(info) = &options.transcription {
            let current = chosen_model.clone().unwrap_or_else(|| info.default.clone());
            ui.add_enabled_ui(!busy, |ui| {
                let shown = info.model(&current).map_or_else(|| current.clone(), model_line);
                egui::ComboBox::from_label("Model").selected_text(shown).width(380.0).show_ui(ui, |ui| {
                    for m in &info.models {
                        if menu_row(ui, m.name == current, &model_line(m)).clicked() {
                            queue.push(OptionsAction::SetModel(m.name.clone()));
                        }
                    }
                });
            });
            if let Some(m) = info.model(&current) {
                egui::Grid::new("model-detail").num_columns(2).spacing([12.0, 2.0]).show(ui, |ui| {
                    ui.label(RichText::new("Download").color(t.text_dim));
                    ui.label(match (m.downloaded, m.size_on_disk_gb, m.download_gb) {
                        (true, Some(d), _) => format!("already downloaded ({d:.2} GB on disk)"),
                        (true, None, _) => "already downloaded".to_string(),
                        (false, _, Some(g)) => format!("not downloaded yet: about {g:.1} GB (approximate)"),
                        _ => "not downloaded yet".to_string(),
                    });
                    ui.end_row();
                    ui.label(RichText::new("GPU memory").color(t.text_dim));
                    ui.label(m.vram_gb.map_or("unknown".to_string(), |v| format!("about {v:.1} GB (approximate)")));
                    ui.end_row();
                    ui.label(RichText::new("Speed").color(t.text_dim));
                    ui.label(m.relative_speed.map_or("unknown".to_string(), |s| format!("about {s:.1}× the speed of the default model")));
                    ui.end_row();
                });
                if let Some(w) = &m.warning {
                    ui.add_space(2.0);
                    ui.add(egui::Label::new(RichText::new(format!("Warning: {w}")).color(amber)).wrap());
                }
            }
            if let Some(g) = &info.gpu {
                ui.label(RichText::new(format!("GPU: {} · {:.1} GB, {:.1} GB free now", g.name, g.total_gb, g.free_gb)).color(t.text_dim).small());
            } else {
                ui.label(RichText::new("No NVIDIA GPU found: transcription runs on the CPU.").color(t.text_dim).small());
            }
        }
        ui.add_space(10.0);

        // ── the silence gap
        ui.label(RichText::new("Sentence splitting").strong());
        if let Some(info) = options.silence_gap {
            let mut v = chosen_gap.unwrap_or(info.default);
            ui.add_enabled_ui(!busy, |ui| {
                ui.horizontal(|ui| {
                    ui.spacing_mut().slider_width = 200.0;
                    ui.visuals_mut().widgets.inactive.bg_fill = t.text_dim.gamma_multiply(0.45); // the slider's rail: the theme's own is invisible on this window
                    ui.visuals_mut().widgets.hovered.bg_fill = t.text_dim.gamma_multiply(0.6);
                    let r = ui.add(egui::Slider::new(&mut v, info.min..=info.max).step_by(info.step).suffix(" s").text("Silence gap"));
                    if r.changed() {
                        queue.push(OptionsAction::SetSilenceGap(v));
                    }
                    if r.drag_stopped() || r.lost_focus() {
                        queue.push(OptionsAction::CommitSilenceGap);
                    }
                    if ui.small_button("Default").on_hover_text(format!("Back to {:.2} s", info.default)).clicked() {
                        queue.push(OptionsAction::SetSilenceGap(info.default));
                        queue.push(OptionsAction::CommitSilenceGap);
                    }
                });
            });
            if let Some(p) = &preview {
                ui.label(format!("At {:.2} s the transcript gives {} sentences (now {}).", p.gap, p.units, p.current));
                if status.as_ref().is_some_and(|s| s.has_units)
                    && p.units != p.current
                    && ask.is_none()
                    && ui
                        .add_enabled(!busy, egui::Button::new("Re-split sentences"))
                        .on_hover_text("Uses the saved word timestamps: no re-transcribing")
                        .clicked()
                {
                    queue.push(OptionsAction::ResplitNow);
                }
            }
        }

        // ── start over
        ui.add_space(10.0);
        ui.label(RichText::new("Start over").strong());
        ui.horizontal(|ui| {
            if ui.add_enabled(!working, egui::Button::new("Reset transcription…")).on_hover_text("Set the transcription aside").clicked() {
                queue.push(OptionsAction::AskReset(WorkKind::Transcription));
            }
            if ui.add_enabled(!working, egui::Button::new("Reset keeper groups…")).on_hover_text("Set the AI's choices aside").clicked() {
                queue.push(OptionsAction::AskReset(WorkKind::KeeperGroups));
            }
        });

        // ── stale results and questions
        if let Some(st) = &status {
            let stale: Vec<String> = st.stale.iter().filter(|n| !st.reset.contains(n)).cloned().collect();
            if !stale.is_empty() {
                ui.add_space(8.0);
                ui.label(RichText::new(format!("Out of date: {}", friendly(&stale))).color(amber));
            }
        }
        match &ask {
            Some(Ask::Retranscribe { from, to }) => {
                ui.add_space(8.0);
                ui.separator();
                ui.add(egui::Label::new(format!("The transcript was made with “{from}”. Re-transcribe with “{to}”?")).wrap());
                ui.horizontal(|ui| {
                    if ui.button("Re-transcribe now").clicked() {
                        queue.push(OptionsAction::ConfirmRetranscribe);
                    }
                    if ui.button("Keep the old transcript").clicked() {
                        queue.push(OptionsAction::KeepOldTranscript);
                    }
                });
            }
            Some(Ask::ResplitOffer { .. }) => {
                ui.add_space(8.0);
                ui.separator();
                ui.add(egui::Label::new("Re-split the sentences at this gap?").wrap());
                ui.horizontal(|ui| {
                    if ui.button("Re-split now").clicked() {
                        queue.push(OptionsAction::ResplitNow);
                    }
                    if ui.button("Later").clicked() {
                        queue.push(OptionsAction::CancelAsk);
                    }
                });
            }
            Some(Ask::ResplitConfirm { gap, will_replace }) => {
                ui.add_space(8.0);
                ui.separator();
                ui.add(egui::Label::new(format!("Re-split at {gap:.2} s? This replaces: {}.", friendly(will_replace))).wrap());
                ui.horizontal(|ui| {
                    if ui.button("Re-split").clicked() {
                        queue.push(OptionsAction::ConfirmResplit);
                    }
                    if ui.button("Cancel").clicked() {
                        queue.push(OptionsAction::CancelAsk);
                    }
                });
            }
            Some(Ask::Reset { what }) => {
                ui.add_space(8.0);
                ui.separator();
                let what_text = match what {
                    WorkKind::Transcription => "the transcription",
                    WorkKind::KeeperGroups => "the keeper groups",
                };
                ui.add(egui::Label::new(format!("Reset {what_text}?")).wrap());
                ui.horizontal(|ui| {
                    if ui.button("Reset").clicked() {
                        queue.push(OptionsAction::ConfirmReset);
                    }
                    if ui.button("Cancel").clicked() {
                        queue.push(OptionsAction::CancelAsk);
                    }
                });
            }
            None => {}
        }

        // ── old data
        if let Some(st) = &status
            && !st.legacy.is_empty()
        {
            ui.add_space(8.0);
            ui.separator();
            ui.add(egui::Label::new(format!("Earlier work on this video was found ({}). Copy it into this project?", friendly(&st.legacy))).wrap());
            if ui.button("Import into this project").clicked() {
                queue.push(OptionsAction::ImportLegacy);
            }
        }

        if let Some(n) = &note {
            ui.add_space(8.0);
            ui.add(egui::Label::new(RichText::new(n.as_str()).color(t.text_dim)).wrap());
        }
        ui.add_space(10.0);
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if ui.button("Close").clicked() {
                queue.push(OptionsAction::Close);
            }
        });
    });
    for a in queue {
        app.queue(Action::Options(a));
    }
}
