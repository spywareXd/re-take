//! What the retake service offers beyond finding groups, as the editor shows it: the transcription models (with size, memory, speed and whether they are already
//! downloaded), the silence-gap range, and the settings that ride along with an analysis. Everything from the service is treated as hostile: numbers are checked,
//! texts capped, lists bounded.

use serde_json::Value;

const MAX_MODELS: usize = 24;
const MAX_TEXT: usize = 300;

fn text(v: &Value, key: &str) -> Option<String> {
    v.get(key).and_then(Value::as_str).map(|s| s.chars().take(MAX_TEXT).collect())
}

fn number(v: &Value, key: &str) -> Option<f64> {
    v.get(key).and_then(Value::as_f64).filter(|n| n.is_finite() && *n >= 0.0)
}

/// One transcription model the service can run.
#[derive(Debug, Clone, PartialEq)]
pub struct AsrModel {
    /// The name to send back, exactly as the service wrote it (`large-v2`, `turbo`, ...).
    pub name: String,
    pub default: bool,
    /// Already in the local cache (the service checked without downloading).
    pub downloaded: bool,
    /// Approximate download size (GB), and the real size on disk once downloaded.
    pub download_gb: Option<f64>,
    pub size_on_disk_gb: Option<f64>,
    /// Approximate GPU memory (GB) and speed relative to the default model (1.0 = same, 5.0 = five times faster).
    pub vram_gb: Option<f64>,
    pub relative_speed: Option<f64>,
    /// A warning (never a block): the model probably does not fit this GPU.
    pub warning: Option<String>,
}

/// The GPU the service found (None: no NVIDIA GPU, transcription runs on the CPU).
#[derive(Debug, Clone, PartialEq)]
pub struct GpuInfo {
    pub name: String,
    pub total_gb: f64,
    pub free_gb: f64,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct TranscriptionInfo {
    pub default: String,
    pub models: Vec<AsrModel>,
    pub gpu: Option<GpuInfo>,
    /// How the compute type is chosen, as a sentence.
    pub compute_note: String,
}

impl TranscriptionInfo {
    pub fn from_json(v: &Value) -> Option<Self> {
        let models: Vec<AsrModel> = v
            .get("models")?
            .as_array()?
            .iter()
            .take(MAX_MODELS)
            .filter_map(|m| {
                Some(AsrModel {
                    name: text(m, "name").filter(|n| !n.trim().is_empty())?,
                    default: m.get("default").and_then(Value::as_bool).unwrap_or(false),
                    downloaded: m.get("downloaded").and_then(Value::as_bool).unwrap_or(false),
                    download_gb: number(m, "download_gb"),
                    size_on_disk_gb: number(m, "size_on_disk_gb"),
                    vram_gb: number(m, "vram_gb"),
                    relative_speed: number(m, "relative_speed"),
                    warning: text(m, "warning").filter(|w| !w.trim().is_empty()),
                })
            })
            .collect();
        if models.is_empty() {
            return None;
        }
        let gpu = v
            .get("gpu")
            .filter(|g| g.is_object())
            .and_then(|g| Some(GpuInfo { name: text(g, "name")?, total_gb: number(g, "total_gb")?, free_gb: number(g, "free_gb").unwrap_or(0.0) }));
        Some(Self {
            default: text(v, "default").unwrap_or_else(|| models.first().map(|m| m.name.clone()).unwrap_or_default()),
            models,
            gpu,
            compute_note: text(v, "compute_type").unwrap_or_default(),
        })
    }

    pub fn model(&self, name: &str) -> Option<&AsrModel> {
        self.models.iter().find(|m| m.name == name)
    }
}

/// The range of the silence gap that splits sentences (seconds).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SilenceInfo {
    pub default: f64,
    pub min: f64,
    pub max: f64,
    pub step: f64,
}

impl SilenceInfo {
    pub fn from_json(v: &Value) -> Option<Self> {
        let (default, min, max, step) = (number(v, "default")?, number(v, "min")?, number(v, "max")?, number(v, "step")?);
        (min < max && default >= min && default <= max && step > 0.0).then_some(Self { default, min, max, step })
    }

    /// A value from the user, forced into the range and onto the step grid.
    pub fn clamp(&self, value: f64) -> f64 {
        if !value.is_finite() {
            return self.default;
        }
        let v = value.clamp(self.min, self.max);
        let snapped = ((v / self.step).round() * self.step).clamp(self.min, self.max);
        (snapped * 10_000.0).round() / 10_000.0 // no 0.7000000000000001 on the wire
    }
}

/// Everything the service advertised that the editor offers as a setting.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ServiceOptions {
    pub transcription: Option<TranscriptionInfo>,
    pub silence_gap: Option<SilenceInfo>,
}

impl ServiceOptions {
    /// Read the capabilities reply (`GET /v1/capabilities`). Missing parts stay None: an older service simply offers fewer settings.
    pub fn from_capabilities(v: &Value) -> Self {
        Self {
            transcription: v.get("transcription").and_then(TranscriptionInfo::from_json),
            silence_gap: v.get("silence_gap").and_then(SilenceInfo::from_json),
        }
    }
}

/// One export preset (built in, or saved by the user in their own config folder).
#[derive(Debug, Clone, PartialEq)]
pub struct ExportPreset {
    pub name: String,
    pub category: String,
    pub description: String,
    pub builtin: bool,
    /// The preset's settings as the service stores them (`video`, `resolution`, `fps`, `audio`), used to fill the form.
    pub settings: Value,
}

/// What the export window needs from the service (`GET /v1/export/options`). Hardware encoders are listed only when the service's test encode worked.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ExportOptionsInfo {
    pub presets: Vec<ExportPreset>,
    pub heights: Vec<String>,
    pub canvases: Vec<String>,
    pub fits: Vec<String>,
    pub dim_min: u32,
    pub dim_max: u32,
    pub fps_choices: Vec<u32>,
    pub fps_range: (f64, f64),
    pub codecs: Vec<String>,
    pub crf_range: (u32, u32),
    pub bitrate_range: (u32, u32),
    pub speeds: Vec<String>,
    /// Codecs (`h264`, `h265`) the GPU can encode; empty = the option is not offered.
    pub hardware: Vec<String>,
    pub audio_bitrates: Vec<u32>,
    pub audio_formats: Vec<String>,
    pub text_formats: Vec<(String, Vec<String>)>,
    pub fade_default: f64,
    pub fade_max: f64,
}

fn strings(v: &Value) -> Vec<String> {
    v.as_array().map(|a| a.iter().filter_map(Value::as_str).map(|s| s.chars().take(MAX_TEXT).collect()).take(MAX_MODELS * 2).collect()).unwrap_or_default()
}

fn uints(v: &Value) -> Vec<u32> {
    v.as_array().map(|a| a.iter().filter_map(Value::as_u64).filter_map(|n| u32::try_from(n).ok()).take(MAX_MODELS * 2).collect()).unwrap_or_default()
}

fn pair_u32(v: &Value) -> (u32, u32) {
    let a = uints(v);
    (a.first().copied().unwrap_or(0), a.get(1).copied().unwrap_or(0))
}

impl ExportOptionsInfo {
    pub fn from_json(v: &Value) -> Option<Self> {
        let mut presets: Vec<ExportPreset> = v
            .get("presets")?
            .as_object()?
            .iter()
            .take(100)
            .map(|(name, p)| ExportPreset {
                name: name.chars().take(80).collect(),
                category: text(p, "category").unwrap_or_default(),
                description: text(p, "description").unwrap_or_default(),
                builtin: p.get("builtin").and_then(Value::as_bool).unwrap_or(false),
                settings: serde_json::json!({"video": p.get("video"), "resolution": p.get("resolution"), "fps": p.get("fps"), "audio": p.get("audio")}),
            })
            .collect();
        // the service sends the order (a JSON object has none): built-in presets first, then the user's own
        let order = strings(&v["preset_order"]);
        presets.sort_by_key(|p| order.iter().position(|n| *n == p.name).unwrap_or(usize::MAX));
        let r = v.get("resolutions").unwrap_or(&Value::Null);
        let fps = v.get("fps").unwrap_or(&Value::Null);
        let vid = v.get("video").unwrap_or(&Value::Null);
        let aud = v.get("audio").unwrap_or(&Value::Null);
        Some(Self {
            presets,
            heights: strings(&r["heights"]),
            canvases: r["canvases"].as_object().map(|o| o.keys().cloned().collect()).unwrap_or_default(),
            fits: strings(&r["fits"]),
            dim_min: r["min"].as_u64().and_then(|n| u32::try_from(n).ok()).unwrap_or(16),
            dim_max: r["max"].as_u64().and_then(|n| u32::try_from(n).ok()).unwrap_or(7680),
            fps_choices: uints(&fps["choices"]),
            fps_range: (number(fps, "custom_min").unwrap_or(1.0), number(fps, "custom_max").unwrap_or(240.0)),
            codecs: strings(&vid["codecs"]),
            crf_range: pair_u32(&vid["crf_range"]),
            bitrate_range: pair_u32(&vid["bitrate_kbps_range"]),
            speeds: strings(&vid["speeds"]),
            hardware: vid["hardware"].as_object().map(|o| o.keys().cloned().collect()).unwrap_or_default(),
            audio_bitrates: uints(&aud["bitrates_kbps"]),
            audio_formats: strings(&v["audio_only_formats"]),
            text_formats: v["text_formats"].as_object().map(|o| o.iter().map(|(k, f)| (k.clone(), strings(f))).collect()).unwrap_or_default(),
            fade_default: number(aud, "fade_ms_default").unwrap_or(15.0),
            fade_max: number(aud, "fade_ms_max").unwrap_or(100.0),
        })
    }

    pub fn preset(&self, name: &str) -> Option<&ExportPreset> {
        self.presets.iter().find(|p| p.name == name)
    }
}

/// An export to make: what to export and how, as the JSON the service takes (`kind`, `format`, `preset`, `resolution`, `fps`, `video`, `audio`, `name`). The source
/// file and the project folder are added by the client.
#[derive(Debug, Clone, PartialEq)]
pub struct ExportRequest {
    pub item: crate::ItemId,
    /// The source video on disk.
    pub path: String,
    pub body: Value,
}

/// A request about the user's own (per-user, not per-project) export presets and the export options.
#[derive(Debug, Clone, PartialEq)]
pub enum UserCall {
    /// `GET /v1/export/options`.
    ExportOptions,
    /// `PUT /v1/export/presets/{name}`.
    SavePreset { name: String, preset: Value },
    /// `DELETE /v1/export/presets/{name}`.
    DeletePreset { name: String },
}

/// The settings that go with one analysis. None = whatever the project (or the service) has: the default model, the default gap.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct AnalysisSettings {
    pub model: Option<String>,
    pub silence_gap_s: Option<f64>,
    /// The user confirmed that an existing transcript / sentence split may be replaced (the old ones are kept as numbered copies by the service).
    pub replace_existing: bool,
}

/// A request about the project's stored settings and results that is not an analysis. The job's result is `{"status": <http status>, "body": <the reply>}`,
/// so a "please confirm" answer (409) reaches the editor with its details.
#[derive(Debug, Clone, PartialEq)]
pub enum ProjectCall {
    /// Settings, stale marks and whether the source video is still the same (`GET /v1/project/status`).
    Status,
    /// Change stored settings; a changed model marks the transcript stale and asks (`PUT /v1/project/settings`).
    SetSettings { model: Option<String>, silence_gap_s: Option<f64> },
    /// How many sentences a gap would give, from the saved word timestamps (`GET /v1/project/units-preview`).
    UnitsPreview { silence_gap_s: f64 },
    /// Re-split the saved transcript at the gap without re-transcribing; `confirm` false only reports what would be replaced (`POST /v1/project/resplit`).
    Resplit { silence_gap_s: f64, confirm: bool },
    /// Copy the old per-machine data of the video at `path` into the project (`POST /v1/project/import-legacy`); the service records the video as the project's
    /// source. The originals stay.
    ImportLegacy { path: String },
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn models_are_read_defensively() {
        let v = json!({"default": "large-v2", "gpu": {"name": "RTX", "total_gb": 8.0, "free_gb": 5.2}, "compute_type": "auto: tries int8_float16",
        "models": [
            {"name": "tiny", "downloaded": false, "download_gb": 0.08, "vram_gb": 1.0, "relative_speed": 12.0},
            {"name": "large-v2", "default": true, "downloaded": true, "size_on_disk_gb": 3.09, "vram_gb": 5.5, "relative_speed": 1.0, "warning": "Needs about 5.5 GB"},
            {"name": "", "vram_gb": 1}, {"vram_gb": 1}, 7, null,
            {"name": "weird", "vram_gb": -4, "download_gb": "lots", "relative_speed": null}
        ]});
        let t = TranscriptionInfo::from_json(&v).unwrap();
        assert_eq!(t.models.iter().map(|m| m.name.as_str()).collect::<Vec<_>>(), vec!["tiny", "large-v2", "weird"], "unusable entries are dropped");
        assert_eq!(t.default, "large-v2");
        assert!(t.model("large-v2").unwrap().default && t.model("large-v2").unwrap().downloaded);
        assert_eq!(t.model("large-v2").unwrap().warning.as_deref(), Some("Needs about 5.5 GB"));
        let w = t.model("weird").unwrap();
        assert!(w.vram_gb.is_none() && w.download_gb.is_none() && w.relative_speed.is_none(), "negative, textual and missing numbers are ignored");
        assert_eq!(t.gpu, Some(GpuInfo { name: "RTX".into(), total_gb: 8.0, free_gb: 5.2 }));
        assert!(TranscriptionInfo::from_json(&json!({"models": []})).is_none() && TranscriptionInfo::from_json(&json!("x")).is_none());
        assert!(TranscriptionInfo::from_json(&json!({"models": Vec::<Value>::new()})).is_none());
        let many: Vec<Value> = (0..100).map(|i| json!({"name": format!("m{i}")})).collect();
        assert_eq!(TranscriptionInfo::from_json(&json!({"models": many})).unwrap().models.len(), MAX_MODELS);
    }

    #[test]
    fn the_silence_range_is_checked_and_values_snap_to_it() {
        let s = SilenceInfo::from_json(&json!({"default": 0.7, "min": 0.1, "max": 3.0, "step": 0.05})).unwrap();
        assert_eq!(s.clamp(0.72), 0.7);
        assert_eq!(s.clamp(-4.0), 0.1);
        assert_eq!(s.clamp(99.0), 3.0);
        assert_eq!(s.clamp(f64::NAN), 0.7);
        assert!((s.clamp(1.234) - 1.25).abs() < 1e-9);
        for bad in [
            json!({"default": 0.7, "min": 3.0, "max": 0.1, "step": 0.05}),
            json!({"default": 9.0, "min": 0.1, "max": 3.0, "step": 0.05}),
            json!({"default": 0.7, "min": 0.1, "max": 3.0, "step": 0}),
            json!({"default": 0.7, "min": 0.1, "max": 3.0}),
        ] {
            assert!(SilenceInfo::from_json(&bad).is_none(), "{bad}");
        }
    }

    #[test]
    fn an_older_service_offers_fewer_settings() {
        let o = ServiceOptions::from_capabilities(&json!({"adjudicate": true, "refine": true}));
        assert!(o.transcription.is_none() && o.silence_gap.is_none());
    }
}
