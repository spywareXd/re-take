//! A frozen view of the project for rendering.
//!
//! The project is copy-on-write (`Arc<Project>`), so a snapshot costs nothing to take and the
//! render and audio threads can keep using it while the user keeps editing. Playback and preview
//! only ever read from a snapshot. A snapshot looks at either the open sequence or one media file
//! (the library's preview), and answers the same questions for both.

use std::sync::Arc;

use filmcraft_engine::{MediaPool, Services};
use filmcraft_frame::AudioBuffer;
use filmcraft_media::MediaInfo;
use filmcraft_project::{ItemId, ItemKind, Project};
use filmcraft_render::RenderOptions;
use filmcraft_time::{FrameRate, Tick};

use crate::editor::Editor;

/// Sample rate media is played at in the library preview.
const MEDIA_SAMPLE_RATE: u32 = 48_000;

/// One picture for the screen: opaque sRGB RGBA8, `width * height * 4` bytes.
#[derive(Debug, Clone, PartialEq)]
pub struct PreviewFrame {
    pub width: usize,
    pub height: usize,
    pub rgba: Vec<u8>,
}

impl PreviewFrame {
    /// A black picture, for when nothing can be drawn.
    pub fn black(width: usize, height: usize) -> Self {
        let (width, height) = (width.max(1), height.max(1));
        let mut rgba = vec![0u8; width * height * 4];
        for px in rgba.as_chunks_mut::<4>().0.iter_mut() {
            px[3] = 255;
        }
        Self { width, height, rgba }
    }
}

/// What a snapshot looks at.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Target {
    /// The open sequence: the edit.
    Sequence(ItemId),
    /// One imported file on its own.
    Media(ItemId),
}

#[derive(Clone)]
pub struct Snapshot {
    project: Arc<Project>,
    target: Target,
    pool: Arc<MediaPool>,
    services: Arc<dyn Services>,
    /// The editor's revision when the snapshot was taken; a different one means the project changed.
    pub revision: u64,
}

impl Snapshot {
    /// The open sequence as it is now, or None when no sequence is open.
    pub fn of(editor: &Editor) -> Option<Snapshot> {
        let s = editor.session();
        let sequence = s.state.active_sequence?;
        s.project.sequence(sequence)?;
        Some(Snapshot {
            project: s.project.clone(),
            target: Target::Sequence(sequence),
            pool: s.media.clone(),
            services: s.services.clone(),
            revision: s.revision,
        })
    }

    /// One imported file, or None when `item` is not a media file of the project.
    pub fn of_media(editor: &Editor, item: ItemId) -> Option<Snapshot> {
        let s = editor.session();
        match &s.project.item(item)?.kind {
            ItemKind::Media(_) => {}
            _ => return None,
        }
        Some(Snapshot { project: s.project.clone(), target: Target::Media(item), pool: s.media.clone(), services: s.services.clone(), revision: s.revision })
    }

    pub fn target(&self) -> Target {
        self.target
    }

    fn media_info(&self, item: ItemId) -> Option<&MediaInfo> {
        match &self.project.item(item)?.kind {
            ItemKind::Media(mc) => Some(&mc.info),
            _ => None,
        }
    }

    /// How long one media file of the project is.
    pub fn media_duration(&self, item: ItemId) -> Tick {
        self.media_info(item).map(|i| i.duration).unwrap_or_default()
    }

    /// Length of the sequence (the end of its last clip) or of the media file.
    pub fn duration(&self) -> Tick {
        match self.target {
            Target::Sequence(id) => self.project.sequence(id).map(|q| q.duration()).unwrap_or_default(),
            Target::Media(id) => self.media_info(id).map(|i| i.duration).unwrap_or_default(),
        }
    }

    pub fn frame_rate(&self) -> FrameRate {
        match self.target {
            Target::Sequence(id) => self.project.sequence(id).map(|q| q.settings.frame_rate).unwrap_or_default(),
            Target::Media(id) => self.media_info(id).and_then(|i| i.video.as_ref()).map(|v| v.frame_rate).unwrap_or_default(),
        }
    }

    /// The rate [`Self::mix_audio`] counts samples in (never 0).
    pub fn sample_rate(&self) -> u32 {
        match self.target {
            Target::Sequence(id) => self.project.sequence(id).map(|q| q.settings.sample_rate).filter(|r| *r > 0).unwrap_or(MEDIA_SAMPLE_RATE),
            Target::Media(_) => MEDIA_SAMPLE_RATE,
        }
    }

    /// The picture at `t`, scaled (1.0 = full size). For the sequence this includes captions; for
    /// a media file with no picture it is a black frame.
    pub fn render_frame(&self, t: Tick, scale: f32) -> PreviewFrame {
        match self.target {
            Target::Sequence(id) => {
                let provider = self.pool.provider(self.project.clone(), self.services.clone());
                let opts = RenderOptions { scale: clamp_scale(scale), captions: true, ..Default::default() };
                let img = filmcraft_render::render_sequence(&self.project, id, Tick(t.0.max(0)), opts, &provider);
                PreviewFrame { width: img.w, height: img.h, rgba: img.over_black_rgba8() }
            }
            Target::Media(id) => self.render_media_frame(id, t, scale).unwrap_or_else(|| PreviewFrame::black(16, 9)),
        }
    }

    /// One media file's picture at media time `t` (the library thumbnail), or None for audio-only.
    pub fn render_media_frame(&self, item: ItemId, t: Tick, scale: f32) -> Option<PreviewFrame> {
        let provider = self.pool.provider(self.project.clone(), self.services.clone());
        let img = filmcraft_render::render_item(&self.project, item, Tick(t.0.max(0)), clamp_scale(scale), &provider)?;
        Some(PreviewFrame { width: img.w, height: img.h, rgba: img.over_black_rgba8() })
    }

    /// `frames` stereo-or-mono samples starting at sample `start` (at [`Self::sample_rate`]). For
    /// the sequence this is the full mixer: volumes, mutes and music included.
    pub fn mix_audio(&self, start: i64, frames: usize) -> Option<AudioBuffer> {
        match self.target {
            Target::Sequence(id) => {
                let q = self.project.sequence(id)?;
                let provider = self.pool.provider(self.project.clone(), self.services.clone());
                Some(filmcraft_render::audio::mix_sequence(&self.project, q, start.max(0), frames, &provider))
            }
            Target::Media(id) => {
                let source = self.pool.source_for(&self.project, id, &*self.services)?;
                source.audio(start.max(0), frames, MEDIA_SAMPLE_RATE).ok()
            }
        }
    }
}

fn clamp_scale(scale: f32) -> f32 {
    if scale.is_finite() { scale.clamp(0.05, 1.0) } else { 0.5 }
}
