//! Ctrl+Z and Ctrl+Y for the pipeline's own steps. A step that created files (the transcription, the keeper choices) is undone like an edit: Ctrl+Z sets what it made
//! aside, so doing the process again with other settings starts from scratch instead of finding the old results. Ctrl+Y brings them back.
//!
//! Nothing is deleted: the service keeps the files (the old ones become numbered copies when the work is redone) and never touches the user's answers. The editor's own
//! undo stack (cuts, trims, ...) is separate: Ctrl+Z undoes the pipeline step only when nothing was edited after it; otherwise the edit comes first, as usual.

use simplecut_core::{WorkCall, WorkKind};

use crate::app::SimpleCutApp;

/// What a step made, with what is needed to undo and redo it in the review.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum StepKind {
    /// The transcript, sentences and pair scores were made (not found in the cache).
    Analysis,
    /// The keeper (AI) filled in the ticks: (sentence id, kept) for every sentence it answered for, the groups it was unsure about, and its reasons.
    Keepers { applied: Vec<(String, bool)>, unclear: Vec<String>, reasons: Vec<(String, String)> },
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Step {
    pub kind: StepKind,
    /// The editor's undo depth when the step finished: the step is the latest thing done only while the depth is still this.
    pub depth: usize,
}

#[derive(Debug, Default)]
pub(crate) struct StepHistory {
    pub done: Vec<Step>,
    pub undone: Vec<Step>,
}

impl SimpleCutApp {
    fn editor_depth(&self) -> usize {
        self.editor.session().history.undo.len()
    }

    /// A pipeline step finished and made files (or choices): it can now be undone with Ctrl+Z. A new step ends the redo chain, as with any edit.
    pub(crate) fn record_step(&mut self, kind: StepKind) {
        let depth = self.editor_depth();
        self.steps.undone.clear();
        self.steps.done.push(Step { kind, depth });
    }

    /// Ctrl+Z: undo the latest pipeline step if nothing was edited after it. False when there is no such step (the editor's own undo is next).
    pub(crate) fn undo_pipeline_step(&mut self) -> bool {
        if self.pipeline_running() {
            return false;
        }
        let depth = self.editor_depth();
        if !self.steps.done.last().is_some_and(|s| s.depth == depth) {
            return false;
        }
        let Some(step) = self.steps.done.pop() else { return false };
        match &step.kind {
            StepKind::Analysis => {
                self.work(WorkCall::Reset { what: WorkKind::Transcription, path: String::new() });
                self.perform(crate::app::Action::CloseReview);
                self.info_text("Transcription undone.");
            }
            StepKind::Keepers { applied, .. } => {
                self.remove_ai_choices(applied.iter().map(|(id, _)| id.as_str()));
                self.work(WorkCall::Reset { what: WorkKind::KeeperGroups, path: String::new() });
                self.info_text("AI choices undone.");
            }
        }
        self.steps.undone.push(step);
        true
    }

    /// Ctrl+Y: redo the pipeline step that was just undone, if nothing was edited since.
    pub(crate) fn redo_pipeline_step(&mut self) -> bool {
        if self.pipeline_running() {
            return false;
        }
        let depth = self.editor_depth();
        if !self.steps.undone.last().is_some_and(|s| s.depth == depth) {
            return false;
        }
        let Some(step) = self.steps.undone.pop() else { return false };
        match &step.kind {
            StepKind::Analysis => {
                self.work(WorkCall::Restore { what: WorkKind::Transcription, path: String::new() });
                self.info_text("Transcription restored.");
            }
            StepKind::Keepers { applied, unclear, reasons } => {
                self.reapply_ai_choices(applied, unclear, reasons);
                self.work(WorkCall::Restore { what: WorkKind::KeeperGroups, path: String::new() });
                self.info_text("AI choices restored.");
            }
        }
        self.steps.done.push(step);
        true
    }

    /// Ask the service to set work aside or bring it back. The video's path is filled in here (the service needs it when the project has no folder yet).
    pub(crate) fn work(&mut self, call: WorkCall) {
        let Some(path) = self.source_video_path() else { return };
        let call = match call {
            WorkCall::Reset { what, .. } => WorkCall::Reset { what, path },
            WorkCall::Restore { what, .. } => WorkCall::Restore { what, path },
        };
        self.make_work_call(call);
    }
}
