//! Retake Editor, the desktop app: a window, a sound card and file dialogs around `retake-editor-ui`.

mod audio;
mod dialogs;

use std::path::PathBuf;

use retake_editor_core::AudioOutput;
use retake_editor_ui::RetakeEditorApp;

fn main() -> eframe::Result {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn")).init();
    install_panic_logging();
    // hardware video decoding where the machine has it; our own decoders otherwise
    let hardware = filmcraft_platform::register();
    log::info!("hardware decoding: {hardware:?}");

    let open: Option<PathBuf> = std::env::args_os().nth(1).map(PathBuf::from);
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("Retake")
            .with_icon(retake_editor_icon())
            .with_inner_size([1360.0, 860.0])
            .with_min_inner_size([960.0, 600.0])
            .with_drag_and_drop(true),
        ..Default::default()
    };
    eframe::run_native(
        "Retake",
        options,
        Box::new(move |cc| {
            let output = audio::CpalOut::new().map(|o| Box::new(o) as Box<dyn AudioOutput>);
            if output.is_none() {
                log::warn!("no sound card found; playing without sound");
            }
            let mut app = RetakeEditorApp::new(&cc.egui_ctx, Box::new(dialogs::NativeDialogs), output);
            let data_dir = std::env::var_os("RETAKE_EDITOR_DATA_DIR")
                .map(PathBuf::from)
                .or_else(|| std::env::var_os("APPDATA").map(|a| PathBuf::from(a).join("RetakeEditor")))
                .unwrap_or_else(|| PathBuf::from(".retake-editor"));
            app.set_label_log(Some(data_dir.join("labels.jsonl")));
            app.set_span_log(Some(data_dir.join("span_edits.jsonl")));
            app.enable_autosave(data_dir);
            // a stand-in service so the whole retake flow can be tried before a real one exists
            if std::env::var_os("RETAKE_EDITOR_FAKE_RETAKES").is_some() {
                app.set_service(Some(Box::new(retake_editor_core::FakeService { adjudicate: true, ..Default::default() })));
            } else if let Some(url) = retake_service_url() {
                // the real service (`python -m service`); a dead one is reported when a request is made
                let service = retake_editor_service::HttpService::connect(&url);
                if !service.is_reachable() {
                    log::warn!("the retake service at {url} did not answer; start it with `python -m service`");
                }
                app.set_service(Some(Box::new(service)));
            }
            if let Some(path) = open {
                app.open_project_path(path);
            }
            Ok(Box::new(app))
        }),
    )
}

/// Where the retake service is: `RETAKE_EDITOR_SERVICE_URL` is an address, or `1` / empty for the default one.
/// Unset means no service (the Remove Retakes button stays hidden).
fn retake_service_url() -> Option<String> {
    let value = std::env::var("RETAKE_EDITOR_SERVICE_URL").ok()?;
    let value = value.trim();
    if value.is_empty() || value == "1" { Some(retake_editor_service::DEFAULT_URL.to_string()) } else { Some(value.to_string()) }
}

/// Window icon: a dark slate tile with an amber cut mark.
fn retake_editor_icon() -> egui::IconData {
    const W: u32 = 32;
    const H: u32 = 32;
    let mut rgba = vec![0u8; (W * H * 4) as usize];
    for y in 0..H {
        for x in 0..W {
            let idx = ((y * W + x) * 4) as usize;
            let dx = (x as i32 - 16).abs();
            let dy = (y as i32 - 16).abs();
            if dx * dx + dy * dy <= 15 * 15 {
                rgba[idx] = 28;
                rgba[idx + 1] = 30;
                rgba[idx + 2] = 38;
                rgba[idx + 3] = 255;

                let cut_dist = ((x as i32 - y as i32) as f32).abs();
                if cut_dist < 2.0 && (6..=26).contains(&x) && (6..=26).contains(&y) {
                    rgba[idx] = 235;
                    rgba[idx + 1] = 140;
                    rgba[idx + 2] = 40;
                    rgba[idx + 3] = 255;
                }
            }
        }
    }
    egui::IconData { rgba, width: W, height: H }
}

/// Record where a panic happened before the default handler runs. The interface and the background
/// threads catch their own panics, so this is for finding out why.
fn install_panic_logging() {
    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let thread = std::thread::current();
        let place = info.location().map(|l| format!("{}:{}", l.file(), l.line())).unwrap_or_default();
        let msg = format!("panic on thread '{}' at {place}: {info}", thread.name().unwrap_or("unnamed"));
        log::error!("{msg}");
        retake_editor_ui::write_crash_log(&msg, None);
        previous(info);
    }));
}
