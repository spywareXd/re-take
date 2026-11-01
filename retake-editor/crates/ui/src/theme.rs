//! The dark look: a handful of colours, applied once.

use egui::{Color32, CornerRadius, FontFamily, FontId, Stroke, TextStyle, Visuals};

/// Every colour the interface uses, in one place.
#[derive(Clone, Copy)]
pub struct Tokens {
    pub bg: Color32,
    pub panel: Color32,
    pub panel_raised: Color32,
    pub border: Color32,
    pub text: Color32,
    pub text_dim: Color32,
    pub accent: Color32,
    pub danger: Color32,
    pub video_clip: Color32,
    pub dialogue_clip: Color32,
    pub music_clip: Color32,
    pub lane: Color32,
    pub playhead: Color32,
}

pub const DARK: Tokens = Tokens {
    bg: Color32::from_rgb(0x12, 0x14, 0x18),
    panel: Color32::from_rgb(0x1a, 0x1d, 0x23),
    panel_raised: Color32::from_rgb(0x23, 0x27, 0x2f),
    border: Color32::from_rgb(0x2e, 0x33, 0x3d),
    text: Color32::from_rgb(0xe6, 0xe8, 0xec),
    text_dim: Color32::from_rgb(0x96, 0x9c, 0xa8),
    accent: Color32::from_rgb(0x4c, 0x8d, 0xff),
    danger: Color32::from_rgb(0xff, 0x6b, 0x6b),
    video_clip: Color32::from_rgb(0x3a, 0x6e, 0xd8),
    dialogue_clip: Color32::from_rgb(0x2f, 0x9e, 0x8f),
    music_clip: Color32::from_rgb(0x9b, 0x6b, 0xd6),
    lane: Color32::from_rgb(0x16, 0x19, 0x1e),
    playhead: Color32::from_rgb(0xff, 0x5d, 0x5d),
};

pub fn tokens() -> Tokens {
    DARK
}

/// Eight distinct colors for retake groups, clearly readable on dark panels and distinct from danger/accent.
pub const GROUP_COLORS: [Color32; 8] = [
    Color32::from_rgb(245, 158, 11), // Amber / Gold
    Color32::from_rgb(16, 185, 129), // Emerald
    Color32::from_rgb(168, 85, 247), // Violet
    Color32::from_rgb(6, 182, 212),  // Cyan
    Color32::from_rgb(217, 70, 239), // Magenta
    Color32::from_rgb(132, 204, 22), // Lime
    Color32::from_rgb(20, 184, 166), // Teal
    Color32::from_rgb(249, 115, 22), // Orange
];

/// Apply the dark theme to the whole app.
pub fn apply(ctx: &egui::Context) {
    let t = DARK;
    let mut v = Visuals::dark();
    v.panel_fill = t.panel;
    v.window_fill = t.panel_raised;
    v.extreme_bg_color = t.bg;
    v.faint_bg_color = t.panel_raised;
    v.window_stroke = Stroke::new(1.0, t.border);
    v.override_text_color = Some(t.text);
    v.selection.bg_fill = t.accent.gamma_multiply(0.45);
    v.selection.stroke = Stroke::new(1.0, t.accent);
    v.hyperlink_color = t.accent;
    let r = CornerRadius::same(6);
    for w in [&mut v.widgets.noninteractive, &mut v.widgets.inactive, &mut v.widgets.hovered, &mut v.widgets.active, &mut v.widgets.open] {
        w.corner_radius = r;
    }
    v.widgets.noninteractive.bg_stroke = Stroke::new(1.0, t.border);
    v.widgets.inactive.bg_fill = t.panel_raised;
    v.widgets.inactive.weak_bg_fill = t.panel_raised;
    v.widgets.inactive.bg_stroke = Stroke::new(1.0, t.border);
    v.widgets.hovered.bg_fill = t.border;
    v.widgets.hovered.weak_bg_fill = t.border;
    v.widgets.active.bg_fill = t.accent.gamma_multiply(0.7);
    v.widgets.active.weak_bg_fill = t.accent.gamma_multiply(0.7);
    ctx.set_visuals(v);

    ctx.global_style_mut(|style| {
        style.spacing.item_spacing = egui::vec2(8.0, 6.0);
        style.spacing.button_padding = egui::vec2(10.0, 5.0);
        style.text_styles.insert(TextStyle::Body, FontId::new(14.0, FontFamily::Proportional));
        style.text_styles.insert(TextStyle::Button, FontId::new(14.0, FontFamily::Proportional));
        style.text_styles.insert(TextStyle::Small, FontId::new(12.0, FontFamily::Proportional));
        style.text_styles.insert(TextStyle::Heading, FontId::new(16.0, FontFamily::Proportional));
    });
}
