//! A small software renderer for the window tests: draws what egui painted (its tessellated triangles and its font/image textures) into an RGBA picture, so a test can
//! save a screenshot of the real window layout without a GPU, a desktop session or any extra dependency. Same maths as egui's own software reference: premultiplied
//! colours, blending in gamma space, bilinear texture sampling. Good for layout review (text, boxes, borders); not pixel-identical to the GL/wgpu output.
#![allow(dead_code, clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex};

use egui::epaint::Primitive;
use egui::{ClippedPrimitive, Color32, Context, FullOutput, TextureId, TexturesDelta};
use egui_kittest::TestRenderer;

struct Tex {
    w: usize,
    h: usize,
    px: Vec<Color32>,
}

/// Hand this to `HarnessBuilder::renderer`; keep a clone of it to draw pictures afterwards.
#[derive(Clone, Default)]
pub struct SoftRenderer {
    textures: Arc<Mutex<HashMap<TextureId, Tex>>>,
}

impl TestRenderer for SoftRenderer {
    fn handle_delta(&mut self, delta: &mut TexturesDelta) {
        let mut t = self.textures.lock().unwrap();
        let delta = std::mem::take(delta); // egui insists every delta is handled (taken) exactly once
        for (id, deltas) in &delta.set {
            for d in deltas {
                let egui::ImageData::Color(img) = &d.image;
                let [w, h] = img.size;
                match d.pos {
                    None => {
                        t.insert(*id, Tex { w, h, px: img.pixels.clone() });
                    }
                    Some([x0, y0]) => {
                        if let Some(tex) = t.get_mut(id) {
                            for y in 0..h {
                                for x in 0..w {
                                    if let Some(dst) = tex.px.get_mut((y0 + y) * tex.w + x0 + x) {
                                        *dst = img.pixels[y * w + x];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        for id in &delta.free {
            t.remove(id);
        }
        let mut delta = delta;
        delta.clear();
    }
}

pub struct Picture {
    pub w: usize,
    pub h: usize,
    pub rgba: Vec<u8>,
}

impl SoftRenderer {
    /// Draw the last frame of `output` at its pixel density into a `width` x `height` picture (in points).
    pub fn draw(&self, ctx: &Context, output: &FullOutput, width: f32, height: f32, background: Color32) -> Picture {
        let ppp = output.pixels_per_point;
        let (w, h) = ((width * ppp).round() as usize, (height * ppp).round() as usize);
        let mut buf: Vec<[f32; 4]> = vec![[background.r() as f32, background.g() as f32, background.b() as f32, 255.0]; w * h];
        let prims: Vec<ClippedPrimitive> = ctx.tessellate(output.shapes.clone(), ppp);
        let textures = self.textures.lock().unwrap();
        for ClippedPrimitive { clip_rect, primitive } in prims {
            let Primitive::Mesh(mesh) = primitive else { continue };
            let Some(tex) = textures.get(&mesh.texture_id) else { continue };
            let clip = [
                (clip_rect.min.x * ppp).floor().max(0.0) as i64,
                (clip_rect.min.y * ppp).floor().max(0.0) as i64,
                (clip_rect.max.x * ppp).ceil().min(w as f32) as i64,
                (clip_rect.max.y * ppp).ceil().min(h as f32) as i64,
            ];
            for tri in mesh.indices.as_chunks::<3>().0 {
                let v = [mesh.vertices[tri[0] as usize], mesh.vertices[tri[1] as usize], mesh.vertices[tri[2] as usize]];
                let p = [(v[0].pos.x * ppp, v[0].pos.y * ppp), (v[1].pos.x * ppp, v[1].pos.y * ppp), (v[2].pos.x * ppp, v[2].pos.y * ppp)];
                let area = (p[1].0 - p[0].0) * (p[2].1 - p[0].1) - (p[2].0 - p[0].0) * (p[1].1 - p[0].1);
                if area.abs() < 1e-6 {
                    continue;
                }
                let min_x = (p.iter().map(|q| q.0).fold(f32::MAX, f32::min).floor() as i64).max(clip[0]);
                let max_x = (p.iter().map(|q| q.0).fold(f32::MIN, f32::max).ceil() as i64).min(clip[2]);
                let min_y = (p.iter().map(|q| q.1).fold(f32::MAX, f32::min).floor() as i64).max(clip[1]);
                let max_y = (p.iter().map(|q| q.1).fold(f32::MIN, f32::max).ceil() as i64).min(clip[3]);
                for y in min_y..max_y {
                    for x in min_x..max_x {
                        let (cx, cy) = (x as f32 + 0.5, y as f32 + 0.5);
                        let w0 = ((p[1].0 - cx) * (p[2].1 - cy) - (p[2].0 - cx) * (p[1].1 - cy)) / area;
                        let w1 = ((p[2].0 - cx) * (p[0].1 - cy) - (p[0].0 - cx) * (p[2].1 - cy)) / area;
                        let w2 = 1.0 - w0 - w1;
                        if w0 < -1e-4 || w1 < -1e-4 || w2 < -1e-4 {
                            continue;
                        }
                        let uv = (w0 * v[0].uv.x + w1 * v[1].uv.x + w2 * v[2].uv.x, w0 * v[0].uv.y + w1 * v[1].uv.y + w2 * v[2].uv.y);
                        let t = sample(tex, uv.0, uv.1);
                        let col = |f: fn(&Color32) -> u8| w0 * f(&v[0].color) as f32 + w1 * f(&v[1].color) as f32 + w2 * f(&v[2].color) as f32;
                        let (vr, vg, vb, va) = (col(Color32::r), col(Color32::g), col(Color32::b), col(Color32::a));
                        // premultiplied: source = vertex colour * texel
                        let s = [vr * t[0] / 255.0, vg * t[1] / 255.0, vb * t[2] / 255.0, va * t[3] / 255.0];
                        let d = &mut buf[y as usize * w + x as usize];
                        let k = 1.0 - s[3] / 255.0;
                        for c in 0..3 {
                            d[c] = s[c] + d[c] * k;
                        }
                    }
                }
            }
        }
        let mut rgba = Vec::with_capacity(w * h * 4);
        for px in buf {
            rgba.extend_from_slice(&[px[0].clamp(0.0, 255.0) as u8, px[1].clamp(0.0, 255.0) as u8, px[2].clamp(0.0, 255.0) as u8, 255]);
        }
        Picture { w, h, rgba }
    }
}

fn sample(tex: &Tex, u: f32, v: f32) -> [f32; 4] {
    let x = (u * tex.w as f32 - 0.5).clamp(0.0, tex.w as f32 - 1.0);
    let y = (v * tex.h as f32 - 0.5).clamp(0.0, tex.h as f32 - 1.0);
    let (x0, y0) = (x.floor() as usize, y.floor() as usize);
    let (x1, y1) = ((x0 + 1).min(tex.w - 1), (y0 + 1).min(tex.h - 1));
    let (fx, fy) = (x - x0 as f32, y - y0 as f32);
    let at = |xx: usize, yy: usize| {
        let c = tex.px[yy * tex.w + xx];
        [c.r() as f32, c.g() as f32, c.b() as f32, c.a() as f32]
    };
    let (a, b, c, d) = (at(x0, y0), at(x1, y0), at(x0, y1), at(x1, y1));
    let mut out = [0.0; 4];
    for i in 0..4 {
        out[i] = (a[i] * (1.0 - fx) + b[i] * fx) * (1.0 - fy) + (c[i] * (1.0 - fx) + d[i] * fx) * fy;
    }
    out
}

impl Picture {
    /// A rectangle of the picture (clamped), enlarged `scale` times with hard pixels.
    pub fn crop_scaled(&self, x: usize, y: usize, cw: usize, ch: usize, scale: usize) -> Picture {
        let (x, y) = (x.min(self.w.saturating_sub(1)), y.min(self.h.saturating_sub(1)));
        let (cw, ch) = (cw.min(self.w - x), ch.min(self.h - y));
        let mut rgba = Vec::with_capacity(cw * ch * scale * scale * 4);
        for yy in 0..ch * scale {
            for xx in 0..cw * scale {
                let i = ((y + yy / scale) * self.w + x + xx / scale) * 4;
                rgba.extend_from_slice(&self.rgba[i..i + 4]);
            }
        }
        Picture { w: cw * scale, h: ch * scale, rgba }
    }

    pub fn save_png(&self, path: &Path) {
        std::fs::write(path, png(self.w as u32, self.h as u32, &self.rgba)).unwrap();
    }
}

// ---- a tiny PNG writer (stored deflate blocks, no compression), so no image crate is needed ----
fn crc32(data: &[u8]) -> u32 {
    let mut table = [0u32; 256];
    for (n, slot) in table.iter_mut().enumerate() {
        let mut c = n as u32;
        for _ in 0..8 {
            c = if c & 1 != 0 { 0xEDB8_8320 ^ (c >> 1) } else { c >> 1 };
        }
        *slot = c;
    }
    !data.iter().fold(!0u32, |c, b| table[((c ^ *b as u32) & 0xFF) as usize] ^ (c >> 8))
}

fn adler32(data: &[u8]) -> u32 {
    let (mut a, mut b) = (1u32, 0u32);
    for d in data {
        a = (a + *d as u32) % 65521;
        b = (b + a) % 65521;
    }
    (b << 16) | a
}

fn chunk(out: &mut Vec<u8>, kind: &[u8; 4], body: &[u8]) {
    out.extend_from_slice(&(body.len() as u32).to_be_bytes());
    let mut crc_in = kind.to_vec();
    crc_in.extend_from_slice(body);
    out.extend_from_slice(&crc_in);
    out.extend_from_slice(&crc32(&crc_in).to_be_bytes());
}

fn png(w: u32, h: u32, rgba: &[u8]) -> Vec<u8> {
    let mut raw = Vec::with_capacity((w as usize * 4 + 1) * h as usize);
    for row in rgba.chunks_exact(w as usize * 4) {
        raw.push(0);
        raw.extend_from_slice(row);
    }
    let mut z = vec![0x78, 0x01];
    let mut blocks = raw.chunks(65_535).peekable();
    while let Some(b) = blocks.next() {
        z.push(u8::from(blocks.peek().is_none()));
        z.extend_from_slice(&(b.len() as u16).to_le_bytes());
        z.extend_from_slice(&(!(b.len() as u16)).to_le_bytes());
        z.extend_from_slice(b);
    }
    z.extend_from_slice(&adler32(&raw).to_be_bytes());
    let mut out = vec![0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];
    let mut ihdr = Vec::new();
    ihdr.extend_from_slice(&w.to_be_bytes());
    ihdr.extend_from_slice(&h.to_be_bytes());
    ihdr.extend_from_slice(&[8, 6, 0, 0, 0]);
    chunk(&mut out, b"IHDR", &ihdr);
    chunk(&mut out, b"IDAT", &z);
    chunk(&mut out, b"IEND", &[]);
    out
}
