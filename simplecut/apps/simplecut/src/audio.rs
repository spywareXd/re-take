//! The sound card, through cpal. Adapted from FilmCraft's player, without the device settings:
//! it uses the system's default output, or the first device that opens.

use std::panic::{AssertUnwindSafe, catch_unwind};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use simplecut_core::{AudioOutput, FillFn};

pub struct CpalOut {
    device: cpal::Device,
    stream: Option<cpal::Stream>,
}

impl CpalOut {
    /// The default output device, or the first one that has a usable format. None: no sound card.
    pub fn new() -> Option<Self> {
        let host = cpal::default_host();
        let usable = |d: &cpal::Device| d.default_output_config().is_ok();
        let device = host.default_output_device().filter(usable).or_else(|| host.output_devices().ok()?.find(usable))?;
        Some(Self { device, stream: None })
    }
}

impl AudioOutput for CpalOut {
    fn start(&mut self, fill: FillFn) -> Result<u32, String> {
        self.stop();
        let supported = self.device.default_output_config().map_err(|e| e.to_string())?;
        let channels = usize::from(supported.channels());
        let rate = supported.sample_rate().0;
        if channels == 0 || rate == 0 {
            return Err("the sound card reported no channels or no sample rate".into());
        }
        let config: cpal::StreamConfig = supported.clone().into();
        let stream = match supported.sample_format() {
            cpal::SampleFormat::F32 => build::<f32>(&self.device, &config, fill, channels),
            cpal::SampleFormat::I16 => build::<i16>(&self.device, &config, fill, channels),
            cpal::SampleFormat::U16 => build::<u16>(&self.device, &config, fill, channels),
            cpal::SampleFormat::I32 => build::<i32>(&self.device, &config, fill, channels),
            cpal::SampleFormat::U32 => build::<u32>(&self.device, &config, fill, channels),
            cpal::SampleFormat::I8 => build::<i8>(&self.device, &config, fill, channels),
            cpal::SampleFormat::U8 => build::<u8>(&self.device, &config, fill, channels),
            cpal::SampleFormat::F64 => build::<f64>(&self.device, &config, fill, channels),
            other => return Err(format!("unsupported sample format {other:?}")),
        }
        .map_err(|e| e.to_string())?;
        stream.play().map_err(|e| e.to_string())?;
        self.stream = Some(stream);
        Ok(rate)
    }

    fn stop(&mut self) {
        // dropping the stream stops it and its thread
        self.stream = None;
    }
}

/// A stream whose callback runs `fill` in `f32` and converts to the card's sample type. A panic
/// in `fill` plays silence instead of taking the audio thread down.
fn build<T>(device: &cpal::Device, config: &cpal::StreamConfig, mut fill: FillFn, channels: usize) -> Result<cpal::Stream, cpal::BuildStreamError>
where
    T: cpal::SizedSample + cpal::FromSample<f32>,
{
    let mut scratch: Vec<f32> = Vec::new();
    device.build_output_stream(
        config,
        move |out: &mut [T], _| {
            scratch.resize(out.len(), 0.0);
            if catch_unwind(AssertUnwindSafe(|| fill(&mut scratch, channels))).is_err() {
                scratch.fill(0.0);
            }
            for (o, s) in out.iter_mut().zip(&scratch) {
                *o = T::from_sample(if s.is_finite() { s.clamp(-1.0, 1.0) } else { 0.0 });
            }
        },
        |e| log::warn!("audio stream error: {e}"),
        None,
    )
}
