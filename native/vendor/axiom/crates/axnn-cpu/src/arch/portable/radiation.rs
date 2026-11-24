//! Allocation-free thermal-radiation kernels for rendering and simulation.

/// RGB samples of Planck spectral radiance relative to a fixed reference
/// blackbody, plus the corresponding relative bolometric intensity.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BlackbodyRadiance {
    /// Spectral radiance at 610, 550, and 460 nm. All channels share one
    /// normalization, so both chromaticity and temperature-driven intensity
    /// are retained.
    pub spectral_rgb: [f32; 3],
    /// Stefan-Boltzmann intensity ratio `(T / T_reference)^4`.
    pub relative_bolometric_intensity: f32,
}

impl BlackbodyRadiance {
    pub const ZERO: Self = Self {
        spectral_rgb: [0.0; 3],
        relative_bolometric_intensity: 0.0,
    };
}

const SECOND_RADIATION_M_K: f64 = 0.014_387_768_77;
const RGB_WAVELENGTHS_NM: [f64; 3] = [610.0, 550.0, 460.0];

#[inline]
fn planck_shape(temperature_kelvin: f64, wavelength_nm: f64) -> f64 {
    let wavelength = wavelength_nm * 1.0e-9;
    let wavelength_squared = wavelength * wavelength;
    let wavelength_fourth = wavelength_squared * wavelength_squared;
    let wavelength_fifth = wavelength_fourth * wavelength;
    let exponent = SECOND_RADIATION_M_K / (wavelength * temperature_kelvin);
    let denominator = wavelength_fifth * axmath::expm1_f64(exponent);
    if denominator.is_nan() || denominator <= 0.0 {
        0.0
    } else {
        denominator.recip()
    }
}

/// Evaluate a compact blackbody RGB spectrum and total intensity.
///
/// Planck's common `2hc^2` numerator cancels during normalization. Unlike a
/// per-temperature color normalization, the shared reference preserves the
/// large radiance increase of hotter material. The bolometric component uses
/// the Stefan-Boltzmann `T^4` ratio directly.
pub fn blackbody_radiance_rgb(
    temperature_kelvin: f32,
    reference_temperature_kelvin: f32,
) -> BlackbodyRadiance {
    if !temperature_kelvin.is_finite()
        || !reference_temperature_kelvin.is_finite()
        || temperature_kelvin <= 0.0
        || reference_temperature_kelvin <= 0.0
    {
        return BlackbodyRadiance::ZERO;
    }

    let temperature = f64::from(temperature_kelvin);
    let reference_temperature = f64::from(reference_temperature_kelvin);
    let reference =
        RGB_WAVELENGTHS_NM.map(|wavelength| planck_shape(reference_temperature, wavelength));
    let reference_maximum = reference[0].max(reference[1]).max(reference[2]);
    if !reference_maximum.is_finite() || reference_maximum <= 0.0 {
        return BlackbodyRadiance::ZERO;
    }

    let spectral_rgb = RGB_WAVELENGTHS_NM
        .map(|wavelength| (planck_shape(temperature, wavelength) / reference_maximum) as f32);
    let ratio = temperature_kelvin / reference_temperature_kelvin;
    let ratio_squared = ratio * ratio;
    BlackbodyRadiance {
        spectral_rgb,
        relative_bolometric_intensity: ratio_squared * ratio_squared,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invalid_blackbody_inputs_are_dark() {
        assert_eq!(
            blackbody_radiance_rgb(0.0, 10_000.0),
            BlackbodyRadiance::ZERO
        );
        assert_eq!(
            blackbody_radiance_rgb(5_000.0, f32::NAN),
            BlackbodyRadiance::ZERO
        );
    }

    #[test]
    fn stefan_boltzmann_ratio_is_fourth_power() {
        let sample = blackbody_radiance_rgb(10_000.0, 5_000.0);
        assert_eq!(sample.relative_bolometric_intensity, 16.0);
    }

    #[test]
    fn planck_spectrum_retains_temperature_driven_color_and_intensity() {
        let cool = blackbody_radiance_rgb(3_000.0, 10_000.0);
        let reference = blackbody_radiance_rgb(10_000.0, 10_000.0);
        assert!(cool.spectral_rgb[0] > cool.spectral_rgb[2]);
        assert!(reference.spectral_rgb[2] > reference.spectral_rgb[0]);
        let maximum = reference.spectral_rgb.into_iter().fold(0.0_f32, f32::max);
        assert!((maximum - 1.0).abs() < 2.0e-6, "{maximum}");
        for channel in 0..3 {
            assert!(reference.spectral_rgb[channel] > cool.spectral_rgb[channel]);
        }
    }
}
