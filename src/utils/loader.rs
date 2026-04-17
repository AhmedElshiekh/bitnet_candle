use std::path::Path;
use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use anyhow::Result;

pub fn load_weights<P: AsRef<Path>>(path: P, device: &Device) -> Result<VarBuilder<'_>> {
    // نستخدم F32 لضمان الدقة، يمكن تحويلها لـ F16 لتقليل استهلاك الرام
    let dtype = DType::F32; 
    unsafe {
        let vb = VarBuilder::from_mmaped_safetensors(&[path.as_ref()], dtype, device)?;
        Ok(vb)
    }
}
