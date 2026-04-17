use candle_core::{Device, Tensor, Module, IndexOp};
mod model;
mod utils;

use crate::model::llama_1bit::SmolLM;
use crate::utils::loader::load_weights;
use tokenizers::Tokenizer;

fn main() -> anyhow::Result<()> {
    let device = Device::Cpu;
    let weights_path = "weights/model.safetensors";
    let tokenizer_path = "weights/tokenizer.json";

    println!("⚡ Loading SmolLM2 with Candle 0.10.2...");

    let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(anyhow::Error::msg)?;
    let vb = load_weights(weights_path, &device)?;

    // الأبعاد الدقيقة لـ SmolLM2-135M
    let model = SmolLM::load(vb, 576, 49152)?;

    let prompt = "Rust programming is";
    let tokens = tokenizer.encode(prompt, true).map_err(anyhow::Error::msg)?;
    let mut token_ids = tokens.get_ids().to_vec();

    print!("\n{}: ", prompt);

    for _ in 0..20 {
        let input = Tensor::new(token_ids.as_slice(), &device)?.unsqueeze(0)?;
        let logits = model.forward(&input)?;
        
        // التعديل: استخدام IndexOp للحصول على آخر Logit بكفاءة
        let logits = logits.i((0, logits.dim(1)? - 1))?;
        let next_token = logits.argmax(0)?.to_scalar::<u32>()?;

        token_ids.push(next_token);
        let word = tokenizer.decode(&[next_token], true).map_err(anyhow::Error::msg)?;
        print!("{}", word);
        
        if next_token == 0 { break; }
    }

    Ok(())
}



