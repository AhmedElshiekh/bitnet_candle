use candle_core::{Result, Tensor, Module};
use candle_nn::{VarBuilder, RmsNorm, Linear, rms_norm, linear};

pub struct SmolBlock {
    pub input_layernorm: RmsNorm,
    pub self_attn_q: Linear,
    pub post_attention_layernorm: RmsNorm,
}

pub struct SmolLM {
    pub blocks: Vec<SmolBlock>,
    pub norm: RmsNorm,
    pub lm_head: Linear,
}

impl SmolLM {
    pub fn load(vb: VarBuilder, d_model: usize, vocab_size: usize) -> Result<Self> {
        let mut blocks = Vec::new();
        
        // تحميل طبقة واحدة للتجربة (يمكن زيادة العدد لـ 30 لـ SmolLM2)
        for i in 0..1 {
            let p = vb.pp(&format!("model.layers.{}", i));
            
            // تصحيح: الإصدار 0.10.2 يتطلب VarBuilder كمرجع أو استهلاك منظم
            let self_attn_q = linear(d_model, d_model, p.pp("self_attn.q_proj"))?;
            let input_layernorm = rms_norm(d_model, 1e-5, p.pp("input_layernorm"))?;
            let post_attention_layernorm = rms_norm(d_model, 1e-5, p.pp("post_attention_layernorm"))?;

            blocks.push(SmolBlock {
                input_layernorm,
                self_attn_q,
                post_attention_layernorm,
            });
        }

        let norm = rms_norm(d_model, 1e-5, vb.pp("model.norm"))?;
        let lm_head = linear(d_model, vocab_size, vb.pp("lm_head"))?;

        Ok(Self { blocks, norm, lm_head })
    }
}

impl Module for SmolLM {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut x = xs.clone();
        for block in &self.blocks {
            let residual = x.clone();
            // استخدام المرجع &x في forward ليتوافق مع Module trait
            let norm_x = block.input_layernorm.forward(&x)?;
            let attn_out = block.self_attn_q.forward(&norm_x)?;
            x = (residual + attn_out)?;
        }
        x = self.norm.forward(&x)?;
        self.lm_head.forward(&x)
    }
}
