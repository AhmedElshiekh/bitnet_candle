use tokenizers::Tokenizer;
use anyhow::Result;

pub struct LlamaTokenizer {
    inner: Tokenizer,
}

impl LlamaTokenizer {
    pub fn new(path: &str) -> Result<Self> {
        // استخدام FromFile مباشرة
        let inner = Tokenizer::from_file(path).map_err(|e| anyhow::anyhow!(e))?;
        Ok(Self { inner })
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let encoding = self.inner.encode(text, true).map_err(|e| anyhow::anyhow!(e))?;
        Ok(encoding.get_ids().to_vec())
    }

    pub fn decode(&self, ids: Vec<u32>) -> Result<String> {
        self.inner.decode(&ids, true).map_err(|e| anyhow::anyhow!(e))
    }
}
