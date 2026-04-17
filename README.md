# BitNet Candle: SmolLM2-135M Inference on Android (Termux)

هذا المشروع يهدف إلى تشغيل نموذج اللغة المصغر **SmolLM2-135M** على بيئة أندرويد باستخدام مكتبة **Candle** المكتوبة بلغة **Rust**. المشروع مصمم ليكون خفيفاً وسريعاً ومتوافقاً مع موارد الهواتف الذكية المحدودة.

## 🚀 المميزات
- **أداء عالٍ:** مبني باستخدام لغة Rust ومكتبة Candle (التي طورتها HuggingFace).
- **موجه للهواتف:** يدعم التشغيل على أندرويد عبر تطبيق Termux.
- **كفاءة الذاكرة:** يستخدم تقنيات Memory Mapping لتحميل الأوزان دون استهلاك الرام بالكامل.
- **دقة SmolLM2:** يعتمد على معمارية Llama-based المصغرة من فريق HuggingFace.

## 🛠 المتطلبات التقنية
- **بيئة التشغيل:** أندرويد (Termux).
- **المكتبات الأساسية:**
  - `candle-core v0.10.2`
  - `tokenizers v0.22.2`
  - `safetensors v0.7.0`
- **النموذج:** SmolLM2-135M (أوزان بتنسيق Safetensors).

## 📁 هيكل المشروع
```text
bitnet_candle/
├── Cargo.toml           # ملف إعدادات التبعيات (النسخ المثبتة 0.10.2)
├── weights/             # مجلد الأوزان
│   ├── model.safetensors
│   └── tokenizer.json
└── src/
    ├── main.rs          # المحرك الرئيسي لعملية التوليد (Inference)
    ├── model/
    │   └── llama_1bit.rs # تعريف معمارية SmolLM2 (RMSNorm, Linear)
    └── utils/
        └── loader.rs     # أداة تحميل الأوزان بكفاءة
```

## ⚙️ معمارية النموذج
المشروع يحاكي معمارية Transformer المصغرة التي تستخدمها SmolLM2:
- **Embedding Dim:** 576
- **Normalization:** RMSNorm
- **Tokenizer:** Llama-based (Fast Tokenizer)

## 🏗 كيفية التشغيل

1. **تثبيت Rust في Termux:**
   ```bash
   pkg install rust
   ```

2. **تحميل التبعيات وبناء المشروع:**
   ```bash
   export CARGO_TARGET_DIR=/root/cargo_target
   rm -f Cargo.lock
   cargo build --release
   ```

3. **تشغيل النموذج:**
   ```bash
   ./target/release/bitnet_candle
   ```

## ⚠️ ملاحظات هامة
- تأكد من أن الأوزان موجودة في مسار `weights/` الصحيح.
- في حالة حدوث تعارض في التبعيات (مثل خطأ `bf16`), تأكد من استخدام النسخة المثبتة في `Cargo.toml` باستخدام علامة `=0.10.2`.
- تم تعطيل ميزة `dotprod` في بعض الأجهزة لضمان التوافقية مع معالجات ARM القديمة.

## 📝 الرخصة
هذا المشروع مفتوح المصدر للاستخدام التعليمي والتطويري.
