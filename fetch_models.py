from transformers import GPT2TokenizerFast, ViTImageProcessor, VisionEncoderDecoderModel

model = VisionEncoderDecoderModel.from_pretrained(
    "nlpconnect/vit-gpt2-image-captioning")
tokenizer = GPT2TokenizerFast.from_pretrained(
    "nlpconnect/vit-gpt2-image-captioning")
image_processor = ViTImageProcessor.from_pretrained(
    "nlpconnect/vit-gpt2-image-captioning")

model.save_pretrained('./models/vit-gpt2-image-captioning')
tokenizer.save_pretrained('./models/vit-gpt2-image-captioning')
image_processor.save_pretrained('./models/vit-gpt2-image-captioning')
