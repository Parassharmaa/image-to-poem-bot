import requests
from PIL import Image
from transformers import GPT2TokenizerFast, ViTImageProcessor, VisionEncoderDecoderModel
from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, ContextTypes, filters
import openai
import os
from dotenv import load_dotenv


load_dotenv()

model = VisionEncoderDecoderModel.from_pretrained(
    "./models/vit-gpt2-image-captioning")
tokenizer = GPT2TokenizerFast.from_pretrained(
    "./models/vit-gpt2-image-captioning")
image_processor = ViTImageProcessor.from_pretrained(
    "./models/vit-gpt2-image-captioning")

print("Starting...")


openai.api_key = os.getenv("OPENAI_API_KEY")

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")


def describe_image(img):
    image = Image.open(requests.get(img, stream=True).raw)
    pixel_values = image_processor(image, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values, max_length=100)
    generated_text = tokenizer.batch_decode(
        generated_ids, skip_special_tokens=True)[0]
    return generated_text


def generate_captions(img_description):
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=f"Generate detailed captions for social media on an image that has \"{img_description}\", keep the caption creative and unique, few should be in first person:\n1.",
        temperature=0.92,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    return response.choices[0].text


def generate_poem(img_description):
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=f"Write a poem on an image that has \"{img_description}\n",
        temperature=0.92,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    return response.choices[0].text


def generate_quote(img_description):
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=f"Write a quote on an image that has \"{img_description}\n",
        temperature=0.92,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    return response.choices[0].text


async def handle_photo_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    caption = update.effective_message.caption
    print(caption)
    if caption is None or caption not in ['/poem', '/quote', '/captions']:
        await update.message.reply_text(f'Please use commands like /poem, /captions, /quote.')
        return
    if (len(update.effective_message.photo) > 0):
        img = await update.effective_message.photo[0].get_file()
        img_desc = describe_image(img.file_path)
        print(img_desc)
        if caption == "/captions":
            captions = generate_captions(img_desc)
            await update.message.reply_text(f'Here are some captions:\n1.{captions}')
            return
        if caption == '/poem':
            results = generate_poem(img_desc)
            await update.message.reply_text(results)
            return
        if caption == '/quote':
            results = generate_quote(img_desc)
            await update.message.reply_text(results)
            return


async def handle_text_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(f'Hi there, I am Caption Bot. Send me an image with commands /poem, /captions, /quote and I will send back some magic.')


app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

app.add_handler(MessageHandler(
    filters.PHOTO | filters.CAPTION, handle_photo_message))

app.add_handler(MessageHandler(filters.TEXT, handle_text_message))

app.run_polling()
