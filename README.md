## Image to Poem - Telegram Bot




```python

# add your end variables
cp .env.sample .env

python3 fetch_models.py

docker build . -t poem-ai

docker run -d --env-file ./.env poem-ai

```
