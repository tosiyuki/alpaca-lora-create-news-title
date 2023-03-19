# alpaca-lora-create-news-title
[Alpaca-LoRA](https://github.com/tloen/alpaca-lora)をlivedoorニュースコーパスでFineTuningさせるサンプルコード

## 環境構築
```
pip install -r requirements.txt
```

[livedoorニュースコーパス](https://www.rondhuit.com/download.html)をダウンロードしてdataディレクトリの中に格納してください。

## 実行手順
学習済みのデータや整形済みのデータも格納しているため、1や2の手順を飛ばして推論だけを行うことも可能です。  
学習にはRTX3080で10時間ほどかかりました。

1. 学習データの作成
```
python create_alpaca_format.py
```

2. 学習
```
python finetune_livedoor_news.py
```

3. 推論
```
python generate_news_title.py
```

## 推論結果
![推論結果](/img/img1.PNG) 