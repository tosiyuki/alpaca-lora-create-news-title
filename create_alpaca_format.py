# livedoorのニュースコーパスからアルパカに学習させるデータを作成する
import os
import json


def parse_file(file_path):
    with open(file_path, 'r') as f:
        title = ''
        news = ''

        for i, line in enumerate(f.readlines()):
            if i < 2:
                # urlと日付は無視
                continue
            elif i == 2:
                title = line.strip()
            else:
                news += line.strip()

        return title, news


def main():
    DATADIR = 'data/ldcc-20140209/text'
    LICENCE_FILENAME = 'LICENSE.txt'
    learn_datas = []
    news_list = []
    
    for item in os.listdir(DATADIR):
        if not os.path.isdir(os.path.join(DATADIR, item)):
            # ディレクトリ以外は無視
            continue

        for filename in os.listdir(f'{DATADIR}/{item}'):
            learn_data = {
                "instruction": "次のニュースのタイトルを考えて",
                "input": "",
                "output": ""
            }

            if filename == LICENCE_FILENAME:
                continue

            title, news = parse_file(f'{DATADIR}/{item}/{filename}')

            news_list.append(news)
            learn_data["input"] = news
            learn_data["output"] = title

            learn_datas.append(learn_data)

    json_learn_data = json.dumps(learn_datas, indent=4, ensure_ascii=False)
    with open('news_data_ja.json', 'w', encoding="utf-8") as f:
        f.write(json_learn_data)
        

if __name__ == '__main__':
    main()