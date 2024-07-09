import os
import re
from bs4 import BeautifulSoup
import aiohttp
import asyncio
from ckip_transformers.nlp import CkipWordSegmenter
from collections import Counter
import numpy as np
import json

def remove_metadata(content):
    metadata_start = "作者"
    metadata_end = "時間"
    weekdays = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    pattern = r"\b(?:\d{1,2}\s+)?(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2}(?:\s+\d{4})?\s+(?:\d{1,2}:){2}\d{1,2}\b"
    content = re.sub(pattern, "", content)

    content = re.sub(r"http\S+|www\S+|https\S+", "", content, flags=re.MULTILINE)
    content = re.sub(r"[^\w\s]+", "", content)
    content = re.sub(f"{metadata_start}.*{metadata_end}", "", content)

    for weekday in weekdays:
        content = content.replace(weekday, "")
    return content

async def fetch(session, url):
    async with session.get(url, cookies={'over18': '1'}) as response:
        response.raise_for_status()
        content = await response.text()
        return BeautifulSoup(content, 'html.parser')

async def get_article_content(session, url):
    soup = await fetch(session, url)
    article_content = soup.select_one('#main-content').text.split('--')[0]
    article_content = remove_metadata(article_content).strip('\n')
    return article_content

async def get_replies(session, url):
    soup = await fetch(session, url)
    replies = soup.select('.push')
    push_contents = []

    if len(replies) == 0:
        return []
    else:
        for reply in replies:
            push_content = reply.select_one('.push-content').text.strip()
            push_content = push_content.strip(':').strip()
            push_content = re.sub(r"\s+", "", push_content)
            push_contents.append(push_content)
    return push_contents

def calculate_worth_reply(push_contents, article_content):
    ws_article = ws_driver([article_content])
    ws_push_contents = ws_driver(push_contents)

    article_words = ws_article[0]
    push_words = [word for ws in ws_push_contents for word in ws]

    article_word_counts = Counter(article_words)

    best_reply = None
    best_count = 0

    for ws_push in ws_push_contents:
        push_word_count = sum(article_word_counts.get(word, 0) for word in ws_push)
        if push_word_count > best_count:
            best_count = push_word_count
            best_reply = push_contents[ws_push_contents.index(ws_push)]

    if best_reply is None:
        rand_int = np.random.randint(0, len(push_contents))
        best_reply = push_contents[rand_int]

    return best_reply

def save_state(index, filename):
    with open(filename, 'w') as f:
        json.dump({'index': index}, f)

def load_state(filename, default_index):
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            state = json.load(f)
            return state.get('index', default_index)
    return default_index

async def process_article(session, article, ignore_type, punctuation_pattern, data_path):
    title = article.select_one('.title').text.strip()
    title_split = re.split('[ | ]', title)
    title_complete = ''

    confirm_flag = True
    for word in title_split:
        if word in ignore_type:
            confirm_flag = False

    for w in title_split:
        if len(w.split('[')) > 1 or w == 'Re:':
            continue
        else:
            title_complete = title_complete + w

    clean_title = re.sub(punctuation_pattern, '', title_complete, flags=re.IGNORECASE)
    clean_title = re.sub(r"\s+", "", clean_title)

    print('Title:', clean_title)

    if confirm_flag:
        try:
            link = "https://www.ptt.cc" + article.select_one('.title a')['href']
            push_contents = await get_replies(session, link)
            article_content = await get_article_content(session, link)
        except Exception as e:
            print(f"Error processing article: {e}")
            return

        if len(push_contents) > 0:
            try:
                best_reply = calculate_worth_reply(push_contents, clean_title)
                print("Reply:", best_reply)

                with open(f'{data_path}/PTT_Gossiping_Sex.txt', 'a+') as fp:
                    fp.write(clean_title + '  ' + best_reply + '\n')
            except Exception as e:
                print(f"Error calculating worth reply: {e}")

async def process_url(url_pattern, start_index, end_index, data_path, ignore_type, punctuation_pattern, state_file, max_workers=20):
    index_value = load_state(state_file, start_index)
    async with aiohttp.ClientSession() as session:
        while index_value >= end_index:
            try:
                print('Processing index:', index_value)
                url = url_pattern.format(index=index_value)

                soup = await fetch(session, url)
                articles = soup.select('.r-ent')

                tasks = [process_article(session, article, ignore_type, punctuation_pattern, data_path) for article in articles]
                await asyncio.gather(*tasks)

                save_state(index_value, state_file)
                index_value -= 1
                # await asyncio.sleep(0.5)  # 減少延遲時間
            except Exception as e:
                print(f"Error processing index {index_value}: {e}")
                continue

async def main():
    print("Initializing drivers ... WS")
    global ws_driver
    ws_driver = CkipWordSegmenter(model="albert-base", device=0)
    print("Initializing drivers ... all done")

    CURRENT_PATH = os.path.dirname(__file__)
    data_path = f'{CURRENT_PATH}/data'

    ignore_type = ['[協尋]', '[公告]', '(本文已被刪除)']
    punctuation_pattern = r'[^\w\s]'

    start_index_gossiping = 39361
    end_index_gossiping = 1
    start_index_sex = 3999
    end_index_sex = 1

    tasks = [
        process_url(
            url_pattern="https://www.ptt.cc/bbs/Gossiping/index{index}.html",
            start_index=start_index_gossiping,
            end_index=end_index_gossiping,
            data_path=data_path,
            ignore_type=ignore_type,
            punctuation_pattern=punctuation_pattern,
            state_file='data/gossiping_state.json',
            max_workers=15
        ),
        process_url(
            url_pattern="https://www.ptt.cc/bbs/sex/index{index}.html",
            start_index=start_index_sex,
            end_index=end_index_sex,
            data_path=data_path,
            ignore_type=ignore_type,
            punctuation_pattern=punctuation_pattern,
            state_file='data/sex_state.json',
            max_workers=15
        )
    ]

    await asyncio.gather(*tasks)

if __name__ == '__main__':
    asyncio.run(main())
