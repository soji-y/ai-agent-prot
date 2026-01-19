import asyncio
import requests
import ollama
from playwright.async_api import async_playwright
import re
import json

# --- 設定項目 ---
SEACH_SITE = "https://www.yahoo.co.jp"
LLM_MODEL = "gemma3:4b"
SEARCH_NUM = 1  # 1回で取得したい件数
# NO_THINK = "/no_think"

# 検索してサイトのリストを取得
async def _search_query(query, search_site="https://www.yahoo.co.jp", call_cnt=0):
  async with async_playwright() as p:
    browser = await p.chromium.launch(headless=True)
    page = await browser.new_page()

    await page.goto(search_site)
    await page.fill("input[name='p']", query)
    await page.press("input[name='p']", "Enter")
    await page.wait_for_selector("div.sw-Card__title a")

    results = []
    # 取得範囲の決定
    start_index = SEARCH_NUM * call_cnt
    end_index = SEARCH_NUM * (call_cnt + 1)
    total_collected = 0  # これまでに取得した件数

    while total_collected < (end_index - start_index):
      elements = page.locator("div.sw-Card__title a")
      count = await elements.count()

      # ページ内のインデックスを計算
      for i in range(start_index, end_index):
        if i > count -1:
          break

        # 実際に取得する「グローバルな」件数の調整
        # global_index = start_index + total_collected
        # print(f"Total Count: {global_index}")
        # インデックスに関係なくページ内 i で取得
        
        raw_title = await elements.nth(i).inner_text()
        title = re.sub(r'\s+', ' ', raw_title.strip())
        url = await elements.nth(i).get_attribute("href")
        results.append({"title": title, "url": url})

        total_collected += 1

      if total_collected >= SEARCH_NUM:
        break

      # 「次へ」ボタンを探す（class または aria-label）
      # next_button = page.locator("div.Pagination__next a")
      # if await next_button.count() > 0:
      #   next_href = await next_button.first.get_attribute("href")
      #   if next_href:
      #     await page.goto(next_href)
              
      # await next_button.first.click()
      # await page.wait_for_load_state("load")
      # await page.wait_for_selector("div.sw-Card__title a")

      next_button = page.locator("div.Pagination__next a")
      if await next_button.count() == 0:
        break  # 次ページなし

      try:
        await next_button.first.wait_for(state="visible", timeout=5000)
        await next_button.first.click()
        await page.wait_for_load_state("load")
        await page.wait_for_selector("div.sw-Card__title a")
      except TimeoutError:
        print("次へボタンがクリックできませんでした。終了します。")
        break
      start_index -= count
      end_index -= count
      
    await browser.close()
    return results

async def _summarize_site(site, agent=None):
  url = site["url"]
  title = site["title"].strip()
  print(f"   ▶ 情報を取得・要約中･･･ [サイト] {title}") # , [URL]{url}")
  try:
    summary = await _summarize_page(url, agent)
    # print(summary)
    # return f"{title}:\n{summary}"
    return {"title": title, "url":url, "summary": summary}
  except Exception as e:
      print(f"要約失敗: {e}")
      return None
    
async def _summarize_page(url, agent=None):
  async with async_playwright() as p:
    browser = await p.chromium.launch(headless=True)
    page = await browser.new_page()
    await page.goto(url, timeout=60000)

    # 本文らしきテキストを抜き出す（シンプル例）
    content = await page.inner_text("body")

    await browser.close()

  # prompt = f"以下のページ内容を日本語で簡潔に要約してください:\n\n{content}" # [:5000]} # {NO_THINK}"
  prompt = f"以下のページ内容を日本語でなるべく情報量を落とさずに要約してください:\n{content[:5000]}" # {NO_THINK}"
  if agent is not None:
    response = agent.generate(prompt, sys_use=False)
  else:
    res = ollama.chat(model=LLM_MODEL, messages=[{"role":"user","content":prompt}])    
    response =  res["message"]["content"] if res else ""

  return response

# ウェブ検索
def web_search(query, num=0, agent=None):
  # global call_cnt
  selected_sites = asyncio.run(_search_query(query, SEACH_SITE, call_cnt=num))
  # call_cnt += 1
  
  # 各サイトを要約してまとめる
  results = ""
  # if agent is not None:
  res_list = []
  for site in selected_sites:
    res = asyncio.run(_summarize_site(site, agent)) 
    if res:
      res_list.append(res)
      results += res["summary"] + "\n"
      
  # else:
  #   results = "\n".join([s[] for s in selected_sites])
        
  # tasks = [summarize_site(site) for site in selected_sites_dic]
  # summaries_results_dic = await asyncio.gather(*tasks)
  # None を除外してリストにまとめる
  # summaries_dic = [s for s in summaries_results_dic if s is not None]
  
  return results
    
if __name__ == "__main__":
  result = web_search("天気")
  print(result)