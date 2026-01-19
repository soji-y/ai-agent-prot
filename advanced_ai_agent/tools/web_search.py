import asyncio
import requests
import ollama
# from playwright.async_api import async_playwright
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError

import re
import json
from readability import Document
import httpx
from bs4 import BeautifulSoup


# --- 設定項目 ---
SEACH_SITE = "https://www.yahoo.co.jp"
LLM_MODEL = "gemma3:4b"
SEARCH_NUM = 1  # 1回で取得したい件数
# NO_THINK = "/no_think"

TOOL_NAME = "search"
TOOL_ALIAS = {"web": "search", "search": "search"}
TOOL_USING = """
search:
 - Description: Search the web for up-to-date information.
 - Input: search queries (string) (example: "today news weather")
 - Output: A short summary of search results.
"""
TOOL_TITLE = "Web検索"
TOOL_TYPE = "tool"

class WebSearch:
  @staticmethod
  def name():
    return TOOL_NAME
  
  @staticmethod
  def alias():
    return TOOL_ALIAS

  @staticmethod
  def using():
    return TOOL_USING

  @staticmethod
  def title():
    return TOOL_TITLE

  @staticmethod
  def type():
    return TOOL_TYPE
  
  # 検索してサイトのリストを取得
  async def _search_query(self, query, search_site="https://www.yahoo.co.jp", call_cnt=0):
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

  # サイトを要約する
  async def _summarize_site_0(self, site, agent=None):
    url = site["url"]
    title = site["title"].strip()
    print(f"   ▶ 情報を取得・要約中･･･ [サイト] {title}") # , [URL]{url}")
    try:
      summary = await self._summarize_page(url, agent)
      # print(summary)
      # return f"{title}:\n{summary}"
      return {"title": title, "url":url, "summary": summary}
    except Exception as e:
        print(f"要約失敗: {e}")
        return None
  
  # ページを要約する
  async def _summarize_page_0(self, url, agent=None):
    async with async_playwright() as p:
      browser = await p.chromium.launch(headless=True)
      page = await browser.new_page()
      await page.goto(url, timeout=60000)

      # 本文らしきテキストを抜き出す（シンプル例）
      content = await page.inner_text("body")

      await browser.close()

    # prompt = f"以下のページ内容を日本語で簡潔に要約してください:\n\n{content}" # [:5000]} # {NO_THINK}"
    prompt = f"以下のページ内容を日本語でなるべく情報量を落とさずに要約してください:\n{content}" # [:5000]} # {NO_THINK}"
    if agent is not None:
      response = agent.generate(prompt, sys_use=False)
    else:
      res = ollama.chat(model=LLM_MODEL, messages=[{"role":"user","content":prompt}])    
      response =  res["message"]["content"] if res else ""

    return response

  async def _summarize_site_1(self, site, agent=None):
    url = site["url"]
    title = site["title"].strip()
    print(f"   ▶ 要約中･･･ [サイト] {title}")

    try:
      # Playwrightを使わずHTTPで本文取得
      async with httpx.AsyncClient(timeout=15.0) as client:
        r = await client.get(url)
        if r.status_code != 200:
          raise Exception(f"HTTP {r.status_code}")

      # Readabilityで本文抽出
      doc = Document(r.text)
      readable_html = doc.summary()
      soup = BeautifulSoup(readable_html, "html.parser")
      content = soup.get_text(separator="\n").strip()

      prompt = f"以下のページ内容を日本語でなるべく情報量を落とさずに要約してください:\n{content[:8000]}"
      if agent:
        response = agent.generate(prompt, sys_use=False)
      else:
        res = ollama.chat(model=LLM_MODEL, messages=[{"role": "user", "content": prompt}])
        response = res["message"]["content"] if res else ""
      return {"title": title, "url": url, "summary": response}

    except Exception as e:
      print(f"要約失敗: {e}")
      return None


  async def _summarize_site_2(self, site, agent=None):
    url = site["url"]
    title = site["title"].strip()
    print(f"   ▶ 要約中･･･ [サイト] {title}")

    try:
      # httpxセッションを再利用（DNSキャッシュ・TLS再利用）
      async with httpx.AsyncClient(
        timeout=15.0,
        follow_redirects=True,
        headers={"User-Agent": "Mozilla/5.0 (compatible; SiteSummarizer/1.0)"}
      ) as client:
        r = await client.get(url)
        r.raise_for_status()

      # BeautifulSoupで本文抽出（readability不要・安定）
      soup = BeautifulSoup(r.text, "lxml")

      # <article>タグや<p>タグ中心に本文抽出
      article = soup.find("article")
      if article:
        text = article.get_text(separator="\n")
      else:
        paragraphs = [p.get_text() for p in soup.find_all("p")]
        text = "\n".join(paragraphs)

      # 不要な空白・改行を整理
      text = re.sub(r"\n{2,}", "\n", text).strip()

      if not text or len(text) < 100:
        raise Exception("本文が取得できませんでした")

      # LLMへ要約依頼
      prompt = (
        "以下のページ内容を日本語でなるべく情報量を落とさずに要約してください:\n"
        f"{text[:8000]}"
      )

      if agent:
        response = agent.generate(prompt, sys_use=False)
      else:
        import ollama
        res = ollama.chat(model=self.model_name, messages=[{"role": "user", "content": prompt}])
        response = res["message"]["content"] if res else ""

      return {"title": title, "url": url, "summary": response}

    except Exception as e:
      print(f"要約失敗: {e}")
      return None
    
      
  async def _summarize_site(self, site, agent=None):
    url = site["url"]
    title = site["title"].strip()
    print(f"   ▶ 要約中･･･ [サイト] {title}")

    html = None

    # 1) まずは軽量な httpx で取得（通常はこれでOK）
    try:
      async with httpx.AsyncClient(
        timeout=15.0,
        follow_redirects=True,
        headers={"User-Agent": "Mozilla/5.0 (compatible; SiteSummarizer/1.0)"}
      ) as client:
        r = await client.get(url)
        r.raise_for_status()
        html = r.text
    except Exception as e:
      err_str = str(e).lower()
      # SSL系や transport エラーなどは Playwright にフォールバックする
      if "unsafe legacy renegotiation" in err_str or "ssl" in err_str or "tls" in err_str or "certificate" in err_str:
        print(f"⚠ httpx取得でSSL/TLSエラー発生（{e}）。Playwrightで回避します: {url}")
        try:
          # Playwright を使ってページの HTML を取得（ブラウザの TLS スタックを利用）
          async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            # ページ全体の取得を試みる（タイムアウト短め）
            try:
              await page.goto(url, timeout=30000, wait_until="domcontentloaded")
            except PlaywrightTimeoutError:
              # それでもタイムアウトしたら try: page.content() を試す
              pass
            # ページの完全な HTML を取得
            html = await page.content()
            await browser.close()
        except Exception as e2:
          print(f"Playwright でも取得失敗: {e2}")
          print(f"要約失敗: {e}")
          return None
      else:
        # httpxの他のエラー（接続拒否等）は失敗とする
        print(f"要約失敗: {e}")
        return None

    # 2) 取得した HTML を解析して本文を抜く（lxml + BeautifulSoup）
    try:
      # soup = BeautifulSoup(html, "lxml")
      soup = BeautifulSoup(html, "html.parser")

      # 優先: <article> を使う。なければ <main>、なければ <p> を連結
      article = soup.find("article") or soup.find("main")
      if article:
        text = article.get_text(separator="\n")
      else:
        paragraphs = [p.get_text() for p in soup.find_all("p")]
        text = "\n".join(paragraphs)

      # ノイズ除去
      text = re.sub(r"\n{2,}", "\n", text).strip()

      if not text or len(text) < 80:
        # 代替: body のテキストを最終手段で使う
        body = soup.body
        text = body.get_text(separator="\n").strip() if body else text
        text = re.sub(r"\n{2,}", "\n", text).strip()

      if not text:
        print("要約失敗: 本文が抽出できませんでした")
        return None

    except Exception as e:
      print(f"要約失敗(パースエラー): {e}")
      return None

    # 3) LLM で要約（既存の agent / ollama 呼び出しを使う）
    try:
      prompt = (
        "以下のページ内容を日本語でなるべく情報量を落とさずに要約してください:\n"
        f"{text[:8000]}"
      )

      if agent:
        response = agent.generate(prompt, sys_use=False)
      else:
        import ollama
        res = ollama.chat(model=LLM_MODEL, messages=[{"role": "user", "content": prompt}])
        response = res["message"]["content"] if res else ""

      return {"title": title, "url": url, "summary": response}

    except Exception as e:
      print(f"要約失敗(要約処理): {e}")
      return None
    
  # ウェブ検索
  def __call__(self, query, num=0, agent=None):
    # global call_cnt
    selected_sites = asyncio.run(self._search_query(query, SEACH_SITE, call_cnt=num))
    # call_cnt += 1
    
    # 各サイトを要約してまとめる
    results = ""
    # if agent is not None:
    res_list = []
    for site in selected_sites:
      res = asyncio.run(self._summarize_site(site, agent)) 
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
  web = WebSearch()
  result = web("天気")
  print(result)