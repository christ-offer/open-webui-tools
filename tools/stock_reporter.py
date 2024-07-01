"""
title: Stock Market Helper
description: A comprehensive stock analysis tool that gathers data from Finnhub API and compiles a detailed report.
author: Pyotr Growpotkin
github: https://github.com/christ-offer/open-webui-tools
version: 0.0.6
license: MIT
requirements: finnhub-python
"""

import finnhub
import requests
import aiohttp
import asyncio
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
from bs4 import BeautifulSoup
from pydantic import BaseModel
from datetime import datetime, timedelta
from typing import Dict, Any, List, Union, Generator, Iterator, Tuple, Optional
from functools import lru_cache


def _format_date(date: datetime) -> str:
    """Helper function to format date for Finnhub API"""
    return date.strftime("%Y-%m-%d")


# Caching for expensive operations
@lru_cache(maxsize=128)
def _get_sentiment_model():
    model_name = "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model


def _get_basic_info(client: finnhub.Client, ticker: str) -> Dict[str, Any]:
    """
    Fetch comprehensive company information from Finnhub API.
    """
    profile = client.company_profile2(symbol=ticker)
    basic_financials = client.company_basic_financials(ticker, "all")
    peers = client.company_peers(ticker)

    return {"profile": profile, "basic_financials": basic_financials, "peers": peers}


def _get_current_price(client: finnhub.Client, ticker: str) -> Dict[str, float]:
    """
    Fetch current price and daily change from Finnhub API.
    """
    quote = client.quote(ticker)
    return {
        "current_price": quote["c"],
        "change": quote["dp"],
        "change_amount": quote["d"],
        "high": quote["h"],
        "low": quote["l"],
        "open": quote["o"],
        "previous_close": quote["pc"],
    }


def _get_company_news(client: finnhub.Client, ticker: str) -> List[Dict[str, str]]:
    """
    Fetch recent news articles about the company from Finnhub API.
    Returns a list of dictionaries containing news item details.
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    news = client.company_news(ticker, _format_date(start_date), _format_date(end_date))

    news_items = news[:10]  # Get the first 10 news items

    return [{"url": item["url"], "title": item["headline"]} for item in news_items]


# Asynchronous web scraping
async def _async_web_scrape(session: aiohttp.ClientSession, url: str) -> Optional[str]:
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    try:
        async with session.get(url, headers=headers) as response:
            response.raise_for_status()
            content = await response.text()

        soup = BeautifulSoup(content, "html.parser")

        for element in soup(
            ["script", "style", "nav", "footer", "header", "a", "button"]
        ):
            element.decompose()

        text_elements = soup.find_all(["p", "h1", "h2", "h3", "h4", "h5", "h6", "li"])
        text = " ".join(element.get_text().strip() for element in text_elements)
        text = " ".join(text.split())

        if len(text) < 100:
            print(f"Skipping article (too short): {url}")
            return None

        return text

    except Exception as e:
        print(f"Error scraping web page: {str(e)}")
        return None


# Asynchronous sentiment analysis
async def _async_sentiment_analysis(content: str) -> Dict[str, Union[str, float]]:
    tokenizer, model = _get_sentiment_model()

    inputs = tokenizer(content, return_tensors="pt", truncation=True, max_length=512)

    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    sentiment_scores = probabilities.tolist()[0]

    # Update sentiment labels to match the new model's output
    sentiments = ["Neutral", "Positive", "Negative"]
    sentiment = sentiments[sentiment_scores.index(max(sentiment_scores))]

    confidence = max(sentiment_scores)

    return {"sentiment": sentiment, "confidence": confidence}


# Asynchronous data gathering
async def _async_gather_stock_data(
    client: finnhub.Client, ticker: str
) -> Dict[str, Any]:
    basic_info = _get_basic_info(client, ticker)
    current_price = _get_current_price(client, ticker)
    news_items = _get_company_news(client, ticker)

    async with aiohttp.ClientSession() as session:
        scrape_tasks = [_async_web_scrape(session, item["url"]) for item in news_items]
        contents = await asyncio.gather(*scrape_tasks)

    sentiment_tasks = [
        _async_sentiment_analysis(content) for content in contents if content
    ]
    sentiments = await asyncio.gather(*sentiment_tasks)

    sentiment_results = [
        {
            "url": news_items[i]["url"],
            "title": news_items[i]["title"],
            # "content": contents[i][:500] + "..." if contents[i] and len(contents[i]) > 500 else contents[i],
            "sentiment": sentiment["sentiment"],
            "confidence": sentiment["confidence"],
        }
        for i, sentiment in enumerate(sentiments)
        if contents[i]
    ]

    return {
        "basic_info": basic_info,
        "current_price": current_price,
        "sentiments": sentiment_results,
    }


def _compile_report(data: Dict[str, Any]) -> str:
    """
    Compile gathered data into a comprehensive structured report.
    """
    profile = data["basic_info"]["profile"]
    financials = data["basic_info"]["basic_financials"]
    peers = data["basic_info"]["peers"]
    price_data = data["current_price"]

    report = f"""
    Comprehensive Stock Analysis Report for {profile['name']} ({profile['ticker']})

    Basic Information:
    Industry: {profile.get('finnhubIndustry', 'N/A')}
    Market Cap: {profile.get('marketCapitalization', 'N/A'):,.0f} M USD
    Share Outstanding: {profile.get('shareOutstanding', 'N/A'):,.0f} M
    Country: {profile.get('country', 'N/A')}
    Exchange: {profile.get('exchange', 'N/A')}
    IPO Date: {profile.get('ipo', 'N/A')}

    Current Trading Information:
    Current Price: ${price_data['current_price']:.2f}
    Daily Change: {price_data['change']:.2f}% (${price_data['change_amount']:.2f})
    Day's Range: ${price_data['low']:.2f} - ${price_data['high']:.2f}
    Open: ${price_data['open']:.2f}
    Previous Close: ${price_data['previous_close']:.2f}

    Key Financial Metrics:
    52 Week High: ${financials['metric'].get('52WeekHigh', 'N/A')}
    52 Week Low: ${financials['metric'].get('52WeekLow', 'N/A')}
    P/E Ratio: {financials['metric'].get('peBasicExclExtraTTM', 'N/A')}
    EPS (TTM): ${financials['metric'].get('epsBasicExclExtraItemsTTM', 'N/A')}
    Return on Equity: {financials['metric'].get('roeRfy', 'N/A')}%
    Debt to Equity: {financials['metric'].get('totalDebtToEquityQuarterly', 'N/A')}
    Current Ratio: {financials['metric'].get('currentRatioQuarterly', 'N/A')}
    Dividend Yield: {financials['metric'].get('dividendYieldIndicatedAnnual', 'N/A')}%

    Peer Companies: {', '.join(peers[:5])}

    Recent News and Sentiment Analysis:
    """

    for item in data["sentiments"]:
        report += f"""
    Title: {item['title']}
    URL: {item['url']}
    Sentiment Analysis: {item['sentiment']} (Confidence: {item['confidence']:.2f})

    """
    # Content Preview: {item['content'][:500]}...
    return report


class Tools:
    class UserValves(BaseModel):
        FINNHUB_API_KEY: str = ""
        pass

    def __init__(self):
        pass

    async def compile_stock_report(self, __user__, ticker: str) -> str:
        """
        Perform a comprehensive stock analysis and compile a detailed report for a given ticker using Finnhub's API.

        This function gathers various data points including:
        - Basic company information (industry, market cap, etc.)
        - Current trading information (price, daily change, etc.)
        - Key financial metrics (P/E ratio, EPS, ROE, etc.)
        - List of peer companies
        - Recent news articles with sentiment analysis using FinBERT

        The gathered data is then compiled into a structured, easy-to-read report.

        :param ticker: The stock ticker symbol (e.g., "AAPL" for Apple Inc.).
        :return: A comprehensive analysis report of the stock as a formatted string.
        """
        self.client = finnhub.Client(api_key=__user__["valves"].FINNHUB_API_KEY)
        data = await _async_gather_stock_data(self.client, ticker)
        report = _compile_report(data)
        return report
