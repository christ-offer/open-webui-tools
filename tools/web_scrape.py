"""
title: Web Scrape
author: Pyotr Growpotkin
version: 0.0.2
github: https://github.com/christ-offer/open-webui-tools
license: MIT
description: A simple web scraping tool that extracts text content using BeautifulSoup.
"""

import requests
import aiohttp
from bs4 import BeautifulSoup
from typing import Optional


class Tools:
    def __init__(self):
        pass

    async def web_scrape(self, url: str) -> Optional[str]:
        """
        Scrape a web page and extract its text content.
        :param url: The URL of the web page to scrape.
        :return: The text content of the web page, or None if the scraping fails.
        """
        session = aiohttp.ClientSession()
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }

        try:
            async with session.get(url, headers=headers) as response:
                response.raise_for_status()
                content = await response.text()

            soup = BeautifulSoup(content, "html.parser")

            for element in soup(
                [
                    "script",
                    "style",
                    "nav",
                    "footer",
                    "header",
                    "aside",
                    "button",
                    "img",
                ]
            ):
                element.decompose()

            text_elements = soup.find_all(
                [
                    "p",
                    "h1",
                    "h2",
                    "h3",
                    "h4",
                    "h5",
                    "h6",
                    "li",
                    "span",
                    "strong",
                    "em",
                    "small",
                ]
            )
            text = " ".join(element.get_text().strip() for element in text_elements)
            text = " ".join(text.split())

            return text

        except Exception as e:
            print(f"Error scraping web page: {str(e)}")
            return None
