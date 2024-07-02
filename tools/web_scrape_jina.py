"""
title: Web Scrape
author: Pyotr Growpotkin
version: 0.0.2
github: https://github.com/christ-offer/open-webui-tools
license: MIT
description: A simple web scraping tool that extracts text content using Jina Reader.
"""

import requests


class Tools:
    def __init__(self):
        pass

    def web_scrape(self, url: str) -> str:
        """
        Scrape and process a web page using r.jina.ai

        :param url: The URL of the web page to scrape.
        :return: The scraped and processed content without the Links/Buttons section, or an error message.
        """
        jina_url = f"https://r.jina.ai/{url}"

        headers = {
            "X-No-Cache": "true",
            "X-With-Images-Summary": "true",
            "X-With-Links-Summary": "true",
        }

        try:
            response = requests.get(jina_url, headers=headers)
            response.raise_for_status()

            # Extract content and remove Links/Buttons section as its too many tokens
            content = response.text
            links_section_start = content.rfind("Images:")
            if links_section_start != -1:
                content = content[:links_section_start].strip()

            return content

        except requests.RequestException as e:
            return f"Error scraping web page: {str(e)}"
