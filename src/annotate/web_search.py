# src/web_search.py

import urllib.request
import urllib.parse
import json
import re
import os
import openai
os.environ.setdefault("OPENAI_API_KEY", open("./data/openai_key.txt").read().strip())


_LANG_SITES = {
    "Python":     "site:docs.python.org OR site:numpy.org OR site:pandas.pydata.org",
    "JavaScript": "site:developer.mozilla.org OR site:nodejs.org",
    "TypeScript": "site:typescriptlang.org OR site:developer.mozilla.org",
    "Rust":       "site:doc.rust-lang.org",
    "Go":         "site:pkg.go.dev",
    "Java":       "site:docs.oracle.com",
    "C#":         "site:learn.microsoft.com",
    "Kotlin":     "site:kotlinlang.org",
    "Swift":      "site:developer.apple.com",
    "PHP":        "site:php.net",
    "Ruby":       "site:ruby-doc.org",
    "Scala":      "site:scala-lang.org",
    "R":          "site:rdocumentation.org",
    "Lua":        "site:lua.org",
}


def search_docs(query: str, language: str = "Python", max_results: int = 3) -> str:
    site_filter = _LANG_SITES.get(language, "")
    scoped_query = f"{query} {site_filter}".strip()

    client = openai.OpenAI()
    resp = client.responses.create(
        model="gpt-4o-mini",
        tools=[{"type": "web_search_preview"}],
        input=scoped_query,
    )
    for block in resp.output:
        if block.type == "message":
            return block.content[0].text
    return "[openai_search] No results"