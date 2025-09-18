"""Mock Naver Search tools returning deterministic, realistic-looking data.

Implements three separate tools (option A):
- naver_web_search
- naver_blog_search
- naver_news_search

Each tool validates input, clamps ranges, and returns results that mimic
Naver Open API response structure without performing any network calls.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List
from urllib.parse import quote_plus

try:
    # Prefer the project's BaseTool if available
    from .base_tool import BaseTool  # type: ignore
except Exception:
    # Minimal fallback to allow standalone usage if BaseTool is missing
    class BaseTool:  # type: ignore
        name: str
        description: str

        def __init__(self, name: str, description: str = "") -> None:
            self.name = name
            self.description = description

        def get_schema(self) -> Dict[str, Any]:
            return {
                "name": self.name,
                "description": self.description,
                "parameters": getattr(self, "_get_parameters_schema")(),
            }


KST = timezone(timedelta(hours=9))


def _clamp(value: int, min_value: int, max_value: int) -> int:
    return max(min_value, min(value, max_value))


def _format_rfc2822(dt: datetime) -> str:
    # Example: Thu, 18 Sep 2025 12:34:56 +0900
    return dt.strftime("%a, %d %b %Y %H:%M:%S +0900")


def _seed_from(*parts: str) -> int:
    hasher = hashlib.md5()
    for part in parts:
        hasher.update(part.encode("utf-8"))
    # Use lower 8 bytes for a stable seed range
    return int(hasher.hexdigest()[:16], 16)


def _deterministic_random(parts: List[str]) -> random.Random:
    return random.Random(_seed_from(*parts))


def _generate_total(rng: random.Random) -> int:
    # Skew towards mid-range totals
    base = rng.randint(100, 5000)
    return base


def _pick_one(rng: random.Random, items: List[str]) -> str:
    return items[rng.randrange(len(items))]


def _gen_title_variants(query: str) -> List[str]:
    return [
        f"{query} — 핵심 개념과 가이드",
        f"{query} 입문자를 위한 요약",
        f"{query} 활용 사례와 베스트 프랙티스",
        f"{query} 비교와 선택 기준",
        f"{query} 문제 해결 체크리스트",
    ]


def _gen_summary_variants(query: str) -> List[str]:
    return [
        f"{query}의 개념과 활용 포인트를 간단히 정리했습니다.",
        f"{query} 관련 핵심 특징과 주의사항을 요약합니다.",
        f"{query}를 실제 상황에서 적용할 때 유용한 팁을 제공합니다.",
        f"{query}를 이해하는 데 필요한 기본 맥락과 참고 자료를 소개합니다.",
    ]


@dataclass
class _CommonParams:
    query: str
    display: int
    start: int

    @classmethod
    def from_kwargs(cls, **kwargs: Any) -> "_CommonParams":
        query = str(kwargs.get("query", "")).strip()
        display = kwargs.get("display", 10)
        start = kwargs.get("start", 1)

        if not query:
            raise ValueError("'query'는 비어 있을 수 없습니다.")

        try:
            display_int = int(display)
            start_int = int(start)
        except Exception:
            raise ValueError("'display'와 'start'는 정수여야 합니다.")

        # Clamp ranges to match Naver API constraints
        display_int = _clamp(display_int, 1, 100)
        start_int = _clamp(start_int, 1, 1000)

        return cls(query=query, display=display_int, start=start_int)


class NaverWebSearchMock(BaseTool):
    """Mock tool for Naver Web Search API."""

    def __init__(self) -> None:
        super().__init__(
            name="naver_web_search",
            description="네이버 웹 검색 모의(Mock) 도구: 네트워크 호출 없이 유사 결과를 반환"
        )

    def execute(self, **kwargs: Any) -> Dict[str, Any]:
        params = _CommonParams.from_kwargs(**kwargs)
        rng = _deterministic_random(["web", params.query, str(params.start), str(params.display)])

        now = datetime.now(KST)
        last_build_date = _format_rfc2822(now)
        total = _generate_total(rng)
        encoded_query = quote_plus(params.query)

        domains = [
            "https://en.wikipedia.org/wiki/",
            "https://namu.wiki/w/",
            "https://github.com/search?q=",
            "https://medium.com/search?q=",
            "https://www.youtube.com/results?search_query=",
            "https://ko.wikipedia.org/wiki/",
        ]

        items: List[Dict[str, Any]] = []
        for i in range(params.display):
            title = _pick_one(rng, _gen_title_variants(params.query))
            summary = _pick_one(rng, _gen_summary_variants(params.query))
            base = _pick_one(rng, domains)
            link = f"{base}{encoded_query}&i={params.start + i}"
            items.append({
                "title": title,
                "link": link,
                "description": summary
            })

        return {
            "lastBuildDate": last_build_date,
            "total": total,
            "start": params.start,
            "display": params.display,
            "items": items
        }

    def validate_input(self, **kwargs: Any) -> bool:
        try:
            _CommonParams.from_kwargs(**kwargs)
            return True
        except Exception:
            return False

    def _get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "검색어"},
                "display": {"type": "integer", "minimum": 1, "maximum": 100, "description": "표시 결과 수"},
                "start": {"type": "integer", "minimum": 1, "maximum": 1000, "description": "시작 위치"}
            },
            "required": ["query"]
        }


class NaverBlogSearchMock(BaseTool):
    """Mock tool for Naver Blog Search API."""

    def __init__(self) -> None:
        super().__init__(
            name="naver_blog_search",
            description="네이버 블로그 검색 모의(Mock) 도구: 네트워크 호출 없이 유사 결과를 반환"
        )

    def execute(self, **kwargs: Any) -> Dict[str, Any]:
        params = _CommonParams.from_kwargs(**kwargs)
        rng = _deterministic_random(["blog", params.query, str(params.start), str(params.display)])

        now = datetime.now(KST)
        last_build_date = _format_rfc2822(now)
        total = _generate_total(rng)
        encoded_query = quote_plus(params.query)

        blog_domains = [
            "https://blog.naver.com/",
            "https://tistory.com/",
            "https://velog.io/@",
            "https://medium.com/@",
        ]
        blogger_names = [
            "데브노트", "테크로그", "데이터장인", "코딩하는고양이", "프로그래밍초록", "모킹버드"
        ]

        items: List[Dict[str, Any]] = []
        for i in range(params.display):
            title = _pick_one(rng, _gen_title_variants(params.query))
            summary = _pick_one(rng, _gen_summary_variants(params.query))
            domain = _pick_one(rng, blog_domains)
            bloggername = _pick_one(rng, blogger_names)

            # Simulate recent post date within 30 days
            days_ago = rng.randint(0, 30)
            post_dt = (now - timedelta(days=days_ago)).astimezone(KST)
            postdate = post_dt.strftime("%Y%m%d")

            # Stabilize author handle for velog/medium style
            handle_seed = _seed_from(params.query, str(i), "blogger")
            handle_rng = random.Random(handle_seed)
            handle = f"user{handle_rng.randint(1000, 9999)}"

            if domain.endswith("/@"):
                link = f"{domain}{handle}/{encoded_query}-{params.start + i}"
                bloggerlink = f"{domain}{handle}"
            else:
                link = f"{domain}{encoded_query}-{params.start + i}"
                bloggerlink = domain

            items.append({
                "title": title,
                "link": link,
                "description": summary,
                "bloggername": bloggername,
                "bloggerlink": bloggerlink,
                "postdate": postdate
            })

        return {
            "lastBuildDate": last_build_date,
            "total": total,
            "start": params.start,
            "display": params.display,
            "items": items
        }

    def validate_input(self, **kwargs: Any) -> bool:
        try:
            _CommonParams.from_kwargs(**kwargs)
            return True
        except Exception:
            return False

    def _get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "검색어"},
                "display": {"type": "integer", "minimum": 1, "maximum": 100, "description": "표시 결과 수"},
                "start": {"type": "integer", "minimum": 1, "maximum": 1000, "description": "시작 위치"}
            },
            "required": ["query"]
        }


class NaverNewsSearchMock(BaseTool):
    """Mock tool for Naver News Search API."""

    def __init__(self) -> None:
        super().__init__(
            name="naver_news_search",
            description="네이버 뉴스 검색 모의(Mock) 도구: 네트워크 호출 없이 유사 결과를 반환"
        )

    def execute(self, **kwargs: Any) -> Dict[str, Any]:
        params = _CommonParams.from_kwargs(**kwargs)
        rng = _deterministic_random(["news", params.query, str(params.start), str(params.display)])

        now = datetime.now(KST)
        last_build_date = _format_rfc2822(now)
        total = _generate_total(rng)
        encoded_query = quote_plus(params.query)

        news_domains = [
            "https://news.example.com/",
            "https://www.hani.co.kr/arti/",
            "https://www.chosun.com/site/data/html_dir/",
            "https://www.joongang.co.kr/article/",
            "https://www.khan.co.kr/national/",
        ]

        items: List[Dict[str, Any]] = []
        for i in range(params.display):
            title = _pick_one(rng, _gen_title_variants(params.query))
            summary = _pick_one(rng, _gen_summary_variants(params.query))
            origin = _pick_one(rng, news_domains)

            # Simulate Naver news link style
            oid = f"{rng.randint(100, 999)}"
            aid = f"{rng.randint(1, 99999999):010d}"
            naver_link = f"https://n.news.naver.com/article/{oid}/{aid}"

            # Pub date within the last 14 days
            days_ago = rng.randint(0, 14)
            hours_offset = rng.randint(0, 23)
            pub_dt = (now - timedelta(days=days_ago, hours=hours_offset)).astimezone(KST)
            pub_date = _format_rfc2822(pub_dt)

            link = f"{origin}{encoded_query}-{params.start + i}"
            items.append({
                "originallink": link,
                "link": naver_link,
                "title": title,
                "description": summary,
                "pubDate": pub_date
            })

        return {
            "lastBuildDate": last_build_date,
            "total": total,
            "start": params.start,
            "display": params.display,
            "items": items
        }

    def validate_input(self, **kwargs: Any) -> bool:
        try:
            _CommonParams.from_kwargs(**kwargs)
            return True
        except Exception:
            return False

    def _get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "검색어"},
                "display": {"type": "integer", "minimum": 1, "maximum": 100, "description": "표시 결과 수"},
                "start": {"type": "integer", "minimum": 1, "maximum": 1000, "description": "시작 위치"}
            },
            "required": ["query"]
        }


__all__ = [
    "NaverWebSearchMock",
    "NaverBlogSearchMock",
    "NaverNewsSearchMock",
]


