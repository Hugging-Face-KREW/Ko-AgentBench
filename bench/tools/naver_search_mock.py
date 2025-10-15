"""Mock Naver Search API returning deterministic, realistic-looking data.

This refactored module exposes a single API class with per-feature methods,
so each method can be wrapped as a tool at runtime (via MethodToolWrapper).
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List
from urllib.parse import quote_plus

try:
    # Optional base API for consistency; not strictly required by wrappers
    from .base_api import BaseAPI  # type: ignore
except Exception:  # pragma: no cover - fallback for standalone usage
    class BaseAPI:  # type: ignore
        def __init__(self, name: str, description: str = "") -> None:
            self.name = name
            self.description = description


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


class NaverSearchMockAPI(BaseAPI):
    """Single mock API exposing web/blog/news search methods."""

    def __init__(self) -> None:
        super().__init__(
            name="naver_search_mock_api",
            description="네이버 검색 모의(Mock) API: 네트워크 호출 없음"
        )

    def WebSearch_naver(self, query: str, display: int = 10, start: int = 1) -> Dict[str, Any]:
        params = _CommonParams.from_kwargs(query=query, display=display, start=start)
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

    def BlogSearch_naver(self, query: str, display: int = 10, start: int = 1) -> Dict[str, Any]:
        params = _CommonParams.from_kwargs(query=query, display=display, start=start)
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

            days_ago = rng.randint(0, 30)
            post_dt = (now - timedelta(days=days_ago)).astimezone(KST)
            postdate = post_dt.strftime("%Y%m%d")

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

    def NewsSearch_naver(self, query: str, display: int = 10, start: int = 1, sort: str = "sim") -> Dict[str, Any]:
        params = _CommonParams.from_kwargs(query=query, display=display, start=start)
        rng = _deterministic_random(["news", params.query, str(params.start), str(params.display), sort])

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

            oid = f"{rng.randint(100, 999)}"
            aid = f"{rng.randint(1, 99999999):010d}"
            naver_link = f"https://n.news.naver.com/article/{oid}/{aid}"

            # Adjust date sorting based on sort parameter
            if sort == "date":
                # More recent dates for date sort
                days_ago = rng.randint(0, 3)
                hours_offset = rng.randint(0, 23)
            else:
                # Wider date range for relevance sort
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

    # Alias for dataset compatibility
    def web_search_naver(self, query: str, display: int = 10, start: int = 1) -> Dict[str, Any]:
        """Alias for WebSearch_naver to match dataset naming."""
        return self.WebSearch_naver(query=query, display=display, start=start)


__all__ = [
    "NaverSearchMockAPI",
]


