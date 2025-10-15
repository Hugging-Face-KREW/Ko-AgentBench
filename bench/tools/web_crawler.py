import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin


def get_full_text_from_url(url: str) -> str | None:
    """
    범용 웹페이지 크롤러.
    모든 페이지에 공통적으로 존재하는 비-본문 태그(헤더, 푸터, 사이드바 등)를
    먼저 제거한 후, 남은 영역에서 핵심 텍스트를 추출합니다.

    Args:
        url (str): 크롤링할 웹 페이지의 URL

    Returns:
        str | None: 추출 및 정제된 텍스트. 실패 시 None.
    """
    try:
        # 1. 확장자로 파일 링크를 사전 필터링
        file_extensions = (
            ".pdf",
            ".hwp",
            ".zip",
            ".rar",
            ".7z",
            ".doc",
            ".docx",
            ".xls",
            ".xlsx",
            ".ppt",
            ".pptx",
            ".jpg",
            ".jpeg",
            ".png",
            ".gif",
            ".mp4",
            ".avi",
        )
        if url.lower().split("?")[0].endswith(file_extensions):
            print(f"  -> Skipped: URL이 파일 확장자({url.split('.')[-1]})를 가집니다.")
            return None

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }

        # 2. HEAD 요청으로 Content-Type 확인 (전체 다운로드 방지)
        # allow_redirects=True 옵션으로 리다이렉션되는 최종 URL의 타입을 확인
        head_response = requests.head(
            url, headers=headers, timeout=5, allow_redirects=True
        )
        head_response.raise_for_status()  # HTTP 에러 체크

        content_type = head_response.headers.get("Content-Type", "")
        if "text/html" not in content_type:
            print(f"  -> Skipped: HTML이 아닌 콘텐츠 타입({content_type})입니다.")
            return None

        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")

        if "blog.naver.com" in url:
            main_frame = soup.find("iframe", id="mainFrame")
            if main_frame and main_frame.get("src"):
                frame_url = urljoin(url, main_frame["src"])
                response = requests.get(frame_url, headers=headers, timeout=10)
                response.raise_for_status()
                soup = BeautifulSoup(response.content, "html.parser")

        # 1. 범용적인 비-본문 태그를 먼저 모두 제거
        for tag in soup(
            ["script", "style", "header", "footer", "nav", "aside", "form", "noscript"]
        ):
            tag.decompose()

        # 2. 태그가 제거된 body 영역에서 텍스트를 추출
        if soup.body:
            body_text = soup.body.get_text(separator="\n", strip=True)
        else:  # body 태그가 없는 예외적인 경우 대비
            body_text = soup.get_text(separator="\n", strip=True)

        # 3. 추출된 텍스트를 후처리하여 최종 본문 생성
        # - 줄 단위로 분리하고, 빈 줄을 제거.
        # - 너무 짧은 줄(메뉴, 버튼 텍스트 등)을 제거하는 간단한 휴리스틱을 적용
        lines = [line for line in body_text.splitlines() if line.strip()]

        # 단어 3개 이하로 구성된 짧은 줄은 메뉴나 UI 요소일 가능성이 높으므로 제거
        content_lines = [line for line in lines if len(line.split()) > 3]

        # 만약 위 필터링으로 모든 내용이 사라졌다면, 원본 줄을 사용 (안전장치)
        if not content_lines:
            content_lines = lines

        full_text = " ".join(content_lines)

        return full_text if full_text else None

    except requests.exceptions.RequestException as e:
        print(f"URL 요청 오류 {url}: {e}")
        return None
    except Exception as e:
        print(f"처리 중 예외 발생 {url}: {e}")
        return None


def postprocess_with_crawling(search_result: dict, max_length=500) -> dict:
    """API 검색 결과(Naver/Daum)를 직접 입력받아 'description' 또는 'contents'를
    URL 크롤링을 통해 얻은 전체 텍스트로 대체합니다.
    """
    # Naver 결과와 Daum 결과를 구분하여 처리할 키와 필드명 설정
    items, key_for_url, field_to_update = (None, None, None)

    if "items" in search_result:  # Naver Web Search
        items = search_result.get("items", [])
        key_for_url = "link"
        field_to_update = "description"
    elif "documents" in search_result:  # Daum Web Search
        items = search_result.get("documents", [])
        key_for_url = "url"
        field_to_update = "contents"
    else:
        print("입력 형식이 Naver 또는 Daum 검색 결과와 일치하지 않습니다.")
        return search_result

    if not items:
        return search_result

    print(f"Processing {len(items)} items...")
    for item in items:
        url = item.get(key_for_url)
        if not url:
            continue

        print(f"Crawling: {url}")
        full_text = get_full_text_from_url(url)

        if full_text:
            item[field_to_update] = full_text[:max_length]
            print(f"  -> Success: {max_length}자 텍스트로 대체됨")
        else:
            print(f"  -> Failed: 텍스트를 추출하지 못함")

    return search_result
