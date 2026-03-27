"""
Layer 0 — URL Fetcher
Detects the platform from a URL, scrapes the post, and returns a
structured PostData object containing the caption text + downloaded
media path(s) — ready to be handed to Layers 1-5.

Supported platforms:
    Instagram    → Instaloader (no API key needed for public posts)
    Facebook     → facebook-scraper (public posts only)

No env var needed for Instagram or Facebook public posts.
"""

import os
import re
import json
import tempfile
import urllib.parse
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import requests

# Load .env so FACEBOOK_COOKIES_FILE etc. are available when running standalone
try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))
except ImportError:
    pass


# ── Data model ─────────────────────────────────────────────────────────────────

class Platform(str, Enum):
    INSTAGRAM = "Instagram"
    FACEBOOK  = "Facebook"
    UNKNOWN   = "Unknown"


@dataclass
class PostData:
    platform:    Platform
    url:         str
    caption:     str                   # post text / caption
    author:      str = ""              # username or page name
    timestamp:   str = ""             # ISO string or human date
    image_path:  Optional[str] = None  # local temp path to downloaded image
    video_path:  Optional[str] = None  # local temp path to downloaded video
    thumbnail_url: Optional[str] = None  # original CDN URL for display
    error:       Optional[str] = None  # set if scraping failed


# ── Platform detection ──────────────────────────────────────────────────────────

_INSTAGRAM_RE   = re.compile(r"(https?://)?(www\.)?instagram\.com/(p|reel)/", re.I)
_FACEBOOK_RE    = re.compile(r"(https?://)?(www\.|m\.)?facebook\.com/", re.I)
_FB_PHOTO_RE    = re.compile(r"facebook\.com/photo/?\?.*fbid=(\d+)", re.I)
_FB_POST_ID_RE  = re.compile(r"facebook\.com/[^/]+/posts/(\d+)", re.I)


def detect_platform(url: str) -> Platform:
    if _INSTAGRAM_RE.search(url):
        return Platform.INSTAGRAM
    if _FACEBOOK_RE.search(url):
        return Platform.FACEBOOK
    return Platform.UNKNOWN


# ── Media download helper ───────────────────────────────────────────────────────

def _download_media(url: str, suffix: str) -> Optional[str]:
    """Download a media file to a temp path. Returns path or None on failure."""
    try:
        headers = {"User-Agent": "Mozilla/5.0 (TruthScan/1.0)"}
        r = requests.get(url, headers=headers, timeout=20, stream=True)
        r.raise_for_status()
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        for chunk in r.iter_content(8192):
            tmp.write(chunk)
        tmp.close()
        return tmp.name
    except Exception as e:
        print(f"[download] Failed {url}: {e}")
        return None


# ── Instagram scraper ───────────────────────────────────────────────────────────

def _extract_shortcode(url: str) -> Optional[str]:
    """Pull the shortcode from an Instagram post URL."""
    m = re.search(r"/p/([A-Za-z0-9_\-]+)", url)
    return m.group(1) if m else None


def _scrape_instagram(url: str) -> PostData:
    """
    Uses the `instaloader` library to scrape public Instagram posts.
    No API key needed — works on any public post.
    Rate-limited by Instagram; add a small sleep between calls in batch mode.
    """
    try:
        import instaloader
    except ImportError:
        return PostData(
            platform=Platform.INSTAGRAM, url=url, caption="",
            error="instaloader not installed. Run: pip install instaloader",
        )

    shortcode = _extract_shortcode(url)
    if not shortcode:
        return PostData(platform=Platform.INSTAGRAM, url=url, caption="",
                        error="Could not extract post shortcode from URL.")

    try:
        L    = instaloader.Instaloader(download_pictures=False, download_videos=False,
                                        download_video_thumbnails=False, quiet=True)
        post = instaloader.Post.from_shortcode(L.context, shortcode)

        caption    = post.caption or ""
        author     = "@" + post.owner_username
        timestamp  = post.date_utc.isoformat()
        thumb_url  = post.url  # CDN URL of image / video thumbnail

        # Download media
        image_path = None
        video_path = None
        if post.is_video:
            video_path = _download_media(post.video_url, ".mp4")
        else:
            image_path = _download_media(post.url, ".jpg")

        return PostData(
            platform=Platform.INSTAGRAM, url=url,
            caption=caption, author=author, timestamp=timestamp,
            image_path=image_path, video_path=video_path,
            thumbnail_url=thumb_url,
        )

    except Exception as e:
        return PostData(platform=Platform.INSTAGRAM, url=url, caption="",
                        error=f"Instagram scrape failed: {e}\n"
                              "Note: Private accounts and some rate-limited profiles cannot be scraped.")


# ── Cookie format converter ─────────────────────────────────────────────────────

def _editthiscookie_to_netscape(json_path: str) -> str:
    """
    Convert an EditThisCookie JSON export to a Netscape cookie file
    that facebook-scraper (and requests) can read.
    Returns the path to a temporary Netscape-format file.
    """
    with open(json_path) as f:
        cookies = json.load(f)

    lines = ["# Netscape HTTP Cookie File", ""]
    for c in cookies:
        domain      = c.get("domain", "")
        # Netscape format needs leading dot for domain-wide cookies
        if domain and not domain.startswith("."):
            domain = "." + domain
        flag        = "TRUE" if domain.startswith(".") else "FALSE"
        path        = c.get("path", "/")
        secure      = "TRUE" if c.get("secure", False) else "FALSE"
        expiry      = str(int(c.get("expirationDate", 0)))
        name        = c.get("name", "")
        value       = c.get("value", "")
        lines.append(f"{domain}\t{flag}\t{path}\t{secure}\t{expiry}\t{name}\t{value}")

    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, prefix="fb_cookies_"
    )
    tmp.write("\n".join(lines))
    tmp.close()
    return tmp.name


# ── Open Graph / generic fallback scraper ──────────────────────────────────────

def _scrape_og_metadata(url: str, cookies_json_path: str = "") -> dict:
    """
    Fetch Open Graph meta tags from any public page.
    If cookies_json_path is provided (EditThisCookie JSON), uses authenticated session.
    Returns a dict with keys: title, description, image (URL).
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Sec-Fetch-User": "?1",
        "Upgrade-Insecure-Requests": "1",
        "Cache-Control": "max-age=0",
        "Connection": "keep-alive",
    }

    session = requests.Session()

    # Load cookies from EditThisCookie JSON if provided
    if cookies_json_path and os.path.isfile(cookies_json_path):
        try:
            with open(cookies_json_path) as f:
                cookie_list = json.load(f)
            for c in cookie_list:
                session.cookies.set(
                    c["name"], c["value"],
                    domain=c.get("domain", ".facebook.com"),
                    path=c.get("path", "/"),
                )
            print(f"[OG] Loaded {len(cookie_list)} cookies for authenticated request")
        except Exception as e:
            print(f"[OG] Cookie load failed: {e}")

    try:
        r = session.get(url, headers=headers, timeout=15, allow_redirects=True)
        r.raise_for_status()
    except Exception as e:
        return {"error": str(e)}

    # Simple regex-based OG parse — avoids bs4 dependency assumption
    og: dict = {}
    for prop, content in re.findall(
        r'<meta[^>]+property=["\']og:(\w+)["\'][^>]+content=["\']([^"\']*)["\']',
        r.text, re.I
    ):
        og[prop] = content
    # also try reversed attribute order
    for content, prop in re.findall(
        r'<meta[^>]+content=["\']([^"\']*)["\'][^>]+property=["\']og:(\w+)["\']',
        r.text, re.I
    ):
        og.setdefault(prop, content)

    # title from <title> tag as last resort
    if "title" not in og:
        m = re.search(r"<title>([^<]+)</title>", r.text, re.I)
        if m:
            og["title"] = m.group(1).strip()

    return og


# ── Playwright-based Facebook scraper ──────────────────────────────────────────

def _scrape_facebook_playwright(url: str, cookies_json_path: str = "") -> dict:
    """
    Use a real Chromium browser (headless) via Playwright with injected cookies.
    Returns a dict with: description, image, title — same shape as _scrape_og_metadata.
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("[playwright] not installed — skipping")
        return {}

    cookies_data = []
    if cookies_json_path and os.path.isfile(cookies_json_path):
        with open(cookies_json_path) as f:
            raw = json.load(f)
        for c in raw:
            entry = {
                "name":   c["name"],
                "value":  c["value"],
                "domain": c.get("domain", ".facebook.com"),
                "path":   c.get("path", "/"),
                "secure": c.get("secure", False),
            }
            if c.get("expirationDate"):
                entry["expires"] = int(c["expirationDate"])
            # Normalize sameSite to values Playwright accepts
            same_site = str(c.get("sameSite") or "").lower()
            if same_site in ("strict",):
                entry["sameSite"] = "Strict"
            elif same_site in ("lax",):
                entry["sameSite"] = "Lax"
            else:
                entry["sameSite"] = "None"  # default: no_restriction / unspecified
            cookies_data.append(entry)

    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(
                headless=True,
                args=[
                    "--disable-blink-features=AutomationControlled",
                    "--no-sandbox",
                    "--disable-dev-shm-usage",
                ],
            )
            ctx = browser.new_context(
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/122.0.0.0 Safari/537.36"
                ),
                locale="en-US",
                viewport={"width": 1280, "height": 900},
            )
            # Hide webdriver flag
            ctx.add_init_script(
                "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
            )
            if cookies_data:
                ctx.add_cookies(cookies_data)
                print(f"[playwright] Injected {len(cookies_data)} cookies")

            # Intercept Facebook GraphQL responses for photo data
            graphql_data: list = []
            def _capture_response(response):
                if "graphql" in response.url and response.status == 200:
                    try:
                        body = response.text()
                        if "photo" in body.lower() and len(body) > 200:
                            graphql_data.append(body)
                    except Exception:
                        pass
            ctx.on("response", _capture_response)

            page = ctx.new_page()
            page.goto(url, wait_until="domcontentloaded", timeout=30000)

            # Wait for photo-specific content
            fbid_match = re.search(r"fbid=(\d+)", url)
            fbid = fbid_match.group(1) if fbid_match else ""
            try:
                if fbid:
                    page.wait_for_selector(
                        f"img[src*='{fbid}'], img[data-visualcompletion='media-vc-image']",
                        timeout=8000
                    )
                else:
                    page.wait_for_timeout(6000)
            except Exception:
                page.wait_for_timeout(5000)

            # Check final URL
            final_url = page.url
            print(f"[playwright] Final URL: {final_url}")

            # ── Try GraphQL intercept data first ──────────────────────────────
            og: dict = {}
            if graphql_data:
                for chunk in graphql_data:
                    # Look for message/caption in GraphQL JSON blobs
                    m = re.search(r'"message"\s*:\s*\{"text"\s*:\s*"([^"]{20,})"', chunk)
                    if m:
                        og["description"] = m.group(1).replace("\\n", "\n")
                        break
                    m = re.search(r'"accessibility_caption"\s*:\s*"([^"]{20,})"', chunk)
                    if m and not og.get("description"):
                        og["description"] = m.group(1)

            # ── Extract OG tags from DOM + raw HTML source ────────────────────
            for tag in page.query_selector_all("meta[property^='og:']"):
                prop    = (tag.get_attribute("property") or "").replace("og:", "")
                content = tag.get_attribute("content") or ""
                if prop and content:
                    og.setdefault(prop, content)

            html = page.content()
            for prop, content in re.findall(
                r'property=["\']og:(\w+)["\'][^>]+content=["\']([^"\']{5,})["\']',
                html, re.I
            ):
                og.setdefault(prop, content)
            for content, prop in re.findall(
                r'content=["\']([^"\']{5,})["\'][^>]+property=["\']og:(\w+)["\']',
                html, re.I
            ):
                og.setdefault(prop, content)

            # ── Caption extraction ───────────────────────────────────────────
            # OG description is the most reliable — Facebook server-renders it
            # correctly. Only fall back to DOM scraping if OG is missing.
            if not og.get("description"):
                # Target the photo caption area: it sits inside the photo viewer
                # pagelet, in a div[dir="auto"] that is NOT inside a sidebar/feed.
                # We try progressively broader selectors.
                caption_selectors = [
                    # Photo viewer caption (most specific)
                    "div[data-pagelet='MediaViewerPhoto'] div[dir='auto']",
                    "div[data-pagelet='photo'] div[dir='auto']",
                    # Generic story attachment text
                    "[data-testid='post-content'] div[dir='auto']",
                    # Grab first substantial div[dir=auto] that actually has text
                ]
                for sel in caption_selectors:
                    els = page.query_selector_all(sel)
                    for el in els:
                        try:
                            text = el.inner_text().strip()
                            if text and len(text) > 20:
                                og["description"] = text
                                break
                        except Exception:
                            pass
                    if og.get("description"):
                        break

            # ── Image extraction ─────────────────────────────────────────────
            # Priority: img with data-visualcompletion="media-vc-image" is the
            # actual post photo. data-imgperflogname='feedImage' is a fallback.
            _AI_ALT_RE = re.compile(
                r'^May be (an? image|a photo|a (close-up|screenshot|cartoon|meme|drawing))',
                re.I
            )
            if not og.get("image"):
                for selector in [
                    "img[data-visualcompletion='media-vc-image']",
                    "img[data-imgperflogname='feedImage']",
                ]:
                    img_el = page.query_selector(selector)
                    if img_el:
                        src = img_el.get_attribute("src") or ""
                        if src.startswith("http"):
                            og["image"] = src
                            # alt text is Facebook's AI caption (e.g. "May be an image of..."),
                            # NOT the post caption — only use it if nothing else is available
                            # and explicitly mark it as image alt text, NOT the description
                            alt = (img_el.get_attribute("alt") or "").strip()
                            if alt and not og.get("image_alt"):
                                og["image_alt"] = alt
                            break

            # NEVER use Facebook's AI alt text ("May be an image of...") as the
            # post description — it describes the image, not the post claim.
            if og.get("description") and _AI_ALT_RE.match(og["description"]):
                og.pop("description", None)

            # ── Video extraction ─────────────────────────────────────────────
            video_el = page.query_selector("video source, video[src]")
            if video_el:
                src = video_el.get_attribute("src") or ""
                if src.startswith("http"):
                    og["video"] = src

            # ── Author from page title ───────────────────────────────────────
            if not og.get("author"):
                title_text = og.get("title") or page.title()
                clean = re.sub(r"^\(\d+\)\s*", "", title_text)
                clean = re.sub(r"\s*\|\s*Facebook$", "", clean, flags=re.I).strip()
                if clean and clean.lower() not in ("facebook", ""):
                    og["author"] = clean

            # Fallback: grab title
            if "title" not in og:
                og["title"] = page.title()

            browser.close()
            print(f"[playwright] OG data: {list(og.keys())}, desc={og.get('description','')[:80]}")
            return og

    except Exception as e:
        print(f"[playwright] failed: {e}")
        return {}


# ── Facebook scraper ────────────────────────────────────────────────────────────

def _scrape_facebook(url: str) -> PostData:
    """
    Uses the `facebook-scraper` library to fetch public Facebook posts.
    Also supports /photo/?fbid= URLs via facebook-scraper and OG fallback.

    To pass cookies (for login-walled content):
        Set FACEBOOK_COOKIES_FILE=/path/to/cookies.json in your .env
        Export cookies with a browser extension like "EditThisCookie".
    """
    try:
        from facebook_scraper import get_posts, set_cookies
    except ImportError:
        return PostData(
            platform=Platform.FACEBOOK, url=url, caption="",
            error="facebook-scraper not installed. Run: pip install facebook-scraper",
        )

    cookies_file = os.environ.get("FACEBOOK_COOKIES_FILE", "")
    if cookies_file and os.path.isfile(cookies_file):
        try:
            # EditThisCookie exports JSON; facebook-scraper needs Netscape format
            if cookies_file.lower().endswith(".json"):
                cookies_file = _editthiscookie_to_netscape(cookies_file)
                print(f"[facebook] Converted cookies to Netscape format: {cookies_file}")
            set_cookies(cookies_file)
            print("[facebook] Cookies loaded successfully")
        except Exception as e:
            print(f"[facebook] Cookie load failed: {e}")

    # ── Attempt 1: facebook-scraper (works for /posts/ and sometimes /photo/) ──
    try:
        import signal

        def _timeout_handler(signum, frame):
            raise TimeoutError("facebook-scraper timed out")

        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(20)  # 20-second hard timeout

        try:
            posts = list(get_posts(post_urls=[url], options={"posts_per_page": 1}))
        finally:
            signal.alarm(0)  # cancel alarm

        if posts:
            post      = posts[0]
            caption   = post.get("text") or post.get("post_text") or ""
            author    = post.get("username") or post.get("user_id") or ""
            timestamp = str(post.get("time") or "")
            image_url = post.get("image") or ""
            video_url = post.get("video") or ""

            # Only return if we actually got useful data
            if caption or image_url or video_url:
                image_path = _download_media(image_url, ".jpg") if image_url else None
                video_path = _download_media(video_url, ".mp4") if video_url else None

                return PostData(
                    platform=Platform.FACEBOOK, url=url,
                    caption=caption, author=author, timestamp=timestamp,
                    image_path=image_path, video_path=video_path,
                    thumbnail_url=image_url or None,
                )

        raise ValueError("facebook-scraper returned no usable content")

    except Exception as e:
        print(f"[facebook] facebook-scraper failed ({e}), trying OG fallback…")

    # ── Attempt 2: Open Graph metadata fallback ───────────────────────────────
    og = _scrape_og_metadata(url, cookies_json_path=os.environ.get("FACEBOOK_COOKIES_FILE", ""))

    # ── Attempt 3: Playwright (real browser with cookies) ─────────────────────
    if not og or og.get("title", "").strip().lower() in ("facebook", "", "error facebook"):
        print("[facebook] OG fallback got login wall, trying Playwright…")
        og = _scrape_facebook_playwright(url, os.environ.get("FACEBOOK_COOKIES_FILE", "")) or og
    if og.get("error"):
        return PostData(
            platform=Platform.FACEBOOK, url=url, caption="",
            error=(
                f"Facebook scrape failed: {og['error']}\n"
                "Facebook restricts scraping. For private or login-required posts, "
                "set FACEBOOK_COOKIES_FILE in your .env (see README)."
            ),
        )

    caption   = og.get("description") or ""
    image_url = og.get("image") or ""
    image_alt = og.get("image_alt") or ""
    video_url = og.get("video") or ""
    author    = og.get("author") or og.get("site_name") or ""

    # If we have no caption text but we have image alt text (Facebook's OCR),
    # use the alt text as a secondary caption for analysis.
    if not caption and image_alt:
        caption = f"[Image text detected by Facebook OCR]\n{image_alt}"

    image_path = _download_media(image_url, ".jpg") if image_url and not video_url else None
    video_path = _download_media(video_url, ".mp4") if video_url else None

    if not caption and not image_path and not video_path:
        return PostData(
            platform=Platform.FACEBOOK, url=url, caption="",
            error=(
                "Facebook returned no readable content.\n"
                "This post may be private or require a login.\n"
                "Set FACEBOOK_COOKIES_FILE in your .env with exported Facebook cookies."
            ),
        )

    return PostData(
        platform=Platform.FACEBOOK, url=url,
        caption=caption, author=author, timestamp="",
        image_path=image_path, video_path=video_path,
        thumbnail_url=image_url or None,
        error="Limited data via Open Graph (post may be partially private)." if not caption else None,
    )


# ── Main entry point ────────────────────────────────────────────────────────────

def fetch_post(url: str) -> PostData:
    """
    Detects platform, scrapes the post, downloads media.
    Always returns a PostData — check .error field for failures.
    """
    url = url.strip()
    if not url.startswith("http"):
        url = "https://" + url

    platform = detect_platform(url)

    if platform == Platform.INSTAGRAM:
        return _scrape_instagram(url)
    elif platform == Platform.FACEBOOK:
        return _scrape_facebook(url)
    else:
        return PostData(
            platform=Platform.UNKNOWN, url=url, caption="",
            error=(
                "Unsupported URL. TruthScan currently supports:\n"
                "  • Instagram    (instagram.com/p/… or instagram.com/reel/…)\n"
                "  • Facebook     (facebook.com/…/posts/…)"
            ),
        )


# ── Smoke test ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    test_url = sys.argv[1] if len(sys.argv) > 1 else "https://www.instagram.com/p/ABC123/"
    print(f"Testing URL: {test_url}")
    result = fetch_post(test_url)
    print(f"Platform   : {result.platform}")
    print(f"Author     : {result.author}")
    print(f"Caption    : {result.caption[:120]}…" if len(result.caption) > 120 else f"Caption: {result.caption}")
    print(f"Image path : {result.image_path}")
    print(f"Video path : {result.video_path}")
    if result.error:
        print(f"⚠️  Error   : {result.error}")
