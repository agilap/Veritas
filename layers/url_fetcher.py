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
import tempfile
import urllib.parse
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import requests


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

_INSTAGRAM_RE = re.compile(r"(https?://)?(www\.)?instagram\.com/(p|reel)/", re.I)
_FACEBOOK_RE  = re.compile(r"(https?://)?(www\.|m\.)?facebook\.com/", re.I)


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


# ── Facebook scraper ────────────────────────────────────────────────────────────

def _scrape_facebook(url: str) -> PostData:
    """
    Uses the `facebook-scraper` library to fetch public Facebook posts.
    Works without login for most public pages; some posts require cookies.

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
            set_cookies(cookies_file)
        except Exception:
            pass

    try:
        # facebook-scraper accepts a post URL directly
        posts = list(get_posts(post_urls=[url], options={"posts_per_page": 1}))
        if not posts:
            raise ValueError("No posts returned — post may be private or login-required.")

        post      = posts[0]
        caption   = post.get("text") or post.get("post_text") or ""
        author    = post.get("username") or post.get("user_id") or ""
        timestamp = str(post.get("time") or "")
        image_url = post.get("image") or ""
        video_url = post.get("video") or ""

        image_path = _download_media(image_url, ".jpg") if image_url else None
        video_path = _download_media(video_url, ".mp4") if video_url else None

        return PostData(
            platform=Platform.FACEBOOK, url=url,
            caption=caption, author=author, timestamp=timestamp,
            image_path=image_path, video_path=video_path,
            thumbnail_url=image_url or None,
        )

    except Exception as e:
        return PostData(
            platform=Platform.FACEBOOK, url=url, caption="",
            error=(
                f"Facebook scrape failed: {e}\n"
                "Facebook heavily restricts scraping. For private or login-required posts, "
                "set FACEBOOK_COOKIES_FILE in your .env (see README)."
            ),
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
