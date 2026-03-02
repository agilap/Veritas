"""
Layer 0 — URL Fetcher
Detects the platform from a URL, scrapes the post, and returns a
structured PostData object containing the caption text + downloaded
media path(s) — ready to be handed to Layers 1-5.

Supported platforms:
    X / Twitter  → Tweepy (Twitter API v2, free Bearer Token)
    Instagram    → Instaloader (no API key needed for public posts)
    Facebook     → facebook-scraper (public posts only)

Environment variables:
    TWITTER_BEARER_TOKEN   — get free at https://developer.x.com

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
    TWITTER   = "X / Twitter"
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

_TWITTER_RE   = re.compile(r"(https?://)?(www\.)?(twitter\.com|x\.com)/", re.I)
_INSTAGRAM_RE = re.compile(r"(https?://)?(www\.)?instagram\.com/p/", re.I)
_FACEBOOK_RE  = re.compile(r"(https?://)?(www\.|m\.)?facebook\.com/", re.I)


def detect_platform(url: str) -> Platform:
    if _TWITTER_RE.search(url):
        return Platform.TWITTER
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


# ── X / Twitter scraper ─────────────────────────────────────────────────────────

def _extract_tweet_id(url: str) -> Optional[str]:
    """Pull the numeric tweet ID from any x.com / twitter.com URL."""
    m = re.search(r"/status/(\d+)", url)
    return m.group(1) if m else None


def _scrape_twitter(url: str) -> PostData:
    """
    Uses the Twitter API v2 (Bearer Token — free tier).
    Fetches tweet text + up to 1 image/video.

    Free tier quota: 500,000 tweet reads/month (plenty for a demo).
    Get your Bearer Token at: https://developer.x.com/en/portal/dashboard
    """
    bearer = os.environ.get("TWITTER_BEARER_TOKEN", "")
    if not bearer:
        return PostData(
            platform=Platform.TWITTER, url=url, caption="",
            error=(
                "TWITTER_BEARER_TOKEN not set.\n"
                "Get a free one at https://developer.x.com and add it to your .env / HF Space secrets."
            ),
        )

    tweet_id = _extract_tweet_id(url)
    if not tweet_id:
        return PostData(platform=Platform.TWITTER, url=url, caption="",
                        error="Could not extract tweet ID from URL.")

    endpoint = f"https://api.twitter.com/2/tweets/{tweet_id}"
    params   = {
        "tweet.fields": "created_at,author_id,text",
        "expansions":   "author_id,attachments.media_keys",
        "media.fields": "url,preview_image_url,type,variants",
        "user.fields":  "username",
    }
    headers  = {"Authorization": f"Bearer {bearer}"}

    try:
        resp = requests.get(endpoint, params=params, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        return PostData(platform=Platform.TWITTER, url=url, caption="",
                        error=f"Twitter API error: {e}")

    tweet   = data.get("data", {})
    caption = tweet.get("text", "")
    ts      = tweet.get("created_at", "")

    # Resolve author username
    author = ""
    users  = data.get("includes", {}).get("users", [])
    if users:
        author = "@" + users[0].get("username", "")

    # Download first media attachment (image or video thumbnail)
    image_path = None
    video_path = None
    thumb_url  = None
    media_list = data.get("includes", {}).get("media", [])
    for m in media_list:
        mtype = m.get("type", "")
        if mtype == "photo":
            thumb_url  = m.get("url", "")
            image_path = _download_media(thumb_url, ".jpg")
            break
        elif mtype in ("video", "animated_gif"):
            # Pick highest-bitrate mp4 variant
            variants = m.get("variants", [])
            mp4s     = [v for v in variants if v.get("content_type") == "video/mp4"]
            if mp4s:
                best = max(mp4s, key=lambda v: v.get("bit_rate", 0))
                video_path = _download_media(best["url"], ".mp4")
            thumb_url = m.get("preview_image_url", "")
            break

    return PostData(
        platform=Platform.TWITTER, url=url,
        caption=caption, author=author, timestamp=ts,
        image_path=image_path, video_path=video_path,
        thumbnail_url=thumb_url,
    )


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

    if platform == Platform.TWITTER:
        return _scrape_twitter(url)
    elif platform == Platform.INSTAGRAM:
        return _scrape_instagram(url)
    elif platform == Platform.FACEBOOK:
        return _scrape_facebook(url)
    else:
        return PostData(
            platform=Platform.UNKNOWN, url=url, caption="",
            error=(
                "Unsupported URL. TruthScan currently supports:\n"
                "  • X / Twitter  (x.com or twitter.com/…/status/…)\n"
                "  • Instagram    (instagram.com/p/…)\n"
                "  • Facebook     (facebook.com/…/posts/…)"
            ),
        )


# ── Smoke test ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    test_url = sys.argv[1] if len(sys.argv) > 1 else "https://x.com/NASA/status/1234567890"
    print(f"Testing URL: {test_url}")
    result = fetch_post(test_url)
    print(f"Platform   : {result.platform}")
    print(f"Author     : {result.author}")
    print(f"Caption    : {result.caption[:120]}…" if len(result.caption) > 120 else f"Caption: {result.caption}")
    print(f"Image path : {result.image_path}")
    print(f"Video path : {result.video_path}")
    if result.error:
        print(f"⚠️  Error   : {result.error}")
