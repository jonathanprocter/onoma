import os
import re
import tempfile
from io import BytesIO

import httpx

ISBN_RE = re.compile(r"\b97[89][- ]?\d{1,5}[- ]?\d{1,7}[- ]?\d{1,7}[- ]?\d\b")


def extract_isbn(text: str) -> str | None:
    if not text:
        return None
    match = ISBN_RE.search(text)
    if not match:
        return None
    return re.sub(r"[- ]", "", match.group(0))


def _save_image_bytes(image_bytes: bytes, output_path: str, overwrite: bool) -> str | None:
    if not image_bytes:
        return None
    if os.path.exists(output_path) and not overwrite:
        return None
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(image_bytes)
    return output_path


def _image_size_ok(image_bytes: bytes, min_size: int) -> bool:
    try:
        from PIL import Image

        with Image.open(BytesIO(image_bytes)) as img:
            return min(img.size) >= min_size
    except Exception:
        return False


def extract_cover_embedded(file_path: str, config: dict) -> bytes | None:
    ext = os.path.splitext(file_path)[1].lower()
    min_size = int(config.get("cover_min_size", 300))
    if ext == ".epub":
        try:
            from ebooklib import epub

            book = epub.read_epub(file_path)
            items = list(book.get_items_of_type(epub.ITEM_COVER))
            if not items:
                items = [
                    item
                    for item in book.get_items()
                    if "cover" in item.get_id().lower()
                    or "cover" in (item.get_name() or "").lower()
                ]
            for item in items:
                data = item.get_content()
                if _image_size_ok(data, min_size):
                    return data
        except Exception:
            return None
    if ext == ".pdf":
        try:
            import fitz

            doc = fitz.open(file_path)
            if doc.page_count == 0:
                return None
            pix = doc.load_page(0).get_pixmap()
            data = pix.tobytes("jpg")
            if _image_size_ok(data, min_size):
                return data
        except Exception:
            return None
    return None


def embed_cover_epub(file_path: str, image_bytes: bytes) -> bool:
    try:
        from ebooklib import epub

        book = epub.read_epub(file_path)
        book.set_cover("cover.jpg", image_bytes)
        with tempfile.TemporaryDirectory() as td:
            temp_path = os.path.join(td, os.path.basename(file_path))
            epub.write_epub(temp_path, book)
            os.replace(temp_path, file_path)
        return True
    except Exception:
        return False


def fetch_cover_openlibrary(isbn: str) -> bytes | None:
    url = f"https://covers.openlibrary.org/b/isbn/{isbn}-L.jpg?default=false"
    try:
        with httpx.Client(timeout=15.0) as client:
            resp = client.get(url)
            if resp.status_code == 200 and resp.headers.get("content-type", "").startswith("image"):
                return resp.content
    except Exception:
        return None
    return None


def fetch_cover_google_books(isbn: str, api_key: str) -> bytes | None:
    if not api_key:
        return None
    try:
        with httpx.Client(timeout=15.0) as client:
            resp = client.get(
                "https://www.googleapis.com/books/v1/volumes",
                params={"q": f"isbn:{isbn}", "key": api_key},
            )
            if resp.status_code != 200:
                return None
            data = resp.json()
            items = data.get("items") or []
            if not items:
                return None
            links = items[0].get("volumeInfo", {}).get("imageLinks", {})
            for key in ("extraLarge", "large", "medium", "thumbnail", "smallThumbnail"):
                if key in links:
                    img_url = links[key]
                    img_resp = client.get(img_url)
                    if img_resp.status_code == 200 and img_resp.headers.get("content-type", "").startswith("image"):
                        return img_resp.content
    except Exception:
        return None
    return None


def cover_workflow(
    original_path: str,
    final_path: str,
    isbn: str | None,
    config: dict,
) -> dict | None:
    if not config.get("cover_enabled", False):
        return None
    overwrite = bool(config.get("cover_overwrite", False))
    output_dir = config.get("cover_output_dir") or os.path.dirname(final_path)
    base_name = os.path.splitext(os.path.basename(final_path))[0]
    output_path = os.path.join(output_dir, f"{base_name}.jpg")

    # 1) Embedded extraction
    img = extract_cover_embedded(original_path, config)
    if img:
        saved = _save_image_bytes(img, output_path, overwrite)
        if saved and config.get("cover_embed", False) and original_path.endswith(".epub"):
            embed_cover_epub(original_path, img)
        return {"status": "cover_embedded", "path": saved}

    # 2) API waterfall by ISBN
    if isbn:
        if config.get("cover_use_openlibrary", True):
            img = fetch_cover_openlibrary(isbn)
            if img:
                saved = _save_image_bytes(img, output_path, overwrite)
                return {"status": "cover_openlibrary", "path": saved}
        if config.get("cover_use_google_books", True):
            img = fetch_cover_google_books(isbn, config.get("google_books_api_key", ""))
            if img:
                saved = _save_image_bytes(img, output_path, overwrite)
                return {"status": "cover_google_books", "path": saved}

    return {"status": "cover_not_found", "path": ""}
