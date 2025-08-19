import asyncio
import concurrent.futures
import itertools
import time
from collections.abc import AsyncGenerator, Generator
from dataclasses import dataclass
from io import BytesIO

import httpx
import requests
from PIL import Image
import random

async_client = httpx.AsyncClient()


DEFAULT_MAX_RETRIES = 3


@dataclass
class TileInfo:
    x: int
    y: int
    fileurl: str


@dataclass
class Tile:
    x: int
    y: int
    image: Image.Image


def get_width_and_height_from_zoom(zoom: int) -> tuple[int, int]:
    """
    Returns the width and height of a panorama at a given zoom level, depends on the
    zoom level.
    """
    return 2**zoom, 2 ** (zoom - 1)


def make_download_url(pano_id: str, zoom: int, x: int, y: int) -> str:
    """
    Returns the URL to download a tile.
    """
    return (
        "https://cbk0.google.com/cbk"
        f"?output=tile&panoid={pano_id}&zoom={zoom}&x={x}&y={y}"
    )


def fetch_panorama_tile(
    tile_info: TileInfo, max_retries: int = DEFAULT_MAX_RETRIES
) -> Image.Image:
    """
    Tries to download a tile, returns a PIL Image.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Referer": "https://www.google.com/maps",
        "Accept": "image/webp,image/*,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "DNT": "1",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "image",
        "Sec-Fetch-Mode": "no-cors",
        "Sec-Fetch-Site": "cross-site",
    }

    for attempt in range(max_retries):
        try:
            # Make request with timeout and headers
            response = requests.get(
                tile_info.fileurl,
                # headers=headers,
                timeout=30,
                stream=True
            )

            # Check HTTP status
            if response.status_code == 429:
                # Rate limited, wait longer
                wait_time = min(2 ** attempt + random.uniform(0, 1), 60)
                print(f"Rate limited (429). Waiting {wait_time:.1f}s before retry {attempt+1}/{max_retries}")
                time.sleep(wait_time)
                continue

            if response.status_code == 403:
                # Forbidden, might be temporary
                wait_time = 5 + random.uniform(0, 2)
                print(f"Access forbidden (403). Waiting {wait_time:.1f}s before retry {attempt+1}/{max_retries}")
                time.sleep(wait_time)
                continue

            if not response.ok:
                print(f"HTTP {response.status_code} for tile {tile_info.fileurl}. Retry {attempt+1}/{max_retries}")
                time.sleep(random.uniform(0.1, 0.5))
                continue

            # Check content type
            content_type = response.headers.get('Content-Type', '').lower()
            if not any(img_type in content_type for img_type in ['image/', 'application/octet-stream']):
                print(f"Non-image content (Content-Type: {content_type}) for tile {tile_info.fileurl}. Retry {attempt+1}/{max_retries}")
                # Log first 200 chars of response for debugging
                try:
                    preview = response.text[:200] if response.text else str(response.content[:200])
                    print(f"Response preview: {preview}")
                except:
                    pass
                time.sleep(2)
                continue

            # Check content length
            content_length = len(response.content)
            if content_length < 100:  # Too small to be a valid tile
                print(f"Response too small ({content_length} bytes) for tile {tile_info.fileurl}. Retry {attempt+1}/{max_retries}")
                time.sleep(2)
                continue

            # Try to open as image
            try:
                image = Image.open(BytesIO(response.content))
                # Verify it's a reasonable size (tiles should be 512x512 typically)
                if image.size[0] < 50 or image.size[1] < 50:
                    print(f"Image too small {image.size} for tile {tile_info.fileurl}. Retry {attempt+1}/{max_retries}")
                    time.sleep(2)
                    continue
                return image

            except Exception as img_error:
                print(f"PIL error for tile {tile_info.fileurl}: {img_error}. Retry {attempt+1}/{max_retries}")
                # Log some response info for debugging
                print(f"Content-Length: {content_length}, Content-Type: {content_type}")
                if content_length < 1000:  # Small response, might be error message
                    try:
                        preview = response.content[:500].decode('utf-8', errors='ignore')
                        print(f"Response content preview: {preview}")
                    except:
                        pass
                time.sleep(2)
                continue

        except requests.RequestException as e:
            print(f"Request error for tile {tile_info.fileurl}: {e}. Retry {attempt+1}/{max_retries}")
            wait_time = min(2 ** attempt, 30) + random.uniform(0, 1)
            time.sleep(wait_time)
            continue

        except Exception as e:
            print(f"Unexpected error for tile {tile_info.fileurl}: {e}. Retry {attempt+1}/{max_retries}")
            time.sleep(2)
            continue

    raise Exception(f"Failed to download tile {tile_info.fileurl} after {max_retries} attempts")


async def fetch_panorama_tile_async(
    tile_info: TileInfo, max_retries: int = DEFAULT_MAX_RETRIES
) -> Image.Image:
    """
    Asynchronously tries to download a tile, returns a PIL Image.
    """
    for _ in range(max_retries):
        try:
            response = await async_client.get(tile_info.fileurl)
            return Image.open(BytesIO(response.content))

        except httpx.RequestError as e:  # noqa: PERF203
            print(f"Request error {e}. Trying again in 2 seconds.")
            await asyncio.sleep(2)

    raise httpx.RequestError("Max retries exceeded.")


def iter_tile_info(pano_id: str, zoom: int) -> Generator[TileInfo, None, None]:
    """
    Generate a list of a panorama's tiles and their position.
    """
    width, height = get_width_and_height_from_zoom(zoom)
    for x, y in itertools.product(range(width), range(height)):
        yield TileInfo(
            x=x,
            y=y,
            fileurl=make_download_url(pano_id=pano_id, zoom=zoom, x=x, y=y),
        )


def iter_tiles(
    pano_id: str,
    zoom: int,
    max_retries: int = DEFAULT_MAX_RETRIES,
    multi_threaded: bool = False,
) -> Generator[Tile, None, None]:
    if not multi_threaded:
        for info in iter_tile_info(pano_id, zoom):
            image = fetch_panorama_tile(info, max_retries)
            print(f"[Downloaded] [{info.fileurl}] tile {info.x}, {info.y}")
            yield Tile(x=info.x, y=info.y, image=image)
        return

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_tile = {
            executor.submit(fetch_panorama_tile, info, max_retries): info
            for info in iter_tile_info(pano_id, zoom)
        }
        for future in concurrent.futures.as_completed(future_to_tile):
            info = future_to_tile[future]
            try:
                image = future.result()
            except Exception as exc:
                msg = f"Failed to download tile {info.fileurl} due to Exception: {exc}"
                raise Exception(msg) from exc
            else:
                yield Tile(x=info.x, y=info.y, image=image)


async def iter_tiles_async(
    pano_id: str, zoom: int, max_retries: int = DEFAULT_MAX_RETRIES
) -> AsyncGenerator[Tile, None]:
    for info in iter_tile_info(pano_id, zoom):
        image = await fetch_panorama_tile_async(info, max_retries)
        yield Tile(x=info.x, y=info.y, image=image)
    return


def get_panorama(
    pano_id: str,
    zoom: int = 5,
    multi_threaded: bool = False,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> Image.Image:
    """
    Downloads a streetview panorama.
    Multi-threaded is a lot faster, but it's also a lot more likely to get you banned.
    """
    tile_width = 512
    tile_height = 512

    total_width, total_height = get_width_and_height_from_zoom(zoom)
    panorama = Image.new("RGB", (total_width * tile_width, total_height * tile_height))

    for tile in iter_tiles(
        pano_id=pano_id,
        zoom=zoom,
        multi_threaded=multi_threaded,
        max_retries=max_retries,
    ):
        panorama.paste(im=tile.image, box=(tile.x * tile_width, tile.y * tile_height))
        del tile

    return panorama


async def get_panorama_async(
    pano_id: str, zoom: int, max_retries: int = DEFAULT_MAX_RETRIES
) -> Image.Image:
    """
    Downloads a streetview panorama by iterating through the tiles asynchronously.
    This runs in about the same speed as `get_panorama` with `multi_threaded=True`.
    """
    tile_width = 512
    tile_height = 512

    total_width, total_height = get_width_and_height_from_zoom(zoom)
    panorama = Image.new("RGB", (total_width * tile_width, total_height * tile_height))

    async for tile in iter_tiles_async(
        pano_id=pano_id, zoom=zoom, max_retries=max_retries
    ):
        panorama.paste(im=tile.image, box=(tile.x * tile_width, tile.y * tile_height))
        del tile

    return panorama
