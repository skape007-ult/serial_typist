"""
Monkeytype API client with per-test JSON caching.

Authenticates via ApeKey (env var MONKEYTYPE_APE_KEY). Fetches your test
history from /results, paginated. Caches each test as a separate JSON file
keyed by test _id, so subsequent runs only fetch tests newer than the
highest cached timestamp.

Rate limit awareness: the /results endpoint allows 30 requests per DAY
(not per minute). With limit=1000 per call, a full historical pull needs
~3 calls for ~2700 tests. Incremental updates need 1.

Usage:
    export MONKEYTYPE_APE_KEY="your_ape_key_here"
    python monkeytype_client.py                    # incremental fetch
    python monkeytype_client.py --full             # force full re-fetch
    python monkeytype_client.py --max-pages 1      # limit pages (testing)
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

API_BASE = "https://api.monkeytype.com"
CACHE_DIR = Path("monkeytype_cache")
RESULTS_PER_PAGE = 1000  # API max
INTER_REQUEST_DELAY = 1.0  # seconds, generous to avoid burst rate-limits


def get_api_key():
    key = os.environ.get('MONKEYTYPE_APE_KEY')
    if not key:
        print("ERROR: set MONKEYTYPE_APE_KEY environment variable.")
        print("  Generate one at monkeytype.com → settings → ape keys.")
        print("  Don't forget to ACTIVATE the key after generating it.")
        sys.exit(1)
    return key


def api_get(path, api_key, params=None):
    """GET request to the Monkeytype API. Returns parsed JSON or raises."""
    url = f"{API_BASE}{path}"
    if params:
        from urllib.parse import urlencode
        url += "?" + urlencode(params)

    req = Request(url, headers={
        'Authorization': f'ApeKey {api_key}',
        'Accept': 'application/json',
    })

    try:
        with urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode('utf-8'))
    except HTTPError as e:
        body = e.read().decode('utf-8', errors='replace')
        if e.code == 429:
            print(f"  RATE LIMIT HIT: {body}")
        raise RuntimeError(f"HTTP {e.code} from {url}: {body}")
    except URLError as e:
        raise RuntimeError(f"Network error fetching {url}: {e}")


def get_cached_test_ids():
    """Set of test _ids already cached locally."""
    if not CACHE_DIR.exists():
        return set()
    return {p.stem for p in CACHE_DIR.glob('*.json')}


def get_max_cached_timestamp():
    """Highest timestamp across cached tests, for incremental fetch."""
    if not CACHE_DIR.exists():
        return 0
    max_ts = 0
    for p in CACHE_DIR.glob('*.json'):
        try:
            with open(p) as f:
                data = json.load(f)
            ts = data.get('timestamp', 0)
            if ts > max_ts:
                max_ts = ts
        except Exception:
            continue
    return max_ts


def fetch_results_page(api_key, offset, limit, on_or_after_ts=None):
    """Fetch one page of /results. Returns list of result dicts."""
    params = {'limit': limit, 'offset': offset}
    if on_or_after_ts:
        # Monkeytype's /results supports `onOrAfterTimestamp` for incremental fetch
        params['onOrAfterTimestamp'] = on_or_after_ts

    print(f"  Fetching offset={offset} limit={limit}"
          + (f" since={on_or_after_ts}" if on_or_after_ts else ""))
    resp = api_get('/results', api_key, params)

    if not isinstance(resp, dict) or 'data' not in resp:
        raise RuntimeError(f"Unexpected response shape: {list(resp.keys())[:5]}")

    return resp['data']


def cache_test(test):
    """Write one test result to cache. Skips if already cached."""
    test_id = test.get('_id')
    if not test_id:
        return False
    out_path = CACHE_DIR / f"{test_id}.json"
    if out_path.exists():
        return False
    with open(out_path, 'w') as f:
        json.dump(test, f, indent=2)
    return True


def fetch_all(api_key, mode='incremental', max_pages=None):
    """Fetch test history. Mode: 'incremental' (only new) or 'full' (refetch all).
    Returns count of new tests cached."""
    CACHE_DIR.mkdir(exist_ok=True)

    if mode == 'incremental':
        on_or_after = get_max_cached_timestamp()
        if on_or_after:
            # +1ms to exclude the boundary test we already have
            on_or_after += 1
            print(f"Incremental fetch: tests after {on_or_after}")
        else:
            print("No cache found — performing full fetch.")
    else:
        on_or_after = None
        print("Full fetch (ignoring cache for retrieval; cache still suppresses writes).")

    new_count = 0
    seen_count = 0
    offset = 0
    page = 0

    while True:
        if max_pages and page >= max_pages:
            print(f"  Reached max_pages={max_pages}, stopping.")
            break

        try:
            results = fetch_results_page(api_key, offset, RESULTS_PER_PAGE,
                                         on_or_after_ts=on_or_after)
        except RuntimeError as e:
            print(f"  ERROR: {e}")
            break

        if not results:
            print("  No more results.")
            break

        page_new = 0
        page_seen = 0
        for test in results:
            if cache_test(test):
                page_new += 1
            else:
                page_seen += 1
        new_count += page_new
        seen_count += page_seen

        print(f"  Page {page}: {len(results)} results "
              f"({page_new} new, {page_seen} already cached)")

        page += 1

        # Pagination: Monkeytype returns up to `limit` results per page.
        # If we got fewer than limit, we're at the end.
        if len(results) < RESULTS_PER_PAGE:
            break

        offset += RESULTS_PER_PAGE
        time.sleep(INTER_REQUEST_DELAY)

    return new_count, seen_count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--full', action='store_true',
                        help='Force full fetch instead of incremental.')
    parser.add_argument('--max-pages', type=int, default=None,
                        help='Limit pages fetched (for testing).')
    args = parser.parse_args()

    api_key = get_api_key()
    mode = 'full' if args.full else 'incremental'

    print(f"Monkeytype client — mode={mode}, cache={CACHE_DIR}/")
    print(f"Currently cached: {len(get_cached_test_ids())} tests")
    print()

    start = time.time()
    new_count, seen_count = fetch_all(api_key, mode=mode,
                                      max_pages=args.max_pages)
    elapsed = time.time() - start

    total_cached = len(get_cached_test_ids())
    print(f"\n{'=' * 60}")
    print(f"FETCH COMPLETE in {elapsed:.1f}s")
    print(f"{'=' * 60}")
    print(f"New tests cached:  {new_count}")
    print(f"Already had:       {seen_count}")
    print(f"Total in cache:    {total_cached}")


if __name__ == '__main__':
    main()