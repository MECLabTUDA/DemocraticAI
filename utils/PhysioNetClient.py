import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

class PhysioNetClient:
    """
    Handles authentication to PhysioNet using username + password
    and allows listing and downloading files from restricted datasets.
    """

    LOGIN_URL = "https://physionet.org/login/"

    def __init__(self, username: str, password: str):
        self.username = username
        self.password = password
        self.session = requests.Session()

    def login(self):
        """Logs in via the PhysioNet login page and stores session cookies."""
        r = self.session.get(self.LOGIN_URL)
        r.raise_for_status()

        csrf = (
            self.session.cookies.get("csrftoken")
            or self.session.cookies.get("csrfmiddlewaretoken")
        )

        payload = {
            "username": self.username,
            "password": self.password,
            "csrfmiddlewaretoken": csrf,
        }

        r = self.session.post(
            self.LOGIN_URL,
            data=payload,
            headers={"Referer": self.LOGIN_URL}
        )

        if "sessionid" not in self.session.cookies:
            raise PermissionError(
                "Login failed. Check credentials or access rights."
            )

        print("Logged in successfully.")

    def get_html(self, url: str):
        r = self.session.get(url)
        if r.status_code == 403:
            raise PermissionError("403 Forbidden — you may not have dataset access.")
        r.raise_for_status()
        return r.text

    def list_dir(self, url: str):
        html = self.get_html(url)
        soup = BeautifulSoup(html, "html.parser")

        dirs, files = [], []

        for a in soup.find_all("a"):
            href = a.get("href")
            if not href or href.startswith("?") or href.startswith("/"):
                continue

            full_url = urljoin(url, href)

            if href.endswith("/"):
                dirs.append(full_url)
            else:
                files.append(full_url)

        return dirs, files

    def _download_chunk(self, url, start, end, session, results, idx):
        headers = {"Range": f"bytes={start}-{end}"}
        r = session.get(url, headers=headers, stream=True)
        r.raise_for_status()
        results[idx] = r.content


    def download_file(self, url: str, save_path: str, num_threads: int = 4):
        """
        Download a file in parallel chunks with a progress bar.
        """
        print(f"Downloading {url}...")

        # Get total size
        r = self.session.head(url)
        r.raise_for_status()
        total_size = int(r.headers.get("Content-Length", 0))

        if total_size == 0:
            raise ValueError("Cannot determine file size. Server may not support HEAD requests.")

        # Calculate byte ranges
        chunk_size = total_size // num_threads
        results = [None] * num_threads

        progress_lock = Lock()  # thread-safe updates
        progress = tqdm(
            total=total_size,
            unit="B",
            unit_scale=True,
            desc=save_path,
            miniters=1,
            leave=True,  # keep bar after completion
        )

        def download_and_update(idx, start, end):
            headers = {"Range": f"bytes={start}-{end}"}
            r = self.session.get(url, headers=headers, stream=True)
            r.raise_for_status()
            chunk_data = b""
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    chunk_data += chunk
                    with progress_lock:
                        progress.update(len(chunk))
            results[idx] = chunk_data

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = []
            for i in range(num_threads):
                start = i * chunk_size
                end = total_size - 1 if i == num_threads - 1 else (start + chunk_size - 1)
                futures.append(executor.submit(download_and_update, i, start, end))

            # Wait for all threads to finish
            for f in futures:
                f.result()

        progress.close()  # closes neatly but leaves bar visible

        # Write to file
        with open(save_path, "wb") as f:
            for chunk in results:
                f.write(chunk)

        print(f"Saved to {save_path}")



