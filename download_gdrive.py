import requests
import re
import sys
import http.cookiejar

def download_file_from_google_drive(id, destination, cookies_file):
    print(f"Downloading {destination}...")
    URL = "https://drive.google.com/uc?export=download&id=" + id

    session = requests.Session()
    
    # Load cookies
    cj = http.cookiejar.MozillaCookieJar(cookies_file)
    cj.load(ignore_discard=True, ignore_expires=True)
    session.cookies.update(cj)

    response = session.get(URL, stream=True)
    
    # Check for virus warning page
    if "text/html" in response.headers.get("Content-Type", ""):
        content = response.content.decode('utf-8', errors='ignore')
        
        # Look for the uuid in the form
        match = re.search(r'name="uuid" value="([^"]+)"', content)
        if match:
            uuid = match.group(1)
            print(f"Found virus warning. UUID: {uuid}. Bypassing...")
            
            # The download URL form action
            download_url = "https://drive.usercontent.google.com/download"
            params = {
                'id': id,
                'export': 'download',
                'confirm': 't',
                'uuid': uuid
            }
            response = session.get(download_url, params=params, stream=True)
        else:
            print("Failed to find UUID in the page. The page content was:")
            print(content[:500])
            return False

    # Save the file
    if response.status_code == 200:
        with open(destination, "wb") as f:
            for chunk in response.iter_content(32768):
                if chunk: # filter out keep-alive new chunks
                    f.write(chunk)
        print(f"Done downloading {destination}.")
        return True
    else:
        print(f"Error downloading file: Status code {response.status_code}")
        print(response.text[:500])
        return False

if __name__ == "__main__":
    files = {
        "1hJZRPyk0fp-6NfacOWkAHuPu16D093Da": "libritts.zip",
        "1GhlqZgTPkL-mxrWaS8wWcoeAZNmXIwBV": "PhoAudioBook_part1.zip",
        "1g-glCgARAjWkzc56WO6shxxH7-A8cfk-": "PhoAudioBook_part2.zip",
        "1WNsR1ba2kNV6nq_guvVS7fzUXjyqexmo": "vivoice.zip",
        "1Ii-TbEeEzh-8YOWlzVK1M5lDli34QQz2": "YT_DT_P1.zip",
        "1gZGDa4ILOduPs8um9tQoE9M6D6PVFizy": "YT_DT_P2.zip"
    }
    
    cookies = "/root/TTS_DATA/drive.google.com_cookies.txt"
    
    for file_id, filename in files.items():
        download_file_from_google_drive(file_id, "/root/TTS_DATA/data/" + filename, cookies)

