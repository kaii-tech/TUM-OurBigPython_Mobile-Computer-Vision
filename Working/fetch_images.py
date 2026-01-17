import requests
import os
import urllib.parse

# === CONFIGURATION ===

# Replace with your Pixabay API key
PIXABAY_API_KEY = "YOUR_PIXABAY_API_KEY"

# Folder where images will be saved
IMAGE_FOLDER = "city_images_pixabay"
os.makedirs(IMAGE_FOLDER, exist_ok=True)

# Cities from your project with human‑readable search queries
city_queries = {
    "Bangkok": "Bangkok skyline city",
    "Barcelona": "Barcelona skyline city",
    "Boston": "Boston skyline city",
    "Brussels": "Brussels skyline city",
    "BuenosAires": "Buenos Aires skyline city",
    "Chicago": "Chicago skyline city",
    "Lisbon": "Lisbon skyline city",
    "London": "London skyline city",
    "LosAngeles": "Los Angeles skyline city",
    "Madrid": "Madrid skyline city",
    "Medellin": "Medellin skyline city",
    "Melbourne": "Melbourne skyline city",
    "MexicoCity": "Mexico City skyline city",
    "Miami": "Miami skyline city",
    "Minneapolis": "Minneapolis skyline city",
    "Oslo": "Oslo skyline city",
    "Osaka": "Osaka skyline city",
    "Prague": "Prague skyline city",   # PRG
    "Paris": "Paris skyline city",     # PRS
    "Phoenix": "Phoenix skyline city",
    "Rome": "Rome skyline city",
    "Toronto": "Toronto skyline city", # TRT
    "WashingtonDC": "Washington DC skyline city"
}

# Pixabay search base URL
BASE_URL = "https://pixabay.com/api/"

for city, query in city_queries.items():
    print(f"Searching for {city}...")

    params = {
        "key": PIXABAY_API_KEY,
        "q": query,
        "image_type": "photo",
        "per_page": 3  # try up to 3 images and use the first
    }

    try:
        response = requests.get(BASE_URL, params=params, timeout=15)
        response.raise_for_status()

        data = response.json()
        hits = data.get("hits", [])

        if not hits:
            print(f"❌ No image found for {city}")
            continue

        image_url = hits[0].get("largeImageURL")
        if not image_url:
            print(f"❌ No downloadable image found for {city}")
            continue

        # Download the image
        print(f"Downloading {city} from: {image_url}")
        img_response = requests.get(image_url, timeout=20)
        img_response.raise_for_status()

        filename = f"{city}.jpg"
        filepath = os.path.join(IMAGE_FOLDER, filename)

        with open(filepath, "wb") as f:
            f.write(img_response.content)

        print(f"✔ Saved {filepath}")

    except Exception as e:
        print(f"❌ Error for {city}: {e}")

print("\n✅ All done!")
