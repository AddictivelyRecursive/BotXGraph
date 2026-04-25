import pandas as pd
import re
from urllib.parse import urlparse


class EntityExtractor:
    def __init__(self, config):
        self.config = config

    # ---------------------------
    # HASHTAGS
    # ---------------------------
    def extract_hashtags(self, tweets_df):
        print("Extracting hashtags...")

        hashtag_nodes = set()
        edges = []

        for _, row in tweets_df.iterrows():
            tweet_id = row["id"]

            tags = []
            entities = row.get("entities", {})

            if isinstance(entities, dict):
                for tag_obj in entities.get("hashtags", []):
                    if isinstance(tag_obj, dict):
                        tag = tag_obj.get("tag") or tag_obj.get("text")
                    else:
                        tag = str(tag_obj)
                    tags.append(tag)

            text = row.get("text", "")
            if not tags and isinstance(text, str):
                tags.extend(re.findall(r"(?<!\w)#([A-Za-z0-9_]+)", text))

            for tag in tags:
                if not tag:
                    continue

                tag = str(tag).lower()

                hashtag_nodes.add(tag)

                edges.append({
                    "tweet_id": tweet_id,
                    "hashtag": tag
                })

        hashtags_df = pd.DataFrame({"hashtag": list(hashtag_nodes)})
        hashtag_edges_df = pd.DataFrame(edges,columns=["tweet_id", "hashtag"])
        
        print(f"Hashtags extracted: {len(hashtags_df)}")
        print(f"Tweet-Hashtag edges: {len(hashtag_edges_df)}")

        return hashtags_df, hashtag_edges_df

    # ---------------------------
    # URLS
    # ---------------------------
    def extract_urls(self, tweets_df):
        print("Extracting URLs...")

        url_nodes = set()
        edges = []

        for _, row in tweets_df.iterrows():
            tweet_id = row["id"]

            urls = []
            entities = row.get("entities", {})

            if isinstance(entities, dict):
                for url_obj in entities.get("urls", []):
                    if isinstance(url_obj, dict):
                        url = url_obj.get("expanded_url") or url_obj.get("url")
                    else:
                        url = str(url_obj)
                    urls.append(url)

            text = row.get("text", "")
            if not urls and isinstance(text, str):
                urls.extend(re.findall(r"https?://[^\s]+", text))

            for url in urls:
                if not url:
                    continue

                # extract domain
                try:
                    domain = urlparse(url).netloc.lower()
                except:
                    continue

                if not domain:
                    continue

                url_nodes.add(domain)

                edges.append({
                    "tweet_id": tweet_id,
                    "url": domain
                })

        urls_df = pd.DataFrame({"url": list(url_nodes)})
        url_edges_df = pd.DataFrame(edges,columns=["tweet_id", "url"])

        print(f"URLs extracted: {len(urls_df)}")
        print(f"Tweet-URL edges: {len(url_edges_df)}")

        return urls_df, url_edges_df

    # ---------------------------
    # FULL PIPELINE
    # ---------------------------
    def extract_all(self, tweets_df):
        hashtags_df, hashtag_edges_df = self.extract_hashtags(tweets_df)
        urls_df, url_edges_df = self.extract_urls(tweets_df)

        return {
            "hashtags": hashtags_df,
            "hashtag_edges": hashtag_edges_df,
            "urls": urls_df,
            "url_edges": url_edges_df
        }
