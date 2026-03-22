import pandas as pd
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

            entities = row.get("entities", {})
            if not isinstance(entities, dict):
                continue

            hashtags = entities.get("hashtags", [])

            for tag_obj in hashtags:
                tag = tag_obj.get("tag") or tag_obj.get("text")
                if not tag:
                    continue

                tag = tag.lower()

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

            entities = row.get("entities", {})
            if not isinstance(entities, dict):
                continue

            urls = entities.get("urls", [])

            for url_obj in urls:
                url = url_obj.get("expanded_url") or url_obj.get("url")
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