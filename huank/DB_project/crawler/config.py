import os
from dataclasses import dataclass


@dataclass
class CrawlConfig:
    user_agent: str = (
        os.environ.get(
            "CRAWL_UA",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/119.0 Safari/537.36",
        )
    )
    min_delay_sec: float = float(os.environ.get("CRAWL_MIN_DELAY", "2"))
    max_delay_sec: float = float(os.environ.get("CRAWL_MAX_DELAY", "3"))
    max_retries: int = int(os.environ.get("CRAWL_MAX_RETRIES", "3"))
    timeout_sec: float = float(os.environ.get("CRAWL_TIMEOUT", "20"))
    output_json_path: str = os.environ.get("OUTPUT_JSON", "output/diseases.json")
    log_path: str = os.environ.get("CRAWL_LOG", "logs/crawl.log")
    stats_path: str = os.environ.get("CRAWL_STATS", "logs/stats.json")
    mongo_url: str = os.environ.get("MONGO_URL", "mongodb://localhost:27017")
    mongo_db: str = os.environ.get("MONGO_DB", "medical")
    mongo_collection: str = os.environ.get("MONGO_COLLECTION", "diseases")
    mysql_url: str = os.environ.get("MYSQL_URL", "mysql+pymysql://user:password@localhost:3306/medical")

