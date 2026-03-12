"""Prompt sets for benchmarking."""

PROMPT_SETS: dict[str, list[str]] = {
    "small": [
        # Quick test set (3 prompts)
        "Write a Python function to calculate the factorial of a number",
        "Explain the difference between HTTP and HTTPS",
        "Write a binary search algorithm in Python",
    ],
    "medium": [
        # Standard test set (5 prompts)
        "Explain the key differences between Kubernetes StatefulSets and Deployments, including when to use each and their specific use cases in production environments.",
        "Compare and contrast DevOps and SRE (Site Reliability Engineering) roles: What are the main responsibilities, skill sets, and philosophies that distinguish these two approaches to managing infrastructure and reliability?",
        "You have 12 identical-looking balls. One ball has a different weight (either heavier or lighter than the others, but you don't know which). Using a balance scale exactly 3 times, how can you identify which ball is different and whether it's heavier or lighter? Explain your strategy step by step with clear reasoning.",
        "Design a URL shortening service (like bit.ly) that can handle 1 billion shortened URLs and 10 million requests per day. Explain your architecture, database design, caching strategy, and how you would ensure high availability, reliability, and scalability.",
        "Explain why some programming languages are significantly faster than others at runtime. Compare compiled languages (like C++, Rust) versus interpreted languages (like Python, JavaScript), discuss JIT compilation, and explain the trade-offs between development speed and execution speed with real-world examples.",
    ],
    "large": [
        # Comprehensive test set (11 prompts)
        "Explain the key differences between Kubernetes StatefulSets and Deployments, including when to use each and their specific use cases in production environments.",
        "Compare and contrast DevOps and SRE (Site Reliability Engineering) roles: What are the main responsibilities, skill sets, and philosophies that distinguish these two approaches to managing infrastructure and reliability?",
        "You have 12 identical-looking balls. One ball has a different weight (either heavier or lighter than the others, but you don't know which). Using a balance scale exactly 3 times, how can you identify which ball is different and whether it's heavier or lighter? Explain your strategy step by step with clear reasoning.",
        "Design a URL shortening service (like bit.ly) that can handle 1 billion shortened URLs and 10 million requests per day. Explain your architecture, database design, caching strategy, and how you would ensure high availability, reliability, and scalability.",
        "Explain why some programming languages are significantly faster than others at runtime. Compare compiled languages (like C++, Rust) versus interpreted languages (like Python, JavaScript), discuss JIT compilation, and explain the trade-offs between development speed and execution speed with real-world examples.",
        "Design a distributed caching system like Redis or Memcached. Explain cache eviction policies (LRU, LFU, FIFO), consistency strategies, sharding approaches, and how you would handle cache invalidation in a microservices architecture.",
        "Explain the CAP theorem in distributed systems. Provide real-world examples of databases that prioritize Consistency+Partition tolerance (CP) versus Availability+Partition tolerance (AP), and explain when you would choose each approach.",
        "Write a Python function to implement a rate limiter using the token bucket algorithm. Explain how it works and why it's better than simple request counting for API rate limiting.",
        "Design a real-time notification system (like Firebase Cloud Messaging or AWS SNS) that can handle millions of concurrent users. Explain your technology choices, message queue design, and how you would ensure message delivery guarantees.",
        "Explain how modern search engines like Elasticsearch work. Cover inverted indexes, relevance scoring (TF-IDF, BM25), sharding, and how they achieve near real-time search across billions of documents.",
        "Design the backend architecture for a social media platform like Twitter. Explain how you would handle the feed generation problem (fan-out on write vs fan-out on read), storage strategy for tweets, and how to scale to millions of active users.",
    ],
}


def get_prompts(set_name: str) -> list[str]:
    """Return prompts for the given set name.

    Args:
        set_name: One of "small", "medium", or "large".

    Returns:
        List of prompt strings.

    Raises:
        ValueError: If set_name is not a recognized prompt set.
    """
    if set_name not in PROMPT_SETS:
        available = ", ".join(sorted(PROMPT_SETS.keys()))
        raise ValueError(
            f"Unknown prompt set '{set_name}'. Available sets: {available}"
        )
    return PROMPT_SETS[set_name]
