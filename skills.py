# skills.py
# Categorized skill taxonomy for advanced matching and radar chart visualization
# Expanded to 270+ skills for improved detection accuracy

SKILLS_TAXONOMY = {
    "Programming Languages": [
        "python", "java", "c++", "c#", "javascript", "typescript", "ruby", "php",
        "swift", "kotlin", "go", "golang", "rust", "scala", "r", "matlab", "perl",
        "dart", "objective-c", "haskell", "elixir", "clojure", "lua", "groovy",
        "fortran", "cobol", "assembly", "shell", "bash", "powershell", "vba",
        "julia", "solidity", "sql", "plsql", "t-sql"
    ],
    "Web & Frontend": [
        "html", "html5", "css", "css3", "react", "react.js", "reactjs", "angular",
        "angularjs", "vue", "vue.js", "vuejs", "svelte", "next.js", "nextjs",
        "nuxt.js", "nuxtjs", "tailwind", "tailwind css", "bootstrap", "sass",
        "scss", "less", "webpack", "vite", "babel", "jquery", "redux", "zustand",
        "mobx", "graphql", "rest api", "restful api", "restful", "rest",
        "websocket", "web components", "pwa", "progressive web app", "three.js",
        "d3.js", "chart.js", "storybook", "cypress", "playwright", "jest",
        "vitest", "material ui", "ant design", "chakra ui"
    ],
    "Backend & Frameworks": [
        "node.js", "nodejs", "express", "express.js", "django", "flask", "fastapi",
        "spring", "spring boot", "ruby on rails", "rails", "asp.net", ".net",
        "laravel", "grpc", "microservices", "serverless", "rabbitmq", "celery",
        "pydantic", "sqlalchemy", "alembic", "prisma", "sequelize", "mongoose",
        "nestjs", "nest.js", "hapi", "koa", "gin", "fiber", "echo",
        "actix", "axum", "rocket", "tornado", "aiohttp", "starlette",
        "graphql api", "soap", "api gateway", "message queue"
    ],
    "Data & Databases": [
        "sql", "nosql", "mongodb", "postgresql", "postgres", "mysql", "oracle",
        "redis", "elasticsearch", "cassandra", "sqlite", "dynamodb", "firebase",
        "supabase", "data analysis", "data science", "data mining", "big data",
        "etl", "data warehousing", "data modeling", "database design",
        "mariadb", "cockroachdb", "neo4j", "influxdb", "clickhouse", "duckdb",
        "pinecone", "weaviate", "qdrant", "chroma", "milvus", "vector database",
        "faiss", "data pipeline", "data lake", "data mesh"
    ],
    "AI & Machine Learning": [
        "machine learning", "deep learning", "natural language processing", "nlp",
        "computer vision", "tensorflow", "keras", "pytorch", "scikit-learn",
        "sklearn", "pandas", "numpy", "matplotlib", "seaborn", "transformers",
        "hugging face", "huggingface", "llm", "large language model",
        "generative ai", "gen ai", "reinforcement learning", "feature engineering",
        "xgboost", "lightgbm", "catboost", "opencv", "spacy", "nltk",
        "langchain", "llamaindex", "llama index", "llama", "openai",
        "openai api", "gpt", "gpt-4", "anthropic", "claude", "cohere",
        "gemini", "rag", "retrieval augmented generation", "vector search",
        "embedding", "embeddings", "fine-tuning", "finetuning", "prompt engineering",
        "bert", "roberta", "t5", "gpt-2", "stable diffusion", "diffusion models",
        "object detection", "image segmentation", "yolo", "resnet", "vgg",
        "random forest", "gradient boosting", "svm", "support vector machine",
        "neural network", "cnn", "rnn", "lstm", "gru", "transformer",
        "attention mechanism", "transfer learning", "model deployment",
        "mlflow", "weights and biases", "wandb", "optuna", "hyperparameter tuning",
        "data augmentation", "model evaluation", "a/b testing", "experiment tracking",
        "streamlit", "gradio", "plotly", "bokeh", "altair"
    ],
    "Cloud & DevOps": [
        "aws", "amazon web services", "azure", "microsoft azure",
        "google cloud platform", "gcp", "google cloud", "docker", "kubernetes",
        "k8s", "terraform", "ansible", "jenkins", "ci/cd", "github actions",
        "circleci", "gitlab ci", "travis ci", "prometheus", "grafana", "nginx",
        "linux", "unix", "bash scripting", "shell scripting", "powershell",
        "helm", "istio", "service mesh", "load balancing", "auto scaling",
        "lambda", "aws lambda", "ec2", "s3", "rds", "vpc", "iam",
        "cloudformation", "pulumi", "cloud run", "cloud functions",
        "azure functions", "aks", "eks", "gke", "serverless framework",
        "monitoring", "logging", "observability", "elk stack", "splunk",
        "datadog", "new relic", "pagerduty", "sre", "site reliability",
        "infrastructure as code", "iac", "devsecops", "sonarqube", "trivy",
        "vault", "consul", "envoy", "argocd", "flux"
    ],
    "Data Engineering": [
        "hadoop", "spark", "apache spark", "kafka", "apache kafka", "airflow",
        "apache airflow", "dbt", "databricks", "snowflake", "looker",
        "tableau", "power bi", "powerbi", "dask", "flink", "apache flink",
        "beam", "apache beam", "nifi", "apache nifi", "hive", "presto",
        "trino", "redshift", "bigquery", "azure synapse", "delta lake",
        "iceberg", "hudi", "parquet", "avro", "orc", "data catalog",
        "data governance", "data quality", "great expectations",
        "looker studio", "google data studio", "qlik", "metabase",
        "superset", "apache superset"
    ],
    "Version Control & Collaboration": [
        "git", "github", "gitlab", "bitbucket", "jira", "confluence",
        "trello", "agile", "scrum", "kanban", "notion", "linear",
        "asana", "monday.com", "slack", "teams", "code review",
        "pull request", "pair programming", "tdd", "test driven development",
        "bdd", "behavior driven development", "pair programming", "devops culture"
    ],
    "Security & Testing": [
        "unit testing", "integration testing", "e2e testing", "pytest",
        "unittest", "mocha", "jasmine", "junit", "selenium", "cypress",
        "playwright", "postman", "swagger", "openapi", "api testing",
        "load testing", "performance testing", "security testing",
        "penetration testing", "owasp", "ssl", "tls", "oauth",
        "oauth2", "jwt", "authentication", "authorization", "rbac",
        "cryptography", "encryption", "vulnerability assessment"
    ],
    "Soft Skills": [
        "communication", "teamwork", "problem solving", "critical thinking",
        "leadership", "project management", "time management", "adaptability",
        "creativity", "mentoring", "collaboration", "presentation",
        "negotiation", "analytical thinking", "attention to detail",
        "cross-functional", "stakeholder management", "decision making",
        "strategic thinking", "customer focus", "empathy"
    ]
}

# Known false-positive skill pairs that should NOT be treated as matches
# Format: frozenset({skill_a, skill_b}) - bidirectional
# NOTE: Do NOT add alias pairs here (go/golang, react/reactjs, etc.) —
#       those are the SAME skill and are handled by SKILL_ALIASES.
SKILL_FALSE_POSITIVES = [
    frozenset({"java", "javascript"}),
    frozenset({"java", "typescript"}),
    frozenset({"flask", "spark"}),      # Apache Spark vs Flask web framework
    frozenset({"r", "rust"}),           # R language vs Rust language
    frozenset({"c", "c++"}),            # C vs C++
    frozenset({"c#", "c++"}),           # C# vs C++
    frozenset({"sql", "nosql"}),        # Opposites — do not cross-match
    frozenset({"c#", "c"}),
    frozenset({"python", "pytorch"}),   # PyTorch contains "py" but is not Python
    frozenset({"java", "java"}),        # Self-pair guard (no-op but explicit)
    frozenset({"go", "groovy"}),        # Go language vs Groovy
    frozenset({"scala", "sql"}),        # Different languages
    frozenset({"r", "ruby"}),           # R language vs Ruby
    frozenset({"hadoop", "haskell"}),   # Different domains entirely
    frozenset({"dart", "data"}),        # Dart language vs data keyword
]

# Canonical aliases — if either appears, treat as the same skill
SKILL_ALIASES = {
    "reactjs": "react",
    "react.js": "react",
    "nodejs": "node.js",
    "vuejs": "vue.js",
    "nextjs": "next.js",
    "nuxtjs": "nuxt.js",
    "golang": "go",
    "sklearn": "scikit-learn",
    "postgres": "postgresql",
    "powerbi": "power bi",
    "huggingface": "hugging face",
    "rails": "ruby on rails",
    "k8s": "kubernetes",
    "gen ai": "generative ai",
}


def get_skills():
    """Return flat list of all skills."""
    all_skills = []
    for category, skills in SKILLS_TAXONOMY.items():
        all_skills.extend(skills)
    return list(set(all_skills))  # deduplicate


def get_skills_taxonomy():
    """Return the full categorized taxonomy."""
    return SKILLS_TAXONOMY


def get_skill_category(skill):
    """Find which category a skill belongs to."""
    skill_lower = skill.lower()
    # Check aliases first
    canonical = SKILL_ALIASES.get(skill_lower, skill_lower)
    for category, skills in SKILLS_TAXONOMY.items():
        if canonical in skills or skill_lower in skills:
            return category
    return "Other"


def are_false_positive_pair(skill_a, skill_b):
    """Check if two skills are a known false-positive match pair."""
    pair = frozenset({skill_a.lower(), skill_b.lower()})
    return pair in SKILL_FALSE_POSITIVES