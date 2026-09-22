"""Pools of synthetic values used by the deterministic dataset generator.

All values are fictional. No real person's resume data is used anywhere in
this project (see docs/DATASET.md).
"""

from __future__ import annotations

FIRST_NAMES = [
    "Aarav", "Aditi", "Aisha", "Akash", "Amit", "Ananya", "Anjali", "Arjun",
    "Bhavna", "Carlos", "Deepak", "Diya", "Elena", "Emma", "Farhan", "Gaurav",
    "Ishaan", "Jasmine", "Kabir", "Karan", "Kiran", "Lakshmi", "Liam", "Manish",
    "Meera", "Michael", "Neha", "Nikhil", "Olivia", "Priya", "Rahul", "Rajesh",
    "Riya", "Rohan", "Samuel", "Sandra", "Sara", "Shreya", "Sneha", "Sofia",
    "Suresh", "Tanvi", "Tara", "Uma", "Varun", "Vikram", "Yuki", "Zara",
]

LAST_NAMES = [
    "Agarwal", "Anderson", "Bhatt", "Brown", "Chen", "Chopra", "Davis", "Desai",
    "Fernandez", "Garcia", "Gupta", "Iyer", "Jain", "Johnson", "Joshi", "Kapoor",
    "Kaur", "Khan", "Kim", "Kumar", "Lee", "Malhotra", "Martin", "Mehta",
    "Menon", "Mishra", "Mittal", "Nair", "Nguyen", "Patel", "Patil", "Pillai",
    "Rao", "Reddy", "Rodriguez", "Sharma", "Shah", "Silva", "Singh", "Smith",
    "Srinivasan", "Tanaka", "Taylor", "Thomas", "Verma", "Williams", "Wong",
    "Yadav", "Zhang", "Kulkarni",
]

# (city, region) - rendered as "City, Region"
CITIES = [
    ("Hyderabad", "Telangana"), ("Bengaluru", "Karnataka"), ("Pune", "Maharashtra"),
    ("Chennai", "Tamil Nadu"), ("Mumbai", "Maharashtra"), ("New Delhi", "Delhi"),
    ("Kolkata", "West Bengal"), ("Ahmedabad", "Gujarat"), ("Jaipur", "Rajasthan"),
    ("Kochi", "Kerala"), ("Indore", "Madhya Pradesh"), ("Nagpur", "Maharashtra"),
    ("Visakhapatnam", "Andhra Pradesh"), ("Lucknow", "Uttar Pradesh"),
    ("Chandigarh", "Punjab"), ("Bhopal", "Madhya Pradesh"),
    ("Coimbatore", "Tamil Nadu"), ("London", "Greater London"),
    ("Manchester", "Greater Manchester"), ("Berlin", "Berlin"),
    ("Munich", "Bavaria"), ("Toronto", "Ontario"),
    ("Vancouver", "British Columbia"), ("Singapore", "Central Region"),
    ("Sydney", "New South Wales"), ("Melbourne", "Victoria"),
    ("Austin", "Texas"), ("Seattle", "Washington"), ("Boston", "Massachusetts"),
    ("Chicago", "Illinois"), ("Dublin", "Leinster"), ("Amsterdam", "North Holland"),
]

COMPANIES = [
    "Nova Analytics", "QuantumLeaf Technologies", "BlueOrbit Systems",
    "Zenith Data Labs", "CoreWave Solutions", "PixelForge Studio",
    "SilverLine Software", "BrightPath AI", "Trueline Consulting",
    "Vertex Cloud Works", "IronPeak Software", "MapleCore Systems",
    "Sunstone Labs", "CloudNine Analytics", "RapidPixel Games",
    "EchoBridge Tech", "FalconSoft", "NimbusWorks", "GraniteByte",
    "LatticeMind", "OceanGrid Systems", "Redwood Data Co",
    "Stellar Code Works", "UrbanByte Solutions", "CopperField Analytics",
    "Wavecrest Labs", "NorthStar Software", "DeltaStream Tech", "ArgonSoft",
    "Pinnacle Insights", "Skyline Analytics", "Modulus Labs", "HorizonSoft",
    "QuantumQuill", "BrightByte Systems", "ClearWater Tech",
    "SummitLine Analytics", "OakRidge Software", "BlueRiver AI", "FalconSight",
    "TerraByte", "Vertexa Labs", "SkillForge Tech", "MindBridge AI",
    "DataNest", "AlgoWorks", "PrimeLogic Software", "NextGen Coders",
]

UNIVERSITIES = [
    "XYZ Institute of Technology", "ABC National University",
    "Sunrise Engineering College", "Greenfield University",
    "Riverdale Institute of Science", "St. Mary's Technical University",
    "National Institute of Technology", "Oakbrook University",
    "Lakeside Institute of Technology", "Maplewood University",
    "Silverdale College of Engineering", "Crestwood Institute",
    "Harborview University", "Kingsland Institute of Technology",
    "Redcliff University", "Westfield State University", "Northgate Institute",
    "Eastvale Engineering College", "Bluecrest University", "Highfield College",
    "Stonebridge University", "Clearwater Institute of Technology",
    "Fairview University", "Grandview Institute", "Rosewood College",
    "Pinewood International University", "Silver Oak University",
    "Techforge Institute",
]

# (degree, field)
UNDERGRAD_DEGREES = [
    ("B.Tech", "Computer Science"), ("B.Tech", "Information Technology"),
    ("B.E.", "Electronics and Communication"), ("B.E.", "Computer Engineering"),
    ("B.Sc", "Computer Science"), ("B.Sc", "Mathematics"),
    ("BCA", "Computer Applications"),
]
POSTGRAD_DEGREES = [
    ("M.Tech", "Data Science"), ("M.Tech", "Software Engineering"),
    ("M.Sc", "Statistics"), ("M.Sc", "Computer Science"),
    ("MCA", "Computer Applications"), ("MBA", "Business Analytics"),
]
ALL_DEGREE_PAIRS = UNDERGRAD_DEGREES + POSTGRAD_DEGREES

JOB_TITLES = [
    "Machine Learning Engineer", "Data Scientist", "Software Engineer",
    "Senior Software Engineer", "Backend Developer", "Frontend Developer",
    "Full Stack Developer", "Data Engineer", "Data Analyst",
    "DevOps Engineer", "Cloud Engineer", "MLOps Engineer",
    "AI Research Engineer", "NLP Engineer", "Computer Vision Engineer",
    "Business Intelligence Analyst", "QA Engineer", "Automation Engineer",
    "Product Analyst", "Database Administrator", "Systems Engineer",
    "Mobile Application Developer", "Site Reliability Engineer",
    "Analytics Consultant", "Junior Data Scientist", "Python Developer",
    "Solutions Architect", "Research Assistant",
]

SKILLS = [
    "Python", "Java", "C++", "C#", "JavaScript", "TypeScript", "SQL", "NoSQL",
    "HTML", "CSS", "React", "Angular", "Vue.js", "Node.js", "Django", "Flask",
    "FastAPI", "Spring Boot", ".NET", "Go", "Rust", "Kotlin", "Swift", "PHP",
    "Ruby", "R", "MATLAB", "Scala", "Bash", "PowerShell", "PyTorch",
    "TensorFlow", "Keras", "Scikit-learn", "XGBoost", "LightGBM", "OpenCV",
    "NLTK", "spaCy", "Hugging Face Transformers", "Pandas", "NumPy", "SciPy",
    "Matplotlib", "Seaborn", "Plotly", "Tableau", "Power BI", "Excel",
    "MySQL", "PostgreSQL", "MongoDB", "Redis", "Elasticsearch", "Cassandra",
    "Snowflake", "BigQuery", "Redshift", "Spark", "Hadoop", "Kafka", "Airflow",
    "dbt", "Docker", "Kubernetes", "Jenkins", "Git", "GitHub Actions",
    "Terraform", "Ansible", "AWS", "Azure", "GCP", "EC2", "S3", "Lambda",
    "Linux", "CI/CD", "REST APIs", "GraphQL", "Microservices",
    "Machine Learning", "Deep Learning", "NLP", "Computer Vision",
    "Data Visualization", "Statistical Analysis", "A/B Testing", "ETL",
    "Data Warehousing", "Unit Testing", "Agile", "Jira",
]

# Category mapping used by the skills-heavy template (skills embedded in
# labelled paragraphs). Every skill listed here must exist in SKILLS.
SKILL_CATEGORIES = {
    "Languages": ["Python", "Java", "C++", "C#", "JavaScript", "TypeScript",
                  "SQL", "Go", "Rust", "Kotlin", "PHP", "R", "Scala", "Bash"],
    "Frameworks": ["React", "Angular", "Vue.js", "Node.js", "Django", "Flask",
                   "FastAPI", "Spring Boot", ".NET", "Keras"],
    "Machine Learning": ["PyTorch", "TensorFlow", "Scikit-learn", "XGBoost",
                         "LightGBM", "OpenCV", "NLTK", "spaCy",
                         "Hugging Face Transformers", "Machine Learning",
                         "Deep Learning", "NLP", "Computer Vision"],
    "Data": ["Pandas", "NumPy", "SciPy", "Matplotlib", "Seaborn", "Plotly",
             "Spark", "Hadoop", "Kafka", "Airflow", "dbt", "ETL",
             "Data Warehousing", "Statistical Analysis"],
    "Databases": ["MySQL", "PostgreSQL", "MongoDB", "Redis", "Elasticsearch",
                  "Cassandra", "Snowflake", "BigQuery", "Redshift", "NoSQL"],
    "Cloud & DevOps": ["AWS", "Azure", "GCP", "EC2", "S3", "Lambda", "Docker",
                       "Kubernetes", "Jenkins", "Terraform", "Ansible", "Linux",
                       "CI/CD", "Git", "GitHub Actions"],
    "Tools & Other": ["Tableau", "Power BI", "Excel", "REST APIs", "GraphQL",
                      "Microservices", "Data Visualization", "A/B Testing",
                      "Unit Testing", "Agile", "Jira"],
}

CERTIFICATIONS = [
    ("AWS Certified Solutions Architect - Associate", "Amazon Web Services"),
    ("AWS Certified Developer - Associate", "Amazon Web Services"),
    ("Azure Fundamentals AZ-900", "Microsoft"),
    ("Azure Data Scientist Associate DP-100", "Microsoft"),
    ("Google Cloud Professional Data Engineer", "Google Cloud"),
    ("Google Cloud Associate Cloud Engineer", "Google Cloud"),
    ("Professional Machine Learning Engineer", "Google Cloud"),
    ("Certified Kubernetes Administrator", "CNCF"),
    ("Certified Kubernetes Application Developer", "CNCF"),
    ("Docker Certified Associate", "Docker"),
    ("TensorFlow Developer Certificate", "Google"),
    ("Tableau Desktop Specialist", "Tableau"),
    ("Power BI Data Analyst Associate", "Microsoft"),
    ("Project Management Professional PMP", "Project Management Institute"),
    ("Certified ScrumMaster", "Scrum Alliance"),
    ("Salesforce Certified Administrator", "Salesforce"),
    ("Databricks Certified Machine Learning Associate", "Databricks"),
    ("Certified Analytics Professional", "INFORMS"),
]

PROJECT_NAMES = [
    "Customer Churn Prediction", "Sales Forecasting Dashboard",
    "Resume Screening Tool", "Sentiment Analysis Engine",
    "Image Classification Pipeline", "Recommendation System",
    "Fraud Detection Model", "Chatbot Assistant", "Stock Price Predictor",
    "Healthcare Data Pipeline", "E-commerce Analytics Platform",
    "Real-time Log Analyzer", "Face Recognition System",
    "Text Summarization Tool", "Energy Consumption Forecaster",
    "Traffic Prediction Model", "Invoice Processing System",
    "Movie Recommendation Engine", "Weather Data Dashboard",
    "Speech Recognition Prototype", "Document Classification System",
    "Expense Tracker App", "Job Market Analyzer",
    "Social Media Analytics Tool", "Anomaly Detection Framework",
    "Price Optimization Engine", "Inventory Management System",
    "Learning Management Platform", "Fitness Tracking App",
    "News Aggregator", "Recipe Recommendation App", "Task Automation Bot",
    "Geospatial Mapping Tool", "Audio Transcription Service",
    "Crypto Price Tracker", "Credit Risk Scoring Model",
    "Supply Chain Optimizer", "Sports Statistics Dashboard",
    "Language Learning App", "Parking Availability Predictor",
]

AREAS = [
    "machine learning", "data engineering", "web development",
    "cloud infrastructure", "data analytics", "software development",
    "MLOps", "natural language processing", "computer vision",
    "backend systems", "distributed systems", "devops automation",
]

RESPONSIBILITY_TOPICS = [
    "data", "machine learning", "NLP", "ETL", "cloud", "CI/CD", "backend",
    "analytics", "testing", "monitoring", "API", "database",
]

RESPONSIBILITY_TEMPLATES = [
    "Worked on {topic} pipelines for production workloads",
    "Built and maintained {topic} services",
    "Developed {topic} models that improved accuracy by {pct}%",
    "Collaborated with cross-functional teams on {topic} initiatives",
    "Automated {topic} workflows and reduced manual effort by {pct}%",
    "Designed and implemented {topic} solutions for enterprise clients",
    "Optimized {topic} performance for large-scale datasets",
    "Migrated legacy systems to a modern {topic} stack",
    "Created dashboards and reports for {topic} metrics",
    "Performed code reviews and mentored interns on {topic} practices",
    "Integrated third-party {topic} APIs",
    "Wrote unit and integration tests for {topic} modules",
]

PROJECT_DESC_TEMPLATES = [
    "Built as a personal project using {t1} and {t2}.",
    "A {adj} project focused on {area}, developed end to end.",
    "Developed to explore {area} with a focus on practical deployment.",
]

MONTHS_FULL = [
    "January", "February", "March", "April", "May", "June", "July", "August",
    "September", "October", "November", "December",
]

EMAIL_DOMAINS = [
    "gmail.com", "outlook.com", "yahoo.com", "protonmail.com", "hotmail.com",
    "rediffmail.com", "mail.com", "fastmail.com", "icloud.com", "example.org",
]

# Fixed reference year used for deterministic career timelines.
REFERENCE_YEAR = 2025
