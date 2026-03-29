import numpy as np
import joblib
import re
import urllib.parse as urlparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# -----------------------------
# URL FEATURE EXTRACTION
# -----------------------------
def extract_url_features(url):
    features = {}

    parsed = urlparse.urlparse(url)
    domain = parsed.netloc

    features["url_length"] = len(url)
    features["domain_length"] = len(domain)
    features["num_digits"] = sum(c.isdigit() for c in url)
    features["num_special_chars"] = len(re.findall(r'[\W_]', url))
    features["num_dots"] = url.count(".")
    features["num_hyphens"] = url.count("-")
    features["num_subdirs"] = url.count("/")
    features["has_https"] = int(url.startswith("https"))
    features["has_ip"] = int(bool(re.match(r"\d+\.\d+\.\d+\.\d+", domain)))
    features["has_at"] = int("@" in url)

    suspicious_words = ["login", "verify", "secure", "update", "bank"]
    features["suspicious_words"] = sum(word in url.lower() for word in suspicious_words)

    return list(features.values()), list(features.keys())


# -----------------------------
# GENERATE MOCK DATA
# -----------------------------
def generate_mock_url_data(n_samples=1000):
    safe_urls = ["https://google.com", "https://github.com", "https://wikipedia.org"]
    malicious_urls = [
        "http://secure-login.bank-update.com",
        "http://192.168.1.1/login",
        "http://verify-account.security-update.net"
    ]

    urls = []
    labels = []

    for _ in range(n_samples):
        if np.random.rand() > 0.5:
            url = np.random.choice(safe_urls)
            labels.append(0)
        else:
            url = np.random.choice(malicious_urls)
            labels.append(1)
        urls.append(url)

    return urls, np.array(labels)


# -----------------------------
# TRAIN MODEL
# -----------------------------
urls, labels = generate_mock_url_data()

X = []
for url in urls:
    f, _ = extract_url_features(url)
    X.append(f)

X = np.array(X)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = RandomForestClassifier(n_estimators=100)
model.fit(X_scaled, labels)

# Save model
joblib.dump(model, "url_model.pkl")
joblib.dump(scaler, "url_scaler.pkl")

print("✅ Advanced URL Model Trained Successfully")