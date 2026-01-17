import os
from pathlib import Path
from cryptography import x509
from cryptography.x509.oid import NameOID
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import rsa
from datetime import datetime, timedelta

# -----------------------------
# Paths
# -----------------------------
secure_folder = Path("Server/Secure")
secure_folder.mkdir(parents=True, exist_ok=True)

cert_file = secure_folder / "cert.pem"
key_file = secure_folder / "key.pem"

# -----------------------------
# Check if files exist
# -----------------------------
if cert_file.exists() and key_file.exists():
    print("Certificate and key already exist.")
    exit(0)

# -----------------------------
# Generate private key
# -----------------------------
key = rsa.generate_private_key(
    public_exponent=65537,
    key_size=2048
)

# Save private key
with open(key_file, "wb") as f:
    f.write(key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.TraditionalOpenSSL,
        encryption_algorithm=serialization.NoEncryption()
    ))

# -----------------------------
# Generate self-signed certificate
# -----------------------------
subject = issuer = x509.Name([
    x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
    x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "State"),
    x509.NameAttribute(NameOID.LOCALITY_NAME, "City"),
    x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Organization"),
    x509.NameAttribute(NameOID.COMMON_NAME, "localhost"),
])

cert = x509.CertificateBuilder().subject_name(
    subject
).issuer_name(
    issuer
).public_key(
    key.public_key()
).serial_number(
    x509.random_serial_number()
).not_valid_before(
    datetime.utcnow()
).not_valid_after(
    # Certificate valid for 365 days
    datetime.utcnow() + timedelta(days=365)
).add_extension(
    x509.SubjectAlternativeName([x509.DNSName("localhost")]),
    critical=False,
).sign(key, hashes.SHA256())

# Save certificate
with open(cert_file, "wb") as f:
    f.write(cert.public_bytes(serialization.Encoding.PEM))

print(f"Certificate saved -> {cert_file}")
print(f"Private key saved -> {key_file}")
