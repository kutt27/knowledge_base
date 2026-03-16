import secrets

random_bytes = secrets.token_bytes(32)
print(random_bytes.hex())
