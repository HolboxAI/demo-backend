import os
import requests
from fastapi import HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import jwt, jwk
from jose.exceptions import JWTError
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Clerk environment variables
clerk_issuer = os.getenv("CLERK_ISSUER")
clerk_jwks_url = os.getenv("CLERK_JWKS_URL")

# Setup for Bearer Token Authentication
security = HTTPBearer()

# Function to get JWKS (JSON Web Key Set) from Clerk
def get_jwks():
    response = requests.get(clerk_jwks_url)
    if response.status_code != 200:
        raise HTTPException(status_code=500, detail="Unable to fetch JWKS from Clerk")
    return response.json()

# Function to get the public key for the JWT token (based on 'kid' from token header)
def get_public_key(kid):
    jwks = get_jwks()
    for key in jwks['keys']:
        if key['kid'] == kid:
            return jwk.construct(key)
    raise HTTPException(status_code=401, detail="Invalid token")

# Function to decode the JWT token and verify its validity
def decode_token(token: str):
    try:
        # Get the 'kid' (key id) from the token's unverified header
        headers = jwt.get_unverified_headers(token)
        kid = headers['kid']
        
        # Get the public key for that kid
        public_key = get_public_key(kid)
        
        # Decode the token using the public key and verify it
        payload = jwt.decode(
            token,
            public_key.to_pem().decode('utf-8'),
            algorithms=['RS256'],
            audience="your_audience",  # Replace with your audience
            issuer=clerk_issuer
        )
        
        return payload
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Dependency to handle Bearer token verification in routes
def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials  # Extract Bearer token from the request
    payload = decode_token(token)  # Decode and verify the token
    user_id = payload.get('sub')  # 'sub' is the user ID in the JWT token
    
    if not user_id:
        raise HTTPException(status_code=401, detail="User ID not found in token")

    return user_id
