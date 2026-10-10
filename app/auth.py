"""Sign in with Google: checking Google's ID token, and this server's own login tokens.

The flow, with no public address needed for the server:
1. The page shows Google's button; Google's popup signs the user in and hands the page an ID
   token, a short message signed by Google: "this is user <sub>, email ..., for app <client id>,
   valid until ...".
2. The page sends it to POST /auth/google. The server checks it itself (verify_google_token):
   the browser could send anything, so nothing in it is trusted until Google's signature, the
   audience (our client id), the issuer and the expiry all check out.
3. The server then starts its own login: a random token in a cookie, which every request
   carries from then on. Google isn't asked again until the login expires.

Users are keyed on Google's `sub`, which never changes, not on their email, which can.
"""

import hashlib
import secrets
from dataclasses import dataclass

from google.auth.transport import requests as google_requests
from google.oauth2 import id_token


class SignInError(Exception):
    """The ID token didn't check out; the message says why, for the log, not the user."""


@dataclass(frozen=True)
class GoogleIdentity:
    sub: str  # Google's permanent id for the account
    email: str
    email_verified: bool
    name: str


def verify_google_token(credential: str, client_id: str) -> GoogleIdentity:
    """Check an ID token from Google's button and return who it names.

    google-auth fetches Google's public keys and checks the signature, that the token was issued
    for this app (audience = our client id), by Google (issuer), and hasn't expired. Any failure
    raises: a forged, expired or someone-else's-app token never gets past here.
    """
    try:
        claims = id_token.verify_oauth2_token(credential, google_requests.Request(), client_id)
    except ValueError as error:  # bad signature, wrong audience or issuer, expired, malformed
        raise SignInError(str(error)) from error
    return GoogleIdentity(
        sub=claims["sub"],
        email=claims.get("email", ""),
        email_verified=bool(claims.get("email_verified", False)),
        name=claims.get("name") or claims.get("email", "").split("@")[0],
    )


def new_login_token() -> str:
    """A login token for the cookie: 32 random bytes from the operating system's secure
    source, so it can't be guessed."""
    return secrets.token_urlsafe(32)


def token_hash(token: str) -> str:
    """What the database stores instead of the token. A leaked database then holds no token
    anyone could log in with, the same reason passwords are stored hashed. A plain SHA-256 is
    enough here (unlike for passwords): the token is random and long, so there's nothing to
    guess."""
    return hashlib.sha256(token.encode()).hexdigest()
