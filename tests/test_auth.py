"""Tests for Google sign-in and per-user data, with Google's token check replaced by a fake.

The fake knows two users, "token-a" and "token-b"; anything else is a bad token. The analysis,
rendering and video probe are faked as in test_app.py, so these tests are about who may see what.
"""

from contextlib import contextmanager
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import db as store
from app import main
from app.auth import GoogleIdentity, SignInError, token_hash
from pipeline.models import Rep, SessionResult, VideoInfo

INFO = VideoInfo(path="uploaded.mp4", fps=30.0, width=640, height=360, n_frames=300)
REP = Rep(1, 0, 30, 60, 85.0, 4.0, 170.0, 2.0, 1.0, 0.0, 0.0, 28, 30, 59, 40, 50)
PEOPLE = {
    "token-a": GoogleIdentity("sub-a", "a@example.com", True, "Avi"),
    "token-b": GoogleIdentity("sub-b", "b@example.com", True, "Bea"),
    "token-unverified": GoogleIdentity("sub-c", "c@example.com", False, "Cal"),
}


def fake_verify(credential: str, client_id: str) -> GoogleIdentity:
    assert client_id == "test-client-id"  # the server checks tokens against its own client id
    if credential not in PEOPLE:
        raise SignInError("bad signature")
    return PEOPLE[credential]


@pytest.fixture
def server(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Sign-in on, a temporary database and upload folder, and a fake analysis."""
    monkeypatch.setattr(main, "GOOGLE_CLIENT_ID", "test-client-id")
    monkeypatch.setattr(main, "ALLOWED_EMAILS", None)  # "*": any Google account
    monkeypatch.setattr(main, "verify_google_token", fake_verify)
    monkeypatch.setattr(main, "UPLOADS", tmp_path)
    monkeypatch.setattr(main, "DB_PATH", tmp_path / "history.db")
    monkeypatch.setattr(main, "probe_video", lambda path: INFO)
    monkeypatch.setattr(main, "analyze", lambda path, config: SessionResult(INFO, [REP], [], []))

    def fake_annotate(result, titles, write):
        write(b"video")
        return {1: 0.0}

    @contextmanager
    def file_writer(out, video):
        yield out.write_bytes

    monkeypatch.setattr(main, "annotate", fake_annotate)
    monkeypatch.setattr(main, "h264_writer", file_writer)
    return tmp_path


def signed_in(credential: str) -> TestClient:
    """A browser of its own (its own cookies), signed in as the token's user."""
    browser = TestClient(main.app)
    response = browser.post("/auth/google", json={"credential": credential})
    assert response.status_code == 200, response.text
    return browser


def upload(browser: TestClient, profile: str | None = None) -> dict:
    data = {"profile": profile} if profile else {}
    response = browser.post(
        "/analyze", files={"video": ("set.mp4", b"bytes", "video/mp4")}, data=data
    )
    assert response.status_code == 200, response.text
    return response.json()


def test_signing_in_sets_a_cookie_scripts_and_other_sites_cant_use(server: Path):
    browser = TestClient(main.app)
    response = browser.post("/auth/google", json={"credential": "token-a"})
    assert response.json() == {"name": "Avi", "email": "a@example.com"}
    cookie = response.headers["set-cookie"].lower()
    assert "httponly" in cookie and "samesite=lax" in cookie
    assert browser.get("/me").json()["name"] == "Avi"


def test_the_database_keeps_a_hash_of_the_login_token_not_the_token(server: Path):
    browser = signed_in("token-a")
    token = browser.cookies[main.LOGIN_COOKIE]
    db = store.connect(server / "history.db")
    stored = [row["token_hash"] for row in db.execute("SELECT token_hash FROM logins")]
    db.close()
    assert stored == [token_hash(token)] and token not in stored


@pytest.mark.parametrize("path", ["/me", "/sessions", "/progress"])
def test_without_signing_in_nothing_is_served(server: Path, path: str):
    assert TestClient(main.app).get(path).status_code == 401


def test_without_signing_in_nothing_is_analysed(server: Path):
    # A request without the file gets 422 first: FastAPI checks the request's shape before the
    # endpoint runs. With the file, as a browser sends it, the sign-in check refuses it.
    files = {"video": ("set.mp4", b"bytes", "video/mp4")}
    assert TestClient(main.app).post("/analyze", files=files).status_code == 401
    assert list(server.glob("*.mp4")) == []  # nothing was saved or rendered


def test_a_token_google_didnt_sign_is_refused(server: Path):
    response = TestClient(main.app).post("/auth/google", json={"credential": "forged"})
    assert response.status_code == 401


def test_sign_in_must_be_json_so_other_sites_cant_post_it(server: Path):
    """An HTML form on another site can only send form data; JSON from another site needs this
    server's permission first, which it never gives."""
    response = TestClient(main.app).post("/auth/google", data={"credential": "token-a"})
    assert response.status_code == 422


def test_the_setting_reads_a_list_or_star():
    assert main.allowed_emails(" A@x.com, b@y.com ,") == {"a@x.com", "b@y.com"}
    assert main.allowed_emails("*") is None  # any Google account, said explicitly
    assert main.allowed_emails("") == set()  # nobody


def test_with_no_list_nobody_signs_in_and_the_server_wont_start(
    server: Path, monkeypatch: pytest.MonkeyPatch
):
    """Deny by default: forgetting the setting must not mean "everyone"."""
    monkeypatch.setattr(main, "ALLOWED_EMAILS", set())
    refused = TestClient(main.app).post("/auth/google", json={"credential": "token-a"})
    assert refused.status_code == 403
    with pytest.raises(RuntimeError, match="ALLOWED_EMAILS"), TestClient(main.app):
        pass  # entering the client runs the startup checks


def test_the_email_list_limits_who_signs_in(server: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(main, "ALLOWED_EMAILS", {"a@example.com", "c@example.com"})
    assert signed_in("token-a").get("/me").status_code == 200
    outsider = TestClient(main.app).post("/auth/google", json={"credential": "token-b"})
    assert outsider.status_code == 403
    # on the list, but Google hasn't confirmed the address is theirs
    unverified = TestClient(main.app).post("/auth/google", json={"credential": "token-unverified"})
    assert unverified.status_code == 403


def test_each_user_sees_only_their_own_sessions_and_videos(server: Path):
    avi, bea = signed_in("token-a"), signed_in("token-b")
    report = upload(avi)
    session, video = report["session_id"], report["video_url"]

    assert avi.get(f"/sessions/{session}").status_code == 200
    assert avi.get(video).status_code == 200
    # someone else's session and video don't exist, as far as Bea can tell
    assert bea.get(f"/sessions/{session}").status_code == 404
    assert bea.get(video).status_code == 404
    assert bea.get("/sessions").json() == []
    assert [s["id"] for s in avi.get("/sessions").json()] == [session]


def test_naming_someone_elses_profile_changes_nothing(server: Path):
    """With sign-in on, the request doesn't choose whose data it is: the cookie does."""
    avi, bea = signed_in("token-a"), signed_in("token-b")
    upload(avi)
    assert bea.get("/sessions", params={"profile": "google:sub-a"}).json() == []
    upload(bea, profile="google:sub-a")  # trying to write into Avi's history
    assert len(avi.get("/sessions").json()) == 1


def test_signing_out_ends_the_login_on_the_server(server: Path):
    browser = signed_in("token-a")
    token = browser.cookies[main.LOGIN_COOKIE]
    assert browser.post("/auth/logout").json() == {"signed_out": True}
    assert browser.get("/me").status_code == 401
    # a copy of the old cookie, kept by someone, no longer works either
    copy = TestClient(main.app, cookies={main.LOGIN_COOKIE: token})
    assert copy.get("/me").status_code == 401


def test_an_expired_login_is_refused(server: Path):
    browser = signed_in("token-a")
    db = store.connect(server / "history.db")
    with db:
        db.execute("UPDATE logins SET expires_at = '2000-01-01T00:00:00+00:00'")
    db.close()
    assert browser.get("/me").status_code == 401


def test_profiles_are_off_when_sign_in_is_on(server: Path):
    browser = signed_in("token-a")
    assert browser.get("/profiles").status_code == 404
    assert browser.post("/profiles", data={"profile": "Tal"}).status_code == 404


def test_the_pages_learn_the_mode_from_config_js(server: Path, monkeypatch: pytest.MonkeyPatch):
    assert "test-client-id" in TestClient(main.app).get("/config.js").text
    monkeypatch.setattr(main, "GOOGLE_CLIENT_ID", None)
    assert '"googleClientId": null' in TestClient(main.app).get("/config.js").text
