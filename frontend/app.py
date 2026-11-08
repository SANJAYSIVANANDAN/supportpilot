import os
import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://api:8000")
KEYCLOAK_URL = os.environ.get("KEYCLOAK_URL", "http://keycloak:8080")
KEYCLOAK_REALM = os.environ.get("KEYCLOAK_REALM", "support")
KEYCLOAK_CLIENT_ID = os.environ.get("KEYCLOAK_CLIENT_ID", "support-frontend")
KEYCLOAK_CLIENT_SECRET = os.environ.get("KEYCLOAK_CLIENT_SECRET", "changeme-frontend")

st.set_page_config(page_title="AI Support Agent", page_icon="🎧", layout="centered")
st.title("AI Support Agent")


def get_token(username: str, password: str) -> str | None:
    token_url = f"{KEYCLOAK_URL}/realms/{KEYCLOAK_REALM}/protocol/openid-connect/token"
    try:
        resp = requests.post(
            token_url,
            data={
                "grant_type": "password",
                "client_id": KEYCLOAK_CLIENT_ID,
                "client_secret": KEYCLOAK_CLIENT_SECRET,
                "username": username,
                "password": password,
            },
            timeout=10,
        )
        if resp.status_code == 200:
            return resp.json()["access_token"]
        st.error(f"Login failed: {resp.json().get('error_description', 'Unknown error')}")
    except requests.RequestException as e:
        st.error(f"Could not reach auth server: {e}")
    return None


if "token" not in st.session_state:
    st.session_state.token = None
if "thread_id" not in st.session_state:
    st.session_state.thread_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []

if not st.session_state.token:
    st.subheader("Login")
    with st.form("login"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Sign in")
        if submitted and username and password:
            token = get_token(username, password)
            if token:
                st.session_state.token = token
                st.rerun()
else:
    col1, col2 = st.columns([4, 1])
    with col2:
        if st.button("Logout"):
            st.session_state.token = None
            st.session_state.thread_id = None
            st.session_state.messages = []
            st.rerun()
    with col1:
        if st.button("New conversation"):
            st.session_state.thread_id = None
            st.session_state.messages = []
            st.rerun()

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("How can I help you?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    resp = requests.post(
                        f"{API_URL}/chat",
                        json={
                            "message": prompt,
                            "thread_id": st.session_state.thread_id,
                        },
                        headers={"Authorization": f"Bearer {st.session_state.token}"},
                        timeout=120,
                    )
                    if resp.status_code == 401:
                        st.error("Session expired. Please log in again.")
                        st.session_state.token = None
                        st.rerun()
                    resp.raise_for_status()
                    data = resp.json()
                    reply = data["reply"]
                    st.session_state.thread_id = data["thread_id"]
                    st.markdown(reply)
                    st.caption(f"Latency: {data['latency_ms']:.0f} ms")
                    st.session_state.messages.append({"role": "assistant", "content": reply})
                except requests.RequestException as e:
                    st.error(f"Error: {e}")
