import re
import os
import numpy as np
import pandas as pd
import yake
import streamlit as st
from datetime import datetime, timedelta
from sentence_transformers import SentenceTransformer


st.set_page_config(page_title="Application Tracker", layout="wide")

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SP_CSV = os.path.join(BASE_DIR, "selling_points.csv") #selling_points doit être dans le même dossier que app.py (avec la même structure, je mets celui dont je me suis servi) + pour ouvrir, bash: streamlit run 'chemin de app.py' (si ça marche pas, rajouter 'http://' dans l'url)
LOG_FILE = os.path.join(BASE_DIR, "applications.csv")
INTERVIEW_FILE = os.path.join(BASE_DIR, "interviews.csv")


st.markdown("""
            <style>.card {background: #ffffff;
            border: 1px solid #eee;
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 12px;
            }.score {
            font-weight: bold;
            color: #4c3fcf;
            }</style>""", 
            unsafe_allow_html=True)


@st.cache_resource
def load_model():
    return SentenceTransformer(MODEL_NAME)


def clean_text(text):
    return re.sub(r"\s+", " ", text).strip()


def build_sp_text(df):
    return [" ".join([str(r.get("title","")), str(r.get("keywords","")), str(r.get("pitch_2lines",""))]) for _, r in df.iterrows()]

def run_matching(jd_text, top_k=5):

    sp = pd.read_csv(SP_CSV, sep=";")
    model = load_model()
    jd_clean = clean_text(jd_text)
    sp_texts = build_sp_text(sp)

    sp_emb = model.encode(sp_texts, normalize_embeddings=True)
    jd_emb = model.encode([jd_clean], normalize_embeddings=True)[0]

    scores = sp_emb @ jd_emb
    idx = np.argsort(-scores)[:top_k]

    out = sp.iloc[idx].copy()
    out["score"] = (scores[idx] * 100).round(1)

    return out

def log_application(company, role, skills):

    if not os.path.exists(LOG_FILE):
        df = pd.DataFrame(columns=["date","company","role","skills","status","note"])
    else:
        df = pd.read_csv(LOG_FILE)
        if "note" not in df.columns:
            df["note"] = ""

    new_row = {"date": datetime.now().strftime("%Y-%m-%d"),
               "company": company,
               "role": role,
               "skills": ", ".join(skills),
               "status": "Applied",
               "note": ""}
    
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    df.to_csv(LOG_FILE, index=False)

    return df  



def clean_old_records(days=30, preview=True):

    if not os.path.exists(LOG_FILE):
        st.info("No applications to clean.")
        return
    
    df = pd.read_csv(LOG_FILE)

    if df.empty:
        st.info("Log is empty.")
        return

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    cutoff = datetime.now() - timedelta(days=days)
    mask = (df["date"] < cutoff) & (df["status"] == "Applied")
    to_delete = df[mask]

    if to_delete.empty:
        st.info("No records to clean.")
        return

    st.write("The following records will be cleaned:")
    st.dataframe(to_delete)

    if preview:
        st.info("Preview mode: records not deleted.")
        return

    df = df[~mask]
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    df.to_csv(LOG_FILE, index=False, encoding="utf-8-sig")
    st.success(f"{len(to_delete)} records deleted.")





def load_interviews():

    if not os.path.exists(INTERVIEW_FILE):
        return pd.DataFrame(columns=["date","time","company","role"])
    return pd.read_csv(INTERVIEW_FILE)

def save_interview(date, time, company, role):

    df = load_interviews()
    df = pd.concat([df, pd.DataFrame([{"date": str(date),
                                       "time": str(time),
                                       "company": company,
                                       "role": role
                                       }])], ignore_index=True)
    df.to_csv(INTERVIEW_FILE, index=False)





with st.sidebar:

    page = st.radio("Menu", ["Analyse job", "Applications"])

    st.markdown("---")
    st.markdown("### Interviews")

    df_iv = load_interviews()

    if not df_iv.empty:
        df_iv["date"] = pd.to_datetime(df_iv["date"])
        upcoming = df_iv.sort_values("date")
        for _, row in upcoming.iterrows():
            st.markdown(f"**{row['company']}**  \n{row['date'].strftime('%d %b')} at {row['time']}")

    st.markdown("---")
    with st.form("interview"):
        c = st.text_input("Company")
        r = st.text_input("Role")
        d = st.date_input("Date")
        t = st.time_input("Time")
        if st.form_submit_button("Add"):
            save_interview(d, t, c, r)
            st.success("Added")




if page == "Analyse job":

    st.title("Analyse job posting")
    company = st.text_input("Company")
    role = st.text_input("Role")
    jd = st.text_area("Job description", height=300)

    if st.button("Analyse"):
        st.session_state['results'] = run_matching(jd)

    if 'results' in st.session_state:

        results = st.session_state['results']
        st.subheader("Recommended skills")

        for _, row in results.iterrows():
            st.markdown(f"""<div class="card">
                        <div class="score">{row['score']:.1f}%</div>
                        <b>{row['title']}</b><br>
                        {row.get('pitch_2lines','')}
                        </div>""", 
                        unsafe_allow_html=True)

        if st.button("Save application"):

            df_app = log_application(company, role, results["title"].tolist())
            st.success("Saved")

            del st.session_state['results']

            st.dataframe(df_app[["date","company","role","skills","status","note"]], use_container_width=True)




elif page == "Applications":

    st.title("My applications")

    if not os.path.exists(LOG_FILE):
        st.info("No applications yet.")

    else:
        df_app = pd.read_csv(LOG_FILE)

        if "note" not in df_app.columns:
            df_app["note"] = ""

        st.dataframe(df_app[["date","company","role","skills","status","note"]], use_container_width=True)

        st.subheader("Update application")

        selected = st.selectbox("Select application", df_app.index, format_func=lambda i: f"{df_app.loc[i,'company']} - {df_app.loc[i,'role']}")

        status = st.selectbox("Status", ["Applied","Interview","Rejected","Offer"], index=["Applied","Interview","Rejected","Offer"].index(df_app.loc[selected,"status"]))

        note = st.text_area("Note", df_app.loc[selected,"note"], height=100)

        if st.button("Update"):

            df_app.loc[selected, "status"] = status
            df_app.loc[selected, "note"] = note
            df_app.to_csv(LOG_FILE, index=False)

            st.success("Updated")
            st.dataframe(df_app[["date","company","role","skills","status","note"]], use_container_width=True)

        st.subheader("Clean old applications")

        days = st.number_input("Days threshold", min_value=1, value=30)
        preview = st.checkbox("Preview only", value=True)

        if st.button("Clean"):
            clean_old_records(days=days, preview=preview)