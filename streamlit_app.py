import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from datetime import date, timedelta

API_URL = "https://edf-api.orangebeach-b5cf8765.francecentral.azurecontainerapps.io"

st.set_page_config(
    page_title="EDF — Prévision consommation électrique", page_icon="⚡", layout="wide"
)

st.title("⚡ Prévision de consommation électrique")
st.caption("Basé sur le modèle ML entraîné sur les données RTE éco2mix")

# ── Sidebar ──
with st.sidebar:
    st.header("Paramètres")

    # Date max = dernière date d'entraînement + 1 an
    max_date = date.today() + timedelta(days=365)
    min_date = date(2024, 1, 1)

    start_date = st.date_input(
        "Date de début",
        value=date.today(),
        min_value=min_date,
        max_value=max_date,
    )
    end_date = st.date_input(
        "Date de fin",
        value=date.today() + timedelta(days=30),
        min_value=min_date,
        max_value=max_date,
    )

    model_name = st.selectbox(
        "Modèle",
        ["random_forest", "decision_tree", "knn", "mlp"],
        index=0,
    )

    predict_btn = st.button(
        "Lancer la prévision", type="primary", use_container_width=True
    )

# ── Contenu principal ──
if predict_btn:
    if (end_date - start_date).days > 365:
        st.error("La plage ne peut pas dépasser 365 jours.")
    elif end_date < start_date:
        st.error("La date de fin doit être après la date de début.")
    else:
        with st.spinner("Calcul des prévisions..."):
            try:
                response = requests.post(
                    f"{API_URL}/forecast",
                    json={
                        "start_date": start_date.strftime("%Y-%m-%d"),
                        "end_date": end_date.strftime("%Y-%m-%d"),
                        "model_name": model_name,
                    },
                    timeout=60,
                )
                response.raise_for_status()
                data = response.json()

                df = pd.DataFrame(data["predictions"])
                df["date"] = pd.to_datetime(df["date"])

                # ── Métriques ──
                col1, col2, col3 = st.columns(3)
                col1.metric("Nombre de jours", data["count"])
                col2.metric("R²", f"{data['r2_score']:.4f}")
                col3.metric("MAPE", f"{data['mape_percent']:.2f}%")

                # ── Graphe ──
                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(
                        x=df["date"],
                        y=df["prediction_mw"],
                        mode="lines",
                        name="Prévision",
                        line=dict(color="#E8593C", width=2),
                    )
                )
                fig.update_layout(
                    title="Prévision de consommation électrique (MW)",
                    xaxis_title="Date",
                    yaxis_title="Consommation (MW)",
                    hovermode="x unified",
                    height=450,
                )
                st.plotly_chart(fig, use_container_width=True)

                # ── Tableau ──
                st.subheader("Détail des prévisions")
                df_display = df.copy()
                df_display["date"] = df_display["date"].dt.strftime("%Y-%m-%d")
                df_display["prediction_mw"] = df_display["prediction_mw"].apply(
                    lambda x: f"{x:,.0f} MW"
                )
                df_display.columns = ["Date", "Consommation prévue"]
                st.dataframe(df_display, use_container_width=True, hide_index=True)

                # ── Export ──
                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Télécharger CSV",
                    csv,
                    f"forecast_{start_date}_{end_date}.csv",
                    "text/csv",
                )

            except requests.exceptions.ConnectionError:
                st.error(
                    "Impossible de contacter l'API. Vérifiez qu'elle est démarrée."
                )
            except Exception as e:
                st.error(f"Erreur : {e}")
else:
    st.info("Sélectionnez une plage de dates et cliquez sur **Lancer la prévision**.")
