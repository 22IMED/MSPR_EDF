import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from datetime import date, timedelta

API_URL = "https://edf-api.orangebeach-b5cf8765.francecentral.azurecontainerapps.io"

st.set_page_config(
    page_title="EDF — Prévision consommation électrique",
    page_icon="⚡",
    layout="wide",
)

st.title("⚡ Prévision de consommation électrique")
st.caption("Basé sur le modèle ML entraîné sur les données RTE éco2mix")


# ── Chargement dynamique des modèles ──
@st.cache_data(ttl=300)
def get_available_models():
    try:
        r = requests.get(f"{API_URL}/models", timeout=10)
        return [m["model_name"] for m in r.json()["available_models"]]
    except Exception:
        return ["random_forest"]


available_models = get_available_models()

# ── Tabs ──
tab1, tab2 = st.tabs(["🔍 Prédiction simple", "📈 Prévision dans le temps"])

# ════════════════════════════════════════════════════════
# TAB 1 — Prédiction simple
# ════════════════════════════════════════════════════════
with tab1:
    st.subheader("Prédiction pour une date donnée")

    with st.form("predict_form"):
        col1, col2 = st.columns(2)

        with col1:
            pred_date = st.date_input(
                "Date de prédiction",
                value=date.today(),
                min_value=date(2020, 1, 1),
                max_value=date.today() + timedelta(days=365),
            )
            prevision_j1 = st.number_input(
                "Prévision RTE J-1 (MW)",
                min_value=0.0,
                value=55000.0,
                step=500.0,
                help="Prévision de consommation de la veille fournie par RTE",
            )
            lag_1 = st.number_input(
                "Consommation J-1 (MW)",
                min_value=0.0,
                value=54000.0,
                step=500.0,
                help="Consommation réelle de la veille",
            )
            lag_7 = st.number_input(
                "Consommation J-7 (MW)",
                min_value=0.0,
                value=53000.0,
                step=500.0,
                help="Consommation réelle il y a 7 jours",
            )

        with col2:
            st.markdown("**Énergies (optionnel)**")
            nucleaire = st.number_input(
                "Nucléaire (MW)", min_value=0.0, value=40000.0, step=500.0
            )
            eolien = st.number_input(
                "Éolien (MW)", min_value=0.0, value=5000.0, step=100.0
            )
            solaire = st.number_input(
                "Solaire (MW)", min_value=0.0, value=3000.0, step=100.0
            )
            hydraulique = st.number_input(
                "Hydraulique (MW)", min_value=0.0, value=8000.0, step=100.0
            )
            gaz = st.number_input("Gaz (MW)", min_value=0.0, value=6000.0, step=100.0)
            taux_co2 = st.number_input(
                "Taux CO2 (g/kWh)", min_value=0.0, value=50.0, step=1.0
            )

        model_name_1 = st.selectbox("Modèle", available_models, key="model_tab1")
        submitted = st.form_submit_button(
            "Prédire", type="primary", use_container_width=True
        )

    if submitted:
        with st.spinner("Calcul en cours..."):
            try:
                response = requests.post(
                    f"{API_URL}/predict",
                    json={
                        "date": pred_date.strftime("%Y-%m-%d"),
                        "prevision_j1": prevision_j1,
                        "lag_1": lag_1,
                        "lag_7": lag_7,
                        "nucleaire": nucleaire,
                        "eolien": eolien,
                        "solaire": solaire,
                        "hydraulique": hydraulique,
                        "gaz": gaz,
                        "taux_co2": taux_co2,
                        "model_name": model_name_1,
                    },
                    timeout=30,
                )
                response.raise_for_status()
                data = response.json()

                prediction = data["prediction_mw"]

                st.success(
                    f"Prédiction calculée avec succès — modèle : **{data['model_name']}**"
                )

                col_a, col_b, col_c, col_d = st.columns(4)
                col_a.metric("Consommation prédite", f"{prediction:,.0f} MW")
                col_b.metric("R²", f"{data['r2_score']:.4f}")
                col_c.metric("MAPE", f"{data['mape_percent']:.2f}%")
                col_d.metric("Latence", f"{data['latency_ms']:.0f} ms")

                # ── Suggestion utilisateur ──
                st.divider()
                st.markdown("#### Votre estimation")
                st.caption(
                    "Si la valeur prédite ne vous semble pas correcte, vous pouvez saisir votre propre estimation."
                )

                col_s1, col_s2 = st.columns([2, 1])
                with col_s1:
                    user_value = st.number_input(
                        "Votre estimation (MW)",
                        min_value=0.0,
                        value=float(round(prediction)),
                        step=500.0,
                        key="user_estimate",
                    )
                with col_s2:
                    diff = user_value - prediction
                    diff_pct = (diff / prediction * 100) if prediction else 0
                    st.metric(
                        "Écart avec le modèle",
                        f"{diff:+,.0f} MW",
                        delta=f"{diff_pct:+.1f}%",
                        delta_color="inverse",
                    )

                st.info(
                    "ℹ️ Votre estimation n'est pas sauvegardée — elle sert uniquement à comparer avec la prédiction du modèle."
                )

            except requests.exceptions.ConnectionError:
                st.error("Impossible de contacter l'API.")
            except Exception as e:
                st.error(f"Erreur : {e}")

# ════════════════════════════════════════════════════════
# TAB 2 — Forecast dans le temps
# ════════════════════════════════════════════════════════
with tab2:
    st.subheader("Prévision sur une plage de dates")

    with st.sidebar:
        st.header("Paramètres forecast")
        max_date = date.today() + timedelta(days=365)
        min_date = date(2024, 1, 1)

        start_date = st.date_input(
            "Date de début",
            value=date.today(),
            min_value=min_date,
            max_value=max_date,
            key="start",
        )
        end_date = st.date_input(
            "Date de fin",
            value=date.today() + timedelta(days=30),
            min_value=min_date,
            max_value=max_date,
            key="end",
        )
        model_name_2 = st.selectbox("Modèle", available_models, key="model_tab2")
        forecast_btn = st.button(
            "Lancer la prévision", type="primary", use_container_width=True
        )

    if forecast_btn:
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
                            "model_name": model_name_2,
                        },
                        timeout=60,
                    )
                    response.raise_for_status()
                    data = response.json()

                    df = pd.DataFrame(data["predictions"])
                    df["date"] = pd.to_datetime(df["date"])

                    # ── Métriques ──
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Nombre de jours", data["count"])
                    col2.metric("R²", f"{data['r2_score']:.4f}")
                    col3.metric("MAPE", f"{data['mape_percent']:.2f}%")
                    col4.metric(
                        "Moy. prédite",
                        f"{df['prediction_mw'].mean():,.0f} MW",
                    )

                    # ── Graphe ──
                    fig = go.Figure()
                    fig.add_trace(
                        go.Scatter(
                            x=df["date"],
                            y=df["prediction_mw"],
                            mode="lines",
                            name="Prévision",
                            line=dict(color="#E8593C", width=2),
                            fill="tozeroy",
                            fillcolor="rgba(232,89,60,0.08)",
                        )
                    )
                    fig.update_layout(
                        title="Prévision de consommation électrique (MW)",
                        xaxis_title="Date",
                        yaxis_title="Consommation (MW)",
                        hovermode="x unified",
                        height=450,
                        margin=dict(l=0, r=0, t=40, b=0),
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # ── Suggestion utilisateur ──
                    st.divider()
                    st.markdown("#### Ajuster une valeur")
                    st.caption(
                        "Sélectionnez une date pour comparer la prédiction avec votre estimation."
                    )

                    col_u1, col_u2, col_u3 = st.columns(3)
                    with col_u1:
                        selected_date = st.selectbox(
                            "Date",
                            df["date"].dt.strftime("%Y-%m-%d").tolist(),
                            key="selected_date",
                        )
                    model_val = df[df["date"] == pd.Timestamp(selected_date)][
                        "prediction_mw"
                    ].values[0]
                    with col_u2:
                        st.metric("Valeur modèle", f"{model_val:,.0f} MW")
                    with col_u3:
                        user_val = st.number_input(
                            "Votre estimation (MW)",
                            min_value=0.0,
                            value=float(round(model_val)),
                            step=500.0,
                            key="user_forecast_val",
                        )

                    diff = user_val - model_val
                    diff_pct = (diff / model_val * 100) if model_val else 0
                    if abs(diff) > 0:
                        st.info(
                            f"Écart : **{diff:+,.0f} MW** ({diff_pct:+.1f}%) — "
                            "cette valeur n'est pas sauvegardée."
                        )

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
                        use_container_width=True,
                    )

                except requests.exceptions.ConnectionError:
                    st.error("Impossible de contacter l'API.")
                except Exception as e:
                    st.error(f"Erreur : {e}")
    else:
        st.info(
            "Sélectionnez une plage de dates dans la sidebar et cliquez sur **Lancer la prévision**."
        )
