import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import math
from sympy import symbols, diff, sympify, lambdify


st.set_page_config(
    page_title="Métodos Numéricos - Raíces de Ecuaciones",
    page_icon="🧮",
    layout="wide"
)

st.title("🧮 Raíces de Ecuaciones No Lineales")
st.markdown("**Sesión 3: Método del Punto Fijo, Newton-Raphson y Secante**")

st.sidebar.header("⚙️ Configuración")

metodo = st.sidebar.selectbox(
    "Selecciona el método:",
    ["Punto Fijo", "Newton-Raphson", "Secante"],
    index=0
)

opcion = st.sidebar.radio(
    "Usar problema:",
    ["Problemas de la guía", "Función personalizada"]
)

if opcion == "Problemas de la guía":
    if metodo == "Punto Fijo":
        st.sidebar.info("Problema 1: Control de temperatura")
        funcion_str = "18 + 8*exp(-0.15*x)"
        x0_default = 20.0
        x1_default = None
    elif metodo == "Newton-Raphson":
        st.sidebar.info("Problema 2: Sistema de almacenamiento")
        funcion_str = "x**3 - 7*x - 5"
        x0_default = 3.0
        x1_default = None
    else:
        st.sidebar.info("Problema 3: Rendimiento de servidor")
        funcion_str = "exp(-x) - x**2 + 0.2"
        x0_default = 0.75
        x1_default = 1.0
else:
    funcion_str = st.sidebar.text_input(
        "Ingresa la función f(x):",
        value="x**3 - 7*x - 5",
        help="Usa sintaxis de Python. Ejemplo: x**2 - 4, exp(-x) - x, sin(x) - x/2"
    )
    x0_default = 1.0
    x1_default = 2.0

st.sidebar.subheader("📊 Parámetros")

if metodo == "Secante":
    col1, col2 = st.sidebar.columns(2)
    with col1:
        x0 = st.number_input("x₀ (inicial):", value=x0_default, format="%.6f")
    with col2:
        x1 = st.number_input("x₁ (segundo):", value=x1_default, format="%.6f")
else:
    x0 = st.number_input("Valor inicial x₀:", value=x0_default, format="%.6f")
    x1 = None

epsilon = st.sidebar.slider("Tolerancia εa (%):", 0.001, 1.0, 0.01, 0.001) / 100
max_iter = st.sidebar.slider("Máximo de iteraciones:", 10, 100, 50, 5)


# ============================================================
# MÉTODO DEL PUNTO FIJO
# ============================================================
def punto_fijo(func_str, x0, epsilon, max_iter):
    x = symbols('x')
    g_expr = sympify(func_str)
    g = lambdify(x, g_expr, modules=['numpy', 'math'])

    g_prime_expr = diff(g_expr, x)
    g_prime = lambdify(x, g_prime_expr, modules=['numpy', 'math'])

    iteraciones = []
    x_n = x0
    iteraciones.append({'Iteración': 0, 'x_n': x_n,
                         'f(x_n)': x_n - g(x_n), 'Error %': np.nan})

    for i in range(1, max_iter + 1):
        try:
            x_n1 = g(x_n)
            error = abs(x_n1 - x_n) / abs(x_n1) if x_n1 != 0 else 0.0
            f_x = x_n1 - g(x_n1)

            iteraciones.append({'Iteración': i, 'x_n': x_n1,
                                 'f(x_n)': f_x, 'Error %': error * 100})

            x_n = x_n1
            if error * 100 <= epsilon * 100:
                break
        except Exception as e:
            st.error(f"Error en iteración {i}: {e}")
            break

    df = pd.DataFrame(iteraciones)
    return df, x_n, g_prime(x_n)


# ============================================================
# MÉTODO DE NEWTON-RAPHSON
# ============================================================
def newton_raphson(func_str, x0, epsilon, max_iter):
    x = symbols('x')
    f_expr = sympify(func_str)
    f = lambdify(x, f_expr, modules=['numpy', 'math'])

    f_prime_expr = diff(f_expr, x)
    f_prime = lambdify(x, f_prime_expr, modules=['numpy', 'math'])

    iteraciones = []
    x_n = x0
    iteraciones.append({'Iteración': 0, 'x_n': x_n,
                         'f(x_n)': f(x_n), 'Error %': np.nan})

    for i in range(1, max_iter + 1):
        try:
            f_x = f(x_n)
            f_prime_x = f_prime(x_n)

            if abs(f_prime_x) < 1e-10:
                st.warning(f"⚠️ Derivada muy pequeña en iteración {i}")
                break

            x_n1 = x_n - f_x / f_prime_x
            error = abs(x_n1 - x_n) / abs(x_n1) if x_n1 != 0 else 0.0

            iteraciones.append({'Iteración': i, 'x_n': x_n1,
                                 'f(x_n)': f(x_n1), 'Error %': error * 100})

            x_n = x_n1
            if error * 100 <= epsilon * 100:
                break
        except Exception as e:
            st.error(f"Error en iteración {i}: {e}")
            break

    df = pd.DataFrame(iteraciones)
    return df, x_n, str(f_prime_expr)


# ============================================================
# MÉTODO DE LA SECANTE
# ============================================================
def secante(func_str, x0, x1, epsilon, max_iter):
    x = symbols('x')
    f_expr = sympify(func_str)
    f = lambdify(x, f_expr, modules=['numpy', 'math'])

    iteraciones = []
    x_n_1, x_n = x0, x1
    iteraciones.append({'Iteración': 0, 'x_n': x_n, 'f(x_n)': f(x_n), 'Error %': np.nan})

    for i in range(1, max_iter + 1):
        try:
            f_x_n_1, f_x_n = f(x_n_1), f(x_n)
            denominador = f_x_n - f_x_n_1

            if abs(denominador) < 1e-10:
                st.warning(f"⚠️ Denominador muy pequeño en iteración {i}")
                break

            x_n1 = x_n - f_x_n * (x_n - x_n_1) / denominador
            error = abs(x_n1 - x_n) / abs(x_n1) if x_n1 != 0 else 0.0

            iteraciones.append({'Iteración': i, 'x_n': x_n1,
                                 'f(x_n)': f(x_n1), 'Error %': error * 100})

            x_n_1, x_n = x_n, x_n1
            if error * 100 <= epsilon * 100:
                break
        except Exception as e:
            st.error(f"Error en iteración {i}: {e}")
            break

    df = pd.DataFrame(iteraciones)
    return df, x_n


st.markdown("---")

if st.sidebar.button("🚀 Calcular", type="primary", use_container_width=True):

    st.subheader(f"📈 Resultados - Método: {metodo}")

    if metodo == "Punto Fijo":
        df, raiz, g_prime_val = punto_fijo(funcion_str, x0, epsilon, max_iter)
        st.success(f"✅ **Raíz encontrada:** {raiz:.6f}")
        st.info(f"📐 **Función de iteración:** g(x) = {funcion_str}")
        st.info(f"📐 **Derivada g'(x) en la raíz:** {abs(g_prime_val):.6f}")
        if abs(g_prime_val) < 1:
            st.success("✅ **Convergencia garantizada:** |g'(x)| < 1")
        else:
            st.error("❌ **No converge:** |g'(x)| ≥ 1")

    elif metodo == "Newton-Raphson":
        df, raiz, derivada = newton_raphson(funcion_str, x0, epsilon, max_iter)
        st.success(f"✅ **Raíz encontrada:** {raiz:.6f}")
        st.info(f"📐 **Función:** f(x) = {funcion_str}")
        st.info(f"📐 **Derivada:** f'(x) = {derivada}")

    else:  # Secante
        df, raiz = secante(funcion_str, x0, x1, epsilon, max_iter)
        st.success(f"✅ **Raíz encontrada:** {raiz:.6f}")
        st.info(f"📐 **Función:** f(x) = {funcion_str}")

    # ---------------------------------------------------------
    # TABLA CON 6 DECIMALES EXACTOS (column_config, sin redondeo
    # silencioso de pandas y sin perder precisión al mostrar)
    # ---------------------------------------------------------
    st.markdown("### 📋 Tabla de Iteraciones")
    st.dataframe(
        df,
        use_container_width=True,
        column_config={
            "Iteración": st.column_config.NumberColumn("Iteración", format="%d"),
            "x_n": st.column_config.NumberColumn("x_n", format="%.6f"),
            "f(x_n)": st.column_config.NumberColumn("f(x_n)", format="%.6e"),
            "Error %": st.column_config.NumberColumn("Error %", format="%.6f"),
        },
        hide_index=True
    )

    # ---------------------------------------------------------
    # GRÁFICAS INTERACTIVAS CON PLOTLY (NO son imágenes estáticas)
    # Al pasar el mouse sobre cada punto se ve el valor EXACTO
    # con 6 decimales, sin depender de la resolución de un PNG.
    # ---------------------------------------------------------
    st.markdown("### 📊 Gráfica de Convergencia (interactiva)")

    col1, col2 = st.columns(2)

    with col1:
        df_err = df.dropna(subset=['Error %'])
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(
            x=df_err['Iteración'], y=df_err['Error %'],
            mode='lines+markers',
            marker=dict(size=9, color='royalblue'),
            line=dict(width=2),
            name='Error %',
            hovertemplate='Iteración %{x}<br>Error = %{y:.6f} %<extra></extra>'
        ))
        fig1.add_hline(
            y=epsilon * 100, line_dash="dash", line_color="red",
            annotation_text=f"Tolerancia ({epsilon*100:.6f}%)"
        )
        fig1.update_yaxes(type="log", title="Error Relativo (%) [escala log]")
        fig1.update_xaxes(title="Iteración", dtick=1)
        fig1.update_layout(title="Error vs Iteración", height=450)
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=df['Iteración'], y=df['x_n'],
            mode='lines+markers',
            marker=dict(size=9, color='seagreen'),
            line=dict(width=2),
            name='x_n',
            hovertemplate='Iteración %{x}<br>x_n = %{y:.6f}<extra></extra>'
        ))
        fig2.add_hline(
            y=raiz, line_dash="dash", line_color="orange",
            annotation_text=f"Raíz ≈ {raiz:.6f}"
        )
        fig2.update_xaxes(title="Iteración", dtick=1)
        fig2.update_yaxes(title="Valor de x")
        fig2.update_layout(title="Convergencia de x", height=450)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("### 🔍 Análisis de Resultados")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Iteraciones", len(df) - 1)
    with col2:
        st.metric("Error Final", f"{df['Error %'].iloc[-1]:.6f}%")
    with col3:
        st.metric("Raíz Aproximada", f"{raiz:.6f}")

    if opcion == "Problemas de la guía":
        st.markdown("#### 📝 Interpretación Física")
        if metodo == "Punto Fijo":
            st.info(f"La temperatura de equilibrio del centro de datos es **{raiz:.6f}°C**, "
                    f"dentro del rango óptimo para servidores (18-27°C).")
        elif metodo == "Newton-Raphson":
            st.info(f"El tiempo de respuesta del sistema de almacenamiento es **{raiz:.6f} ms**, "
                    f"indicando una condición operativa eficiente.")
        else:
            st.info(f"El nivel de carga normalizado del servidor es **{raiz:.6f}**, "
                    f"representando el punto óptimo de operación.")

st.markdown("---")
st.markdown("""
### ℹ️ Información de los Métodos

| Método | Orden de Convergencia | Requiere Derivada | Velocidad |
|--------|----------------------|-------------------|-----------|
| Punto Fijo | Lineal (1) | No | Lenta |
| Newton-Raphson | Cuadrática (2) | Sí | Muy rápida |
| Secante | Superlineal (1.618) | No | Rápida |
""")

st.markdown("---")
st.markdown("**Curso:** Métodos Numéricos | **Sesión 3** | Docente: Jorge Luis Manrique Plasencia")
