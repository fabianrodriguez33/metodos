
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
        titulo_problema = "MÉTODO DE PUNTO FIJO: Control de Temperatura"
        subtitulo_problema = "Ecuación: T = 18 + 8 * exp(-0.15 * T)"
        x0_default = 20.0
        x1_default = None
    elif metodo == "Newton-Raphson":
        st.sidebar.info("Problema 2: Sistema de almacenamiento")
        funcion_str = "x**3 - 7*x - 5"
        titulo_problema = "MÉTODO DE NEWTON-RAPHSON: Dimensionamiento de Almacenamiento"
        subtitulo_problema = "Ecuación: f(t) = t^3 - 7t - 5 = 0"
        x0_default = 3.0
        x1_default = None
    else:
        st.sidebar.info("Problema 3: Rendimiento de servidor")
        funcion_str = "exp(-x) - x**2 + 0.2"
        titulo_problema = "MÉTODO DE LA SECANTE: Rendimiento del Servidor"
        subtitulo_problema = "Ecuación: f(x) = exp(-x) - x^2 + 0.2 = 0"
        x0_default = 0.0
        x1_default = 1.0
else:
    funcion_str = st.sidebar.text_input(
        "Ingresa la función f(x):",
        value="x**3 - 7*x - 5",
        help="Usa sintaxis de Python. Ejemplo: x**2 - 4, exp(-x) - x, sin(x) - x/2"
    )
    titulo_problema = f"MÉTODO: {metodo.upper()}"
    subtitulo_problema = f"Ecuación: f(x) = {funcion_str} = 0"
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

epsilon = st.sidebar.slider("Tolerancia εa (%):", 0.001, 1.0, 0.01, 0.001)
max_iter = st.sidebar.slider("Máximo de iteraciones:", 10, 100, 50, 5)


# ============================================================
# MÉTODO DEL PUNTO FIJO
# Columnas: Iteración (i) | T_n | g(T_n) | Error Relativo (%)
# Convención (igual a la tabla de referencia):
#   fila i muestra T_n = x_i,  g(T_n) = x_(i+1)
#   Error de la fila i = error de la transición que produjo x_i
#   (es decir, compara x_i contra x_(i-1)); la fila 0 no tiene error.
# ============================================================
def punto_fijo(func_str, x0, epsilon, max_iter):
    x = symbols('x')
    g_expr = sympify(func_str)
    g = lambdify(x, g_expr, modules=['numpy', 'math'])
    g_prime_expr = diff(g_expr, x)
    g_prime = lambdify(x, g_prime_expr, modules=['numpy', 'math'])

    xs = [x0]
    for i in range(max_iter):
        xs.append(g(xs[-1]))
        err = abs(xs[-1] - xs[-2]) / abs(xs[-1]) * 100 if xs[-1] != 0 else 0.0
        if err <= epsilon:
            break

    rows = []
    for i in range(len(xs) - 1):
        Tn, gTn = xs[i], xs[i + 1]
        err = np.nan if i == 0 else abs(xs[i] - xs[i - 1]) / abs(xs[i]) * 100
        rows.append({'Iteración (i)': i, 'T_n': Tn, 'g(T_n)': gTn, 'Error Relativo (%)': err})

    df = pd.DataFrame(rows)
    raiz = xs[-1]
    return df, raiz, g_prime(raiz)


# ============================================================
# MÉTODO DE NEWTON-RAPHSON
# Columnas: Iteración (i) | t_n | f(t_n) | f'(t_n) | t_(n+1) | Error Relativo (%)
# Convención: Error de la fila i = error de SU PROPIA transición
#   t_n(i) -> t_(n+1)(i); la fila 0 no tiene error mostrado.
# ============================================================
def newton_raphson(func_str, x0, epsilon, max_iter):
    x = symbols('x')
    f_expr = sympify(func_str)
    f = lambdify(x, f_expr, modules=['numpy', 'math'])
    f_prime_expr = diff(f_expr, x)
    f_prime = lambdify(x, f_prime_expr, modules=['numpy', 'math'])

    xs = [x0]
    rows = []
    for i in range(max_iter):
        tn = xs[-1]
        ft = f(tn)
        fpt = f_prime(tn)
        if abs(fpt) < 1e-12:
            st.warning(f"⚠️ Derivada muy pequeña en iteración {i}")
            break
        tn1 = tn - ft / fpt
        err = np.nan if i == 0 else abs(tn1 - tn) / abs(tn1) * 100
        rows.append({'Iteración (i)': i, 't_n': tn, 'f(t_n)': ft,
                      "f'(t_n)": fpt, 't_(n+1)': tn1, 'Error Relativo (%)': err})
        xs.append(tn1)
        err_stop = abs(tn1 - tn) / abs(tn1) * 100 if tn1 != 0 else 0.0
        if err_stop <= epsilon:
            break

    df = pd.DataFrame(rows)
    raiz = xs[-1]
    return df, raiz, str(f_prime_expr)


# ============================================================
# MÉTODO DE LA SECANTE
# Columnas: Iteración (i) | x_(n-1) | x_n | f(x_(n-1)) | f(x_n) | x_(n+1) | Error Relativo (%)
# Convención: Error de la fila i = error de SU PROPIA transición
#   x_n(i) -> x_(n+1)(i). La numeración empieza en i = 1.
# ============================================================
def secante(func_str, x0, x1, epsilon, max_iter):
    x = symbols('x')
    f_expr = sympify(func_str)
    f = lambdify(x, f_expr, modules=['numpy', 'math'])

    xs = [x0, x1]
    rows = []
    for i in range(1, max_iter + 1):
        xn_1, xn = xs[-2], xs[-1]
        fxn_1, fxn = f(xn_1), f(xn)
        denom = fxn - fxn_1
        if abs(denom) < 1e-12:
            st.warning(f"⚠️ Denominador muy pequeño en iteración {i}")
            break
        xn1 = xn - fxn * (xn - xn_1) / denom
        err = abs(xn1 - xn) / abs(xn1) * 100 if xn1 != 0 else 0.0
        rows.append({'Iteración (i)': i, 'x_(n-1)': xn_1, 'x_n': xn,
                      'f(x_(n-1))': fxn_1, 'f(x_n)': fxn, 'x_(n+1)': xn1,
                      'Error Relativo (%)': err})
        xs.append(xn1)
        if err <= epsilon:
            break

    df = pd.DataFrame(rows)
    raiz = xs[-1]
    return df, raiz


st.markdown("---")

if st.sidebar.button("🚀 Calcular", type="primary", use_container_width=True):

    st.markdown(f"## {titulo_problema}")
    st.markdown(f"*{subtitulo_problema} | Tolerancia: < {epsilon:.2f}%*")

    if metodo == "Punto Fijo":
        df, raiz, g_prime_val = punto_fijo(funcion_str, x0, epsilon, max_iter)
        col_cfg = {
            "Iteración (i)": st.column_config.NumberColumn(format="%d"),
            "T_n": st.column_config.NumberColumn(format="%.6f"),
            "g(T_n)": st.column_config.NumberColumn(format="%.6f"),
            "Error Relativo (%)": st.column_config.NumberColumn(format="%.6f"),
        }
        raiz_label = raiz
    elif metodo == "Newton-Raphson":
        df, raiz, derivada = newton_raphson(funcion_str, x0, epsilon, max_iter)
        col_cfg = {
            "Iteración (i)": st.column_config.NumberColumn(format="%d"),
            "t_n": st.column_config.NumberColumn(format="%.6f"),
            "f(t_n)": st.column_config.NumberColumn(format="%.6f"),
            "f'(t_n)": st.column_config.NumberColumn(format="%.6f"),
            "t_(n+1)": st.column_config.NumberColumn(format="%.6f"),
            "Error Relativo (%)": st.column_config.NumberColumn(format="%.6f"),
        }
        raiz_label = raiz
    else:
        df, raiz = secante(funcion_str, x0, x1, epsilon, max_iter)
        col_cfg = {
            "Iteración (i)": st.column_config.NumberColumn(format="%d"),
            "x_(n-1)": st.column_config.NumberColumn(format="%.6f"),
            "x_n": st.column_config.NumberColumn(format="%.6f"),
            "f(x_(n-1))": st.column_config.NumberColumn(format="%.6f"),
            "f(x_n)": st.column_config.NumberColumn(format="%.6f"),
            "x_(n+1)": st.column_config.NumberColumn(format="%.6f"),
            "Error Relativo (%)": st.column_config.NumberColumn(format="%.6f"),
        }
        raiz_label = raiz

    st.dataframe(df, use_container_width=True, column_config=col_cfg, hide_index=True)

    st.success(f"✅ **Raíz encontrada:** {raiz_label:.6f}")
    if metodo == "Punto Fijo":
        st.info(f"📐 **g(x)** = {funcion_str}  |  **|g'(raíz)|** = {abs(g_prime_val):.6f}"
                f"  {'✅ converge' if abs(g_prime_val) < 1 else '❌ no converge'}")
    elif metodo == "Newton-Raphson":
        st.info(f"📐 **f'(x)** = {derivada}")

    # Gráficas interactivas (Plotly, no imágenes estáticas)
    st.markdown("### 📊 Gráfica de Convergencia")
    col1, col2 = st.columns(2)

    err_col = 'Error Relativo (%)'
    x_col = {'Punto Fijo': 'T_n', 'Newton-Raphson': 't_n', 'Secante': 'x_n'}[metodo]

    with col1:
        df_err = df.dropna(subset=[err_col])
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(
            x=df_err['Iteración (i)'], y=df_err[err_col],
            mode='lines+markers', marker=dict(size=9, color='royalblue'),
            line=dict(width=2),
            hovertemplate='Iteración %{x}<br>Error = %{y:.6f} %<extra></extra>'
        ))
        fig1.add_hline(y=epsilon, line_dash="dash", line_color="red",
                        annotation_text=f"Tolerancia ({epsilon:.6f}%)")
        fig1.update_yaxes(type="log", title="Error Relativo (%) [log]")
        fig1.update_xaxes(title="Iteración", dtick=1)
        fig1.update_layout(title="Error vs Iteración", height=430)
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=df['Iteración (i)'], y=df[x_col],
            mode='lines+markers', marker=dict(size=9, color='seagreen'),
            line=dict(width=2),
            hovertemplate='Iteración %{x}<br>Valor = %{y:.6f}<extra></extra>'
        ))
        fig2.add_hline(y=raiz, line_dash="dash", line_color="orange",
                        annotation_text=f"Raíz ≈ {raiz:.6f}")
        fig2.update_xaxes(title="Iteración", dtick=1)
        fig2.update_yaxes(title="Valor")
        fig2.update_layout(title="Convergencia del valor", height=430)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("### 🔍 Análisis de Resultados")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Iteraciones", len(df))
    with col2:
        st.metric("Error Final", f"{df[err_col].iloc[-1]:.6f}%")
    with col3:
        st.metric("Raíz Aproximada", f"{raiz:.6f}")

    if opcion == "Problemas de la guía":
        st.markdown("#### 📝 Interpretación Física")
        if metodo == "Punto Fijo":
            st.info(f"La temperatura de equilibrio del centro de datos es **{raiz:.6f}°C**, "
                    f"dentro del rango óptimo para servidores (18-27°C).")
        elif metodo == "Newton-Raphson":
            st.info(f"El tiempo de respuesta del sistema de almacenamiento es **{raiz:.6f} ms**.")
        else:
            st.info(f"El nivel de carga normalizado del servidor es **{raiz:.6f}**.")

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
