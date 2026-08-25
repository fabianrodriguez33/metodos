import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
        x0_default = 0.75      # ajustado: f(0)*f(1) NO cambia de signo cerca de la raíz real
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
        x0 = st.number_input("x₀ (inicial):", value=x0_default, format="%.4f")
    with col2:
        x1 = st.number_input("x₁ (segundo):", value=x1_default, format="%.4f")
else:
    x0 = st.number_input("Valor inicial x₀:", value=x0_default, format="%.4f")
    x1 = None

epsilon = st.sidebar.slider("Tolerancia εa (%):", 0.001, 1.0, 0.01, 0.001) / 100
max_iter = st.sidebar.slider("Máximo de iteraciones:", 10, 100, 50, 5)


# ============================================================
# MÉTODO DEL PUNTO FIJO — CORREGIDO
# ============================================================
def punto_fijo(func_str, x0, epsilon, max_iter):
    """
    x_{n+1} = g(x_n).
    CORRECCIÓN: cuando se cumple el criterio de parada, se agrega
    una fila final con el valor CONVERGIDO (x_n1), no solo con el
    valor anterior. Así la tabla siempre coincide con la raíz reportada.
    """
    x = symbols('x')
    g_expr = sympify(func_str)
    g = lambdify(x, g_expr, modules=['numpy', 'math'])

    g_prime_expr = diff(g_expr, x)
    g_prime = lambdify(x, g_prime_expr, modules=['numpy', 'math'])

    iteraciones = []
    x_n = x0
    # Fila 0: valor inicial, sin error definido todavía
    iteraciones.append({'Iteración': 0, 'x_n': x_n,
                         'f(x_n)': x_n - g(x_n), 'Error %': np.nan})

    for i in range(1, max_iter + 1):
        try:
            x_n1 = g(x_n)
            error = abs(x_n1 - x_n) / abs(x_n1) if x_n1 != 0 else 0.0
            f_x = x_n1 - g(x_n1)  # f(x_n1) = x_n1 - g(x_n1)

            iteraciones.append({'Iteración': i, 'x_n': x_n1,
                                 'f(x_n)': f_x, 'Error %': error * 100})

            x_n = x_n1
            if error * 100 <= epsilon * 100:
                break
        except Exception as e:
            st.error(f"Error en iteración {i}: {e}")
            break

    df = pd.DataFrame(iteraciones)
    raiz = x_n
    return df, raiz, g_prime(raiz)


# ============================================================
# MÉTODO DE NEWTON-RAPHSON — CORREGIDO
# ============================================================
def newton_raphson(func_str, x0, epsilon, max_iter):
    """
    x_{n+1} = x_n - f(x_n)/f'(x_n).
    CORRECCIÓN: se agrega la fila con el valor convergido antes de
    salir del bucle, y se evita el NameError si la derivada se anula
    en la primera iteración.
    """
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
    raiz = x_n
    return df, raiz, str(f_prime_expr)


# ============================================================
# MÉTODO DE LA SECANTE — CORREGIDO
# ============================================================
def secante(func_str, x0, x1, epsilon, max_iter):
    """
    x_{n+1} = x_n - f(x_n)*(x_n - x_{n-1}) / (f(x_n) - f(x_{n-1})).
    CORRECCIÓN: se agrega la fila con el valor convergido antes de salir.
    """
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
    raiz = x_n
    return df, raiz


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

    else:  # Secante — CORREGIDO: usaba func_str (no existía)
        df, raiz = secante(funcion_str, x0, x1, epsilon, max_iter)

        st.success(f"✅ **Raíz encontrada:** {raiz:.6f}")
        st.info(f"📐 **Función:** f(x) = {funcion_str}")

    st.markdown("### 📋 Tabla de Iteraciones")
    st.dataframe(df, use_container_width=True)

    st.markdown("### 📊 Gráfica de Convergencia")

    col1, col2 = st.columns(2)

    with col1:
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        # se omite la fila 0 (error NaN) al graficar el error
        df_err = df.dropna(subset=['Error %'])
        ax1.plot(df_err['Iteración'], df_err['Error %'], 'bo-', linewidth=2, markersize=8)
        ax1.set_xlabel('Iteración', fontsize=12)
        ax1.set_ylabel('Error Relativo (%)', fontsize=12)
        ax1.set_title('Error vs Iteración', fontsize=14, fontweight='bold')
        ax1.set_yscale('log')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=epsilon * 100, color='r', linestyle='--',
                     label=f'Tolerancia ({epsilon*100:.3f}%)')
        ax1.legend()
        st.pyplot(fig1)

    with col2:
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        ax2.plot(df['Iteración'], df['x_n'], 'go-', linewidth=2, markersize=8)
        ax2.set_xlabel('Iteración', fontsize=12)
        ax2.set_ylabel('Valor de x', fontsize=12)
        ax2.set_title('Convergencia de x', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        st.pyplot(fig2)

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
            st.info(f"La temperatura de equilibrio del centro de datos es **{raiz:.2f}°C**, "
                    f"dentro del rango óptimo para servidores (18-27°C).")
        elif metodo == "Newton-Raphson":
            st.info(f"El tiempo de respuesta del sistema de almacenamiento es **{raiz:.2f} ms**, "
                    f"indicando una condición operativa eficiente.")
        else:
            st.info(f"El nivel de carga normalizado del servidor es **{raiz:.4f}**, "
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
