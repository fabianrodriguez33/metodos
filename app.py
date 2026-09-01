import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import cmath
from sympy import symbols, diff, sympify, lambdify

# ==========================================
# CONFIGURACIÓN DE LA PÁGINA
# ==========================================
st.set_page_config(
    page_title="Métodos Numéricos - Raíces",
    page_icon="🧮",
    layout="wide"
)

st.title("🧮 Métodos Numéricos: Raíces de Ecuaciones y Polinomios")
st.markdown("**Curso:** Métodos Numéricos | **Docente:** Jorge Luis Manrique Plasencia")

st.sidebar.header("⚙️ Configuración Global")
sesion = st.sidebar.selectbox(
    "Selecciona la Sesión:",
    ["Sesión 3: Ecuaciones No Lineales", "Sesión 4: Raíces de Polinomios (Müller)"],
    index=1
)

st.sidebar.markdown("---")

# ==========================================
# FUNCIONES AUXILIARES (SESIÓN 3)
# ==========================================
def evaluar_funcion(func_str, x_val):
    try:
        x = symbols('x')
        expr = sympify(func_str)
        f = lambdify(x, expr, modules=['numpy', 'math'])
        return f(x_val)
    except Exception as e:
        st.error(f"Error en la función: {e}")
        return None

def punto_fijo(func_str, x0, epsilon, max_iter):
    x = symbols('x')
    g_expr = sympify(func_str)
    g = lambdify(x, g_expr, modules=['numpy', 'math'])
    g_prime_expr = diff(g_expr, x)
    g_prime = lambdify(x, g_prime_expr, modules=['numpy', 'math'])
    
    iteraciones = []
    x_n = x0
    x_n1 = x_n 
    
    for i in range(max_iter):
        try:
            x_n1 = g(x_n)
            f_x = x_n - x_n1
            error = abs(x_n1 - x_n) / abs(x_n1) if x_n1 != 0 else 0
            iteraciones.append({'Iteración': i, 'x_n': x_n, 'f(x_n)': f_x, 'Error %': error * 100})
            if error <= epsilon: break
            x_n = x_n1
        except Exception as e:
            st.error(f"Error en iteración {i}: {e}")
            break
    return pd.DataFrame(iteraciones), x_n1, g_prime(x_n1)

def newton_raphson(func_str, x0, epsilon, max_iter):
    x = symbols('x')
    f_expr = sympify(func_str)
    f = lambdify(x, f_expr, modules=['numpy', 'math'])
    f_prime_expr = diff(f_expr, x)
    f_prime = lambdify(x, f_prime_expr, modules=['numpy', 'math'])
    
    iteraciones = []
    x_n = x0
    x_n1 = x_n
    
    for i in range(max_iter):
        try:
            f_x = f(x_n)
            f_prime_x = f_prime(x_n)
            if abs(f_prime_x) < 1e-10:
                st.warning(f"⚠️ Derivada muy pequeña en iteración {i}")
                break
            x_n1 = x_n - f_x / f_prime_x
            error = abs(x_n1 - x_n) / abs(x_n1) if x_n1 != 0 else 0
            iteraciones.append({'Iteración': i, 'x_n': x_n, 'f(x_n)': f_x, 'Error %': error * 100})
            if error <= epsilon: break
            x_n = x_n1
        except Exception as e:
            st.error(f"Error en iteración {i}: {e}")
            break
    return pd.DataFrame(iteraciones), x_n1, str(f_prime_expr)

def secante(func_str, x0, x1, epsilon, max_iter):
    x = symbols('x')
    f_expr = sympify(func_str)
    f = lambdify(x, f_expr, modules=['numpy', 'math'])
    
    iteraciones = []
    x_n_1 = x0
    x_n = x1
    x_n1 = x_n
    
    for i in range(max_iter):
        try:
            f_x_n_1 = f(x_n_1)
            f_x_n = f(x_n)
            denominador = f_x_n - f_x_n_1
            if abs(denominador) < 1e-10:
                st.warning(f"⚠️ Denominador muy pequeño en iteración {i}")
                break
            x_n1 = x_n - f_x_n * (x_n - x_n_1) / denominador
            error = abs(x_n1 - x_n) / abs(x_n1) if x_n1 != 0 else 0
            iteraciones.append({'Iteración': i, 'x_n': x_n, 'f(x_n)': f_x_n, 'Error %': error * 100})
            if error <= epsilon: break
            x_n_1 = x_n
            x_n = x_n1
        except Exception as e:
            st.error(f"Error en iteración {i}: {e}")
            break
    return pd.DataFrame(iteraciones), x_n1

# ==========================================
# FUNCIONES AUXILIARES (SESIÓN 4 - MÜLLER Y TEORÍA)
# ==========================================
def evaluar_polinomio(coeffs, z):
    resultado = coeffs[0]
    for i in range(1, len(coeffs)):
        resultado = resultado * z + coeffs[i]
    return resultado

def fmt_tabla(val):
    """Formatea reales o complejos para la tabla"""
    if isinstance(val, complex) and abs(val.imag) > 1e-6:
        return f"{val.real:.5f} {'+' if val.imag >= 0 else '-'} {abs(val.imag):.5f}j"
    return f"{val.real:.5f}" if isinstance(val, complex) else f"{val:.5f}"

def fmt_error(e):
    """Imita el formato 'General' de Excel del docente: 22.2131 / 2.1E-05"""
    if e < 1e-3:
        return f"{e:.1E}"
    return f"{e:.6g}"

def muller(coeffs, z0, z1, z2, tol, max_iter):
    historial = []
    z3 = z2
    for i in range(max_iter):
        # La ventana ACTUAL de esta iteración (para la tabla del profe)
        x_i, x_i1, x_i2 = z0, z1, z2
        
        f0 = evaluar_polinomio(coeffs, z0)
        f1 = evaluar_polinomio(coeffs, z1)
        f2 = evaluar_polinomio(coeffs, z2)
        
        h0 = z1 - z0
        h1 = z2 - z1
        delta0 = (f1 - f0) / h0
        delta1 = (f2 - f1) / h1
        
        a = (delta1 - delta0) / (h1 + h0)
        b = a * h1 + delta1
        c = f2
        
        discriminante = cmath.sqrt(b**2 - 4*a*c)
        denom1 = b + discriminante
        denom2 = b - discriminante
        # Criterio de la Guía Teórica: maximizar |denominador| (evita cancelación sustractiva)
        denom = denom1 if abs(denom1) > abs(denom2) else denom2
        
        if abs(denom) < 1e-15: break
            
        z3 = z2 - (2 * c) / denom
        
        # Error relativo aproximado (%) igual que el Excel del docente.
        # La fila i=0 queda en blanco porque aún no hay x3 previo.
        ea = abs((z3 - x_i2) / z3) * 100 if (i > 0 and abs(z3) > 1e-15) else None
        
        historial.append({
            'i': i,                      # <-- ahora empieza en 0, como en clase
            'xi': fmt_tabla(x_i),
            'xi+1': fmt_tabla(x_i1),
            'xi+2': fmt_tabla(x_i2),
            'xi+3': fmt_tabla(z3),
            'Error': '' if ea is None else fmt_error(ea)
        })
        
        if ea is not None and ea < tol:
            return z3, pd.DataFrame(historial), True
        
        z0, z1, z2 = z1, z2, z3  # desplazar ventana
        
    return z3, pd.DataFrame(historial), False

def deflacion(coeffs, raiz):
    nuevos = [coeffs[0]]
    for i in range(1, len(coeffs) - 1):
        nuevos.append(coeffs[i] + raiz * nuevos[-1])
    return nuevos

def fmt(c):
    if isinstance(c, complex) and abs(c.imag) > 1e-6:
        return f"{c.real:.4f} {'+' if c.imag >= 0 else '-'} {abs(c.imag):.4f}j"
    return f"{c.real:.4f}" if isinstance(c, complex) else f"{c:.4f}"

def calcular_descartes(coeffs):
    signos = [np.sign(c) for c in coeffs if c != 0]
    cambios_pos = sum(1 for i in range(len(signos)-1) if signos[i] != signos[i+1])
    n = len(coeffs) - 1
    signos_neg = []
    for i, c in enumerate(coeffs):
        if c == 0: continue
        potencia = n - i
        signos_neg.append(-np.sign(c) if potencia % 2 != 0 else np.sign(c))
    cambios_neg = sum(1 for i in range(len(signos_neg)-1) if signos_neg[i] != signos_neg[i+1])
    return cambios_pos, cambios_neg

def calcular_lagrange(coeffs):
    if coeffs[0] < 0: coeffs = [-c for c in coeffs]
    a_n = coeffs[0]
    neg_coeffs = [abs(c) for c in coeffs if c < 0]
    if not neg_coeffs: return 0.0
    K = max(neg_coeffs)
    k = 0
    for i in range(1, len(coeffs)):
        if coeffs[i] < 0:
            k = i
            break
    if k == 0: return 1.0
    return 1 + (K / a_n)**(1/k)

def formatear_polinomio_latex(coeffs):
    terms = []
    n = len(coeffs) - 1
    for i, c in enumerate(coeffs):
        if c == 0: continue
        pot = n - i
        if pot == 0: terms.append(f"{c:+g}")
        elif pot == 1: terms.append(f"{c:+g}z")
        else: terms.append(f"{c:+g}z^{pot}")
    poly_str = " ".join(terms)
    if poly_str.startswith("+"): poly_str = poly_str[1:].strip()
    return poly_str

# ==========================================
# LÓGICA DE LA INTERFAZ
# ==========================================

if sesion == "Sesión 3: Ecuaciones No Lineales":
    st.markdown("---")
    st.subheader("📌 Sesión 3: Método del Punto Fijo, Newton-Raphson y Secante")
    
    metodo = st.sidebar.selectbox("Selecciona el método:", ["Punto Fijo", "Newton-Raphson", "Secante"], index=0)
    opcion = st.sidebar.radio("Usar problema:", ["Problemas de la guía", "Función personalizada"])
    
    if opcion == "Problemas de la guía":
        if metodo == "Punto Fijo":
            st.sidebar.info("Problema 1: Control de temperatura")
            funcion_str, x0_default, x1_default = "18 + 8*exp(-0.15*x)", 20.0, None
        elif metodo == "Newton-Raphson":
            st.sidebar.info("Problema 2: Sistema de almacenamiento")
            funcion_str, x0_default, x1_default = "x**3 - 7*x - 5", 3.0, None
        else:
            st.sidebar.info("Problema 3: Rendimiento de servidor")
            funcion_str, x0_default, x1_default = "exp(-x) - x**2 + 0.2", 0.0, 1.0
    else:
        funcion_str = st.sidebar.text_input("Ingresa la función f(x):", value="x**3 - 7*x - 5")
        x0_default, x1_default = 1.0, 2.0

    st.sidebar.subheader("📊 Parámetros")
    if metodo == "Secante":
        col1, col2 = st.sidebar.columns(2)
        with col1: x0 = st.number_input("x₀ (inicial):", value=x0_default, format="%.4f")
        with col2: x1 = st.number_input("x₁ (segundo):", value=x1_default, format="%.4f")
    else:
        x0 = st.sidebar.number_input("Valor inicial x₀:", value=x0_default, format="%.4f")
        x1 = None

    epsilon = st.sidebar.slider("Tolerancia εa (%):", 0.001, 1.0, 0.01, 0.001) / 100
    max_iter = st.sidebar.slider("Máximo de iteraciones:", 10, 100, 50, 5)

    if st.sidebar.button("🚀 Calcular", type="primary", use_container_width=True):
        st.subheader(f"📈 Resultados - Método: {metodo}")
        
        if metodo == "Punto Fijo":
            df, raiz, g_prime_val = punto_fijo(funcion_str, x0, epsilon, max_iter)
            st.success(f"✅ **Raíz encontrada:** {raiz:.6f}")
            st.info(f"📐 **Función de iteración:** g(x) = {funcion_str}")
            st.info(f"📐 **Derivada g'(x) en la raíz:** {abs(g_prime_val):.6f}")
            if abs(g_prime_val) < 1: st.success("✅ **Convergencia garantizada:** |g'(x)| < 1")
            else: st.error("❌ **No converge:** |g'(x)| ≥ 1")
        elif metodo == "Newton-Raphson":
            df, raiz, derivada = newton_raphson(funcion_str, x0, epsilon, max_iter)
            st.success(f"✅ **Raíz encontrada:** {raiz:.6f}")
            st.info(f"📐 **Función:** f(x) = {funcion_str}")
            st.info(f"📐 **Derivada:** f'(x) = {derivada}")
        else:
            df, raiz = secante(funcion_str, x0, x1, epsilon, max_iter)
            st.success(f"✅ **Raíz encontrada:** {raiz:.6f}")
            st.info(f"📐 **Función:** f(x) = {funcion_str}")
            
        st.markdown("### 📋 Tabla de Iteraciones")
        st.dataframe(df, use_container_width=True)
        
        st.markdown("### 📊 Gráfica de Convergencia")
        col1, col2 = st.columns(2)
        with col1:
            fig1, ax1 = plt.subplots(figsize=(10, 6))
            ax1.plot(df['Iteración'], df['Error %'], 'bo-', linewidth=2, markersize=8)
            ax1.set_xlabel('Iteración'); ax1.set_ylabel('Error Relativo (%)')
            ax1.set_title('Error vs Iteración', fontweight='bold'); ax1.grid(True, alpha=0.3)
            ax1.axhline(y=epsilon*100, color='r', linestyle='--', label=f'Tolerancia ({epsilon*100:.3f}%)')
            ax1.legend(); st.pyplot(fig1)
        with col2:
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            ax2.plot(df['Iteración'], df['x_n'], 'go-', linewidth=2, markersize=8)
            ax2.set_xlabel('Iteración'); ax2.set_ylabel('Valor de x')
            ax2.set_title('Convergencia de x', fontweight='bold'); ax2.grid(True, alpha=0.3)
            st.pyplot(fig2)
            
        st.markdown("### 🔍 Análisis de Resultados")
        col1, col2, col3 = st.columns(3)
        with col1: st.metric("Iteraciones", len(df))
        with col2: st.metric("Error Final", f"{df['Error %'].iloc[-1]:.6f}%")
        with col3: st.metric("Raíz Aproximada", f"{raiz:.6f}")
        
        if opcion == "Problemas de la guía":
            st.markdown("#### 📝 Interpretación Física")
            if metodo == "Punto Fijo": st.info(f"La temperatura de equilibrio es **{raiz:.2f}°C**.")
            elif metodo == "Newton-Raphson": st.info(f"El tiempo de respuesta es **{raiz:.2f} ms**.")
            else: st.info(f"El nivel de carga normalizado es **{raiz:.2f}**.")

elif sesion == "Sesión 4: Raíces de Polinomios (Müller)":
    st.markdown("---")
    st.subheader("📌 Sesión 4: Raíces de Polinomios y Estabilidad de Sistemas")
    st.markdown("Herramienta para la búsqueda de raíces (reales y complejas) mediante el Método de Müller, con deflación automática y análisis en el Plano Z.")
    
    st.sidebar.subheader("🔧 Configuración del Polinomio")
    opcion = st.sidebar.radio("Tipo de entrada:", ["Caso de Estudio: Filtro IIR", "Polinomio Personalizado"])
    
    if opcion == "Caso de Estudio: Filtro IIR":
        st.sidebar.info("Caso de Estudio (Filtro Digital):\n$D(z) = 8z^4 - 6z^3 - 3z^2 + 3z - 1$")
        coeffs_default = [8.0, -6.0, -3.0, 3.0, -1.0]
        z0_def, z1_def, z2_def = 0.0, 0.5, 1.0
        tol_def = 1e-5
    else:
        coeffs_input = st.sidebar.text_input("Coeficientes (de mayor a menor grado, separados por coma):", "1, 0, -3, 0, 2")
        try:
            coeffs_default = [float(c.strip()) for c in coeffs_input.split(",")]
        except:
            st.sidebar.error("Formato inválido. Usando ejemplo por defecto.")
            coeffs_default = [1.0, 0.0, -3.0, 0.0, 2.0]
        z0_def, z1_def, z2_def = 0.0, 0.5, 1.0
        tol_def = 1e-5

    st.sidebar.subheader("📊 Parámetros de Müller")
    col1, col2, col3 = st.sidebar.columns(3)
    with col1: z0 = st.number_input("z₀ (xi):", value=z0_def, format="%.4f")
    with col2: z1 = st.number_input("z₁ (xi+1):", value=z1_def, format="%.4f")
    with col3: z2 = st.number_input("z₂ (xi+2):", value=z2_def, format="%.4f")
    
    tol = st.sidebar.number_input("Tolerancia (ε):", value=tol_def, format="%.e")
    max_iter = st.sidebar.slider("Máx. iteraciones por raíz:", 10, 100, 50, 5)

    st.markdown("---")
    if st.sidebar.button("🚀 Ejecutar Análisis Completo", type="primary", use_container_width=True):
        coeffs_actuales = coeffs_default.copy()
        grado = len(coeffs_actuales) - 1
        raices = []
        
        st.markdown(f"#### 📥 Polinomio de entrada (Grado {grado})")
        st.latex(f"P(z) = {formatear_polinomio_latex(coeffs_actuales)} = 0")
        
        # 1. ANÁLISIS PREVIO (DINÁMICO)
        st.markdown("### 🧠 1. Delimitación Teórica (Análisis Previo)")
        st.markdown("*Antes de iterar, el sistema aplica teoremas para predecir la naturaleza de las raíces.*")
        c_pos, c_neg = calcular_descartes(coeffs_actuales)
        cota_L = calcular_lagrange(coeffs_actuales)
        
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            st.markdown("**Regla de los Signos de Descartes**")
            st.markdown(f"- Variaciones en $P(z)$: **{c_pos}** $\implies$ Raíces positivas: **{c_pos}** (o menos en cantidad par).")
            st.markdown(f"- Variaciones en $P(-z)$: **{c_neg}** $\implies$ Raíces negativas: **{c_neg}** (o menos en cantidad par).")
        with col_t2:
            st.markdown("**Cota Superior de Lagrange**")
            st.markdown(f"- Módulo máximo estimado: **$|z| \le {cota_L:.4f}$**")
            st.caption("Garantiza que todas las raíces reales positivas se encuentran dentro de este límite.")
            
        st.markdown("---")
        
        # 2. BÚSQUEDA NUMÉRICA
        st.markdown("### ⚙️ 2. Búsqueda Numérica (Müller y Deflación)")
        
        for i in range(grado):
            with st.expander(f"🔍 Paso {i+1}: Búsqueda de la Raíz {i+1} (Grado actual: {len(coeffs_actuales)-1})", expanded=(i==0)):
                if len(coeffs_actuales) == 2: 
                    raiz = -coeffs_actuales[1] / coeffs_actuales[0]
                    raices.append(raiz)
                    st.success(f"✅ Raíz directa (lineal): **{fmt(raiz)}**")
                    break
                    
                raiz, df_iter, convergio = muller(coeffs_actuales, z0, z1, z2, tol, max_iter)
                
                if convergio:
                    st.success(f"✅ Convergió en {len(df_iter)} iteraciones (i = 0 a {len(df_iter)-1}). Raíz encontrada: **{fmt(raiz)}**")
                    raices.append(raiz)
                    
                    st.markdown("**📋 Tabla de Convergencia (formato de clase: i | xi | xi+1 | xi+2 | xi+3 | Error)**")
                    st.dataframe(df_iter, use_container_width=True, hide_index=True)
                    
                    coeffs_actuales = deflacion(coeffs_actuales, raiz)
                    st.info(f"➡️ Polinomio deflacionado resultante: `{[fmt(c) for c in coeffs_actuales]}`")
                else:
                    st.error("❌ El método no convergió con los parámetros actuales.")
                    break
                    
        st.markdown("---")
        
        # 3. ANÁLISIS DE INGENIERÍA
        if len(raices) == grado:
            st.markdown("### 📡 3. Análisis de Ingeniería (Estabilidad en el Plano Z)")
            st.markdown("Para que un sistema discreto (como un filtro IIR) sea estable, todos sus polos (raíces) deben estar estrictamente dentro del círculo unitario: **$|z| < 1$**.")
            
            datos, estable = [], True
            for i, r in enumerate(raices):
                mod = abs(r)
                if mod >= 1.0: estable = False
                r_str = f"{r.real:.4f} {'+' if r.imag >= 0 else '-'} {abs(r.imag):.4f}j" if isinstance(r, complex) and abs(r.imag) > 1e-6 else f"{r.real:.4f}"
                datos.append({"Raíz (Polo)": f"$z_{i+1}$", "Valor": r_str, "Módulo |z|": f"{mod:.5f}", "Estado": "✅ Estable" if mod < 1.0 else "❌ Inestable"})
                
            st.dataframe(pd.DataFrame(datos), use_container_width=True, hide_index=True)
            
            if estable:
                st.success("🎉 **CONCLUSIÓN DEL SISTEMA:** Todos los polos están dentro del círculo unitario. El sistema es **ESTABLE**.")
            else:
                st.error("⚠️ **CONCLUSIÓN DEL SISTEMA:** Al menos un polo está fuera o en el borde del círculo unitario. El sistema es **INESTABLE**.")
            
            st.markdown("#### 🌍 Mapa de Polos en el Plano Z")
            fig, ax = plt.subplots(figsize=(6, 6))
            theta = np.linspace(0, 2*np.pi, 100)
            ax.plot(np.cos(theta), np.sin(theta), 'k--', label='Círculo Unitario (|z|=1)')
            ax.fill(np.cos(theta), np.sin(theta), color='green', alpha=0.1)
            
            for i, r in enumerate(raices):
                ax.plot(r.real, r.imag, 'rx', markersize=10, markeredgewidth=2)
                ax.text(r.real + 0.05, r.imag + 0.05, f'$z_{i+1}$', fontsize=12, color='red', fontweight='bold')
                
            ax.set_xlabel("Parte Real"); ax.set_ylabel("Parte Imaginaria")
            ax.set_title("Diagrama de Polos"); ax.grid(True, alpha=0.3)
            ax.set_aspect('equal'); ax.legend()
            st.pyplot(fig)

st.markdown("---")
st.markdown("**Curso:** Métodos Numéricos | Docente: Jorge Luis Manrique Plasencia")
