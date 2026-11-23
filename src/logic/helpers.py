import numpy as np


def identify_high_correlations(features_by_region, threshold=0.9):
    """
    Identifica los pares de variables con una correlación mayor al umbral, calculada entre
    las regiones analizadas (cada región es una muestra).
    features_by_region: lista de diccionarios {característica: valor}, uno por región.
    Retorna una lista de tuplas (var1, var2, valor_correlacion).
    """
    if len(features_by_region) < 2:
        return []

    # Solo las características presentes en todas las regiones
    names = [k for k in features_by_region[0] if all(k in f for f in features_by_region)]
    values = np.array([[float(f[k]) for k in names] for f in features_by_region])
    # Las columnas constantes no tienen correlación definida
    varying = values.std(axis=0) > 0
    names = [n for n, keep in zip(names, varying) if keep]
    corr = np.abs(np.corrcoef(values[:, varying], rowvar=False))

    rows, cols = np.where(np.triu(corr, k=1) > threshold)
    return [(names[i], names[j], float(corr[i, j])) for i, j in zip(rows, cols)]


def remove_redundant_variables(high_corr_pairs, current_vars_to_remove):
    """
    Agrega a la lista de variables a eliminar (redundantes) aquellas que superen la alta correlación.
    """
    new_vars_to_remove = set(current_vars_to_remove)
    for v1, v2, val in high_corr_pairs:
        if v1 not in new_vars_to_remove:
            new_vars_to_remove.add(v2)
    return new_vars_to_remove
