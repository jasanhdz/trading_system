def last_closed_and_series(cs):
    """
    Devuelve (L_cerrada, serie_sin_formacion).
    Si la última vela no está cerrada (attr is_closed==False), usa la previa.
    Si no existe el atributo, asumimos que viene cerrada.
    """
    if not cs:
        raise ValueError("No candles")
    L = cs[-1]
    if getattr(L, "is_closed", True):
        return L, cs
    if len(cs) < 2:
        return L, cs
    return cs[-2], cs[:-1]