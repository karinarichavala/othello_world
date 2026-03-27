"""
Board State Properties (BSP) Identifier for Othello

Este módulo implementa un sistema de 198 BSPs para identificar el estado
completo de un tablero de Othello desde la perspectiva del jugador actual.

BSPs incluidas:
- 192 BSPs de casillas (64 casillas × 3 estados: vacía, mía, oponente)
- 6 BSPs especiales (inicial, terminada, gané, perdí, empate, en_curso)

Referencia:
"Measuring Progress in Dictionary Learning for Language Model Interpretability 
with Board Game Models" (Karvonen et al., NeurIPS 2024)
"""

import numpy as np


def identificador(tablero, color_jugador):
    """
    Genera las 198 Board State Properties (BSPs) para un tablero de Othello.
    
    Args:
        tablero (np.ndarray): Array 8x8 representando el tablero.
            - 0: casilla vacía
            - 1: ficha negra (estándar de OthelloBoardState)
            - -1: ficha blanca (estándar de OthelloBoardState)
        color_jugador (int): Color del jugador actual (1 para negras, -1 para blancas)
    
    Returns:
        dict[str, bool]: Diccionario con las 198 BSPs. Claves en formato:
            - 'BSP{casilla}{estado}': Para casillas (ej: 'BSPA10', 'BSPA11', 'BSPA12')
            - 'BSP_INICIAL', 'BSP_TERMINADA', etc.: Para estados especiales
    
    Propiedades:
        - Cada casilla tiene EXACTAMENTE 1 BSP activa (one-hot encoding)
        - Total: 192 BSPs de casillas + 6 BSPs especiales = 198 BSPs
    
    Ejemplo:
        >>> tablero = np.zeros((8, 8))
        >>> tablero[3, 4] = 1; tablero[4, 3] = 1  # Negras
        >>> tablero[3, 3] = -1; tablero[4, 4] = -1  # Blancas
        >>> bsps = identificador(tablero, color_jugador=1)  # Perspectiva de negras
        >>> bsps['BSPD40']  # D4 vacía
        False
        >>> bsps['BSP_INICIAL']
        True
    """
    bsps = {}
    
    # Mapeo de índices a coordenadas de ajedrez
    filas = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    columnas = ['1', '2', '3', '4', '5', '6', '7', '8']
    
    # 1. BSPs de casillas (192 BSPs)
    for i in range(8):
        for j in range(8):
            casilla = filas[i] + columnas[j]
            valor = tablero[i, j]
            
            # Estado 0: vacía
            bsps[f'BSP{casilla}0'] = (valor == 0)
            
            # Estado 1: ficha mía
            bsps[f'BSP{casilla}1'] = (valor == color_jugador)
            
            # Estado 2: ficha del oponente
            bsps[f'BSP{casilla}2'] = (valor == -color_jugador)

    # 2. BSPs absolutas (128 BSPs)
    for i in range(8):
        for j in range(8):
            casilla = filas[i] + columnas[j]
            valor = tablero[i, j]
            bsps[f'BSP{casilla}B'] = (valor == 1)
            bsps[f'BSP{casilla}W'] = (valor == -1)
    
    # 3. BSPs especiales (6 BSPs)
    # Contar fichas
    total_fichas = np.sum(tablero != 0)
    fichas_mias = np.sum(tablero == color_jugador)
    fichas_oponente = np.sum(tablero == -color_jugador)
    
    # BSP_INICIAL: configuración inicial (4 fichas centrales)
    configuracion_inicial = (
        tablero[3, 3] != 0 and tablero[3, 4] != 0 and
        tablero[4, 3] != 0 and tablero[4, 4] != 0 and
        total_fichas == 4
    )
    bsps['BSP_INICIAL'] = configuracion_inicial
    
    # BSP_TERMINADA: tablero lleno o sin movimientos válidos
    tablero_lleno = (total_fichas == 64)
    bsps['BSP_TERMINADA'] = tablero_lleno  # Simplificado
    
    # BSP_GANE, BSP_PERDI, BSP_EMPATE (solo relevantes si terminada)
    if bsps['BSP_TERMINADA']:
        bsps['BSP_GANE'] = (fichas_mias > fichas_oponente)
        bsps['BSP_PERDI'] = (fichas_mias < fichas_oponente)
        bsps['BSP_EMPATE'] = (fichas_mias == fichas_oponente)
    else:
        bsps['BSP_GANE'] = False
        bsps['BSP_PERDI'] = False
        bsps['BSP_EMPATE'] = False
    
    # BSP_EN_CURSO: juego no terminado
    bsps['BSP_EN_CURSO'] = not bsps['BSP_TERMINADA']

    return bsps


def imprimir_tablero(tablero):
    """
    Imprime el tablero en formato visual.
    
    Args:
        tablero (np.ndarray): Array 8x8 del tablero
    """
    print("   1 2 3 4 5 6 7 8")
    print("  " + "-" * 17)
    filas = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    
    for i, fila in enumerate(filas):
        print(f"{fila} |", end="")
        for j in range(8):
            if tablero[i, j] == 1:
                print("●", end=" ")  # Negra (1 = black en OthelloBoardState)
            elif tablero[i, j] == -1:
                print("○", end=" ")  # Blanca (-1 = white en OthelloBoardState)
            else:
                print("·", end=" ")  # Vacía
        print("|")
    print("  " + "-" * 17)


def imprimir_resumen(bsps):
    """
    Imprime un resumen de las BSPs activas.
    
    Args:
        bsps (dict[str, bool]): Diccionario de BSPs
    """
    # Contar BSPs activas por tipo
    vacias = sum(1 for k, v in bsps.items() if v and k.endswith('0') and k.startswith('BSP') and len(k) == 6)
    mias = sum(1 for k, v in bsps.items() if v and k.endswith('1') and k.startswith('BSP') and len(k) == 6)
    oponente = sum(1 for k, v in bsps.items() if v and k.endswith('2') and k.startswith('BSP') and len(k) == 6)
    
    print(f"\n Resumen de BSPs activas:")
    print(f"   Casillas vacías: {vacias}")
    print(f"   Fichas mías: {mias}")
    print(f"   Fichas oponente: {oponente}")
    print(f"   Total casillas: {vacias + mias + oponente} (debe ser 64)")
    
    # Estados especiales
    print(f"\n Estados especiales:")
    estados = ['BSP_INICIAL', 'BSP_TERMINADA', 'BSP_GANE', 'BSP_PERDI', 'BSP_EMPATE', 'BSP_EN_CURSO']
    for estado in estados:
        if bsps.get(estado, False):
            print(f"   ✓ {estado}")


def imprimir_bsps_activas_con_fichas(bsps):
    """
    Imprime solo las BSPs activas que corresponden a casillas con fichas (sin vacías).
    Útil para ver qué piezas detecta el SAE.
    
    Args:
        bsps (dict[str, bool]): Diccionario de BSPs
    """
    print("\n BSPs activas con fichas (sin vacías):")
    
    mias = []
    oponente = []
    
    for k, v in bsps.items():
        if v and k.startswith('BSP') and len(k) == 6:
            if k.endswith('1'):  # Fichas mías
                mias.append(k[3:5])  # Extraer casilla (ej: 'A1')
            elif k.endswith('2'):  # Fichas oponente
                oponente.append(k[3:5])
    
    print(f"\n   Mis fichas ({len(mias)}):")
    if mias:
        print(f"   {', '.join(sorted(mias))}")
    
    print(f"\n   Fichas oponente ({len(oponente)}):")
    if oponente:
        print(f"   {', '.join(sorted(oponente))}")


def imprimir_todas_bsps_activas(bsps):
    """
    Imprime TODAS las BSPs activas (incluyendo vacías), separadas por tipo.
    
    Args:
        bsps (dict[str, bool]): Diccionario de BSPs
    """
    print("\n TODAS las BSPs activas:")
    
    vacias = []
    mias = []
    oponente = []
    especiales = []
    
    for k, v in bsps.items():
        if v:
            if k.startswith('BSP') and len(k) == 6:
                casilla = k[3:5]
                if k.endswith('0'):
                    vacias.append(casilla)
                elif k.endswith('1'):
                    mias.append(casilla)
                elif k.endswith('2'):
                    oponente.append(casilla)
            elif k.startswith('BSP_'):
                especiales.append(k)
    
    print(f"\n   Casillas vacías ({len(vacias)}):")
    if vacias:
        print(f"   {', '.join(sorted(vacias))}")
    
    print(f"\n   Mis fichas ({len(mias)}):")
    if mias:
        print(f"   {', '.join(sorted(mias))}")
    
    print(f"\n   Fichas oponente ({len(oponente)}):")
    if oponente:
        print(f"   {', '.join(sorted(oponente))}")
    
    print(f"\n   Estados especiales:")
    for estado in especiales:
        print(f"   ✓ {estado}")
    
    print(f"\n   Total: {len(vacias) + len(mias) + len(oponente)} casillas + {len(especiales)} especiales")


def verificar_consistencia(bsps):
    """
    Verifica que cada casilla tenga exactamente 1 BSP activa (one-hot encoding).
    
    Args:
        bsps (dict[str, bool]): Diccionario de BSPs
    
    Returns:
        bool: True si la consistencia es válida
    
    Raises:
        AssertionError: Si alguna casilla no cumple la propiedad one-hot
    """
    filas = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    columnas = ['1', '2', '3', '4', '5', '6', '7', '8']
    
    for i in filas:
        for j in columnas:
            casilla = i + j
            vacia = bsps[f'BSP{casilla}0']
            mia = bsps[f'BSP{casilla}1']
            oponente = bsps[f'BSP{casilla}2']
            
            activas = sum([vacia, mia, oponente])
            assert activas == 1, f"Casilla {casilla} tiene {activas} BSPs activas (debe ser 1)"
    
    print(" Consistencia verificada: Todas las casillas tienen exactamente 1 BSP activa")
    return True


def obtener_bsps_solo_piezas(bsps):
    """
    Extrae solo las BSPs de piezas (sin vacías).
    Útil para calcular Coverage-pieces (métrica principal).
    
    Args:
        bsps (dict[str, bool]): Diccionario completo de BSPs
    
    Returns:
        dict[str, bool]: Diccionario con solo las 128 BSPs de piezas (sin vacías)
    """
    bsps_piezas = {}
    
    for k, v in bsps.items():
        # Solo BSPs de casillas que NO son vacías (estado 1 y 2, no 0)
        if k.startswith('BSP') and len(k) == 6 and not k.endswith('0'):
            bsps_piezas[k] = v
    
    return bsps_piezas


def contar_bsps_activas(bsps):
    """
    Cuenta cuántas BSPs están activas en total y por categoría.
    
    Args:
        bsps (dict[str, bool]): Diccionario de BSPs
    
    Returns:
        dict: Diccionario con conteos por categoría
    """
    conteos = {
        'vacias': 0,
        'mias': 0,
        'oponente': 0,
        'especiales': 0,
        'total': 0
    }
    
    for k, v in bsps.items():
        if v:
            conteos['total'] += 1
            if k.startswith('BSP') and len(k) == 6:
                if k.endswith('0'):
                    conteos['vacias'] += 1
                elif k.endswith('1'):
                    conteos['mias'] += 1
                elif k.endswith('2'):
                    conteos['oponente'] += 1
            elif k.startswith('BSP_'):
                conteos['especiales'] += 1
    
    return conteos