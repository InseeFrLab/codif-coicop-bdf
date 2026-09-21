"""Manipulation des codes COICOP.

`truncate_code` existait en **quatre exemplaires** (dont deux dans le seul
module `rag-notices`), `get_parents` en deux. Tous fonctionnellement
identiques ; le corps ci-dessous les reproduit à l'identique, y compris leurs
comportements limites — ce n'est pas le lieu de les « améliorer ».
"""

from typing import List, Optional


def truncate_code(code: str, level: int) -> Optional[str]:
    """Tronque un code COICOP pointé au niveau hiérarchique demandé.

    ``truncate_code('08.1.2.3.4', 4)`` → ``'08.1.2.3'``. Renvoie le code
    inchangé s'il est déjà à ce niveau ou en dessous, ``None`` s'il est
    inexploitable (``None``, non-``str``, chaîne vide).
    """
    if code is None or not isinstance(code, str) or code == "":
        return None
    parts = code.split(".")
    if len(parts) <= level:
        return code
    return ".".join(parts[:level])


def get_parents(code: str) -> List[str]:
    """Liste les ancêtres d'un code, du plus général au plus précis, lui exclu.

    ``get_parents('01.1.2')`` → ``['01', '01.1']``.

    Ne garde pas de garde sur ``code`` : les deux implémentations d'origine
    lèvent une ``AttributeError`` sur ``None``, et leurs appelants
    (`0_create_vector_db.py`, la construction de la lignée COICOP) s'appuient
    sur ce fait pour échouer tôt sur une nomenclature abîmée. Rendre la
    fonction tolérante ici masquerait ce signal.
    """
    code_level = len(code.split("."))
    return [truncate_code(code, level) for level in range(1, code_level)]
