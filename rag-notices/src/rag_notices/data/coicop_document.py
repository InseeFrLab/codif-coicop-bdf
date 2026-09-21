from dataclasses import dataclass
from typing import List, Dict, Optional
from enum import Enum

class EmbeddingStrategy(Enum):
    """Stratégies disponibles pour la création de notices"""
    CODE_ONLY = "code_only"  # Code et label uniquement
    BASIC = "basic"  # Code, label, note générale et inclusions
    WITH_EXCLUSIONS = "with_exclusions"  # Basic + exclusions
    WITH_HIERARCHY = "with_hierarchy"  # Basic + lignée hiérarchique
    ALL_INFO = "all_info"  # Toutes les informations disponibles

@dataclass
class CoicopDocument:
    """Structure of a COICOP document to embed"""
    code: str
    label_fr: str
    note_generale_fr: Optional[str] = None
    contenu_central_fr: Optional[str] = None
    contenu_additionnel_fr: Optional[str] = None
    note_exclusion_fr: Optional[str] = None
    parents: Optional[List[str]] = None
    parents_labels: Optional[List[str]] = None
    
    @property
    def inclusions(self) -> Optional[str]:
        """Concatenate central and additional content"""
        parts = []
        if self.contenu_central_fr:
            parts.append(self.contenu_central_fr.strip())
        if self.contenu_additionnel_fr:
            parts.append(self.contenu_additionnel_fr.strip())
        return ". ".join(parts) if parts else None
    
    @property
    def hierarchy_text(self) -> Optional[str]:
        """Generate hierarchy text from parents"""
        if not self.parents or not self.parents_labels:
            return None
        if len(self.parents) != len(self.parents_labels):
            return None
        
        hierarchy_parts = [
            f"{code}: {label}" 
            for code, label in zip(self.parents, self.parents_labels)
        ]
        return " > ".join(hierarchy_parts)
    
    def _build_basic_content(self) -> List[str]:
        """Build basic content sections (note générale + inclusions)"""
        lines = []
        
        if self.note_generale_fr:
            lines.append(f"**Note générale:** {self.note_generale_fr}")
        
        inclusions = self.inclusions
        if inclusions:
            lines.append(f"**Inclusions:** {inclusions}")
        
        return lines
    
    def _build_content_by_strategy(self, strategy: EmbeddingStrategy) -> List[str]:
        """Build content sections according to strategy"""
        lines = []
        
        if strategy == EmbeddingStrategy.CODE_ONLY:
            return lines
        
        # Toutes les stratégies sauf CODE_ONLY incluent le contenu de base
        lines.extend(self._build_basic_content())
        
        # Ajout de la hiérarchie si demandé
        if strategy in (EmbeddingStrategy.WITH_HIERARCHY, EmbeddingStrategy.ALL_INFO):
            hierarchy = self.hierarchy_text
            if hierarchy:
                lines.append(f"**À niveau(x) plus agrégé(s), ce code fait partie des familles de produits suivants :** {hierarchy}")
        
        # Ajout des exclusions si demandé
        if strategy in (EmbeddingStrategy.WITH_EXCLUSIONS, EmbeddingStrategy.ALL_INFO):
            if self.note_exclusion_fr:
                lines.append(f"**Exclusions:** {self.note_exclusion_fr}")
        
        return lines
        
    def to_single_text(self, strategy: str = "all_info") -> str:
        """
        Convert to a single Markdown-formatted text according to strategy.
        Each section is separated by a clear line break for embedding.
        
        Args:
            strategy: Strategy name (string or EmbeddingStrategy enum)
        """
        # Convert string to enum if necessary
        if isinstance(strategy, str):
            strategy = EmbeddingStrategy(strategy)
        
        # Title is always included
        lines = [f"**{self.code}: {self.label_fr}**"]
        
        # Add content according to strategy
        lines.extend(self._build_content_by_strategy(strategy))
        
        # Join sections with two line breaks for clear separation in embeddings
        return "\n\n".join(lines)
    
    def to_text_chunks(self, strategy: str = "all_info") -> Dict[str, str]:
        """
        Convert to text chunk for embedding according to strategy.
        
        Args:
            strategy: Strategy name (string or EmbeddingStrategy enum)
        """
        if isinstance(strategy, str):
            strategy = EmbeddingStrategy(strategy)
        
        chunk = {
            "type": strategy.value,
            "text": self.to_single_text(strategy),
            "code": self.code
        }
        return chunk