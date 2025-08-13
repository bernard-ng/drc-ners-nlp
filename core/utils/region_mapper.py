from typing import Optional, Dict, Tuple

import pandas as pd


class RegionMapper:
    """Reusable region mapping utilities"""

    def __init__(self, mapping: Optional[Dict] = None):
        self.mapping = mapping or REGION_MAPPING

    def map(self, series: pd.Series) -> pd.Series:
        """Vectorized region to province mapping"""
        return series.str.lower().map(
            lambda r: self.mapping.get(r, ("AUTRES", "AUTRES"))[1].lower()
        )

    @staticmethod
    def get_provinces():
        return [
            "kinshasa",
            "bas-congo",
            "bandundu",
            "katanga",
            "equateur",
            "province-orientale",
            "maniema",
            "nord-kivu",
            "sud-kivu",
            "kasai-occidental",
            "kasai-oriental",
            "autres",
        ]


# DRC Region to Province Mapping
REGION_MAPPING: Dict[str, Tuple[str, str]] = {
    # Kinshasa
    "kinshasa": ("KINSHASA", "KINSHASA"),
    "kinshasa-centre": ("KINSHASA", "KINSHASA"),
    "kinshasa-est": ("KINSHASA", "KINSHASA"),
    "kinshasa-funa": ("KINSHASA", "KINSHASA"),
    "kinshasa-lukunga": ("KINSHASA", "KINSHASA"),
    "kinshasa-mont-amba": ("KINSHASA", "KINSHASA"),
    "kinshasa-ouest": ("KINSHASA", "KINSHASA"),
    "kinshasa-plateau": ("KINSHASA", "KINSHASA"),
    "kinshasa-tshangu": ("KINSHASA", "KINSHASA"),
    # Bas-Congo → Kongo-Central → BAS-CONGO
    "bas-congo": ("KONGO-CENTRAL", "BAS-CONGO"),
    "bas-congo-1": ("KONGO-CENTRAL", "BAS-CONGO"),
    "bas-congo-2": ("KONGO-CENTRAL", "BAS-CONGO"),
    "kongo-central": ("KONGO-CENTRAL", "BAS-CONGO"),
    "kongo-central-1": ("KONGO-CENTRAL", "BAS-CONGO"),
    "kongo-central-2": ("KONGO-CENTRAL", "BAS-CONGO"),
    "kongo-central-3": ("KONGO-CENTRAL", "BAS-CONGO"),
    # Kwilu, Kwango, Mai-Ndombe → BANDUNDU
    "bandundu": ("BANDUNDU", "BANDUNDU"),
    "bandundu-1": ("BANDUNDU", "BANDUNDU"),
    "bandundu-2": ("BANDUNDU", "BANDUNDU"),
    "bandundu-3": ("BANDUNDU", "BANDUNDU"),
    "kwilu": ("KWILU", "BANDUNDU"),
    "kwilu-1": ("KWILU", "BANDUNDU"),
    "kwilu-2": ("KWILU", "BANDUNDU"),
    "kwilu-3": ("KWILU", "BANDUNDU"),
    "kwango": ("KWANGO", "BANDUNDU"),
    "kwango-1": ("KWANGO", "BANDUNDU"),
    "kwango-2": ("KWANGO", "BANDUNDU"),
    "mai-ndombe": ("MAI-NDOMBE", "BANDUNDU"),
    "mai-ndombe-1": ("MAI-NDOMBE", "BANDUNDU"),
    "mai-ndombe-2": ("MAI-NDOMBE", "BANDUNDU"),
    "mai-ndombe-3": ("MAI-NDOMBE", "BANDUNDU"),
    # Katanga → HAUT-KATANGA, HAUT-LOMAMI, LUALABA, TANGANYIKA
    "haut-katanga": ("HAUT-KATANGA", "KATANGA"),
    "haut-katanga-1": ("HAUT-KATANGA", "KATANGA"),
    "haut-katanga-2": ("HAUT-KATANGA", "KATANGA"),
    "haut-lomami": ("HAUT-LOMAMI", "KATANGA"),
    "haut-lomami-1": ("HAUT-LOMAMI", "KATANGA"),
    "haut-lomami-2": ("HAUT-LOMAMI", "KATANGA"),
    "lualaba": ("LUALABA", "KATANGA"),
    "lualaba-1": ("LUALABA", "KATANGA"),
    "lualaba-2": ("LUALABA", "KATANGA"),
    "lualaba-74-corrige-922a": ("LUALABA", "KATANGA"),
    "tanganyika": ("TANGANYIKA", "KATANGA"),
    "tanganyika-1": ("TANGANYIKA", "KATANGA"),
    "tanganyika-2": ("TANGANYIKA", "KATANGA"),
    # Equateur → MONGALA, NORD-UBANGI, SUD-UBANGI, TSHUAPA
    "equateur": ("EQUATEUR", "EQUATEUR"),
    "equateur-1": ("EQUATEUR", "EQUATEUR"),
    "equateur-2": ("EQUATEUR", "EQUATEUR"),
    "equateur-3": ("EQUATEUR", "EQUATEUR"),
    "equateur-4": ("EQUATEUR", "EQUATEUR"),
    "equateur-5": ("EQUATEUR", "EQUATEUR"),
    "mongala": ("MONGALA", "EQUATEUR"),
    "mongala-1": ("MONGALA", "EQUATEUR"),
    "mongala-2": ("MONGALA", "EQUATEUR"),
    "nord-ubangi": ("NORD-UBANGI", "EQUATEUR"),
    "nord-ubangi-1": ("NORD-UBANGI", "EQUATEUR"),
    "nord-ubangi-2": ("NORD-UBANGI", "EQUATEUR"),
    "sud-ubangi": ("SUD-UBANGI", "EQUATEUR"),
    "sud-ubangi-1": ("SUD-UBANGI", "EQUATEUR"),
    "sud-ubangi-2": ("SUD-UBANGI", "EQUATEUR"),
    "tshuapa": ("TSHUAPA", "EQUATEUR"),
    "tshuapa-1": ("TSHUAPA", "EQUATEUR"),
    "tshuapa-2": ("TSHUAPA", "EQUATEUR"),
    # Province-Orientale
    "province-orientale": ("PROVINCE-ORIENTALE", "PROVINCE-ORIENTALE"),
    "province-orientale-1": ("PROVINCE-ORIENTALE", "PROVINCE-ORIENTALE"),
    "province-orientale-2": ("PROVINCE-ORIENTALE", "PROVINCE-ORIENTALE"),
    "province-orientale-3": ("PROVINCE-ORIENTALE", "PROVINCE-ORIENTALE"),
    "province-orientale-4": ("PROVINCE-ORIENTALE", "PROVINCE-ORIENTALE"),
    "haut-uele": ("HAUT-UELE", "PROVINCE-ORIENTALE"),
    "haut-uele-1": ("HAUT-UELE", "PROVINCE-ORIENTALE"),
    "haut-uele-2": ("HAUT-UELE", "PROVINCE-ORIENTALE"),
    "bas-uele": ("BAS-UELE", "PROVINCE-ORIENTALE"),
    "bas-uele-1": ("BAS-UELE", "PROVINCE-ORIENTALE"),
    "bas-uele-2": ("BAS-UELE", "PROVINCE-ORIENTALE"),
    "ituri": ("ITURI", "PROVINCE-ORIENTALE"),
    "ituri-1": ("ITURI", "PROVINCE-ORIENTALE"),
    "ituri-2": ("ITURI", "PROVINCE-ORIENTALE"),
    "tshopo": ("TSHOPO", "PROVINCE-ORIENTALE"),
    "tshopo-1": ("TSHOPO", "PROVINCE-ORIENTALE"),
    "tshopo-2": ("TSHOPO", "PROVINCE-ORIENTALE"),
    # Maniema
    "maniema": ("MANIEMA", "MANIEMA"),
    "maniema-1": ("MANIEMA", "MANIEMA"),
    "maniema-2": ("MANIEMA", "MANIEMA"),
    # Nord-Kivu
    "nord-kivu": ("NORD-KIVU", "NORD-KIVU"),
    "nord-kivu-1": ("NORD-KIVU", "NORD-KIVU"),
    "nord-kivu-2": ("NORD-KIVU", "NORD-KIVU"),
    "nord-kivu-3": ("NORD-KIVU", "NORD-KIVU"),
    # Sud-Kivu
    "sud-kivu": ("SUD-KIVU", "SUD-KIVU"),
    "sud-kivu-1": ("SUD-KIVU", "SUD-KIVU"),
    "sud-kivu-2": ("SUD-KIVU", "SUD-KIVU"),
    "sud-kivu-3": ("SUD-KIVU", "SUD-KIVU"),
    # Kasai-Occidental → KASAI, KASAI-CENTRAL
    "kasai-occidental": ("KASAI-OCCIDENTAL", "KASAI-OCCIDENTAL"),
    "kasai-occidental-1": ("KASAI-OCCIDENTAL", "KASAI-OCCIDENTAL"),
    "kasai-occidental-2": ("KASAI-OCCIDENTAL", "KASAI-OCCIDENTAL"),
    "kasai": ("KASAI", "KASAI-OCCIDENTAL"),
    "kasai-1": ("KASAI", "KASAI-OCCIDENTAL"),
    "kasai-2": ("KASAI", "KASAI-OCCIDENTAL"),
    "kasai-central": ("KASAI-CENTRAL", "KASAI-OCCIDENTAL"),
    "kasai-central-1": ("KASAI-CENTRAL", "KASAI-OCCIDENTAL"),
    "kasai-central-2": ("KASAI-CENTRAL", "KASAI-OCCIDENTAL"),
    # Kasai-Oriental → LOMAMI, SANKURU
    "kasai-oriental": ("KASAI-ORIENTAL", "KASAI-ORIENTAL"),
    "kasai-oriental-1": ("KASAI-ORIENTAL", "KASAI-ORIENTAL"),
    "kasai-oriental-2": ("KASAI-ORIENTAL", "KASAI-ORIENTAL"),
    "kasai-oriental-3": ("KASAI-ORIENTAL", "KASAI-ORIENTAL"),
    "lomami": ("LOMAMI", "KASAI-ORIENTAL"),
    "lomami-1": ("LOMAMI", "KASAI-ORIENTAL"),
    "lomami-2": ("LOMAMI", "KASAI-ORIENTAL"),
    "sankuru": ("SANKURU", "KASAI-ORIENTAL"),
    "sankuru-1": ("SANKURU", "KASAI-ORIENTAL"),
    "sankuru-2": ("SANKURU", "KASAI-ORIENTAL"),
}
