import unicodedata
from typing import Optional, Dict, Tuple

import pandas as pd


class RegionMapper:
    """Reusable region mapping utilities"""

    def __init__(self, mapping: Optional[Dict] = None):
        self.mapping = mapping or REGION_MAPPING
        self.mapping = {k.lower(): v[1].upper() for k, v in self.mapping.items()}

    def map(self, series: pd.Series) -> pd.Series:
        return series.str.lower().map(self.mapping).fillna("AUTRES")

    @staticmethod
    def clean_province(series: pd.Series) -> pd.Series:
        return (
            series.str.upper()
            .str.strip()
            .apply(lambda x: unicodedata.normalize("NFKD", x)
                   .encode("ascii", errors="ignore")
                   .decode("utf-8") if isinstance(x, str) else x)
        )

    @staticmethod
    def get_provinces():
        return [
            "kinshasa",
            "bas-congo",
            "bandundu",
            "katanga",
            "equateur",
            "orientale",
            "maniema",
            "nord-kivu",
            "sud-kivu",
            "kasai-occidental",
            "kasai-oriental",
            "autres",
        ]


# DRC Region to Province Mapping
REGION_MAPPING: Dict[str, Tuple[str, str]] = {
    "bandundu": ("BANDUNDU", "BANDUNDU"),
    "bandundu-1": ("BANDUNDU", "BANDUNDU"),
    "bandundu-2": ("BANDUNDU", "BANDUNDU"),
    "bandundu-3": ("BANDUNDU", "BANDUNDU"),
    "bas-congo": ("KONGO-CENTRAL", "BAS-CONGO"),
    "bas-congo-1": ("KONGO-CENTRAL", "BAS-CONGO"),
    "bas-congo-2": ("KONGO-CENTRAL", "BAS-CONGO"),
    "bas-fleuve": ("KONGO-CENTRAL", "BAS-CONGO"),
    "bas-uele": ("BAS-UELE", "ORIENTALE"),
    "bas-uele-1": ("BAS-UELE", "ORIENTALE"),
    "bas-uele-2": ("BAS-UELE", "ORIENTALE"),
    "cataractes": ("KONGO-CENTRAL", "BAS-CONGO"),
    "equateur": ("EQUATEUR", "EQUATEUR"),
    "equateur-1": ("EQUATEUR", "EQUATEUR"),
    "equateur-2": ("EQUATEUR", "EQUATEUR"),
    "equateur-3": ("EQUATEUR", "EQUATEUR"),
    "equateur-4": ("EQUATEUR", "EQUATEUR"),
    "equateur-5": ("EQUATEUR", "EQUATEUR"),
    "haut-katanga": ("HAUT-KATANGA", "KATANGA"),
    "haut-katanga-1": ("HAUT-KATANGA", "KATANGA"),
    "haut-katanga-2": ("HAUT-KATANGA", "KATANGA"),
    "haut-lomami": ("HAUT-LOMAMI", "KATANGA"),
    "haut-lomami-1": ("HAUT-LOMAMI", "KATANGA"),
    "haut-lomami-2": ("HAUT-LOMAMI", "KATANGA"),
    "haut-uele": ("HAUT-UELE", "ORIENTALE"),
    "haut-uele-1": ("HAUT-UELE", "ORIENTALE"),
    "haut-uele-2": ("HAUT-UELE", "ORIENTALE"),
    "ituri": ("ITURI", "ORIENTALE"),
    "ituri-1": ("ITURI", "ORIENTALE"),
    "ituri-2": ("ITURI", "ORIENTALE"),
    "ituri-3": ("ITURI", "ORIENTALE"),
    "kasai": ("KASAI", "KASAI-OCCIDENTAL"),
    "kasai-1": ("KASAI", "KASAI-OCCIDENTAL"),
    "kasai-2": ("KASAI", "KASAI-OCCIDENTAL"),
    "kasai-ce": ("KASAI-CENTRAL", "KASAI-OCCIDENTAL"),
    "kasai-central": ("KASAI-CENTRAL", "KASAI-OCCIDENTAL"),
    "kasai-central-1": ("KASAI-CENTRAL", "KASAI-OCCIDENTAL"),
    "kasai-central-2": ("KASAI-CENTRAL", "KASAI-OCCIDENTAL"),
    "kasai-occidental": ("KASAI-OCCIDENTAL", "KASAI-OCCIDENTAL"),
    "kasai-occidental-1": ("KASAI-OCCIDENTAL", "KASAI-OCCIDENTAL"),
    "kasai-occidental-2": ("KASAI-OCCIDENTAL", "KASAI-OCCIDENTAL"),
    "kasai-oriental": ("KASAI-ORIENTAL", "KASAI-ORIENTAL"),
    "kasai-oriental-1": ("KASAI-ORIENTAL", "KASAI-ORIENTAL"),
    "kasai-oriental-2": ("KASAI-ORIENTAL", "KASAI-ORIENTAL"),
    "kasai-oriental-3": ("KASAI-ORIENTAL", "KASAI-ORIENTAL"),
    "kasai-orientale": ("KASAI-ORIENTAL", "KASAI-ORIENTAL"),
    "katanga": ("KATANGA", "KATANGA"),
    "katanga-1": ("KATANGA", "KATANGA"),
    "katanga-2": ("KATANGA", "KATANGA"),
    "katanga-3": ("KATANGA", "KATANGA"),
    "katanga-4": ("KATANGA", "KATANGA"),
    "kinshasa": ("KINSHASA", "KINSHASA"),
    "kinshasa-centre": ("KINSHASA", "KINSHASA"),
    "kinshasa-est": ("KINSHASA", "KINSHASA"),
    "kinshasa-funa": ("KINSHASA", "KINSHASA"),
    "kinshasa-global": ("KINSHASA", "KINSHASA"),
    "kinshasa-lukunga": ("KINSHASA", "KINSHASA"),
    "kinshasa-mont-amba": ("KINSHASA", "KINSHASA"),
    "kinshasa-ouest": ("KINSHASA", "KINSHASA"),
    "kinshasa-plateau": ("KINSHASA", "KINSHASA"),
    "kinshasa-tshangu": ("KINSHASA", "KINSHASA"),
    "kongo-central": ("KONGO-CENTRAL", "BAS-CONGO"),
    "kongo-central-1": ("KONGO-CENTRAL", "BAS-CONGO"),
    "kongo-central-2": ("KONGO-CENTRAL", "BAS-CONGO"),
    "kongo-central-3": ("KONGO-CENTRAL", "BAS-CONGO"),
    "kwango": ("KWANGO", "BANDUNDU"),
    "kwango-1": ("KWANGO", "BANDUNDU"),
    "kwango-2": ("KWANGO", "BANDUNDU"),
    "kwilu": ("KWILU", "BANDUNDU"),
    "kwilu-1": ("KWILU", "BANDUNDU"),
    "kwilu-2": ("KWILU", "BANDUNDU"),
    "kwilu-3": ("KWILU", "BANDUNDU"),
    "lomami": ("LOMAMI", "KASAI-ORIENTAL"),
    "lomami-1": ("LOMAMI", "KASAI-ORIENTAL"),
    "lomami-2": ("LOMAMI", "KASAI-ORIENTAL"),
    "lualaba": ("LUALABA", "KATANGA"),
    "lualaba-1": ("LUALABA", "KATANGA"),
    "lualaba-2": ("LUALABA", "KATANGA"),
    "lualaba-74-corrige-922a": ("LUALABA", "KATANGA"),
    "lukaya": ("KONGO-CENTRAL", "BAS-CONGO"),
    "mai-ndombe": ("MAI-NDOMBE", "BANDUNDU"),
    "mai-ndombe-1": ("MAI-NDOMBE", "BANDUNDU"),
    "mai-ndombe-2": ("MAI-NDOMBE", "BANDUNDU"),
    "mai-ndombe-3": ("MAI-NDOMBE", "BANDUNDU"),
    "maniema": ("MANIEMA", "MANIEMA"),
    "maniema-1": ("MANIEMA", "MANIEMA"),
    "maniema-2": ("MANIEMA", "MANIEMA"),
    "mongala": ("MONGALA", "EQUATEUR"),
    "mongala-1": ("MONGALA", "EQUATEUR"),
    "mongala-2": ("MONGALA", "EQUATEUR"),
    "nord-kivu": ("NORD-KIVU", "NORD-KIVU"),
    "nord-kivu-1": ("NORD-KIVU", "NORD-KIVU"),
    "nord-kivu-2": ("NORD-KIVU", "NORD-KIVU"),
    "nord-kivu-3": ("NORD-KIVU", "NORD-KIVU"),
    "nord-ubangi": ("NORD-UBANGI", "EQUATEUR"),
    "nord-ubangi-1": ("NORD-UBANGI", "EQUATEUR"),
    "nord-ubangi-2": ("NORD-UBANGI", "EQUATEUR"),
    "province-orientale": ("ORIENTALE", "ORIENTALE"),
    "province-orientale-1": ("ORIENTALE", "ORIENTALE"),
    "province-orientale-2": ("ORIENTALE", "ORIENTALE"),
    "province-orientale-3": ("ORIENTALE", "ORIENTALE"),
    "province-orientale-4": ("ORIENTALE", "ORIENTALE"),
    "sankuru": ("SANKURU", "KASAI-ORIENTAL"),
    "sankuru-1": ("SANKURU", "KASAI-ORIENTAL"),
    "sankuru-2": ("SANKURU", "KASAI-ORIENTAL"),
    "sud-kivu": ("SUD-KIVU", "SUD-KIVU"),
    "sud-kivu-1": ("SUD-KIVU", "SUD-KIVU"),
    "sud-kivu-2": ("SUD-KIVU", "SUD-KIVU"),
    "sud-kivu-3": ("SUD-KIVU", "SUD-KIVU"),
    "sud-ubangi": ("SUD-UBANGI", "EQUATEUR"),
    "sud-ubangi-1": ("SUD-UBANGI", "EQUATEUR"),
    "sud-ubangi-2": ("SUD-UBANGI", "EQUATEUR"),
    "tanganyika": ("TANGANYIKA", "KATANGA"),
    "tanganyika-1": ("TANGANYIKA", "KATANGA"),
    "tanganyika-2": ("TANGANYIKA", "KATANGA"),
    "tshopo": ("TSHOPO", "ORIENTALE"),
    "tshopo-1": ("TSHOPO", "ORIENTALE"),
    "tshopo-2": ("TSHOPO", "ORIENTALE"),
    "tshuapa": ("TSHUAPA", "EQUATEUR"),
    "tshuapa-1": ("TSHUAPA", "EQUATEUR"),
    "tshuapa-2": ("TSHUAPA", "EQUATEUR"),
}
