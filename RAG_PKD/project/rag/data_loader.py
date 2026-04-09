import fitz
import re
import pandas as pd
from config import PDF_FILE, EXCEL_FILE
from langchain_community.document_loaders import DataFrameLoader



class DataLoader:
    def __init__(self, pdf_path: str = None, excel_path: str = None):
        self.pdf_path = pdf_path
        self.excel_path = excel_path
        # Pattern to match PKD codes like 01.11.Z
        self.regex_pattern = r"(?s)(\d{2}\.\d{2}\.[A-Z])(.*?)(?=\d{2}\.\d{2}\.[A-Z]|$)"

    @staticmethod
    def clean_pkd_text(raw_text: str, pkd_code_context: str = None) -> str:
        """
        Czyści i strukturyzuje tekst opisu PKD.
        """
        if not isinstance(raw_text, str) or pd.isna(raw_text):
            return ""

        if not pkd_code_context:
            code_match = re.search(r'\d{2}\.\d{2}\.[A-Z]', raw_text)
            pkd_code = code_match.group(0) if code_match else ""
        else:
            pkd_code = pkd_code_context

        seen = set()
        cleaned_parts = []
        for part in re.split(r'\. |, | - ', raw_text):
            p = part.strip()
            if len(p) > 5 and p.lower() not in seen:
                cleaned_parts.append(p)
                seen.add(p.lower())

        text = " ".join(cleaned_parts)

        text = re.sub(r'sklasyfikowan[ego|ej|ych]+\s+w\s*,?', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\s,', ',', text)

        text = text.replace("Podklasa ta obejmuje", "### Obejmuje:- ")
        text = text.replace("Podklasa ta nie obejmuje", "### Nie obejmuje:- ")

        if "###" in text:
            text = text.replace("; ", "").replace("-", "").strip()
            text = re.sub(r'\s+,', ',', text)

        return f"## Kod PKD: {pkd_code}{text.strip()}"

    def _extract_chapters_from_pdf(self) -> list[str]:
        if not self.pdf_path:
            return []

        doc = fitz.open(self.pdf_path)
        fulltext_parts = [page.get_text("blocks") for page in doc]

        flat_text = ""
        for blocks in fulltext_parts:
            for b in blocks:
                flat_text += b[4]

        doc.close()

        matches = re.findall(self.regex_pattern, flat_text)

        chapters = []
        for pkd_code, content in matches:
            full_chunk = f"{pkd_code} {content}".strip()
            if len(full_chunk) > 20:
                chapters.append(full_chunk)

        print(f"📚 Uzyskano {len(chapters)} fragmentów PKD")
        return chapters

    def _import_excel(self):
        excl_data = pd.read_excel(self.excel_path or EXCEL_FILE, header=1, usecols=["Podklasa", "Nazwa grupowania"])
        excl_data.rename(columns={"Podklasa": "pkd_code",
                          "Nazwa grupowania" : "full_text"}, inplace=True)
        return excl_data

    def load_data(self) -> pd.DataFrame:
        raw_chapters = self._extract_chapters_from_pdf()
        if not raw_chapters:
            return pd.DataFrame()

        df = pd.DataFrame(raw_chapters, columns=["full_text"])
        df["pkd_code"] = df["full_text"].str.slice(0, 7)
        df = df[~df["full_text"].str.contains("2007|Tabela", na=False)]
        df = df.groupby("pkd_code")["full_text"].agg(lambda x: " ".join(dict.fromkeys(x))).reset_index()

        excl_data = self._import_excel()
        valid_codes = set(excl_data.pkd_code)
        df = df[df.pkd_code.isin(valid_codes)].reset_index(drop=True)

        df['full_text'] = df.apply(
            lambda row: self.clean_pkd_text(row['full_text'].replace(row['pkd_code'], ''), ""),
            axis=1
        )

        print(f"Po złączeniu i wyczyszczeniu: {len(df)} unikalnych kodów PKD")
        return df



class PKDDataImporter:
    def __init__(self):
        self.url = "https://klasyfikacje.stat.gov.pl/static/pkd_25/pdf/Wyjasnienia_PKD_2025.xls"
        self.url_codes = "https://klasyfikacje.stat.gov.pl/static/pkd_25/pdf/StrukturaPKD2025.xls"
        self._df = None
        self._codes = None
        self._missing_codes_list = None

    def _get_frames(self):
        """Pobiera surowe dane z URL, jeśli nie zostały jeszcze pobrane."""
        if self._df is None or self._codes is None:
            # Poprawione wczytywanie - usecols musi zgadzać się z listą names lub być indeksowane
            self._codes = pd.read_excel(self.url_codes, names=["c1", "c2", "c3", "code", "desc"], usecols=[3, 4])
            self._df = pd.read_excel(self.url, header=None, names=["code", "desc"])
        return self._df.copy(), self._codes.copy()

    def _prepare_frames(self):
        """Czyści i agreguje dane."""
        df, codes = self._get_frames()
        
        # Agregacja opisów rozbitych na wiele wierszy
        df['group'] = df['code'].notna().cumsum()
        df = df.groupby('group').agg({
            'code': 'first',
            'desc': lambda x: ' '.join(x.dropna().astype(str))
        }).reset_index(drop=True)
        
        df['desc'] = df['desc'].str.replace(r'\s+', ' ', regex=True).str.strip()
        # Filtrowanie tylko konkretnych kodów kończących się na .Z
        df = df[df['code'].astype(str).str.contains(r'\.Z', na=False)]
        
        codes.dropna(inplace=True)
        # Usunięcie nagłówka jeśli istnieje
        if not codes.empty:
            codes = codes.iloc[1:].reset_index(drop=True)

        return df, codes

    def _get_missing_codes(self, df, codes):
        """Identyfikuje kody obecne w strukturze, a nieobecne w wyjaśnieniach."""
        if self._missing_codes_list is None:
            df_codes = set(df.code.unique())
            self._missing_codes_list = [c for c in codes.code.unique() if c not in df_codes]
        return self._missing_codes_list

    def _merge_frames(self):
        """Łączy przetworzone dane w jeden DataFrame."""
        df, codes = self._prepare_frames()
        missing_list = self._get_missing_codes(df, codes)
        
        # Pobieramy brakujące kody z ramki 'codes'
        missing_df = codes[codes.code.isin(missing_list)]
        
        return pd.concat([df, missing_df], axis=0, ignore_index=True)

    def get_loaders(self):
        """Zwraca listę dokumentów gotową dla LangChain."""
        df = self._merge_frames()
        df['code'] = df['code'].astype("string")
        df['desc'] = df['desc'].astype("string")
        loader = DataFrameLoader(df, page_content_column="desc")
        return loader.load()