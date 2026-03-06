import pandas as pd
import numpy as np
import holidays

class DataPreprocessor:
    def __init__(self, fix_cols, target_col):
        self.fix_cols = fix_cols
        self.target_col = target_col
        self.num_cols = []
        self.cat_cols = []

        self.GENRE_MAPPING = {
            'Drama': 'Drama & Romantika', 'Krimi': 'Drama & Romantika', 'Mysteriózní': 'Drama & Romantika', 
            'Thriller': 'Drama & Romantika', 'Psychologický': 'Drama & Romantika', 'Romantický': 'Drama & Romantika',
            'Akční': 'Akce & Sci-Fi', 'Dobrodružný': 'Akce & Sci-Fi', 'Fantasy': 'Akce & Sci-Fi', 
            'Sci-Fi': 'Akce & Sci-Fi', 'Horor': 'Akce & Sci-Fi', 'Western': 'Akce & Sci-Fi', 'Válečný': 'Akce & Sci-Fi',
            'Komedie': 'Zábava & Reality', 'Zábavný': 'Zábava & Reality', 'Reality-TV': 'Zábava & Reality', 
            'Soutěžní': 'Zábava & Reality', 'Talk-show': 'Zábava & Reality', 'Hudební': 'Zábava & Reality',
            'Animovaný': 'Děti & Rodina', 'Pohádka': 'Děti & Rodina', 'Rodinný': 'Děti & Rodina',
            'Publicistický': 'Info & Dokumenty', 'Dokumentární': 'Info & Dokumenty'
        }

        self.ORIGIN_MAP = {
            # CZ / SK
            'Česko': 'CZ/SK', 'Slovensko': 'CZ/SK', 'Československo': 'CZ/SK', 
            'Protektorát Čechy a Morava': 'CZ/SK',
            
            # USA
            'USA': 'USA',
            
            # Evropa (Velcí hráči)
            'Velká Británie': 'Evropa_Big', 'Francie': 'Evropa_Big', 'Německo': 'Evropa_Big', 
            'Itálie': 'Evropa_Big', 'Španělsko': 'Evropa_Big', 'Východní Německo': 'Evropa_Big',
            
            # Evropa (Ostatní)
            'Rakousko': 'Evropa_Ostatni', 'Polsko': 'Evropa_Ostatni', 'Norsko': 'Evropa_Ostatni', 
            'Švédsko': 'Evropa_Ostatni', 'Dánsko': 'Evropa_Ostatni', 'Finsko': 'Evropa_Ostatni', 
            'Nizozemsko': 'Evropa_Ostatni', 'Belgie': 'Evropa_Ostatni', 'Irsko': 'Evropa_Ostatni', 
            'Malta': 'Evropa_Ostatni', 'Rusko': 'Evropa_Ostatni', 'Sovětský svaz': 'Evropa_Ostatni', 
            'Ukrajina': 'Evropa_Ostatni', 'Bulharsko': 'Evropa_Ostatni', 'Srbsko': 'Evropa_Ostatni', 
            'Lotyšsko': 'Evropa_Ostatni',
            
            # Anglosaský svět
            'Kanada': 'Anglosasky_svet', 'Austrálie': 'Anglosasky_svet', 'Nový Zéland': 'Anglosasky_svet',
            
            # Exotika / Ostatní
            'Čína': 'Exotika', 'Hongkong': 'Exotika', 'Japonsko': 'Exotika', 'Jižní Korea': 'Exotika', 
            'Tchaj-wan': 'Exotika', 'Indie': 'Exotika', 'Izrael': 'Exotika', 'Mexiko': 'Exotika', 
            'Argentina': 'Exotika', 'Kolumbie': 'Exotika', 'Malajsie': 'Exotika', 'Botswana': 'Exotika', 
            'Bahamy': 'Exotika', 'Papua-Nová Guinea': 'Exotika'
        }
        

    def add_cluster(self, df, source_col, target_col, mapping_dict):
            """
            Univerzální metoda pro seskupování hodnot. 
            Zachovává null hodnoty jako 'Neznámé' a mapuje zbytek.
            """
            if source_col in df.columns:
                # 1. Mapování podle slovníku (neznámé klíče se změní na NaN)
                mapped_series = df[source_col].map(mapping_dict)
                
                # 2. Rozlišení:
                # - Pokud byl původní záznam NaN -> 'Neznámé'
                # - Pokud původní záznam existoval, ale není ve slovníku -> 'Ostatní'
                # - Jinak ponecháme mapovanou hodnotu
                df[target_col] = np.where(
                    df[source_col].isna(), 
                    'Nefilmovy_obsah', 
                    mapped_series.fillna('Ostatní')
                )
            else:
                df[target_col] = 'Nefilmovy_obsah'
                
            return df
        

    def add_genre_clusters(self, df):
        """Groups individual genres (f_10) into broader categories to reduce noise."""
        if 'f_10' in df.columns:
            df['genre_group'] = df['f_10'].map(self.GENRE_MAPPING).fillna('Ostatní')
        else:
            df['genre_group'] = 'Ostatní'
        return df

    def transform_to_long(self, df):
        """Converts wide data (chX__) to long format."""
        consolidated_list = []
        unique_channels = df['channel_id'].unique()

        for chid in unique_channels:
            prefix = f'ch{chid}__'
            specific_cols = [col for col in df.columns if prefix in col]
            
            # Extract and rename columns for this specific channel
            temp_df = df.loc[df['channel_id'] == chid, self.fix_cols + specific_cols].copy()
            temp_df.columns = self.fix_cols + [col.replace(prefix, '') for col in specific_cols]
            
            consolidated_list.append(temp_df)

        return pd.concat(consolidated_list, ignore_index=True)

    def extract_time_features(self, df):
            """Creates numerical features from the datetime column, including Czech holidays."""
            # Ensure it's datetime
            df['timeslot_datetime_from'] = pd.to_datetime(df['timeslot_datetime_from'])
            
            # Initialize Czech holidays (CZ)
            cz_holidays = holidays.CZ()
            
            # Extract basic features
            df['hour'] = df['timeslot_datetime_from'].dt.hour
            df['day_of_week'] = df['timeslot_datetime_from'].dt.dayofweek
            df['month'] = df['timeslot_datetime_from'].dt.month
            df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
            
            # Add holiday feature (1 if it's a holiday, 0 if not)
            df['is_holiday'] = df['timeslot_datetime_from'].dt.date.apply(
                lambda x: 1 if x in cz_holidays else 0
            )
            
            # Optional: Combine weekend and holiday for a "free day" feature
            df['is_free_day'] = ((df['is_weekend'] == 1) | (df['is_holiday'] == 1)).astype(int)
            
            return df
    
    def add_content_type(self, df):
        """
        Distinguishes between movies, series and other content 
        using the decoded metadata (f_7, f_9, f_10).
        """
        # 1. Definujeme si filmové žánry (podle tvé tabulky ty s dlouhou stopáží)
        movie_genres = [
            'Akční', 'Dobrodružný', 'Drama', 'Fantasy', 'Historický', 
            'Horor', 'Katastrofický', 'Muzikál', 'Pohádka', 'Psychologický', 
            'Sci-Fi', 'Thriller', 'Válečný', 'Western', 'Životopisný'
        ]

        # 2. Vytvoření příznaku 'is_movie'
        # Podmínka: Patří do filmového žánru AND má délku (f_7) nad 60 min AND není to seriál (f_9 == 0)
        if all(col in df.columns for col in ['f_7', 'f_9', 'f_10']):
            df['is_movie'] = (
                (df['f_10'].isin(movie_genres)) & 
                (df['f_7'] > 60) & 
                (df['f_9'] == 0)
            ).astype(int)
            
            # 3. Bonus: Vytvoření příznaku 'is_series' (pro lepší predikci loyalty)
            df['is_series_content'] = (df['f_9'] == 1).astype(int)
            
            # 4. Bonus: Vytvoření příznaku 'is_short_news' (pro ty 15minutové bloky)
            df['is_short_content'] = (df['f_7'] <= 20).astype(int)
        else:
            # Fallback pokud sloupce chybí
            df['is_movie'] = 0
            df['is_series_content'] = 0
            df['is_short_content'] = 0
            
        return df
    
    def aggregate_by_ident(self, df):
        """
        Agreguje sloty do unikátních pořadů podle main_ident.
        """
        # Definujeme, jak se má která skupina sloupců chovat
        aggregation_logic = {
            'timeslot_datetime_from': 'min',    # Začátek pořadu
            'share_15_54': 'mean',              # Průměrný share
            'share_15_54_3mo_mean': 'first',    # Historický průměr (zůstává stejný)
            'channel_id': 'first',
            'day_of_week': 'first',
            'month': 'first',
            'hour':'first',
            'is_weekend': 'first',
            'is_holiday': 'first',
            'is_free_day': 'first',
            'is_movie': 'first',
            'is_series_content': 'first',
            'is_short_content': 'first',
            'genre_group': 'first',
            'origin_group': 'first',
            'f_2': 'first',
            'f_3': 'first',
            'f_4': 'first',
            'f_6': 'first',
        }

        # Najdeme i ostatní f_ sloupce, které v seznamu nejsou, a dáme jim 'first'
        for col in df.columns:
            if col.startswith('f_') and col not in aggregation_logic:
                aggregation_logic[col] = 'first'

        # Provedeme seskupení
        df_aggregated = df.groupby('main_ident', as_index=False).agg(aggregation_logic)
        
        return df_aggregated

    def run(self, df):
        # 1. Basic cleaning
        df = df.drop_duplicates().copy()
        
        # 2. Transform format
        df = self.transform_to_long(df)
        
        # 3. Add features
        df = self.extract_time_features(df)
        df = self.add_content_type(df)
        df = self.add_cluster(df, 'f_10', 'genre_group', self.GENRE_MAPPING)
        df = self.add_cluster(df, 'f_11', 'origin_group', self.ORIGIN_MAP)

        # 4. Final cleaning of the long format
        # df = df.fillna(0)
        df = self.aggregate_by_ident(df)
        return df

