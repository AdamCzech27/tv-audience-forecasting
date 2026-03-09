import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

class TVVisualizer:
    def __init__(self, df):
        self.df = df.copy()

        self.df['day_name'] = self.df['timeslot_datetime_from'].dt.day_name()
        self.days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        # Základní nastavení stylu pro všechny grafy
        sns.set_theme(style="whitegrid")
        
    def plot_correlation_matrix(self):
            """Vykreslí heatmapu korelací bez číselných popisků."""
            numeric_df = self.df.select_dtypes(include=[np.number])
            full_corr = numeric_df.corr()

            plt.figure(figsize=(12, 10))
            # Změna: annot=False odstraní čísla z políček
            sns.heatmap(full_corr, 
                        annot=False,          
                        cmap='coolwarm',    
                        center=0,            
                        linewidths=0.5)

            plt.title('Korelační matice numerických proměnných', fontsize=15)
            plt.show()

    def plot_rolling_share(self, window=30):
        """Vykreslí vyhlazený vývoj share pro všechny kanály."""
        pivot_df = self.df.pivot(index='timeslot_datetime_from', 
                                 columns='channel_id', 
                                 values='share_15_54')
        monthly_rolling = pivot_df.rolling(window=window, center=True, min_periods=1).mean()

        plt.figure(figsize=(14, 7))
        for channel in monthly_rolling.columns:
            plt.plot(monthly_rolling.index, monthly_rolling[channel], 
                     label=f'Kanál {channel} ({window}d průměr)', linewidth=2)

        plt.title(f'{window}denní klouzavý průměr share_15_54 (vyhlazeno)', fontsize=15)
        plt.ylabel('Share (%)')
        plt.grid(True, alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()

    def plot_content_type_profile(self):
        """Srovnání profilů kanálů: Seriálový vs. Neseriálový obsah."""
        pivot_df = pd.crosstab(self.df['channel_id'], self.df['is_series_content'], normalize='index') * 100
        pivot_df = pivot_df.rename(columns={0: 'Neseriálový obsah', 1: 'Seriálový obsah'})
        
        # Seřazení a barvy
        sorted_columns = pivot_df.mean().sort_values(ascending=False).index
        pivot_df = pivot_df[sorted_columns]
        colors = ['#2a9d8f', '#e9c46a']
        
        self._plot_stacked_bar(pivot_df, 'Srovnání profilů: Seriálový vs. Neseriálový obsah', colors)

    def plot_origin_profile(self):
        """Srovnání profilů kanálů podle původu obsahu (origin_group)."""
        pivot_df = pd.crosstab(self.df['channel_id'], self.df['origin_group'], normalize='index') * 100
        
        sorted_columns = pivot_df.mean().sort_values(ascending=False).index
        pivot_df = pivot_df[sorted_columns]
        colors = ['#264653', '#2a9d8f', '#e9c46a', '#f4a261', '#e76f51', '#8ab17d', '#babbc1']
        
        self._plot_stacked_bar(pivot_df, 'Srovnání profilů podle původu (Origin Group)', colors)

    def _plot_stacked_bar(self, pivot_df, title, colors):
        """Interní pomocná metoda pro vykreslování skládaných sloupcových grafů s popisky."""
        ax = pivot_df.plot(kind='bar', stacked=True, figsize=(12, 8), color=colors, width=0.7)
        plt.title(title, fontsize=16)
        plt.xlabel('ID Kanálu', fontsize=12)
        plt.ylabel('Podíl ve vysílání (%)', fontsize=12)
        plt.xticks(rotation=0)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        for n, x in enumerate(pivot_df.index):
            cumulative_sum = 0
            for col in pivot_df.columns:
                proportion = pivot_df.loc[x, col]
                if proportion > 4: 
                    plt.text(x=n, y=cumulative_sum + (proportion / 2),
                             s=f'{proportion:.1f}%', 
                             color="white", fontsize=10, fontweight="bold", ha="center", va="center")
                cumulative_sum += proportion
        plt.tight_layout()
        plt.show()

    def plot_category_impact(self):
        """Vykreslí vliv žánrových a původových skupin na share_15_54 (One-Hot Encoding)."""
        # Výběr relevantních sloupců
        df_groups = self.df[['genre_group', 'origin_group', 'share_15_54']]
        
        # One-Hot Encoding pro kategorické proměnné
        df_encoded = pd.get_dummies(df_groups, columns=['genre_group', 'origin_group'])
        
        # Výpočet korelací a seřazení podle share_15_54
        group_corr = df_encoded.corr()
        share_correlations = group_corr[['share_15_54']].sort_values(by='share_15_54', ascending=False)

        plt.figure(figsize=(10, 8))
        sns.heatmap(share_correlations, annot=True, cmap='RdYlGn', center=0)
        plt.title('Vliv žánrových a původových skupin na share_15_54', fontsize=15)
        plt.show()


    def plot_channel_heatmap(self,channel_id):
        subset = self.df[self.df['channel_id'] == channel_id]
        
        # Agregace dat: průměrný share pro každou hodinu a den
        heatmap_data = subset.pivot_table(
            index='day_name', 
            columns='hour', 
            values='share_15_54', 
            aggfunc='mean'
        ).reindex(self.days_order)

        plt.figure(figsize=(15, 8))
        sns.heatmap(heatmap_data, annot=True, cmap='YlGnBu', fmt=".1f")
        plt.title(f'Heatmapa sledovanosti (share_15_54) - Kanál {channel_id}')
        plt.xlabel('Hodina dne')
        plt.ylabel('Den v týdnu')
        plt.show()
