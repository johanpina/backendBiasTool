
import matplotlib.pyplot as plt
from aequitas.plotting import Plot

class PlotES(Plot):
    def __init__(self):
        super().__init__()
        self.translations = {
            "Group Value": "Valor del Grupo",
            "Group": "Grupo",
            "Value": "Valor",
            "Disparity": "Disparidad",
            "False Negative Rate": "Tasa de Falsos Negativos",
            "False Positive Rate": "Tasa de Falsos Positivos",
            "True Positive Rate": "Tasa de Verdaderos Positivos",
            "True Negative Rate": "Tasa de Verdaderos Negativos",
            "Accuracy": "Precisión",
            "Score": "Puntaje",
            "Label Value": "Valor Real",
        }

    def translate(self, text):
        return self.translations.get(text, text)

    def plot_group_metric(self, xtab, metric, ax=None, **kwargs):
        # Usamos el método original para construir la gráfica
        fig = super().plot_group_metric(xtab, metric, ax=ax, **kwargs)

        ax = plt.gca()

        # Traducir título, ejes, etiquetas
        ax.set_title(self.translate(ax.get_title()))
        ax.set_xlabel(self.translate(ax.get_xlabel()))
        ax.set_ylabel(self.translate(ax.get_ylabel()))

        # Traducir ticks del eje x
        ax.set_xticklabels([self.translate(label.get_text()) for label in ax.get_xticklabels()])

        # Traducir leyenda si existe
        legend = ax.get_legend()
        if legend:
            for text in legend.get_texts():
                text.set_text(self.translate(text.get_text()))

        return fig