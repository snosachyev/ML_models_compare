import numpy as np

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, confusion_matrix

import seaborn as sns


# === 4. Предсказания и точность ===
def predict_and_accuracy(model, X_test, y_test):
    y_pred = model.predict(X_test)

    if isinstance(model, LinearRegression):
        y_pred = np.round(y_pred).astype(int)
        # ограничим значения в допустимых пределах классов
        y_pred = np.clip(y_pred, y_test.min(), y_test.max())

    return accuracy_score(y_test, y_pred)


def plot_models(
    results,
    X_test,
    y_test,
    X,
    xx,
    yy,
    grid=None,
    feature_names=("X1", "X2"),
    cmaps=None,
    class_names=None,
    save_path=None,
    dpi=300,
    show_legend=True,
    per_plot_legend=False,
    show_confusion=False
):
    """Визуализирует границы решений, легенды и матрицы ошибок для набора моделей."""

    unique_classes = np.unique(y_test)
    n_classes = len(unique_classes)
    if class_names is None:
        class_names = [str(c) for c in unique_classes]

    # --- 2️⃣ Настройка сетки подграфиков ---
    n_models = len(results)
    cols = 2 if n_models > 1 else 1
    rows = int(np.ceil(n_models / cols))
    fig_height = 4.5 * rows * (1.4 if show_confusion else 1)
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, fig_height))
    axes = np.array(axes).ravel()

    # --- 3️⃣ Цветовые карты ---
    default_cmaps = ["viridis", "plasma", "coolwarm", "Spectral", "winter", "autumn"]
    if cmaps is None:
        cmaps = [default_cmaps[i % len(default_cmaps)] for i in range(n_models)]
    elif isinstance(cmaps, str):
        cmaps = [cmaps] * n_models
    elif len(cmaps) < n_models:
        cmaps = [cmaps[i % len(cmaps)] for i in range(n_models)]

    # --- 4️⃣ Основной цикл ---
    for i, (ax, (name, (model, acc))) in enumerate(zip(axes, results.items())):
        cmap_name = cmaps[i]
        cmap = plt.get_cmap(cmap_name, n_classes)

        # Предсказания для сетки и теста
        Z = model.predict(grid)
        # 💡 некоторые модели возвращают вероятности
        if np.ndim(Z) > 1 and Z.shape[1] > 1:
            Z = np.argmax(Z, axis=1)
        Z = Z.reshape(xx.shape)

        y_pred = model.predict(X_test)
        if np.issubdtype(y_pred.dtype, np.floating):
            if y_pred.ndim > 1 and y_pred.shape[1] > 1:
                y_pred = np.argmax(y_pred, axis=1)
            else:
                y_pred = (y_pred > 0.5).astype(int)

        # Отрисовка границ
        ax.contourf(xx, yy, Z, alpha=0.3, cmap=cmap)
        ax.scatter(
            X_test[:, 0],
            X_test[:, 1],
            c=y_test,
            cmap=cmap,
            edgecolor='k',
            s=40
        )

        ax.set_title(f"{name}\nТочность: {acc:.2f}", fontsize=11, pad=8)
        ax.set_xlabel(feature_names[0])
        ax.set_ylabel(feature_names[1])
        ax.set_xticks([])
        ax.set_yticks([])

        # --- Локальная легенда ---
        if per_plot_legend:
            handles = [
                plt.Line2D(
                    [0], [0],
                    marker='o',
                    color='w',
                    markerfacecolor=cmap(j),
                    markeredgecolor='k',
                    markersize=8,
                    label=class_names[j]
                )
                for j in range(n_classes)
            ]
            ax.legend(handles=handles, title="Классы", loc="upper right", fontsize=9)

        # --- Матрица ошибок ---
        if show_confusion:
            cm = confusion_matrix(y_test, y_pred, labels=unique_classes)
            inset_ax = ax.inset_axes([0.65, 0.05, 0.3, 0.3])  # внутри графика
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cbar=False,
                cmap="Blues",
                xticklabels=class_names,
                yticklabels=class_names,
                ax=inset_ax
            )
            inset_ax.set_xlabel("Предсказано", fontsize=7)
            inset_ax.set_ylabel("Истинно", fontsize=7)
            inset_ax.tick_params(labelsize=6)

    # --- 5️⃣ Убираем пустые графики ---
    for ax in axes[n_models:]:
        ax.axis("off")

    # --- 6️⃣ Общая легенда ---
    if show_legend:
        cmap = plt.get_cmap(cmaps[0], n_classes)
        handles = [
            plt.Line2D(
                [0], [0],
                marker='o',
                color='w',
                markerfacecolor=cmap(j),
                markeredgecolor='k',
                markersize=8,
                label=class_names[j]
            )
            for j in range(n_classes)
        ]
        fig.legend(
            handles=handles,
            title="Классы",
            loc="upper center",
            ncol=len(class_names),
            fontsize=10,
            title_fontsize=11,
            frameon=False,
            bbox_to_anchor=(0.5, 1.05)
        )

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    # --- 7️⃣ Сохранение или показ ---
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"✅ График сохранён в файл: {save_path}")
    else:
        plt.show()


def get_coordianates_griid_2D(X):
    # Создаем сетку координат для построения границ
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 250),
        np.linspace(y_min, y_max, 250)
    )
    return xx, yy


def get_coordianates_greed(X):
    # Создаем сетку координат для построения границ
    x_min, x_max = X[:, [0,2]].min() - 0.5, X[:, [0, 2]].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx_l, yy_l, xx_r, yy_r = np.meshgrid(
        np.linspace(x_min_l, x_max_l, 250),
        np.linspace(y_min_l, y_max_l, 250),
        np.linspace(x_min_r, x_max_r, 250),
        np.linspace(y_min_r, y_max_r, 250)
    )
    return xx_l, yy_l, xx_r, yy_r


def get_grid(X, xx, yy):
    # Для проекции фиксируем средние значения остальных признаков
    X_mean = X.mean(axis=0)
    grid = np.c_[xx.ravel(), yy.ravel(),
             np.full(xx.ravel().shape, X_mean[0]),
             np.full(xx.ravel().shape, X_mean[1])]
    return grid


def get_grid_2D(X, xx, yy):
    # Для проекции фиксируем средние значения остальных признаков
    
    return np.c_[xx.ravel(), yy.ravel()]

