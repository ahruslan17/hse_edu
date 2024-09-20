import pandas as pd
import matplotlib.pyplot as plt

def open_logs(path: str) -> pd.DataFrame:
    file = pd.read_csv(path)
    return file

def plot_performance_metrics(data: pd.DataFrame, filename: str):
    """
    Функция для построения графиков зависимости GFLOPS и времени выполнения
    от количества потоков для разных размеров матриц.
    
    Parameters:
        data (pd.DataFrame): DataFrame, содержащий колонки ["Threads", "Size", "Time", "GFLOPS"]
        filename (str): Имя файла для сохранения графиков

    """
    sizes = data["Size"].unique()

    # Настройка стиля графиков
    # plt.style.use('seaborn')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

    # Первый график: Зависимость GFLOPS от количества потоков
    for size in sizes:
        subset = data[data['Size'] == size]
        ax1.plot(subset['Threads'], subset['GFLOPS'], label=f'Size {size}', marker='o')
    ax1.set_title('Performance in GFLOPS vs Number of Threads', fontsize=15)
    ax1.set_xlabel('Number of Threads', fontsize=12)
    ax1.set_ylabel('GFLOPS', fontsize=12)
    ax1.legend(title='Matrix Size', fontsize=11)
    ax1.grid(True)

    # Второй график: Зависимость времени выполнения от количества потоков
    for size in sizes:
        subset = data[data['Size'] == size]
        ax2.plot(subset['Threads'], subset['Time'], label=f'Size {size}', linestyle='--', marker='o')
    ax2.set_title('Execution Time vs Number of Threads', fontsize=15)
    ax2.set_xlabel('Number of Threads', fontsize=12)
    ax2.set_ylabel('Time (seconds)', fontsize=12)
    ax2.legend(title='Matrix Size', fontsize=11)
    ax2.grid(True)

    # Отображение графиков
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()

def main() -> None:
    logfile = open_logs(path="/home/ahruslan/hse_edu/HPC/HW1/out/output_2.csv")
    print(logfile)
    plot_performance_metrics(logfile, filename="/home/ahruslan/hse_edu/HPC/HW1/out/plots")

if __name__ == "__main__":
    main()