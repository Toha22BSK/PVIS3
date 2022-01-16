from mpi4py import MPI
import numpy, csv, copy
import math, random
import matplotlib.pyplot as plt
from datetime import datetime

K_MEANS_LAUNCH = 20000
CLASS_INDEX = 5
FILTER_DIM = (CLASS_INDEX, )

data_MAP = {
    "very_low": 0,
    "Low": 0.33,
    "Middle": 0.66,
    "High": 1
}
dataLen = len(data_MAP)


def readCSV():
    dataKnowledge = []
    with open("Knowledge.csv") as f:
        dataKnowledge = [row for row in csv.reader(f, delimiter=';')]
    dataKnowledge.pop(0)
    for i,value in enumerate(dataKnowledge):
       value[-1] = data_MAP[value[-1]]  
    return numpy.array(dataKnowledge, dtype=float).tolist()

def formatOutput(data):
    result = ""
    for element in data:
        lenStr = (10 -len(str(element))) // 2
        result+= " " * lenStr + str(element) + " " * lenStr
    return result

def saveResult(data):
    centroids = data["centroids"]
    clusters = data["clusters"]
    index = data["mb-index"]

    with open("result.txt", "w") as fd:
        fd.write("MB-Index: {:^1.5f}\n\n".format(index))

        for i in range(len(clusters)):
            fd.write("**" + formatOutput(centroids[i]) + "**\n")

            for x in clusters[i]:
                fd.write("  " + formatOutput(x) + "\n")
            fd.write("\n\n")


def distance(x, y):
    dist = sum([(value - y[i]) ** 2 for i, value in enumerate(x) if i not in FILTER_DIM])
    return math.sqrt(dist)


def average(data):
    dim = len(data[0])
    avg_sum = [0.0] * dim
    lenData = len(data)
    for p in data:
        for i in range(dim):
            avg_sum[i] += p[i]
    for i in range(dim):
        avg_sum[i] /= len(data)
    return avg_sum



def addRandom(a, b, N):
    result = set()
    while len(result) != N:
        result.add(random.randint(a, b))
    return result


def k_means(dataset, k, max_iteration=1000, tolerance=0.0001):
    # выбираем начальные центры кластеров
    centroids = [dataset[i] for i in addRandom(0, len(dataset) - 1, k)]
    assert len(centroids) == k

    for _ in range(max_iteration):
        # на каждой итерации пересобираем все кластеры
        clusters = []
        for _ in range(k):
            clusters.append([])

        # определяем для каждой точки ближайший центр кластера
        for point in dataset:
            distances = [distance(point, centroid) for centroid in centroids]
            cluster_number = distances.index(min(distances))
            clusters[cluster_number].append(point)

        previous_centroids = copy.deepcopy(centroids)

        # для каждого кластера вычисляем новый центр
        for i in range(k):
            centroids[i] = average(clusters[i])

        # если центры кластеров сдвинулись незначительно, то выходим
        if all([distance(centroids[i], previous_centroids[i]) < tolerance for i in range(k)]):
            break

    return centroids, clusters


# Индекс Маулика Бундефая
def Maulik_Bandoypadhyay_index(centroids, clusters):
    centroidsLen = len(centroids)
    dim = len(centroids[0])

    #нахождение центрального элемента
    N = 0
    X_centre = [0] * dim
    for c in clusters:
        N += len(c)
        for p in c:
            for i in range(dim):
                X_centre[i] += p[i]
    for i in range(dim):
        X_centre[i] /= N
    
    #нахождение сумма расстояний от каждого элемента до центрального E1
    Sum_distance = 0
    for c in clusters:
        for p in c:
            Sum_distance += distance(p, X_centre)

    #нахождение суммы расстояний между центрами кластеров Ec
    Sum_distance_in_clusters = 0
    for cluster in range(centroidsLen):
        for x in clusters[cluster]:
            Sum_distance_in_clusters += distance(centroids[cluster], x)

    #максимальное расстояние между кластерами
    max_dist = distance(centroids[0], centroids[1])
    for i in range(centroidsLen):
        for j in range(centroidsLen):
            if i == j:
                continue
            dist = distance(centroids[i], centroids[j])
            max_dist = max(max_dist, dist)

    #расчёт самого индекса
    Maulik_Bandoypadhyay = (1 / len(clusters) * Sum_distance / Sum_distance_in_clusters * max_dist) ** 2

    return Maulik_Bandoypadhyay


def main_mpi():
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    
    if rank == 0:
        ds = readCSV()
        reqs = [comm.isend(ds, i, tag=1) for i in range(1, size)]
        for r in reqs:
            r.wait()
    else:
        req = comm.irecv(source=0, tag=1)
        ds = req.wait()

    result = None
    for _ in range(K_MEANS_LAUNCH):
        centroids, clusters = k_means(ds, dataLen, max_iteration=1000)
        index = Maulik_Bandoypadhyay_index(centroids, clusters)

        # MB-index чем больше, тем лучше
        if result is None or index > result["mb-index"]:
            result = {
                "mb-index": index,
                "centroids": centroids,
                "clusters": clusters
            }    

    index_max = comm.reduce(result["mb-index"], op=MPI.MAX, root=0)
    index_max = comm.bcast(index_max, root=0)

    if index_max == result["mb-index"]:
        print(f"Saving cluster in process {rank}")
        saveResult(result)

    print(f"process {rank} exit.")
    comm.barrier()
    MPI.Finalize()
    
start_time = datetime.now()
if __name__ == "__main__":
    main_mpi()
print(datetime.now() - start_time)
