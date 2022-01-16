import csv
import math
import copy
import random
import numpy
import matplotlib.pyplot as plt
from mpi4py import MPI

Knowledge_map = {
    "very_low": 0,
    "Low": 0.33,
    "Middle": 0.66,
    "High": 1
}


K_MEANS_LAUNCH = 20000
CLASS_INDEX = 5
FILTER_DIM = (CLASS_INDEX, )

lenMap = len(Knowledge_map)

# Считывание CSV файла
def readCSV():
    dataKnowledge = []
    with open("Knowledge.csv") as f:
        dataKnowledge = [row for row in csv.reader(f, delimiter=';')]
    dataKnowledge.pop(0)
    for i,value in enumerate(dataKnowledge):
       value[-1] = Knowledge_map[value[-1]]  
    return numpy.array(dataKnowledge, dtype=float).tolist()


def save_cluster(data):
    centroids = data["centroids"]
    clusters = data["clusters"]
    index = data["mb-index"]

    with open("result_ds1.res", "w") as fd:
        fd.write(f"MB-Index: {index}\n\n")

        for i in range(len(clusters)):
            fd.write(f"\n------------------------------------\n")
            fd.write(f"{centroids[i]}")
            fd.write(f"\n------------------------------------\n")

            for x in clusters[i]:
                fd.write(f"{x}\n")
            fd.write(f"====================================\n\n\n")


def calc_distance(x, y):
    dist = sum([(value - y[i]) ** 2 for i, value in enumerate(x) if i not in FILTER_DIM])
    return math.sqrt(dist)


def average(data):
    return numpy.mean(data, axis = 0)

def take_unique_random(a, b, N):
    result = set()
    while len(result) != N:
        result.add(random.randint(a, b))
    return result

def k_means(data, k, max_iteration=1000, tolerance=0.0001):
    centers = [data[i] for i in take_unique_random(0, len(data) - 1, k)]
    assert len(centers) == k

    for null in range(max_iteration):
        clusters = []
        for null in range(k):
            clusters.append([])

        for point in data:
            distances = [calc_distance(point, center) for center in centers]
            cluster_number = distances.index(min(distances))
            clusters[cluster_number].append(point)

        previous_centers = copy.deepcopy(centers)

        for i in range(k):
            centers[i] = average(clusters[i])

        if all([calc_distance(centers[i], previous_centers[i]) < tolerance for i in range(k)]):
            break

    return centers, clusters


# def DBI(X, M, R):

#     K, D = M.shape

#     sigma = numpy.zeros(K)

#     for k in range(K):
#         diffs = X – M[k]
#         squared_distances = (diffs * diffs).sum(axis=1)

#         weighted_squared_distances = R[:,k]*squared_distances

#         sigma[k] = numpy.sqrt( weighted_squared_distances).mean()

#     dbi = 0

#     for k in range(K):

#         max_ratio = 0

#         for j in range(K):

#             if k != j:

#                 numerator = sigma[k] + sigma[j]

#                 denominator = numpy.linalg.norm(M[k] – M[j])

#                 ratio = numerator / denominator

#                 if ratio > max_ratio:

#                     max_ratio = ratio

#         dbi += max_ratio

#     return dbi / K


# Индекс Маулика Бундефая
def Maulik_Bandoypadhyay_index(centroids, clusters):
    K = len(centroids)
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
            Sum_distance += calc_distance(p, X_centre)

    #нахождение суммы расстояний между центрами кластеров Ec
    Sum_distance_in_clusters = 0
    for cluster in range(K):
        for x in clusters[cluster]:
            Sum_distance_in_clusters += calc_distance(centroids[cluster], x)

    #максимальное расстояние между кластерами
    max_dist = calc_distance(centroids[0], centroids[1])
    for i in range(K):
        for j in range(K):
            if i == j:
                continue
            dist = calc_distance(centroids[i], centroids[j])
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
        print('\r', 'выполнение:', (_//K_MEANS_LAUNCH*100)*'|', "%.1f" % (_/K_MEANS_LAUNCH), '%', end='')
        centroids, clusters = k_means(ds, lenMap, max_iteration=1000)
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
        save_cluster(result)

    print(f"process {rank} exit.")
    comm.barrier()
    MPI.Finalize()

if __name__ == "__main__":
    main_mpi()
