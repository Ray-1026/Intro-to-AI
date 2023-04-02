import csv
import heapq

edgeFile = "edges.csv"
heuristicFile = "heuristic.csv"


def astar_time(start, end):
    # Begin your code (Part 6)
    """
    Read and process the data from the csv files same as the part in 'astar.py'.
    Difference :
    1. Record the speed limit and maximal speed limit, change them from kh/hr to m/s. Then 'distance'
       divided by its speed limit is the edge cost.
    2. After finishing 'edges.csv', we obtain the maximal speed limit, so the new heuristic function
       is the estimated distance from heuristic divided by max speed limit, which is sure to be an
       admissible heuristic function since the new heuristic value is always smaller than the actual
       cost.
    """
    edge = {}
    heauristic = {}
    max_speed = 0
    with open(edgeFile) as file1, open(heuristicFile) as file2:
        csv1, csv2 = list(csv.reader(file1)), list(csv.reader(file2))
        csv1.pop(0)
        for i in csv1:
            num1, num2, distance, speed = int(i[0]), int(i[1]), float(i[2]), float(i[3])
            speed = speed * 1000 / 3600
            max_speed = max(max_speed, speed)
            if num1 not in edge:
                edge[num1] = [(num2, float(distance / speed))]
            else:
                edge[num1].append((num2, float(distance / speed)))
            if num2 not in edge:
                edge[num2] = []
        idx = next((i for i in range(1, 4) if csv2[0][i] == str(end)), None)
        csv2.pop(0)
        for i in csv2:
            heauristic[int(i[0])] = float(i[idx]) / max_speed
    """
    Implement A* search same as the part in 'astar.py'.
    """
    parent = {}
    heap = [(heauristic[start], start, None)]
    heapq.heapify(heap)
    visited = set()
    num_visited, isfind, time = 0, 0, 0
    while heap:
        (cost, node, p) = heapq.heappop(heap)
        cost -= heauristic[node]
        if node not in visited:
            num_visited += 1
            visited.add(node)
            parent[node] = p
            if node == end:
                isfind, time = 1, cost
                break
            for i, j in edge[node]:
                heapq.heappush(heap, (cost + j + heauristic[i], i, node))
    """
    Find the path same as the part in 'astar.py'.
    """
    path = []
    if isfind:
        path.append(end)
        while path[-1] != start:
            path.append(parent[path[-1]])
        path.reverse()
    return path, time, num_visited
    # End your code (Part 6)


if __name__ == "__main__":
    path, time, num_visited = astar_time(2270143902, 1079387396)
    print(f"The number of path nodes: {len(path)}")
    print(f"Total second of path: {time}")
    print(f"The number of visited nodes: {num_visited}")
